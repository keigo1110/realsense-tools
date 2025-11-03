"""
Dual RealSense Camera Recording Tool

キャリブレーションデータを使用して2台のRealSenseカメラを同期録画します。

使い方:
    python record_dual_cameras.py
    python record_dual_cameras.py --calibration-file calibration.json --output my_recording
"""

import argparse
import json
import os
import sys
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs

# ===== 設定 =====
DEFAULT_CALIB_FILE = "extrinsics_cam1_to_cam0.json"
STREAM_W, STREAM_H, FPS = 640, 480, 30
PAIRING_TOL_MS = 25.0  # 近傍ペアリングの許容[ms]
QUEUE_SIZE = 2  # 遅延低減用


# ===== フレームペアリング =====
class FramePairer:
    """最近傍時刻でフレームをペアリング"""

    def __init__(self, tol_ms=25.0):
        self.buf0, self.buf1 = deque(maxlen=5), deque(maxlen=5)
        self.tol = tol_ms

    @staticmethod
    def ts(f):
        """フレームのタイムスタンプを取得（グローバルタイム有効時はホスト基準）"""
        return f.get_timestamp()

    def put_and_match(self, f0=None, f1=None):
        """フレームを追加して、ペアが揃ったら返す"""
        if f0 is not None:
            self.buf0.append(f0)
        if f1 is not None:
            self.buf1.append(f1)

        if not self.buf0 or not self.buf1:
            return None

        f0c = self.buf0[0]
        t0 = self.ts(f0c)

        best = None
        best_dt = 1e9
        best_i = -1

        for i, g in enumerate(self.buf1):
            dt = abs(self.ts(g) - t0)
            if dt < best_dt:
                best_dt, best, best_i = dt, g, i

        if best_dt <= self.tol:
            self.buf0.popleft()
            del self.buf1[best_i]
            return f0c, best, best_dt

        # 古い方を進める
        if self.ts(self.buf0[0]) < self.ts(self.buf1[0]):
            self.buf0.popleft()
        else:
            self.buf1.popleft()

        return None


# ===== RealSense初期化 =====
def set_sensor_options(dev):
    """RealSenseセンサーのオプションを設定"""
    for s in dev.query_sensors():
        # グローバルタイム（ホスト補正）
        if rs.option.global_time_enabled in s.get_supported_options():
            try:
                s.set_option(rs.option.global_time_enabled, 1.0)
            except Exception:
                pass
        # フレームキュー（遅延抑制）
        if rs.option.frames_queue_size in s.get_supported_options():
            try:
                s.set_option(rs.option.frames_queue_size, float(QUEUE_SIZE))
            except Exception:
                pass


def start_pipeline(serial, record_file=None):
    """RealSenseパイプラインを開始"""
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, STREAM_W, STREAM_H, rs.format.rgb8, FPS)
    cfg.enable_stream(rs.stream.depth, STREAM_W, STREAM_H, rs.format.z16, FPS)

    # 録画ファイルを設定（指定されている場合）
    if record_file:
        cfg.enable_record_to_file(record_file)

    pipe = rs.pipeline()
    prof = pipe.start(cfg)
    align = rs.align(rs.stream.color)

    return pipe, prof, align


def enumerate_two_devices():
    """2台のRealSenseデバイスを列挙"""
    ctx = rs.context()
    devs = ctx.query_devices()

    if len(devs) < 2:
        raise RuntimeError(
            "2台のRealSenseが見つかりません。USB3.0/電源を確認してください。"
        )

    dev0, dev1 = devs[0], devs[1]

    for d in (dev0, dev1):
        set_sensor_options(d)

    sn0 = dev0.get_info(rs.camera_info.serial_number)
    sn1 = dev1.get_info(rs.camera_info.serial_number)

    return (dev0, dev1), (sn0, sn1)


# ===== メイン =====
def main():
    parser = argparse.ArgumentParser(
        description="Dual RealSense Camera Recording Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用方法:
  1. 事前に calibrate_dual_cameras.py を実行してキャリブレーションを保存
  2. このスクリプトを実行して録画を開始
  3. ESCキーまたは 'q'キーで録画を停止

注意:
  - 2台のカメラは同期して録画されます（最近傍時刻でペアリング）
  - 各カメラのストリームは個別のbagファイルに記録されます
  - キャリブレーションデータを含むメタデータJSONが生成されます
        """,
    )

    parser.add_argument(
        "--calibration-file",
        type=str,
        default=DEFAULT_CALIB_FILE,
        help=f"キャリブレーションファイルのパス（デフォルト: {DEFAULT_CALIB_FILE}）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="NAME",
        help="録画ファイル名（指定しない場合は実行日時で自動生成）",
    )
    parser.add_argument(
        "--pairing-tol",
        type=float,
        default=PAIRING_TOL_MS,
        help=f"フレームペアリングの許容時間[ms]（デフォルト: {PAIRING_TOL_MS}）",
    )

    args = parser.parse_args()

    # キャリブレーションファイルの確認
    if not os.path.exists(args.calibration_file):
        print(
            f"[ERROR] Calibration file not found: {args.calibration_file}",
            file=sys.stderr,
        )
        print(
            f"[ERROR] Please run calibrate_dual_cameras.py first to generate calibration data.",
            file=sys.stderr,
        )
        sys.exit(1)

    # キャリブレーションデータを読み込む
    try:
        with open(args.calibration_file, "r") as f:
            calib_data = json.load(f)
        T1_to_0 = np.array(calib_data["T1_to_0"], dtype=np.float64)
        print(f"[INFO] Loaded calibration from: {args.calibration_file}")
        print(
            f"[INFO] Calibration fitness: {calib_data.get('fitness', 'N/A'):.3f}, RMSE: {calib_data.get('inlier_rmse', 'N/A'):.4f}m"
        )
    except Exception as e:
        print(
            f"[ERROR] Failed to load calibration file: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    # 録画ディレクトリの作成
    record_dir = "bagdata"
    os.makedirs(record_dir, exist_ok=True)

    # ファイル名の決定
    if args.output:
        base_name = args.output
        if base_name.endswith(".bag"):
            base_name = base_name[:-4]
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"dual_recording_{timestamp}"

    record_file0 = os.path.join(record_dir, f"{base_name}_cam0.bag")
    record_file1 = os.path.join(record_dir, f"{base_name}_cam1.bag")
    metadata_file = os.path.join(record_dir, f"{base_name}_metadata.json")

    print("=" * 60)
    print("Dual RealSense Camera Recording")
    print("=" * 60)
    print(f"Output files:")
    print(f"  - Camera 0: {record_file0}")
    print(f"  - Camera 1: {record_file1}")
    print(f"  - Metadata: {metadata_file}")
    print()

    try:
        # 2台のデバイスを検出
        (dev0, dev1), (sn0, sn1) = enumerate_two_devices()
        print(f"[INFO] Device 0: {sn0}")
        print(f"[INFO] Device 1: {sn1}")

        # パイプラインを開始（録画ファイルも設定）
        pipe0, prof0, align0 = start_pipeline(sn0, record_file0)
        pipe1, prof1, align1 = start_pipeline(sn1, record_file1)

        print(f"[INFO] Recording started...")
        print(f"[INFO] Press ESC or 'q' to stop recording")

        pairer = FramePairer(args.pairing_tol)
        frame_count = 0
        paired_count = 0
        skipped_count = 0

        while True:
            # フレームを取得
            frames0 = pipe0.wait_for_frames()
            frames1 = pipe1.wait_for_frames()

            fs0 = align0.process(frames0)
            fs1 = align1.process(frames1)

            # ペアリング
            match = pairer.put_and_match(fs0, fs1)

            if match is None:
                skipped_count += 1
                # 表示のみ（録画は続行）
                if frame_count % 10 == 0:
                    f0 = fs0
                    c0 = np.asanyarray(f0.get_color_frame().get_data())
                    c0_bgr = cv2.cvtColor(c0, cv2.COLOR_RGB2BGR)

                    c1 = np.asanyarray(fs1.get_color_frame().get_data())
                    c1_bgr = cv2.cvtColor(c1, cv2.COLOR_RGB2BGR)

                    cv2.putText(
                        c0_bgr,
                        "WAITING FOR PAIR...",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 165, 255),
                        2,
                    )
                    cv2.putText(
                        c1_bgr,
                        "WAITING FOR PAIR...",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 165, 255),
                        2,
                    )

                    cv2.imshow("cam0", c0_bgr)
                    cv2.imshow("cam1", c1_bgr)
                continue

            f0, f1, dt = match
            frame_count += 1
            paired_count += 1

            # フレームを取得して表示
            c0 = np.asanyarray(f0.get_color_frame().get_data())
            d0 = np.asanyarray(f0.get_depth_frame().get_data())
            c1 = np.asanyarray(f1.get_color_frame().get_data())
            d1 = np.asanyarray(f1.get_depth_frame().get_data())

            c0_bgr = cv2.cvtColor(c0, cv2.COLOR_RGB2BGR)
            c1_bgr = cv2.cvtColor(c1, cv2.COLOR_RGB2BGR)

            # 録画状態を表示
            status_text = f"RECORDING | Frames: {frame_count} | Paired: {paired_count} | Skipped: {skipped_count}"
            cv2.putText(
                c0_bgr,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                c0_bgr,
                f"Pair dt: {dt:.1f}ms",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                c0_bgr,
                "Press ESC or 'q' to stop",
                (10, c0_bgr.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            cv2.putText(
                c1_bgr,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                c1_bgr,
                f"Pair dt: {dt:.1f}ms",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                c1_bgr,
                "Press ESC or 'q' to stop",
                (10, c1_bgr.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            cv2.imshow("cam0", c0_bgr)
            cv2.imshow("cam1", c1_bgr)

            # キー入力処理
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                print("\n[INFO] Stopping recording...")
                break

    except KeyboardInterrupt:
        print("\n\n[INFO] Recording interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        cv2.destroyAllWindows()

        # パイプラインを停止
        if "pipe0" in locals():
            pipe0.stop()
        if "pipe1" in locals():
            pipe1.stop()

        # メタデータJSONを作成
        try:
            metadata = {
                "base_name": base_name,
                "cam0_file": os.path.basename(record_file0),
                "cam1_file": os.path.basename(record_file1),
                "calibration_file": os.path.basename(args.calibration_file),
                "calibration": calib_data,
                "serial_cam0": sn0 if "sn0" in locals() else None,
                "serial_cam1": sn1 if "sn1" in locals() else None,
                "resolution": {"width": STREAM_W, "height": STREAM_H},
                "fps": FPS,
                "pairing_tolerance_ms": args.pairing_tol,
                "total_frames": frame_count,
                "paired_frames": paired_count,
                "skipped_frames": skipped_count,
                "timestamp": datetime.now().isoformat(),
            }

            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            print("\n" + "=" * 60)
            print("Recording Complete!")
            print("=" * 60)
            print(f"Camera 0 bag file: {record_file0}")
            print(f"Camera 1 bag file: {record_file1}")
            print(f"Metadata file: {metadata_file}")
            print(f"Total frames: {frame_count}")
            print(f"Paired frames: {paired_count}")
            print(f"Skipped frames: {skipped_count}")
            print()
            print(
                f"[INFO] Use '{os.path.basename(metadata_file)}' with analysis tools"
            )
            print("=" * 60)

        except Exception as e:
            print(f"[WARN] Failed to create metadata file: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

