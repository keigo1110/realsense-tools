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
import shutil
import time
import platform

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
    parser.add_argument(
        "--wait-start",
        action="store_true",
        help="'s'キーが押されるまでプレビューし、押されたら録画開始",
    )
    parser.add_argument(
        "--minimize-after-start",
        action="store_true",
        help="録画開始後はプレビューウィンドウを閉じて描画負荷を抑える",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="録画中はプレビューを完全に無効化（コンソールからEsc/qで停止）",
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

    # ベース名の決定
    if args.output:
        base_name = args.output
        if base_name.endswith(".bag"):
            base_name = base_name[:-4]
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"dual_recording_{timestamp}"

    # 録画ディレクトリの作成（bagdata/<base_name>/）
    record_root = "bagdata"
    record_dir = os.path.join(record_root, base_name)
    os.makedirs(record_dir, exist_ok=True)

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

        # プレビュー待機モード
        if args.wait_start:
            print("[INFO] Previewing... Press 's' to start recording, ESC/'q' to quit")
            pipe0, prof0, align0 = start_pipeline(sn0)
            pipe1, prof1, align1 = start_pipeline(sn1)

            last_fs0 = None
            last_fs1 = None
            stall_ticks = 0
            while True:
                # キー入力を先にチェック（応答性向上）
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    break
                if key == 27 or key == ord('q'):
                    print("\n[INFO] Exit without recording")
                    return

                got0 = pipe0.poll_for_frames()
                got1 = pipe1.poll_for_frames()

                if got0:
                    try:
                        last_fs0 = align0.process(got0)
                    except Exception:
                        last_fs0 = None
                if got1:
                    try:
                        last_fs1 = align1.process(got1)
                    except Exception:
                        last_fs1 = None

                if last_fs0 is None or last_fs1 is None:
                    # どちらか未取得なら軽く待機して継続（GUI応答を保つ）
                    stall_ticks += 1
                    if stall_ticks % 120 == 0:
                        print("[INFO] Waiting for both camera frames...")
                    time.sleep(0.01)  # CPU負荷軽減
                    continue

                fs0 = last_fs0
                fs1 = last_fs1

                try:
                    c0 = np.asanyarray(fs0.get_color_frame().get_data())
                    c1 = np.asanyarray(fs1.get_color_frame().get_data())
                    c0_bgr = cv2.cvtColor(c0, cv2.COLOR_RGB2BGR)
                    c1_bgr = cv2.cvtColor(c1, cv2.COLOR_RGB2BGR)

                    cv2.putText(c0_bgr, "PREVIEW - Press 's' to start", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(c1_bgr, "PREVIEW - Press 's' to start", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow("cam0", c0_bgr)
                    cv2.imshow("cam1", c1_bgr)
                except Exception as e:
                    # フレーム処理エラーは無視して継続
                    pass

            # 一旦停止して録画モードで再起動
            print("\n[INFO] Stopping preview and preparing for recording...")
            # ウィンドウを先に閉じる（GUI応答を保つ）
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

            # パイプライン停止（ブロッキングを最小化）
            try:
                pipe0.stop()
            except Exception as e:
                print(f"[WARN] Error stopping pipe0: {e}")
            try:
                pipe1.stop()
            except Exception as e:
                print(f"[WARN] Error stopping pipe1: {e}")

            # デバイスの完全解放を待つ（ブロッキング回避）
            time.sleep(0.5)

            print("[INFO] Starting recording pipeline...")

        # パイプラインを開始（録画ファイルも設定）
        try:
            pipe0, prof0, align0 = start_pipeline(sn0, record_file0)
        except Exception as e:
            print(f"[ERROR] Failed to start recording pipeline for cam0: {e}", file=sys.stderr)
            sys.exit(1)
        try:
            pipe1, prof1, align1 = start_pipeline(sn1, record_file1)
        except Exception as e:
            print(f"[ERROR] Failed to start recording pipeline for cam1: {e}", file=sys.stderr)
            pipe0.stop()
            sys.exit(1)

        print(f"[INFO] Recording started...")
        print(f"[INFO] Press ESC or 'q' to stop recording")

        # 録画パイプライン開始後のウォームアップ（最初の数フレームを待つ）
        print("[INFO] Warming up recording pipelines...")
        warmup_count = 0
        max_warmup_attempts = 100  # 最大100回試行（約5秒）
        
        for attempt in range(max_warmup_attempts):
            try:
                frames0 = pipe0.poll_for_frames()
                frames1 = pipe1.poll_for_frames()
                
                if frames0 and frames1:
                    try:
                        align0.process(frames0)
                        align1.process(frames1)
                        warmup_count += 1
                        if warmup_count >= 3:  # 3フレーム成功したら準備完了
                            break
                    except Exception:
                        pass
            except Exception:
                pass
            
            if attempt % 20 == 0 and attempt > 0:
                print(f"[INFO] Waiting for frames... ({attempt}/{max_warmup_attempts})")
            
            time.sleep(0.05)  # 50ms待機
        
        if warmup_count >= 3:
            print("[INFO] Recording pipelines ready")
        else:
            print(f"[WARN] Warm-up incomplete ({warmup_count} frames), but continuing...")

        pairer = FramePairer(args.pairing_tol)
        frame_count = 0
        paired_count = 0
        skipped_count = 0

        last_fs0 = None
        last_fs1 = None
        # 録画中のプレビュー可否
        show_preview = not args.minimize_after_start and not args.no_preview

        # Windowsのコンソールキー入力（非ブロッキング）
        use_console_keys = (not show_preview) and platform.system().lower().startswith("win")
        if use_console_keys:
            try:
                import msvcrt  # type: ignore
            except Exception:
                msvcrt = None
                use_console_keys = False

        consecutive_empty_loops = 0
        max_empty_loops = 100  # 連続でフレームが取得できない場合の上限

        while True:
            # フレームを取得（ノンブロッキング）
            got0 = None
            got1 = None
            try:
                got0 = pipe0.poll_for_frames()
            except Exception as e:
                if consecutive_empty_loops % 50 == 0:
                    print(f"[WARN] Error polling cam0: {e}")
                got0 = None
            try:
                got1 = pipe1.poll_for_frames()
            except Exception as e:
                if consecutive_empty_loops % 50 == 0:
                    print(f"[WARN] Error polling cam1: {e}")
                got1 = None

            if got0:
                try:
                    last_fs0 = align0.process(got0)
                except Exception:
                    last_fs0 = None
            if got1:
                try:
                    last_fs1 = align1.process(got1)
                except Exception:
                    last_fs1 = None

            if last_fs0 is None or last_fs1 is None:
                consecutive_empty_loops += 1
                
                # 連続でフレームが取得できない場合の警告
                if consecutive_empty_loops == max_empty_loops:
                    print(f"[WARN] No frames received for {max_empty_loops} loops. Checking pipelines...")
                    # パイプラインの状態を確認（poll_for_framesで非ブロッキング）
                    try:
                        test_frames0 = pipe0.poll_for_frames()
                        test_frames1 = pipe1.poll_for_frames()
                        if test_frames0 and test_frames1:
                            print("[INFO] Pipelines are active, resuming polling...")
                            consecutive_empty_loops = 0
                        else:
                            print("[WARN] Pipelines may be stalled. Trying to continue...")
                    except Exception as e:
                        print(f"[WARN] Error checking pipeline status: {e}")
                
                # イベント処理のみ行って継続
                if show_preview:
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord("q"):
                        print("\n[INFO] Stopping recording...")
                        break
                elif use_console_keys and 'msvcrt' in globals() and msvcrt:
                    if msvcrt.kbhit():
                        ch = msvcrt.getch()
                        if ch in (b"\x1b", b"q", b"Q"):
                            print("\n[INFO] Stopping recording...")
                            break
                    # CPU過負荷防止（連続空ループが多い場合は少し長めに待機）
                    sleep_time = 0.01 if consecutive_empty_loops > 10 else 0.005
                    time.sleep(sleep_time)
                else:
                    sleep_time = 0.01 if consecutive_empty_loops > 10 else 0.005
                    time.sleep(sleep_time)
                continue
            
            # フレームが取得できたのでリセット
            consecutive_empty_loops = 0

            fs0 = last_fs0
            fs1 = last_fs1

            # ペアリング
            try:
                match = pairer.put_and_match(fs0, fs1)
            except Exception:
                match = None

            if match is None:
                skipped_count += 1
                # 表示のみ（録画は続行）
                if show_preview and frame_count % 10 == 0:
                    try:
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
                    except Exception:
                        pass
                continue

            try:
                f0, f1, dt = match
            except Exception:
                continue

            frame_count += 1
            paired_count += 1

            # 表示用データ（プレビュー有効時のみ）
            if show_preview:
                try:
                    c0 = np.asanyarray(f0.get_color_frame().get_data())
                    c1 = np.asanyarray(f1.get_color_frame().get_data())
                    c0_bgr = cv2.cvtColor(c0, cv2.COLOR_RGB2BGR)
                    c1_bgr = cv2.cvtColor(c1, cv2.COLOR_RGB2BGR)
                except Exception:
                    # フレーム取得エラー時はスキップ
                    continue

            # 録画状態を表示（必要に応じてプレビューを抑制）
            status_text = f"RECORDING | Frames: {frame_count} | Paired: {paired_count} | Skipped: {skipped_count}"
            if show_preview:
                try:
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
                except Exception:
                    pass
            else:
                # 最初のループでウィンドウを閉じる（1回だけ）
                if paired_count == 1:
                    try:
                        cv2.destroyWindow("cam0")
                        cv2.destroyWindow("cam1")
                    except Exception:
                        pass
                # コンソールに軽量な進捗を出力
                if frame_count % 30 == 0:
                    print(status_text)

            if not show_preview:
                # CPU過負荷防止
                time.sleep(0.001)

            # キー入力処理
            if show_preview:
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    print("\n[INFO] Stopping recording...")
                    break
            elif use_console_keys and 'msvcrt' in globals() and msvcrt:
                if msvcrt.kbhit():
                    ch = msvcrt.getch()
                    if ch in (b"\x1b", b"q", b"Q"):
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

        # キャリブレーションファイルをレコードフォルダにコピー
        try:
            calib_basename = os.path.basename(args.calibration_file)
            calib_copy_path = os.path.join(record_dir, calib_basename)
            src = os.path.abspath(args.calibration_file)
            dst = os.path.abspath(calib_copy_path)
            if src != dst:
                shutil.copyfile(src, dst)
        except Exception as e:
            print(f"[WARN] Failed to copy calibration file: {e}", file=sys.stderr)

        # メタデータJSONを作成
        try:
            metadata = {
                "base_name": base_name,
                "directory": os.path.basename(record_dir),
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

