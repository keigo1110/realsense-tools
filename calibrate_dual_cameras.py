"""
Dual RealSense Camera Auto-Calibration Tool

ターゲットレス自動外部標定: 2台のRealSenseカメラ（D455）の外部パラメータを
自然シーンの点群から自動推定します。

使い方:
    python calibrate_dual_cameras.py
    python calibrate_dual_cameras.py --calibration-file calibration.json
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs

# ===== 設定 =====
DEFAULT_CALIB_FILE = "extrinsics_cam1_to_cam0.json"
STREAM_W, STREAM_H, FPS = 640, 480, 30
VOXEL_SIZE = 0.05  # [m] FPFH/ICPのダウンサンプル尺
MAX_DEPTH = 5.0  # [m] 点群作成時の奥行き上限
QUEUE_SIZE = 2  # 遅延低減用


# ===== ユーティリティ =====
def get_intrinsics_from_profile(vsp):
    """RealSenseプロファイルから内部パラメータを取得"""
    intr = vsp.get_intrinsics()
    K = np.array(
        [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float64
    )
    return K, intr.width, intr.height


def make_o3d_intrinsic(K, w, h):
    """Open3Dの内部パラメータオブジェクトを作成"""
    pin = o3d.camera.PinholeCameraIntrinsic()
    pin.set_intrinsics(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    return pin


def rgbd_to_pcd(color_bgr, depth_u16, dscale, pinhole):
    """RGBD画像から点群を作成"""
    # Open3DはRGB想定
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    # 遠すぎる/0の深度をマスク
    depth = depth_u16.copy()
    depth = np.where(
        (depth * dscale > 0) & (depth * dscale < MAX_DEPTH), depth, 0
    ).astype(np.uint16)

    o3c = o3d.geometry.Image(color_rgb)
    o3dpt = o3d.geometry.Image(depth)

    # Open3Dのdepth_scaleは「1mあたりのカウント数」。RealSenseは meters = depth * dscale
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3c,
        o3dpt,
        depth_scale=1.0 / dscale,
        depth_trunc=MAX_DEPTH,
        convert_rgb_to_intensity=False,
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole)
    return pcd


def down_and_feature(pcd, voxel):
    """点群をダウンサンプルして特徴量を計算"""
    pcd_d = pcd.voxel_down_sample(voxel)
    pcd_d.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_d,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5.0, max_nn=100),
    )
    return pcd_d, fpfh


def ransac_icp_T(source, target, voxel):
    """RANSAC + ICPで外部パラメータを推定"""
    # RANSACで初期合わせ
    src_d, src_f = down_and_feature(source, voxel)
    tgt_d, tgt_f = down_and_feature(target, voxel)

    try:
        ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src_d,
            tgt_d,
            src_f,
            tgt_f,
            True,
            max_correspondence_distance=voxel * 1.5,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    voxel * 1.5
                ),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(40000, 1000),
        )

        if ransac is None or ransac.transformation is None:
            return None, 0.0, 0.0

        T_init = ransac.transformation

        # ICPで高精度化（点対平面 or G-ICP）
        source.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30)
        )
        target.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30)
        )

        icp = o3d.pipelines.registration.registration_icp(
            source,
            target,
            max_correspondence_distance=voxel * 1.5,
            init=T_init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )

        return icp.transformation, icp.fitness, icp.inlier_rmse

    except Exception as e:
        print(f"[ERROR] Registration failed: {e}")
        return None, 0.0, 0.0


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


def start_pipeline(serial):
    """RealSenseパイプラインを開始"""
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, STREAM_W, STREAM_H, rs.format.rgb8, FPS)
    cfg.enable_stream(rs.stream.depth, STREAM_W, STREAM_H, rs.format.z16, FPS)

    pipe = rs.pipeline()
    prof = pipe.start(cfg)
    align = rs.align(rs.stream.color)

    # Intrinsics & depth scale
    color_vsp = prof.get_stream(rs.stream.color).as_video_stream_profile()
    K, w, h = get_intrinsics_from_profile(color_vsp)
    depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()

    return pipe, prof, align, (K, w, h), depth_scale


def enumerate_two_devices(max_wait_seconds: float = 20.0, poll_interval: float = 0.5):
    """2台のRealSenseデバイスを列挙（待機・診断付き）"""
    ctx = rs.context()

    start = time.time()
    last_count = -1
    while True:
        devs = ctx.query_devices()
        count = len(devs)

        # 状態変化時のみログ
        if count != last_count:
            if count == 0:
                print("[INFO] Waiting for RealSense devices ...")
            else:
                print(f"[INFO] Detected {count} RealSense device(s)")
                try:
                    for i, d in enumerate(devs):
                        try:
                            name = d.get_info(rs.camera_info.name)
                        except Exception:
                            name = "?"
                        try:
                            sn = d.get_info(rs.camera_info.serial_number)
                        except Exception:
                            sn = "?"
                        try:
                            fw = d.get_info(rs.camera_info.firmware_version)
                        except Exception:
                            fw = "?"
                        try:
                            usb = d.get_info(rs.camera_info.usb_type_descriptor)
                        except Exception:
                            usb = "?"
                        print(f"  - [{i}] name={name}, serial={sn}, fw={fw}, usb={usb}")
                except Exception:
                    pass
            last_count = count

        if count >= 2:
            break

        if time.time() - start > max_wait_seconds:
            # 詳細診断を含むエラーメッセージ
            diag = []
            for i, d in enumerate(devs):
                try:
                    name = d.get_info(rs.camera_info.name)
                except Exception:
                    name = "?"
                try:
                    sn = d.get_info(rs.camera_info.serial_number)
                except Exception:
                    sn = "?"
                try:
                    fw = d.get_info(rs.camera_info.firmware_version)
                except Exception:
                    fw = "?"
                try:
                    usb = d.get_info(rs.camera_info.usb_type_descriptor)
                except Exception:
                    usb = "?"
                diag.append(f"[{i}] name={name}, serial={sn}, fw={fw}, usb={usb}")

            detail = "\n".join(diag) if diag else "(no devices)"
            raise RuntimeError(
                "2台のRealSenseが見つかりません。USB3.0/電源を確認してください。\n"
                f"Detected: {len(devs)}\n{detail}"
            )

        time.sleep(poll_interval)

    dev0, dev1 = devs[0], devs[1]

    for d in (dev0, dev1):
        set_sensor_options(d)

    sn0 = dev0.get_info(rs.camera_info.serial_number)
    sn1 = dev1.get_info(rs.camera_info.serial_number)

    return (dev0, dev1), (sn0, sn1)


# ===== メイン =====
def main():
    parser = argparse.ArgumentParser(
        description="Dual RealSense Camera Auto-Calibration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用方法:
  1. 2台のD455をUSB3.0で接続
  2. 両カメラの視野が少し重なる静止シーンを表示（白壁だけは避ける）
  3. このスクリプトを実行 → 自動で外部パラメータを推定してJSON保存

注意:
  - 自然シーンから特徴量を自動抽出してキャリブレーションします
  - 初回推定の間は人や物を動かさないでください
  - 'r'キーで再推定が可能です
        """,
    )

    parser.add_argument(
        "--calibration-file",
        type=str,
        default=DEFAULT_CALIB_FILE,
        help=f"キャリブレーションファイルの保存先（デフォルト: {DEFAULT_CALIB_FILE}）",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=VOXEL_SIZE,
        help=f"ボクセルサイズ[m]（デフォルト: {VOXEL_SIZE}）",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=MAX_DEPTH,
        help=f"最大深度[m]（デフォルト: {MAX_DEPTH}）",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="非対話モード: 1回だけ推定して結果を保存して終了",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Dual RealSense Camera Auto-Calibration")
    print("=" * 60)

    # 例外発生時でも finally で参照できるように初期化
    T1to0 = None
    stream_started = False

    try:
        # 2台のデバイスを検出
        (dev0, dev1), (sn0, sn1) = enumerate_two_devices()
        print(f"[INFO] Device 0: {sn0}")
        print(f"[INFO] Device 1: {sn1}")

        # パイプラインを開始
        pipe0, prof0, align0, (K0, w0, h0), dscale0 = start_pipeline(sn0)
        pipe1, prof1, align1, (K1, w1, h1), dscale1 = start_pipeline(sn1)
        stream_started = True

        pin0 = make_o3d_intrinsic(K0, w0, h0)
        pin1 = make_o3d_intrinsic(K1, w1, h1)

        # 非対話モード（1回推定して終了）
        if args.once:
            print("\n[INFO] Streaming started. Estimating once...")
            # フレームを少し捨てて安定させる
            for _ in range(5):
                align0.process(pipe0.wait_for_frames())
                align1.process(pipe1.wait_for_frames())

            fs0 = align0.process(pipe0.wait_for_frames())
            fs1 = align1.process(pipe1.wait_for_frames())

            c0 = np.asanyarray(fs0.get_color_frame().get_data())
            d0 = np.asanyarray(fs0.get_depth_frame().get_data())
            c1 = np.asanyarray(fs1.get_color_frame().get_data())
            d1 = np.asanyarray(fs1.get_depth_frame().get_data())

            c0_bgr = cv2.cvtColor(c0, cv2.COLOR_RGB2BGR)
            c1_bgr = cv2.cvtColor(c1, cv2.COLOR_RGB2BGR)

            pcd0 = rgbd_to_pcd(c0_bgr, d0, dscale0, pin0)
            pcd1 = rgbd_to_pcd(c1_bgr, d1, dscale1, pin1)

            T1to0_est, fit, rmse = ransac_icp_T(pcd1, pcd0, args.voxel_size)
            if T1to0_est is None:
                print("[ERROR] Extrinsics estimation failed (no valid transformation)", file=sys.stderr)
                sys.exit(1)
            if not (fit > 0.3 and rmse < 0.05):
                print(
                    f"[ERROR] Extrinsics low confidence (fitness={fit:.3f}, rmse={rmse:.4f}m). Expected: fit>0.3, rmse<0.05m",
                    file=sys.stderr,
                )
                sys.exit(1)

            calib_data = {
                "T1_to_0": T1to0_est.tolist(),
                "fitness": float(fit),
                "inlier_rmse": float(rmse),
                "voxel_size": args.voxel_size,
                "max_depth": args.max_depth,
                "serial_cam0": sn0,
                "serial_cam1": sn1,
            }
            json.dump(calib_data, open(args.calibration_file, "w"), indent=2)
            print(
                f"[INFO] ✓ Extrinsics OK (fitness={fit:.3f}, rmse={rmse:.4f}m) -> saved to {args.calibration_file}"
            )
            # finally でのメッセージ抑制のため
            T1to0 = T1to0_est
            return

        print("\n[INFO] Streaming started. Press 'r' to estimate, 'q' to quit.")
        print(
            "[INFO] Make sure both cameras see overlapping scene (not just white wall)."
        )

        T1to0 = None

        while True:
            # フレームを取得
            frames0 = pipe0.wait_for_frames()
            frames1 = pipe1.wait_for_frames()

            fs0 = align0.process(frames0)
            fs1 = align1.process(frames1)

            f0 = fs0
            f1 = fs1

            c0 = np.asanyarray(f0.get_color_frame().get_data())
            d0 = np.asanyarray(f0.get_depth_frame().get_data())
            c1 = np.asanyarray(f1.get_color_frame().get_data())
            d1 = np.asanyarray(f1.get_depth_frame().get_data())

            c0_bgr = cv2.cvtColor(c0, cv2.COLOR_RGB2BGR)
            c1_bgr = cv2.cvtColor(c1, cv2.COLOR_RGB2BGR)

            # キー入力チェック
            key = cv2.waitKey(1) & 0xFF

            # 自動外部標定（初回 or 'r'）
            if T1to0 is None or key == ord("r"):
                print("\n[INFO] Estimating extrinsics from scene...")
                print("[INFO] Please keep scene still during estimation...")

                pcd0 = rgbd_to_pcd(c0_bgr, d0, dscale0, pin0)
                pcd1 = rgbd_to_pcd(c1_bgr, d1, dscale1, pin1)

                try:
                    T1to0_est, fit, rmse = ransac_icp_T(pcd1, pcd0, args.voxel_size)

                    if T1to0_est is None:
                        print(
                            "[WARN] Extrinsics estimation failed (no valid transformation)"
                        )
                        status_color = (0, 0, 255)  # 赤
                        status_text = "CALIB: FAILED"
                    # 妥当性の簡易チェック
                    elif fit > 0.3 and rmse < 0.05:
                        T1to0 = T1to0_est
                        # JSONに保存
                        calib_data = {
                            "T1_to_0": T1to0.tolist(),
                            "fitness": float(fit),
                            "inlier_rmse": float(rmse),
                            "voxel_size": args.voxel_size,
                            "max_depth": args.max_depth,
                            "serial_cam0": sn0,
                            "serial_cam1": sn1,
                        }
                        json.dump(
                            calib_data, open(args.calibration_file, "w"), indent=2
                        )
                        print(
                            f"[INFO] ✓ Extrinsics OK (fitness={fit:.3f}, rmse={rmse:.4f}m) -> saved to {args.calibration_file}"
                        )
                        status_color = (0, 255, 0)  # 緑
                        status_text = f"CALIB: OK (fit={fit:.2f}, rmse={rmse:.3f}m)"
                    else:
                        print(
                            f"[WARN] Extrinsics low confidence (fitness={fit:.3f}, rmse={rmse:.4f}m)"
                        )
                        print(f"[WARN] Expected: fitness > 0.3, rmse < 0.05m")
                        status_color = (0, 165, 255)  # オレンジ
                        status_text = f"CALIB: LOW (fit={fit:.2f}, rmse={rmse:.3f}m)"

                except Exception as e:
                    print(f"[WARN] Extrinsics estimation failed: {e}")
                    status_color = (0, 0, 255)  # 赤
                    status_text = "CALIB: ERROR"

            else:
                # キャリブレーション済み
                status_color = (0, 255, 0)  # 緑
                status_text = "CALIB: OK (Press 'r' to re-estimate)"

            # 画面に状態表示
            cv2.putText(
                c0_bgr,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
            )
            cv2.putText(
                c0_bgr,
                f"Press 'r' to re-estimate, 'q' to quit",
                (10, 60),
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
                0.7,
                status_color,
                2,
            )
            cv2.putText(
                c1_bgr,
                f"Press 'r' to re-estimate, 'q' to quit",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            cv2.imshow("cam0", c0_bgr)
            cv2.imshow("cam1", c1_bgr)

            if key == 27 or key == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n\n中断されました")
    except RuntimeError as e:
        # 既知の実行時エラー（例: デバイス未検出）はトレースバックを出さずに終了
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # 予期しない例外のみトレースバックを表示
        print(f"\n[ERROR] {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        cv2.destroyAllWindows()
        if "pipe0" in locals():
            pipe0.stop()
        if "pipe1" in locals():
            pipe1.stop()

        # ストリーミングが開始できた場合のみ最終メッセージを表示
        if stream_started:
            if T1to0 is not None:
                print(f"\n[INFO] Calibration saved to: {args.calibration_file}")
            else:
                print("\n[WARN] Calibration not saved (estimation failed or cancelled)")


if __name__ == "__main__":
    main()
