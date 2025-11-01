"""
RealSense Recording Tool

RealSenseカメラから映像と深度データを.bagファイルに録画するシンプルなツール

使い方:
    python pose-record.py --record my_recording.bag
    python pose-record.py --record my_recording.bag --enable-depth
"""

import argparse
import sys

import cv2
import numpy as np
import pyrealsense2 as rs


def main():
    parser = argparse.ArgumentParser(
        description='RealSense Recording Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
録画例:
  # カラーストリームのみ録画
  python pose-record.py --record recording.bag

  # カラー + 深度ストリームを録画（推奨）
  python pose-record.py --record recording.bag --enable-depth

  # 解像度とFPSを指定して録画
  python pose-record.py --record recording.bag --enable-depth --resolution 1280 720 --fps 60

操作方法:
  - ESCキー: 録画を停止して終了
  - スペースキー: 録画を一時停止/再開（画面表示のみ）
        """
    )

    # 録画設定
    parser.add_argument("--record", type=str, required=True, metavar='FILE',
                       help="録画先の.bagファイル名 (例: recording.bag)")
    parser.add_argument("--enable-depth", action="store_true",
                       help="深度ストリームも録画する（推奨）")

    # ストリーム設定
    parser.add_argument("--resolution", default=[640, 480], type=int, nargs=2,
                       metavar=('WIDTH', 'HEIGHT'),
                       help="解像度 (デフォルト: 640x480)")
    parser.add_argument("--fps", default=30, type=int,
                       help="フレームレート (デフォルト: 30)")

    args = parser.parse_args()

    # RealSenseパイプラインの設定
    pipeline = rs.pipeline()
    config = rs.config()

    width, height = args.resolution

    # カラーストリームを有効化
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, args.fps)

    # 深度ストリームを有効化（オプション）
    if args.enable_depth:
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, args.fps)
        print(f"録画モード: カラー + 深度 ({width}x{height} @ {args.fps}fps)")
    else:
        print(f"録画モード: カラーのみ ({width}x{height} @ {args.fps}fps)")

    # 録画ファイルを設定
    config.enable_record_to_file(args.record)

    # パイプライン開始
    try:
        pipeline.start(config)
        print(f"録画を開始します: {args.record}")
        print("ESCキーで停止")

        frame_count = 0
        recording = True

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            frame_count += 1

            # 画像を取得して表示
            image = np.asanyarray(color_frame.get_data())

            # RGBからBGRに変換（OpenCV表示用）
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 録画状態を表示
            status_text = "RECORDING" if recording else "PAUSED"
            color = (0, 0, 255) if recording else (0, 255, 255)
            cv2.putText(image, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            cv2.putText(image, f"Frames: {frame_count}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"File: {args.record}", (10, image.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow('RealSense Recording', image)

            # キー入力処理
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESCキーで終了
                print("\n録画を停止しています...")
                break
            elif key == ord(' '):  # スペースキーで一時停止（表示のみ、録画は続行）
                recording = not recording
                if recording:
                    print("録画を再開しました")
                else:
                    print("録画を一時停止しました（実際の録画は続行中）")

    except RuntimeError as e:
        print(f"エラーが発生しました: {e}", file=sys.stderr)
        print("\nRealSenseカメラが接続されているか確認してください。", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n録画を中断しました")
    finally:
        pipeline.stop()
        print(f"録画が完了しました: {args.record}")
        print(f"総フレーム数: {frame_count}")


if __name__ == "__main__":
    main()

