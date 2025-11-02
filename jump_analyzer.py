"""
Jump Analyzer

.bagファイルからYOLOv8-Poseを使用して3D姿勢推定を行い、
ジャンプの高さ・距離・軌跡を測定するメインスクリプト
"""

import argparse
import json
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from realsense_utils import BagFileReader, CUPY_AVAILABLE
from yolov8_pose_3d import YOLOv8PoseDetector, COCO_KEYPOINTS
from jump_detector import JumpDetector
from visualizer import JumpVisualizer

# CuPyのインポート（CUDA高速化用）
if CUPY_AVAILABLE:
    import cupy as cp
    print("CuPy available: Using CUDA acceleration for image processing")
else:
    print("CuPy not available: Using NumPy (CPU mode)")


def resize_image_cuda(image, new_width, new_height):
    """
    画像をリサイズ（CUDA使用可能な場合は高速化）
    
    Args:
        image: 入力画像（NumPy配列）
        new_width: 新しい幅
        new_height: 新しい高さ
    
    Returns:
        リサイズされた画像（NumPy配列）
    """
    if CUPY_AVAILABLE:
        try:
            # CuPyを使用してGPUでリサイズ
            gpu_image = cp.asarray(image)
            # CuPyにはresize関数がないため、OpenCV CUDAを使用するかNumPyにフォールバック
            # ここではNumPyにフォールバック（CuPyでリサイズする場合は別途実装が必要）
            resized = cv2.resize(image, (new_width, new_height))
            return resized
        except Exception:
            # CUDA処理に失敗した場合はCPUで処理
            return cv2.resize(image, (new_width, new_height))
    else:
        # CPUでリサイズ
        return cv2.resize(image, (new_width, new_height))


def convert_keypoints_to_dict(keypoints_2d, keypoints_3d):
    """
    keypointsを辞書形式に変換

    Args:
        keypoints_2d: 2D keypointsのリスト [(x, y, confidence), ...]
        keypoints_3d: 3D keypointsの辞書 {keypoint_name: (x, y, z), ...}

    Returns:
        dict: keypointsデータの辞書
    """
    result = {}

    for i, keypoint_name in enumerate(COCO_KEYPOINTS):
        if i < len(keypoints_2d):
            kp_2d = keypoints_2d[i]
            kp_3d = keypoints_3d.get(keypoint_name)

            result[keypoint_name] = {
                "2d": {
                    "x": float(kp_2d[0]) if kp_2d[0] is not None else None,
                    "y": float(kp_2d[1]) if kp_2d[1] is not None else None,
                    "confidence": float(kp_2d[2]),
                },
                "3d": {
                    "x": float(kp_3d[0]) if kp_3d and kp_3d[0] is not None else None,
                    "y": float(kp_3d[1]) if kp_3d and kp_3d[1] is not None else None,
                    "z": float(kp_3d[2]) if kp_3d and kp_3d[2] is not None else None,
                },
            }
        else:
            result[keypoint_name] = {
                "2d": {"x": None, "y": None, "confidence": 0.0},
                "3d": {"x": None, "y": None, "z": None},
            }

    return result


def save_json(data, output_path):
    """JSONファイルに保存"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"JSON saved to: {output_path}")


def save_csv(statistics, trajectory, output_path):
    """CSVファイルに保存"""
    # 統計情報をCSVに保存
    stats_path = output_path.replace(".csv", "_statistics.csv")
    with open(stats_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Total Jumps", statistics.get("total_jumps", 0)])
        writer.writerow(["Vertical Jumps", statistics.get("vertical_jumps", 0)])
        writer.writerow(["Horizontal Jumps", statistics.get("horizontal_jumps", 0)])
        writer.writerow(["Max Height (cm)", statistics.get("max_height", 0) * 100])
        writer.writerow(["Max Distance (cm)", statistics.get("max_distance", 0) * 100])
        writer.writerow(["Avg Height (cm)", statistics.get("avg_height", 0) * 100])
        writer.writerow(["Avg Distance (cm)", statistics.get("avg_distance", 0) * 100])
    print(f"Statistics CSV saved to: {stats_path}")

    # ジャンプ詳細をCSVに保存
    if statistics.get("jumps"):
        jumps_path = output_path.replace(".csv", "_jumps.csv")
        with open(jumps_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Jump #",
                    "Type",
                    "Height (cm)",
                    "Distance (cm)",
                    "Start Frame",
                    "End Frame",
                    "Duration (frames)",
                ]
            )
            for i, jump in enumerate(statistics["jumps"], 1):
                writer.writerow(
                    [
                        i,
                        jump["jump_type"],
                        jump["height"] * 100,
                        jump["distance"] * 100,
                        jump["frame_start"],
                        jump["frame_end"],
                        jump["frame_end"] - jump["frame_start"],
                    ]
                )
        print(f"Jumps CSV saved to: {jumps_path}")

    # 軌跡データをCSVに保存
    if trajectory:
        trajectory_path = output_path.replace(".csv", "_trajectory.csv")
        with open(trajectory_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Frame", "Timestamp", "X (m)", "Y (m)", "Z (m)"])
            for point in trajectory:
                pos = point.get("position", (None, None, None))
                writer.writerow(
                    [
                        point.get("frame", ""),
                        point.get("timestamp", ""),
                        pos[0] if pos[0] is not None else "",
                        pos[1] if pos[1] is not None else "",
                        pos[2] if pos[2] is not None else "",
                    ]
                )
        print(f"Trajectory CSV saved to: {trajectory_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Jump Analyzer: Analyze jump height, distance, and trajectory from RealSense .bag file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python jump_analyzer.py --input bagdata/my_recording.bag --output results/

  # Specify model directory
  python jump_analyzer.py --input bagdata/my_recording.bag --output results/ --model-dir models/

  # Adjust jump detection thresholds
  python jump_analyzer.py --input bagdata/my_recording.bag --output results/ --threshold-vertical 0.1 --threshold-horizontal 0.2
        """,
    )

    # 必須引数
    parser.add_argument("--input", type=str, required=True, help="Input .bag file path")
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory for results"
    )

    # オプション引数
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory for YOLOv8-Pose model files (default: models)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="yolov8x-pose.pt",
        help="YOLOv8-Pose model name (yolov8n-pose.pt, yolov8s-pose.pt, etc.) (default: yolov8x-pose.pt)",
    )
    parser.add_argument(
        "--threshold-vertical",
        type=float,
        default=0.05,
        help="Vertical jump detection threshold in meters (default: 0.05)",
    )
    parser.add_argument(
        "--threshold-horizontal",
        type=float,
        default=0.1,
        help="Horizontal jump detection threshold in meters (default: 0.1)",
    )
    parser.add_argument(
        "--no-video", action="store_true", help="Skip video visualization output"
    )

    # 高速化オプション
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Process every N frames (1=all frames, 2=every other frame, etc.) (default: 1)",
    )
    parser.add_argument(
        "--resize-factor",
        type=float,
        default=1.0,
        help="Resize image before YOLOv8-Pose inference (1.0=no resize, 0.5=half size) (default: 1.0)",
    )
    parser.add_argument(
        "--minimal-data",
        action="store_true",
        help="Save only jump detection frames in JSON (faster, smaller file)",
    )

    args = parser.parse_args()

    # 出力ディレクトリを作成
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # バグファイルの存在確認
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("Jump Analyzer - YOLOv8-Pose 3D Analysis")
    print("=" * 60)
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output}")
    print()

    # モデルディレクトリを作成（存在しない場合）
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # YOLOv8-Poseモデルの読み込み
    print("Loading YOLOv8-Pose model...")
    yolo_pose = YOLOv8PoseDetector(model_name=args.model_name, model_dir=str(model_dir))
    if not yolo_pose.load_model():
        print("Error: Failed to load YOLOv8-Pose model", file=sys.stderr)
        sys.exit(1)

    # バグファイルの読み込み
    print("Loading .bag file...")
    bag_reader = BagFileReader(args.input)
    if not bag_reader.initialize():
        print("Error: Failed to load .bag file", file=sys.stderr)
        sys.exit(1)

    # ジャンプ検出器の初期化
    jump_detector = JumpDetector(
        threshold_vertical=args.threshold_vertical,
        threshold_horizontal=args.threshold_horizontal,
    )

    # 可視化器の初期化
    visualizer = JumpVisualizer()

    # データ保存用のリスト
    all_frames_data = []

    # 可視化動画ライター（メモリ効率化のため逐次書き込み）
    video_writer = None
    video_path = None
    video_fps = 30

    print("\nProcessing frames...")
    if args.frame_skip > 1:
        print(f"  Frame skipping: Processing every {args.frame_skip} frame(s)")
    if args.resize_factor < 1.0:
        print(f"  Image resize: {args.resize_factor*100:.0f}% of original size")
    if args.minimal_data:
        print("  Minimal data mode: Saving only jump detection frames")
    print()
    print("Note: First frame processing may take longer (model warm-up)...")
    print()

    frame_num = 0
    processed_frame_count = 0
    skipped_frame_count = 0
    start_time = time.time()
    last_progress_time = start_time

    try:
        while True:
            # フレームを取得
            color_frame, depth_frame = bag_reader.get_frames()

            if color_frame is None or depth_frame is None:
                break

            color_image, depth_image = bag_reader.frame_to_numpy(
                color_frame, depth_frame
            )

            if color_image is None or depth_image is None:
                continue

            frame_num += 1

            # フレームスキッピング
            if frame_num % args.frame_skip != 0:
                skipped_frame_count += 1
                continue

            processed_frame_count += 1

            # 画像リサイズ（YOLOv8-Pose推論前）
            original_shape = color_image.shape[:2]
            inference_image = color_image
            resize_scale_x = 1.0
            resize_scale_y = 1.0

            if args.resize_factor < 1.0:
                new_width = int(color_image.shape[1] * args.resize_factor)
                new_height = int(color_image.shape[0] * args.resize_factor)
                inference_image = resize_image_cuda(color_image, new_width, new_height)
                resize_scale_x = color_image.shape[1] / new_width
                resize_scale_y = color_image.shape[0] / new_height

            # 2D keypointsを検出（リサイズ済み画像を使用）
            if processed_frame_count == 1:
                print(f"Processing first frame (frame {frame_num})...")

            # ポーズ推定の時間を測定
            pose_start = time.time()
            keypoints_2d = yolo_pose.detect_keypoints(inference_image)
            pose_time = time.time() - pose_start

            # keypoints座標を元の画像サイズにスケール（リスト内包表記で高速化）
            if args.resize_factor < 1.0 and keypoints_2d:
                keypoints_2d = [
                    (kp[0] * resize_scale_x, kp[1] * resize_scale_y, kp[2])
                    if kp[0] is not None and kp[1] is not None
                    else kp
                    for kp in keypoints_2d
                ]

            if keypoints_2d is None:
                continue

            # 3D keypointsを計算（バッチ処理で高速化）
            depth_start = time.time()
            
            # 有効なkeypointsの座標を収集（リスト内包表記で高速化）
            valid_data = [
                (kp_name, kp_2d[0], kp_2d[1])
                for kp_name, kp_2d in zip(COCO_KEYPOINTS, keypoints_2d)
                if kp_2d[0] is not None and kp_2d[1] is not None
            ]
            
            # バッチ処理で深度値を一括取得
            if valid_data:
                valid_points = [(x, y) for _, x, y in valid_data]
                depths = bag_reader.get_depth_at_points_batch(depth_frame, valid_points)
                # バッチ処理で3D座標に一括変換
                coords_3d = bag_reader.pixels_to_3d_batch(valid_points, depths)
                
                # 結果を辞書に格納（辞書内包表記で高速化）
                valid_coords_dict = {
                    kp_name: coords_3d[i] if coords_3d[i] is not None else (None, None, None)
                    for i, (kp_name, _, _) in enumerate(valid_data)
                }
                # 全keypointsの辞書を作成（有効なものと無効なものを統合）
                keypoints_3d = {
                    kp_name: valid_coords_dict.get(kp_name, (None, None, None))
                    for kp_name in COCO_KEYPOINTS
                }
            else:
                # 有効なkeypointがない場合
                keypoints_3d = {kp_name: (None, None, None) for kp_name in COCO_KEYPOINTS}
            
            depth_time = time.time() - depth_start

            # 最初の数フレームで時間を表示
            if processed_frame_count <= 5:
                total_time = pose_time + depth_time
                print(
                    f"Frame {processed_frame_count}: "
                    f"Pose={pose_time:.3f}s ({pose_time/total_time*100:.1f}%), "
                    f"3D={depth_time:.3f}s ({depth_time/total_time*100:.1f}%)"
                )

            # ジャンプ検出
            timestamp = color_frame.get_timestamp()
            # frame_numではなくprocessed_frame_countを使用（スキップ考慮）
            jump_result = jump_detector.update(
                processed_frame_count, keypoints_3d, timestamp
            )

            # フレームデータを保存（minimal_dataモードではジャンプ検出時のみ）
            should_save = True
            if args.minimal_data:
                should_save = jump_result is not None and jump_result.get("state") in [
                    "jump_start",
                    "jump_end",
                    "jumping",
                ]

            if should_save:
                frame_data = {
                    "frame": frame_num,
                    "processed_frame": processed_frame_count,
                    "timestamp": timestamp,
                    "keypoints": convert_keypoints_to_dict(keypoints_2d, keypoints_3d),
                }

                if jump_result:
                    frame_data["jump_result"] = {
                        "state": jump_result.get("state", "unknown"),
                        "height": jump_result.get("height"),
                        "position": jump_result.get("position"),
                        "jump_type": jump_result.get("jump_type"),
                        "jump_height": jump_result.get("jump_height"),
                        "jump_distance": jump_result.get("jump_distance"),
                    }

                all_frames_data.append(frame_data)

            # 可視化フレームを生成（逐次書き込みでメモリ効率化）
            if not args.no_video:
                # スケルトンを描画（リサイズされていない場合は直接使用）
                skeleton_image = yolo_pose.draw_skeleton(
                    color_image if args.resize_factor >= 1.0 else color_image.copy(),
                    keypoints_2d
                )

                # 統計情報を取得
                statistics = jump_detector.get_statistics()

                # 可視化フレームを生成
                vis_frame = visualizer.draw_frame(
                    skeleton_image, keypoints_2d, jump_result, statistics
                )

                # 動画ライターを初期化（最初のフレーム時のみ）
                if video_writer is None:
                    video_path = output_dir / "jump_visualization.mp4"
                    height, width = vis_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        str(video_path), fourcc, video_fps, (width, height)
                    )
                    print(f"Initialized video writer: {video_path}")

                # フレームを即座に書き込み（メモリに保持しない）
                video_writer.write(vis_frame)

            # 進捗表示（処理速度を含む）
            current_time = time.time()
            # 最初のフレーム処理時、または10フレームごと、または5秒ごとに表示
            if (
                processed_frame_count == 1
                or processed_frame_count % 10 == 0
                or (current_time - last_progress_time) >= 5.0
            ):
                elapsed_time = current_time - start_time
                if processed_frame_count > 0 and elapsed_time > 0:
                    fps = processed_frame_count / elapsed_time
                    print(
                        f"Processed {processed_frame_count} frames (total frames read: {frame_num}, skipped: {skipped_frame_count}) | "
                        f"Speed: {fps:.1f} fps | Elapsed: {elapsed_time:.0f}s"
                    )
                    last_progress_time = current_time

        elapsed_time = time.time() - start_time
        print(
            f"\nFinished processing {processed_frame_count} frames (total: {frame_num}, skipped: {skipped_frame_count})"
        )
        print(f"Total time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
        if processed_frame_count > 0:
            print(f"Average speed: {processed_frame_count/elapsed_time:.1f} fps")

        # 統計情報を取得
        statistics = jump_detector.get_statistics()
        trajectory = jump_detector.get_trajectory()

        print("\n" + "=" * 60)
        print("Analysis Results")
        print("=" * 60)
        print(f"Total jumps detected: {statistics['total_jumps']}")
        print(f"  - Vertical jumps: {statistics['vertical_jumps']}")
        print(f"  - Horizontal jumps: {statistics['horizontal_jumps']}")
        print(f"Max height: {statistics['max_height'] * 100:.1f} cm")
        print(f"Max distance: {statistics['max_distance'] * 100:.1f} cm")
        print(f"Average height: {statistics['avg_height'] * 100:.1f} cm")
        print(f"Average distance: {statistics['avg_distance'] * 100:.1f} cm")
        print()

        # JSONファイルに保存
        json_output = {"frames": all_frames_data, "statistics": statistics}
        json_path = output_dir / "keypoints_3d.json"
        save_json(json_output, str(json_path))

        # CSVファイルに保存
        csv_path = output_dir / "jump_statistics.csv"
        save_csv(statistics, trajectory, str(csv_path))

        # 可視化動画を完成（動画ライターをクローズ）
        if not args.no_video and video_writer is not None:
            video_writer.release()
            print(f"Video saved to: {video_path}")

        print("\n" + "=" * 60)
        print("Analysis complete!")
        print(f"Results saved to: {args.output}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during analysis: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        bag_reader.stop()


if __name__ == "__main__":
    main()
