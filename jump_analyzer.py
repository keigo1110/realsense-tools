"""
Jump Analyzer

.bagファイルからOpenPoseを使用して3D姿勢推定を行い、
ジャンプの高さ・距離・軌跡を測定するメインスクリプト
"""

import argparse
import json
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from realsense_utils import BagFileReader
from openpose_3d import OpenPoseDetector, COCO_KEYPOINTS
from jump_detector import JumpDetector
from visualizer import JumpVisualizer


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
                    "confidence": float(kp_2d[2])
                },
                "3d": {
                    "x": float(kp_3d[0]) if kp_3d and kp_3d[0] is not None else None,
                    "y": float(kp_3d[1]) if kp_3d and kp_3d[1] is not None else None,
                    "z": float(kp_3d[2]) if kp_3d and kp_3d[2] is not None else None
                }
            }
        else:
            result[keypoint_name] = {
                "2d": {"x": None, "y": None, "confidence": 0.0},
                "3d": {"x": None, "y": None, "z": None}
            }

    return result


def save_json(data, output_path):
    """JSONファイルに保存"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"JSON saved to: {output_path}")


def save_csv(statistics, trajectory, output_path):
    """CSVファイルに保存"""
    # 統計情報をCSVに保存
    stats_path = output_path.replace('.csv', '_statistics.csv')
    with open(stats_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Total Jumps', statistics.get('total_jumps', 0)])
        writer.writerow(['Vertical Jumps', statistics.get('vertical_jumps', 0)])
        writer.writerow(['Horizontal Jumps', statistics.get('horizontal_jumps', 0)])
        writer.writerow(['Max Height (cm)', statistics.get('max_height', 0) * 100])
        writer.writerow(['Max Distance (cm)', statistics.get('max_distance', 0) * 100])
        writer.writerow(['Avg Height (cm)', statistics.get('avg_height', 0) * 100])
        writer.writerow(['Avg Distance (cm)', statistics.get('avg_distance', 0) * 100])
    print(f"Statistics CSV saved to: {stats_path}")

    # ジャンプ詳細をCSVに保存
    if statistics.get('jumps'):
        jumps_path = output_path.replace('.csv', '_jumps.csv')
        with open(jumps_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Jump #', 'Type', 'Height (cm)', 'Distance (cm)',
                           'Start Frame', 'End Frame', 'Duration (frames)'])
            for i, jump in enumerate(statistics['jumps'], 1):
                writer.writerow([
                    i,
                    jump['jump_type'],
                    jump['height'] * 100,
                    jump['distance'] * 100,
                    jump['frame_start'],
                    jump['frame_end'],
                    jump['frame_end'] - jump['frame_start']
                ])
        print(f"Jumps CSV saved to: {jumps_path}")

    # 軌跡データをCSVに保存
    if trajectory:
        trajectory_path = output_path.replace('.csv', '_trajectory.csv')
        with open(trajectory_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame', 'Timestamp', 'X (m)', 'Y (m)', 'Z (m)'])
            for point in trajectory:
                pos = point.get('position', (None, None, None))
                writer.writerow([
                    point.get('frame', ''),
                    point.get('timestamp', ''),
                    pos[0] if pos[0] is not None else '',
                    pos[1] if pos[1] is not None else '',
                    pos[2] if pos[2] is not None else ''
                ])
        print(f"Trajectory CSV saved to: {trajectory_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Jump Analyzer: Analyze jump height, distance, and trajectory from RealSense .bag file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python jump_analyzer.py --input bagdata/my_recording.bag --output results/

  # Specify model directory
  python jump_analyzer.py --input bagdata/my_recording.bag --output results/ --model-dir models/

  # Adjust jump detection thresholds
  python jump_analyzer.py --input bagdata/my_recording.bag --output results/ --threshold-vertical 0.1 --threshold-horizontal 0.2
        """
    )

    # 必須引数
    parser.add_argument('--input', type=str, required=True,
                       help='Input .bag file path')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for results')

    # オプション引数
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory for OpenPose model files (default: models)')
    parser.add_argument('--threshold-vertical', type=float, default=0.05,
                       help='Vertical jump detection threshold in meters (default: 0.05)')
    parser.add_argument('--threshold-horizontal', type=float, default=0.1,
                       help='Horizontal jump detection threshold in meters (default: 0.1)')
    parser.add_argument('--no-video', action='store_true',
                       help='Skip video visualization output')

    args = parser.parse_args()

    # 出力ディレクトリを作成
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # バグファイルの存在確認
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("Jump Analyzer - OpenPose 3D Analysis")
    print("=" * 60)
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output}")
    print()

    # OpenPoseモデルの読み込み
    print("Loading OpenPose model...")
    openpose = OpenPoseDetector(model_dir=args.model_dir)
    if not openpose.load_model():
        print("Error: Failed to load OpenPose model", file=sys.stderr)
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
        threshold_horizontal=args.threshold_horizontal
    )

    # 可視化器の初期化
    visualizer = JumpVisualizer()

    # データ保存用のリスト
    all_frames_data = []
    visualization_frames = []

    print("\nProcessing frames...")
    frame_num = 0

    try:
        while True:
            # フレームを取得
            color_frame, depth_frame = bag_reader.get_frames()

            if color_frame is None or depth_frame is None:
                break

            color_image, depth_image = bag_reader.frame_to_numpy(color_frame, depth_frame)

            if color_image is None or depth_image is None:
                continue

            frame_num += 1

            # 2D keypointsを検出
            keypoints_2d = openpose.detect_keypoints(color_image)

            if keypoints_2d is None:
                continue

            # 3D keypointsを計算
            keypoints_3d = {}
            for i, (kp_name, kp_2d) in enumerate(zip(COCO_KEYPOINTS, keypoints_2d)):
                if kp_2d[0] is not None and kp_2d[1] is not None:
                    # 深度値を取得
                    depth = bag_reader.get_depth_at_point(depth_frame, kp_2d[0], kp_2d[1])

                    if depth is not None:
                        # 3D座標に変換
                        x, y, z = bag_reader.pixel_to_3d(kp_2d[0], kp_2d[1], depth)
                        keypoints_3d[kp_name] = (x, y, z)
                    else:
                        keypoints_3d[kp_name] = (None, None, None)
                else:
                    keypoints_3d[kp_name] = (None, None, None)

            # ジャンプ検出
            timestamp = color_frame.get_timestamp()
            jump_result = jump_detector.update(frame_num, keypoints_3d, timestamp)

            # フレームデータを保存
            frame_data = {
                "frame": frame_num,
                "timestamp": timestamp,
                "keypoints": convert_keypoints_to_dict(keypoints_2d, keypoints_3d)
            }

            if jump_result:
                frame_data["jump_result"] = {
                    "state": jump_result.get("state", "unknown"),
                    "height": jump_result.get("height"),
                    "position": jump_result.get("position"),
                    "jump_type": jump_result.get("jump_type"),
                    "jump_height": jump_result.get("jump_height"),
                    "jump_distance": jump_result.get("jump_distance")
                }

            all_frames_data.append(frame_data)

            # 可視化フレームを生成
            if not args.no_video:
                # スケルトンを描画
                skeleton_image = openpose.draw_skeleton(color_image.copy(), keypoints_2d)

                # 統計情報を取得
                statistics = jump_detector.get_statistics()

                # 可視化フレームを生成
                vis_frame = visualizer.draw_frame(skeleton_image, keypoints_2d, jump_result, statistics)
                visualization_frames.append(vis_frame)

            # 進捗表示
            if frame_num % 30 == 0:
                print(f"Processed {frame_num} frames...")

        print(f"\nFinished processing {frame_num} frames")

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
        json_output = {
            "frames": all_frames_data,
            "statistics": statistics
        }
        json_path = output_dir / "keypoints_3d.json"
        save_json(json_output, str(json_path))

        # CSVファイルに保存
        csv_path = output_dir / "jump_statistics.csv"
        save_csv(statistics, trajectory, str(csv_path))

        # 可視化動画を生成
        if not args.no_video and visualization_frames:
            video_path = output_dir / "jump_visualization.mp4"
            fps = 30  # デフォルトFPS（.bagファイルから取得する場合は調整）
            visualizer.create_video(visualization_frames, str(video_path), fps)

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

