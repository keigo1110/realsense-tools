"""
Visualizer

動画フレーム生成、軌跡描画、動画ファイル出力機能を提供
3Dキーポイントアニメーション生成機能を含む
"""

import cv2
import numpy as np
from collections import deque
import json
from pathlib import Path

# matplotlib関連のインポート（3Dアニメーション用）
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.animation as animation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class JumpVisualizer:
    """ジャンプ分析結果の可視化クラス"""

    def __init__(self, trajectory_length=60):
        """
        Args:
            trajectory_length: 描画する軌跡の最大長さ（フレーム数）
        """
        self.trajectory_length = trajectory_length
        self.trajectory_points = deque(maxlen=trajectory_length)

    def draw_frame(self, image, keypoints_2d, jump_result=None, statistics=None):
        """
        可視化フレームを生成

        Args:
            image: 元の画像
            keypoints_2d: 2D keypointsのリスト [(x, y, confidence), ...]
            jump_result: ジャンプ検出結果（辞書）
            statistics: 統計情報（辞書）

        Returns:
            numpy.ndarray: 可視化済み画像
        """
        vis_image = image.copy()

        # keypointsを描画（簡易版 - スケルトン描画はopenpose_3d.pyで行う）
        if keypoints_2d:
            for i, (x, y, conf) in enumerate(keypoints_2d):
                if conf > 0.1 and x is not None and y is not None:
                    cv2.circle(vis_image, (int(x), int(y)), 4, (0, 0, 255), -1)

        # ジャンプ結果の情報を表示
        if jump_result:
            # 状態を表示
            state = jump_result.get("state", "unknown")
            state_colors = {
                "ground": (0, 255, 0),
                "jumping": (0, 165, 255),
                "jump_start": (255, 0, 0),
                "jump_end": (255, 0, 255)
            }
            state_color = state_colors.get(state, (255, 255, 255))

            # 背景を描画（可読性向上）
            cv2.rectangle(vis_image, (10, 10), (400, 150), (0, 0, 0), -1)
            cv2.rectangle(vis_image, (10, 10), (400, 150), state_color, 2)

            # テキストを描画
            y_offset = 30
            cv2.putText(vis_image, f"State: {state.upper()}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)

            y_offset += 25
            if jump_result.get("height") is not None:
                height_m = jump_result["height"]
                height_cm = height_m * 100
                cv2.putText(vis_image, f"Height: {height_cm:.1f} cm", (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            y_offset += 25
            if jump_result.get("jump_height") is not None and jump_result["jump_height"] > 0:
                jump_height_cm = jump_result["jump_height"] * 100
                cv2.putText(vis_image, f"Jump Height: {jump_height_cm:.1f} cm", (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            y_offset += 25
            if jump_result.get("jump_distance") is not None and jump_result["jump_distance"] > 0:
                jump_distance_cm = jump_result["jump_distance"] * 100
                cv2.putText(vis_image, f"Jump Distance: {jump_distance_cm:.1f} cm", (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            y_offset += 25
            if jump_result.get("jump_type"):
                jump_type = jump_result["jump_type"].upper()
                cv2.putText(vis_image, f"Type: {jump_type}", (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

        # 統計情報を表示
        if statistics:
            stats_y = vis_image.shape[0] - 120
            cv2.rectangle(vis_image, (10, stats_y), (350, vis_image.shape[0] - 10), (0, 0, 0), -1)

            y_offset = stats_y + 25
            cv2.putText(vis_image, f"Total Jumps: {statistics.get('total_jumps', 0)}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            y_offset += 20
            if statistics.get("max_height", 0) > 0:
                max_height_cm = statistics["max_height"] * 100
                cv2.putText(vis_image, f"Max Height: {max_height_cm:.1f} cm", (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            y_offset += 20
            if statistics.get("max_distance", 0) > 0:
                max_distance_cm = statistics["max_distance"] * 100
                cv2.putText(vis_image, f"Max Distance: {max_distance_cm:.1f} cm", (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 軌跡を描画
        if jump_result and jump_result.get("position"):
            pos = jump_result["position"]
            if pos[0] is not None and pos[1] is not None:
                self.trajectory_points.append((int(pos[0]), int(pos[1])))

        # 軌跡を線で描画
        if len(self.trajectory_points) > 1:
            points = list(self.trajectory_points)
            for i in range(1, len(points)):
                # 色を徐々に変化（新しい点ほど明るい）
                alpha = i / len(points)
                color = (int(255 * alpha), int(255 * (1 - alpha)), 128)
                cv2.line(vis_image, points[i-1], points[i], color, 2)

        return vis_image

    def create_video(self, frames, output_path, fps=30):
        """
        可視化フレームから動画ファイルを生成

        Args:
            frames: フレームのリスト
            output_path: 出力ファイルのパス
            fps: フレームレート
        """
        if not frames:
            print("No frames to write")
            return

        height, width = frames[0].shape[:2]

        # 動画ライターを初期化
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in frames:
            out.write(frame)

        out.release()
        print(f"Video saved to: {output_path}")


def create_3d_keypoint_animation(json_path, output_path=None, fps=30, keypoint_names=None, interactive=False):
    """
    3Dキーポイントアニメーションを生成

    Args:
        json_path: keypoints_3d.jsonファイルのパス
        output_path: 出力アニメーションファイルのパス（.gifまたは.mp4）。Noneの場合はインタラクティブ表示
        fps: アニメーションのフレームレート
        keypoint_names: 使用するkeypoint名のリスト（Noneの場合はCOCO_KEYPOINTSを使用）
        interactive: Trueの場合、インタラクティブウィンドウを表示（視点を自由に変更可能）
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is required for 3D animation. Install with: pip install matplotlib")
        return False

    try:
        # JSONファイルを読み込み
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        frames = data.get('frames', [])
        if not frames:
            print("Error: No frame data found in JSON file")
            return False

        # COCO_KEYPOINTSとCOCO_PAIRSをインポート
        try:
            # 相対インポートを優先
            from .yolov8_pose_3d import COCO_KEYPOINTS, COCO_PAIRS
            if keypoint_names is None:
                keypoint_names = COCO_KEYPOINTS
        except ImportError:
            # 絶対インポートを試す（外部から呼ばれた場合）
            try:
                from src.yolov8_pose_3d import COCO_KEYPOINTS, COCO_PAIRS
                if keypoint_names is None:
                    keypoint_names = COCO_KEYPOINTS
            except ImportError:
            # フォールバック: デフォルトのkeypoint名とペアを使用
            if keypoint_names is None:
                keypoint_names = [
                    "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
                    "LShoulder", "LElbow", "LWrist", "RHip", "RKnee", "RAnkle",
                    "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar"
                ]
            COCO_PAIRS = [
                [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
                [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
                [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]
            ]

        # 各フレームから3D keypointsを抽出
        keypoints_3d_list = []
        for frame_data in frames:
            keypoints = frame_data.get('keypoints', {})
            kp_3d = {}
            for kp_name in keypoint_names:
                kp_data = keypoints.get(kp_name, {})
                kp_3d_data = kp_data.get('3d', {})
                x = kp_3d_data.get('x')
                y = kp_3d_data.get('y')
                z = kp_3d_data.get('z')
                kp_3d[kp_name] = (x, y, z)
            keypoints_3d_list.append(kp_3d)

        # 有効な3D座標があるかチェック
        has_valid_data = False
        for kp_3d in keypoints_3d_list:
            for kp_name, (x, y, z) in kp_3d.items():
                if x is not None and y is not None and z is not None:
                    has_valid_data = True
                    break
            if has_valid_data:
                break

        if not has_valid_data:
            print("Error: No valid 3D keypoint data found")
            return False

        # 座標の範囲を計算（アスペクト比を保つため）
        all_x = []
        all_y = []
        all_z = []
        for kp_3d in keypoints_3d_list:
            for x, y, z in kp_3d.values():
                if x is not None and y is not None and z is not None:
                    all_x.append(x)
                    all_y.append(y)
                    all_z.append(z)

        if not all_x:
            print("Error: No valid 3D coordinates found")
            return False

        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        z_min, z_max = min(all_z), max(all_z)

        # マージンを追加
        margin = 0.1
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2

        max_range = max(x_range, y_range, z_range) * (1 + margin)

        # アニメーションを作成
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # 軸ラベルを設定（XZ平面を底面としてY軸が上方向）
        ax.set_xlabel('X (m) - Right', fontsize=10)
        ax.set_ylabel('Z (m) - Forward', fontsize=10)
        ax.set_zlabel('Y (m) - Up', fontsize=10)
        ax.set_title('3D Keypoints Animation', fontsize=14)
        
        # XZ平面を底面として表示するための初期視点設定
        # elev: 仰角（上から見下ろす角度）、azim: 方位角（回転角度）
        initial_view = {'elev': 10, 'azim': 45}
        ax.view_init(**initial_view)
        
        # インタラクティブモード用: プロット要素を追跡（外部変数として定義）
        plot_objects = [] if interactive else None
        
        if interactive:
            # インタラクティブモードでも軸の範囲とラベルを設定
            ax.set_xlim([x_center - max_range/2, x_center + max_range/2])
            ax.set_ylim([z_center - max_range/2, z_center + max_range/2])
            ax.set_zlim([-y_center - max_range/2, -y_center + max_range/2])
            ax.set_xlabel('X (m) - Right', fontsize=10)
            ax.set_ylabel('Z (m) - Forward', fontsize=10)
            ax.set_zlabel('Y (m) - Up', fontsize=10)

        def update_frame(frame_idx):
            # インタラクティブモードでは視点を保持するためclear()を使わない
            if interactive:
                # 前回のプロット要素を削除
                for obj in plot_objects:
                    try:
                        obj.remove()
                    except:
                        pass
                plot_objects.clear()
                ax.set_title(f'3D Keypoints Animation - Frame {frame_idx+1}/{len(keypoints_3d_list)}', fontsize=14)
            else:
                ax.clear()
                ax.view_init(**initial_view)
            
            # 軸の範囲を設定（非インタラクティブモードのみ）
            if not interactive:
                ax.set_xlim([x_center - max_range/2, x_center + max_range/2])
                ax.set_ylim([z_center - max_range/2, z_center + max_range/2])  # Z軸をY軸として表示
                ax.set_zlim([-y_center - max_range/2, -y_center + max_range/2])  # -Y軸をZ軸として表示（上下反転）

                ax.set_xlabel('X (m) - Right', fontsize=10)
                ax.set_ylabel('Z (m) - Forward', fontsize=10)
                ax.set_zlabel('Y (m) - Up', fontsize=10)
                ax.set_title(f'3D Keypoints Animation - Frame {frame_idx+1}/{len(keypoints_3d_list)}', fontsize=14)

            if frame_idx >= len(keypoints_3d_list):
                return []

            kp_3d = keypoints_3d_list[frame_idx]

            # keypointの座標を取得
            kp_coords = {}
            for i, kp_name in enumerate(keypoint_names):
                x, y, z = kp_3d.get(kp_name, (None, None, None))
                if x is not None and y is not None and z is not None:
                    kp_coords[i] = (x, y, z)

            # スケルトンを描画（XZ平面を底面としてY軸が上方向になるように座標変換）
            # RealSense座標: (x, y, z) -> 表示座標: (x, z, -y)
            for pair in COCO_PAIRS:
                idx1, idx2 = pair[0], pair[1]
                if idx1 in kp_coords and idx2 in kp_coords:
                    x1, y1, z1 = kp_coords[idx1]
                    x2, y2, z2 = kp_coords[idx2]
                    # XZ平面を底面として表示（Y軸を上下反転）
                    line = ax.plot([x1, x2], [z1, z2], [-y1, -y2], 'b-', linewidth=2, alpha=0.6)
                    if interactive:
                        plot_objects.extend(line)

            # keypointを描画（座標変換を適用）
            for idx, (x, y, z) in kp_coords.items():
                kp_name = keypoint_names[idx]
                # 重要なkeypointは大きく、その他は小さく
                if kp_name in ['Neck', 'Nose', 'RHip', 'LHip']:
                    size = 50
                else:
                    size = 30
                # XZ平面を底面として表示（Y軸を上下反転）
                scatter = ax.scatter([x], [z], [-y], c='red', s=size, marker='o')
                if interactive:
                    plot_objects.append(scatter)

            return []

        # アニメーション作成
        interval = 1000 / fps  # ミリ秒
        anim = animation.FuncAnimation(
            fig, update_frame, frames=len(keypoints_3d_list),
            interval=interval, blit=False, repeat=True
        )

        # インタラクティブモードの場合
        if interactive or output_path is None:
            print(f"Displaying interactive 3D keypoint animation ({len(keypoints_3d_list)} frames)...")
            print("Controls:")
            print("  - Mouse drag: Rotate view")
            print("  - Mouse wheel: Zoom in/out")
            print("  - Close window to exit")
            plt.ion()  # インタラクティブモードを有効化
            plt.show(block=True)  # ウィンドウが閉じられるまで待機
            plt.ioff()  # インタラクティブモードを無効化
            return True

        # ファイルとして保存
        print(f"Creating 3D keypoint animation ({len(keypoints_3d_list)} frames)...")
        
        output_ext = Path(output_path).suffix.lower()
        
        # GIF形式で保存（Pillowを使用）
        if output_ext != '.gif':
            # 拡張子がGIF以外の場合はGIFとして保存
            output_path = str(Path(output_path).with_suffix('.gif'))
        
        try:
            anim.save(output_path, writer='pillow', fps=fps)
        except Exception as e:
            print(f"Error: Failed to save animation with Pillow. Error: {e}")
            print("Please install Pillow: pip install pillow")
            plt.close(fig)
            return False

        print(f"3D keypoint animation saved to: {output_path}")
        plt.close(fig)
        return True

    except Exception as e:
        print(f"Error creating 3D animation: {e}")
        import traceback
        traceback.print_exc()
        return False

