"""
Visualizer

動画フレーム生成、軌跡描画、動画ファイル出力機能を提供
"""

import cv2
import numpy as np
from collections import deque


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

