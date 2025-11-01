"""
Jump Detector and Measurement

ジャンプ検出アルゴリズム、高さ・距離計算、軌跡計算機能を提供
"""

import numpy as np
from collections import deque


class JumpDetector:
    """ジャンプ検出・測定クラス"""

    def __init__(self, threshold_vertical=0.05, threshold_horizontal=0.1, min_frames=10):
        """
        Args:
            threshold_vertical: 垂直ジャンプ検出の閾値（メートル）
            threshold_horizontal: 水平ジャンプ検出の閾値（メートル）
            min_frames: ジャンプと認識する最小フレーム数
        """
        self.threshold_vertical = threshold_vertical
        self.threshold_horizontal = threshold_horizontal
        self.min_frames = min_frames

        # 履歴データ
        self.height_history = deque(maxlen=30)  # 過去30フレームの高さ
        self.position_history = deque(maxlen=30)  # 過去30フレームの位置

        # ジャンプ検出状態
        self.jump_state = "ground"  # "ground", "jumping", "landed"
        self.jump_start_frame = None
        self.jump_start_position = None
        self.jump_max_height = 0.0
        self.jump_max_distance = 0.0

        # 検出されたジャンプのリスト
        self.detected_jumps = []

        # 軌跡データ
        self.trajectory = []

    def update(self, frame_num, keypoints_3d, timestamp=None):
        """
        新しいフレームのデータを更新し、ジャンプを検出

        Args:
            frame_num: フレーム番号
            keypoints_3d: 3D keypointsの辞書 {keypoint_name: (x, y, z), ...}
            timestamp: タイムスタンプ（オプション）

        Returns:
            dict: 検出結果と測定値
        """
        if keypoints_3d is None:
            return None

        # 基準点として腰（MidHip）または足首の平均を使用
        # 腰がない場合は、左右の股関節の中点を使用
        mid_hip = None
        left_hip = keypoints_3d.get("LHip")
        right_hip = keypoints_3d.get("RHip")

        if left_hip and right_hip and left_hip[2] is not None and right_hip[2] is not None:
            mid_hip = (
                (left_hip[0] + right_hip[0]) / 2 if left_hip[0] is not None and right_hip[0] is not None else None,
                (left_hip[1] + right_hip[1]) / 2 if left_hip[1] is not None and right_hip[1] is not None else None,
                (left_hip[2] + right_hip[2]) / 2 if left_hip[2] is not None and right_hip[2] is not None else None
            )

        # 足首の平均も計算
        left_ankle = keypoints_3d.get("LAnkle")
        right_ankle = keypoints_3d.get("RAnkle")
        mid_ankle = None

        if left_ankle and right_ankle and left_ankle[2] is not None and right_ankle[2] is not None:
            mid_ankle = (
                (left_ankle[0] + right_ankle[0]) / 2 if left_ankle[0] is not None and right_ankle[0] is not None else None,
                (left_ankle[1] + right_ankle[1]) / 2 if left_ankle[1] is not None and right_ankle[1] is not None else None,
                (left_ankle[2] + right_ankle[2]) / 2 if left_ankle[2] is not None and right_ankle[2] is not None else None
            )

        # 基準点（優先順位: 腰 > 足首）
        reference_point = mid_hip if mid_hip and mid_hip[2] is not None else mid_ankle

        if reference_point is None or reference_point[2] is None:
            return None

        x, y, z = reference_point

        # 履歴を更新
        self.height_history.append(z)
        self.position_history.append((x, y, z))

        # 軌跡を記録
        self.trajectory.append({
            "frame": frame_num,
            "timestamp": timestamp,
            "position": (x, y, z),
            "keypoints": keypoints_3d
        })

        # ジャンプ検出
        result = self._detect_jump(frame_num, x, y, z, timestamp)

        return result

    def _detect_jump(self, frame_num, x, y, z, timestamp):
        """ジャンプ検出ロジック"""
        if len(self.height_history) < self.min_frames:
            return {
                "state": "initializing",
                "frame": frame_num,
                "timestamp": timestamp,
                "height": z,
                "position": (x, y, z)
            }

        # 現在の高さと過去の平均高さを比較
        current_height = z
        avg_height = np.mean(list(self.height_history)[-self.min_frames:-1])

        # 高さの変化
        height_change = current_height - avg_height

        # 水平位置の変化（前フレームと比較）
        if len(self.position_history) >= 2:
            prev_pos = self.position_history[-2]
            horizontal_distance = np.sqrt((x - prev_pos[0])**2 + (y - prev_pos[1])**2)
        else:
            horizontal_distance = 0.0

        result = {
            "frame": frame_num,
            "timestamp": timestamp,
            "height": z,
            "position": (x, y, z),
            "height_change": height_change,
            "horizontal_distance": horizontal_distance,
            "jump_type": None,
            "jump_height": 0.0,
            "jump_distance": 0.0
        }

        # ジャンプ状態の遷移
        if self.jump_state == "ground":
            # 地面からジャンプ開始を検出
            if height_change > self.threshold_vertical:
                self.jump_state = "jumping"
                self.jump_start_frame = frame_num
                self.jump_start_position = (x, y, z)
                self.jump_max_height = current_height
                self.jump_max_distance = 0.0
                result["state"] = "jump_start"

        elif self.jump_state == "jumping":
            # ジャンプ中の最大高さ・距離を更新
            if current_height > self.jump_max_height:
                self.jump_max_height = current_height

            if self.jump_start_position:
                jump_distance = np.sqrt(
                    (x - self.jump_start_position[0])**2 +
                    (y - self.jump_start_position[1])**2
                )
                if jump_distance > self.jump_max_distance:
                    self.jump_max_distance = jump_distance

            # 着地を検出（高さが減少し、元の高さに戻る）
            if height_change < -self.threshold_vertical and current_height <= avg_height + 0.02:
                self.jump_state = "landed"
                result["state"] = "jump_end"

                # ジャンプの種類を判定
                if self.jump_max_distance < self.threshold_horizontal:
                    result["jump_type"] = "vertical"
                    result["jump_height"] = self.jump_max_height - self.jump_start_position[2]
                    result["jump_distance"] = 0.0
                else:
                    result["jump_type"] = "horizontal"
                    result["jump_height"] = self.jump_max_height - self.jump_start_position[2]
                    result["jump_distance"] = self.jump_max_distance

                # 検出されたジャンプを記録
                jump_data = {
                    "frame_start": self.jump_start_frame,
                    "frame_end": frame_num,
                    "jump_type": result["jump_type"],
                    "height": result["jump_height"],
                    "distance": result["jump_distance"],
                    "max_height": self.jump_max_height,
                    "start_position": self.jump_start_position,
                    "end_position": (x, y, z)
                }
                self.detected_jumps.append(jump_data)

                # 状態をリセット
                self.jump_state = "ground"
                self.jump_start_frame = None
                self.jump_start_position = None

        elif self.jump_state == "landed":
            # 着地後、少し待ってから地面状態に戻る
            if abs(height_change) < 0.01:
                self.jump_state = "ground"
                result["state"] = "ground"

        # ジャンプ中の場合の情報を追加
        if self.jump_state == "jumping":
            result["state"] = "jumping"
            result["jump_height"] = self.jump_max_height - (self.jump_start_position[2] if self.jump_start_position else avg_height)
            result["jump_distance"] = self.jump_max_distance

        return result

    def get_statistics(self):
        """検出されたジャンプの統計情報を取得"""
        if not self.detected_jumps:
            return {
                "total_jumps": 0,
                "vertical_jumps": 0,
                "horizontal_jumps": 0,
                "max_height": 0.0,
                "max_distance": 0.0,
                "avg_height": 0.0,
                "avg_distance": 0.0
            }

        vertical_jumps = [j for j in self.detected_jumps if j["jump_type"] == "vertical"]
        horizontal_jumps = [j for j in self.detected_jumps if j["jump_type"] == "horizontal"]

        all_heights = [j["height"] for j in self.detected_jumps]
        all_distances = [j["distance"] for j in horizontal_jumps] if horizontal_jumps else [0.0]

        return {
            "total_jumps": len(self.detected_jumps),
            "vertical_jumps": len(vertical_jumps),
            "horizontal_jumps": len(horizontal_jumps),
            "max_height": max(all_heights) if all_heights else 0.0,
            "max_distance": max(all_distances) if all_distances else 0.0,
            "avg_height": np.mean(all_heights) if all_heights else 0.0,
            "avg_distance": np.mean(all_distances) if all_distances else 0.0,
            "jumps": self.detected_jumps
        }

    def get_trajectory(self):
        """軌跡データを取得"""
        return self.trajectory

