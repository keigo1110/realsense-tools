"""
Keypoint Smoother

3Dキーポイントの時系列平滑化を行うクラス
移動平均フィルタと指数移動平均（EMA）フィルタを提供
"""

import numpy as np
from collections import deque


class KeypointSmoother:
    """3Dキーポイントの時系列平滑化クラス"""

    def __init__(self, window_size=5, smoothing_type='moving_average'):
        """
        Args:
            window_size: 平滑化ウィンドウサイズ（フレーム数）
            smoothing_type: 平滑化タイプ
                - 'moving_average': 移動平均（デフォルト）
                - 'ema': 指数移動平均（より反応が速いが若干遅延）
                - 'none': 平滑化なし
        """
        self.window_size = window_size
        self.smoothing_type = smoothing_type
        self.history = {}  # 各キーポイントの履歴 {keypoint_name: deque([(x, y, z), ...])}
        self.ema_values = {}  # EMA用の前回値 {keypoint_name: (x, y, z)}
        self.ema_alpha = 0.3  # EMAの平滑化係数（0-1、小さいほど滑らか）

    def smooth(self, keypoints_3d):
        """
        3Dキーポイントを平滑化

        Args:
            keypoints_3d: キーポイント辞書 {keypoint_name: (x, y, z), ...}

        Returns:
            平滑化されたキーポイント辞書
        """
        if self.smoothing_type == 'none':
            return keypoints_3d

        smoothed_keypoints = {}

        for kp_name, kp_3d in keypoints_3d.items():
            x, y, z = kp_3d

            # 無効なキーポイントはスキップ
            if x is None or y is None or z is None:
                smoothed_keypoints[kp_name] = (None, None, None)
                continue

            # 履歴を初期化
            if kp_name not in self.history:
                self.history[kp_name] = deque(maxlen=self.window_size)

            # 履歴に追加
            self.history[kp_name].append((x, y, z))

            # 平滑化を適用
            if self.smoothing_type == 'moving_average':
                smoothed = self._moving_average(kp_name)
            elif self.smoothing_type == 'ema':
                smoothed = self._ema(kp_name, (x, y, z))
            else:
                smoothed = (x, y, z)

            smoothed_keypoints[kp_name] = smoothed

        return smoothed_keypoints

    def _moving_average(self, kp_name):
        """移動平均を計算"""
        history = self.history[kp_name]

        if len(history) == 0:
            return (None, None, None)

        # 履歴が少ない場合は平均を計算
        if len(history) < self.window_size:
            # 現在の履歴の平均を計算
            x_vals = [h[0] for h in history if h[0] is not None]
            y_vals = [h[1] for h in history if h[1] is not None]
            z_vals = [h[2] for h in history if h[2] is not None]

            if x_vals and y_vals and z_vals:
                return (np.mean(x_vals), np.mean(y_vals), np.mean(z_vals))
            else:
                return history[-1]

        # ウィンドウサイズに達したら移動平均を計算
        x_vals = [h[0] for h in history if h[0] is not None]
        y_vals = [h[1] for h in history if h[1] is not None]
        z_vals = [h[2] for h in history if h[2] is not None]

        if x_vals and y_vals and z_vals:
            return (np.mean(x_vals), np.mean(y_vals), np.mean(z_vals))
        else:
            return history[-1]

    def _ema(self, kp_name, current_value):
        """指数移動平均（EMA）を計算"""
        x, y, z = current_value

        if kp_name not in self.ema_values:
            # 初回はそのまま使用
            self.ema_values[kp_name] = (x, y, z)
            return (x, y, z)

        # 前回のEMA値
        prev_x, prev_y, prev_z = self.ema_values[kp_name]

        # EMA計算: new_value = alpha * current + (1 - alpha) * previous
        new_x = self.ema_alpha * x + (1 - self.ema_alpha) * prev_x
        new_y = self.ema_alpha * y + (1 - self.ema_alpha) * prev_y
        new_z = self.ema_alpha * z + (1 - self.ema_alpha) * prev_z

        # 更新
        self.ema_values[kp_name] = (new_x, new_y, new_z)

        return (new_x, new_y, new_z)

    def reset(self):
        """履歴をリセット"""
        self.history.clear()
        self.ema_values.clear()

