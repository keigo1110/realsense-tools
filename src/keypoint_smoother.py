"""
Keypoint Smoother

3Dキーポイントの時系列平滑化を行うクラス
移動平均フィルタと指数移動平均（EMA）フィルタを提供
"""

import numpy as np
from collections import deque


class KeypointSmoother:
    """3Dキーポイントの時系列平滑化クラス"""

    def __init__(self, window_size=5, smoothing_type='moving_average', use_weighted_average=True):
        """
        Args:
            window_size: 平滑化ウィンドウサイズ（フレーム数）
            smoothing_type: 平滑化タイプ
                - 'moving_average': 移動平均（デフォルト）
                - 'ema': 指数移動平均（より反応が速いが若干遅延）
                - 'gaussian': ガウシアン重み付き移動平均（より滑らか）
                - 'none': 平滑化なし
            use_weighted_average: 重み付き平均を使用するか（中央のフレームに重みを付ける）
        """
        self.window_size = window_size
        self.smoothing_type = smoothing_type
        self.use_weighted_average = use_weighted_average
        self.history = {}  # 各キーポイントの履歴 {keypoint_name: deque([(x, y, z), ...])}
        self.ema_values = {}  # EMA用の前回値 {keypoint_name: (x, y, z)}
        self.ema_alpha = 0.3  # EMAの平滑化係数（0-1、小さいほど滑らか）
        
        # ガウシアン重みの計算（中央に重みを付ける）
        if self.window_size > 1:
            self.gaussian_weights = self._compute_gaussian_weights(self.window_size)
        else:
            self.gaussian_weights = None

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
            elif self.smoothing_type == 'gaussian':
                smoothed = self._gaussian_weighted_average(kp_name)
            else:
                smoothed = (x, y, z)

            smoothed_keypoints[kp_name] = smoothed

        return smoothed_keypoints

    def _moving_average(self, kp_name):
        """移動平均を計算（重み付き平均を使用可能）"""
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
        if self.use_weighted_average and len(history) == self.window_size:
            # 重み付き平均（中央のフレームに重みを付ける）
            weights = self._compute_linear_weights(len(history))
            x_vals = [h[0] for h in history if h[0] is not None]
            y_vals = [h[1] for h in history if h[1] is not None]
            z_vals = [h[2] for h in history if h[2] is not None]
            
            if x_vals and y_vals and z_vals and len(weights) == len(history):
                # 有効な値のみを使用
                valid_indices = [i for i, h in enumerate(history) if h[0] is not None]
                if valid_indices:
                    valid_weights = [weights[i] for i in valid_indices]
                    valid_weights = np.array(valid_weights)
                    valid_weights = valid_weights / np.sum(valid_weights)  # 正規化
                    
                    x_weighted = np.average([history[i][0] for i in valid_indices], weights=valid_weights)
                    y_weighted = np.average([history[i][1] for i in valid_indices], weights=valid_weights)
                    z_weighted = np.average([history[i][2] for i in valid_indices], weights=valid_weights)
                    return (x_weighted, y_weighted, z_weighted)
        
        # 通常の移動平均
        x_vals = [h[0] for h in history if h[0] is not None]
        y_vals = [h[1] for h in history if h[1] is not None]
        z_vals = [h[2] for h in history if h[2] is not None]

        if x_vals and y_vals and z_vals:
            return (np.mean(x_vals), np.mean(y_vals), np.mean(z_vals))
        else:
            return history[-1]
    
    def _gaussian_weighted_average(self, kp_name):
        """ガウシアン重み付き移動平均を計算"""
        history = self.history[kp_name]
        
        if len(history) == 0:
            return (None, None, None)
        
        if len(history) < self.window_size:
            # 履歴が少ない場合は通常の平均
            x_vals = [h[0] for h in history if h[0] is not None]
            y_vals = [h[1] for h in history if h[1] is not None]
            z_vals = [h[2] for h in history if h[2] is not None]
            
            if x_vals and y_vals and z_vals:
                return (np.mean(x_vals), np.mean(y_vals), np.mean(z_vals))
            else:
                return history[-1]
        
        # ガウシアン重みを使用
        if self.gaussian_weights is not None and len(history) == len(self.gaussian_weights):
            valid_indices = [i for i, h in enumerate(history) if h[0] is not None]
            if valid_indices:
                valid_weights = [self.gaussian_weights[i] for i in valid_indices]
                valid_weights = np.array(valid_weights)
                valid_weights = valid_weights / np.sum(valid_weights)  # 正規化
                
                x_weighted = np.average([history[i][0] for i in valid_indices], weights=valid_weights)
                y_weighted = np.average([history[i][1] for i in valid_indices], weights=valid_weights)
                z_weighted = np.average([history[i][2] for i in valid_indices], weights=valid_weights)
                return (x_weighted, y_weighted, z_weighted)
        
        # フォールバック: 通常の平均
        x_vals = [h[0] for h in history if h[0] is not None]
        y_vals = [h[1] for h in history if h[1] is not None]
        z_vals = [h[2] for h in history if h[2] is not None]

        if x_vals and y_vals and z_vals:
            return (np.mean(x_vals), np.mean(y_vals), np.mean(z_vals))
        else:
            return history[-1]
    
    def _compute_gaussian_weights(self, window_size, sigma=None):
        """ガウシアン重みを計算"""
        if sigma is None:
            # 標準偏差をウィンドウサイズに基づいて自動設定
            sigma = window_size / 3.0
        
        center = window_size // 2
        weights = []
        for i in range(window_size):
            weight = np.exp(-0.5 * ((i - center) / sigma) ** 2)
            weights.append(weight)
        
        return np.array(weights)
    
    def _compute_linear_weights(self, window_size):
        """線形重みを計算（中央に重みを付ける）"""
        center = window_size // 2
        weights = []
        for i in range(window_size):
            # 中央に近いほど重みが大きい
            weight = 1.0 - abs(i - center) / (center + 1.0)
            weights.append(max(0.1, weight))  # 最小重みを0.1に設定
        return np.array(weights)

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

