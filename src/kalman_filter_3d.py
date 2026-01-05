"""
Kalman Filter for 3D Keypoint Tracking

研究用途向け: カルマンフィルタによる高精度な時系列平滑化と予測
OpenPoseやMotion Capture研究で広く使用される手法
"""

import numpy as np


class KalmanFilter3D:
    """
    3Dキーポイント用の簡易カルマンフィルタ
    
    研究用途: 動的システムの状態推定により、ノイズを除去しつつ
    真の軌跡を推定。OpenPoseなどの高精度なモーションキャプチャ
    システムで標準的に使用される手法。
    """
    
    def __init__(self, process_noise=0.01, measurement_noise=0.1, adaptive_noise=True):
        """
        Args:
            process_noise: プロセスノイズ（システムの不確実性、小さいほど滑らか）
            measurement_noise: 観測ノイズ（測定の不確実性、大きいほど予測を信頼）
            adaptive_noise: 適応的ノイズ調整を使用するか（デフォルト: True）
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.adaptive_noise = adaptive_noise
        
        # 適応的ノイズ調整用の統計情報
        self.measurement_history = []  # 観測値の履歴
        self.innovation_history = []  # イノベーション（観測値 - 予測値）の履歴
        self.history_size = 10
        
        # 状態: [x, y, z, vx, vy, vz] (位置と速度)
        self.state = None  # 状態ベクトル (6x1)
        self.covariance = None  # 共分散行列 (6x6)
        
        # 状態遷移行列 (等速直線運動モデル)
        dt = 1.0  # フレーム間隔（正規化）
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # 観測行列 (位置のみ観測)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)
        
        # プロセスノイズ共分散
        self.Q = np.eye(6, dtype=np.float32) * process_noise
        
        # 観測ノイズ共分散（初期値、適応的に調整される可能性がある）
        self.R_base = measurement_noise
        self.R = np.eye(3, dtype=np.float32) * measurement_noise
    
    def init(self, x, y, z):
        """フィルタを初期化"""
        # 初期状態: 位置と速度（速度は0で初期化）
        self.state = np.array([x, y, z, 0, 0, 0], dtype=np.float32).reshape(6, 1)
        
        # 初期共分散: 不確実性を大きく設定
        self.covariance = np.eye(6, dtype=np.float32) * 100.0
    
    def update(self, x, y, z):
        """
        観測値でフィルタを更新（予測→更新の2ステップ）
        
        Returns:
            tuple: 平滑化された (x, y, z)
        """
        if self.state is None:
            self.init(x, y, z)
            return (float(x), float(y), float(z))
        
        # 予測ステップ
        predicted_state = self.F @ self.state
        predicted_covariance = self.F @ self.covariance @ self.F.T + self.Q
        
        # 観測値
        z_obs = np.array([x, y, z], dtype=np.float32).reshape(3, 1)
        
        # 適応的ノイズ調整
        if self.adaptive_noise:
            # イノベーション（観測値 - 予測値）を計算
            innovation = z_obs - self.H @ predicted_state
            innovation_norm = np.linalg.norm(innovation)
            
            # イノベーションの履歴を更新
            self.innovation_history.append(innovation_norm)
            if len(self.innovation_history) > self.history_size:
                self.innovation_history.pop(0)
            
            # 観測値の変動を評価
            if len(self.innovation_history) >= 5:
                avg_innovation = np.mean(self.innovation_history)
                std_innovation = np.std(self.innovation_history)
                
                # 変動が大きい場合は観測ノイズを増やす（より滑らかに）
                if innovation_norm > avg_innovation + 2 * std_innovation:
                    # 外れ値の可能性が高い場合、観測ノイズを一時的に増やす
                    adaptive_measurement_noise = self.R_base * 2.0
                elif innovation_norm < avg_innovation - std_innovation:
                    # 変動が小さい場合、観測ノイズを減らす（より反応を速く）
                    adaptive_measurement_noise = self.R_base * 0.8
                else:
                    adaptive_measurement_noise = self.R_base
                
                self.R = np.eye(3, dtype=np.float32) * adaptive_measurement_noise
        
        # カルマンゲインの計算
        S = self.H @ predicted_covariance @ self.H.T + self.R
        K = predicted_covariance @ self.H.T @ np.linalg.inv(S)
        
        # 更新ステップ
        innovation = z_obs - self.H @ predicted_state
        self.state = predicted_state + K @ innovation
        self.covariance = (np.eye(6) - K @ self.H) @ predicted_covariance
        
        # 平滑化された位置を返す
        return (float(self.state[0, 0]), float(self.state[1, 0]), float(self.state[2, 0]))
    
    def predict(self):
        """
        次のフレームの位置を予測（観測値がない場合に使用）
        
        Returns:
            tuple: 予測された (x, y, z)
        """
        if self.state is None:
            return None
        
        predicted_state = self.F @ self.state
        return (float(predicted_state[0, 0]), float(predicted_state[1, 0]), float(predicted_state[2, 0]))


class KalmanSmoother:
    """複数のキーポイントにカルマンフィルタを適用"""
    
    def __init__(self, process_noise=0.01, measurement_noise=0.1, adaptive_noise=True):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.adaptive_noise = adaptive_noise
        self.filters = {}  # {keypoint_name: KalmanFilter3D}
    
    def smooth(self, keypoints_3d):
        """
        キーポイントをカルマンフィルタで平滑化
        
        Args:
            keypoints_3d: {keypoint_name: (x, y, z), ...}
        
        Returns:
            平滑化されたキーポイント辞書
        """
        smoothed = {}
        
        for kp_name, kp_3d in keypoints_3d.items():
            x, y, z = kp_3d
            
            if x is None or y is None or z is None:
                smoothed[kp_name] = (None, None, None)
                continue
            
            # フィルタが存在しない場合は作成
            if kp_name not in self.filters:
                self.filters[kp_name] = KalmanFilter3D(
                    process_noise=self.process_noise,
                    measurement_noise=self.measurement_noise,
                    adaptive_noise=self.adaptive_noise
                )
            
            # カルマンフィルタで更新
            smoothed[kp_name] = self.filters[kp_name].update(x, y, z)
        
        return smoothed
    
    def reset(self):
        """全てのフィルタをリセット"""
        self.filters.clear()

