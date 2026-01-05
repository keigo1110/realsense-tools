"""
Floor Detection

深度データから床平面を検出するクラス
RANSACアルゴリズムを使用した平面検出を提供
"""

import numpy as np


class FloorDetector:
    """床平面検出クラス"""

    def __init__(self, floor_threshold=0.03, min_inliers=500, max_iterations=500, 
                 history_size=20, use_outlier_removal=True, adaptive_threshold=True):
        """
        Args:
            floor_threshold: 床平面からの距離閾値（メートル、デフォルト: 3cm）
            min_inliers: 床平面と判定する最小インライア数（デフォルト: 500）
            max_iterations: RANSACの最大反復回数（デフォルト: 500）
            history_size: 履歴サイズ（デフォルト: 20、複数フレームの統合に使用）
            use_outlier_removal: 外れ値除去を使用するか（デフォルト: True）
            adaptive_threshold: 適応的閾値を使用するか（デフォルト: True）
        """
        self.floor_threshold = floor_threshold
        self.min_inliers = min_inliers
        self.max_iterations = max_iterations
        self.use_outlier_removal = use_outlier_removal
        self.adaptive_threshold = adaptive_threshold
        
        # 検出された床平面のパラメータ (ax + by + cz + d = 0)
        self.floor_plane = None  # (a, b, c, d) または None
        self.floor_normal = None  # (nx, ny, nz) 法線ベクトル
        self.floor_height = None  # 床の高さ（カメラ座標系のZ値）
        
        # 履歴（安定化のため、複数フレームの統合に使用）
        self.floor_history = []  # 過去の検出結果
        self.history_size = history_size
        
        # 統計情報（適応的パラメータ調整に使用）
        self.inlier_count_history = []  # インライア数の履歴
        self.plane_quality_history = []  # 平面の品質スコアの履歴
        
    def detect_floor_from_depth(self, depth_frame, depth_scale, intrinsics):
        """
        深度フレームから床平面を検出（RANSACアルゴリズム）
        
        Args:
            depth_frame: RealSense深度フレーム
            depth_scale: 深度スケール
            intrinsics: カメラ内部パラメータ
        
        Returns:
            bool: 床が検出されたかどうか
        """
        if depth_frame is None:
            return False
        
        # 深度画像を取得
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # 有効な深度値を持つ点を3D座標に変換
        points_3d = []
        height, width = depth_image.shape
        
        # サンプリング（処理を高速化）
        step = 10  # 10ピクセルおきにサンプリング
        
        for y in range(0, height, step):
            for x in range(0, width, step):
                depth_value = depth_image[y, x]
                if depth_value > 0:  # 有効な深度値
                    depth_m = depth_value * depth_scale
                    
                    # 3D座標に変換
                    x_3d = (x - intrinsics.ppx) / intrinsics.fx * depth_m
                    y_3d = (y - intrinsics.ppy) / intrinsics.fy * depth_m
                    z_3d = depth_m
                    
                    # 範囲チェック（現実的な範囲のみ）
                    if 0.3 < z_3d < 5.0:  # 0.3m～5mの範囲
                        points_3d.append([x_3d, y_3d, z_3d])
        
        if len(points_3d) < self.min_inliers:
            return False
        
        points_3d = np.array(points_3d)
        
        # 外れ値の除去（統計的手法）
        if self.use_outlier_removal and len(points_3d) > 100:
            points_3d = self._remove_outliers(points_3d)
        
        if len(points_3d) < self.min_inliers:
            return False
        
        # 適応的閾値の調整
        current_threshold = self.floor_threshold
        if self.adaptive_threshold and len(self.inlier_count_history) > 5:
            # 過去のインライア数の平均から適切な閾値を推定
            avg_inliers = np.mean(self.inlier_count_history[-5:])
            if avg_inliers > len(points_3d) * 0.5:  # インライアが多い場合
                # 閾値を厳格化（より正確な平面を検出）
                current_threshold = self.floor_threshold * 0.8
            elif avg_inliers < len(points_3d) * 0.2:  # インライアが少ない場合
                # 閾値を緩和（検出を容易に）
                current_threshold = self.floor_threshold * 1.2
        
        # RANSACで平面を検出
        best_plane = None
        best_inliers = []
        max_inlier_count = 0
        
        for _ in range(self.max_iterations):
            # 3点をランダムに選択
            sample_indices = np.random.choice(len(points_3d), 3, replace=False)
            p1, p2, p3 = points_3d[sample_indices]
            
            # 2つのベクトル
            v1 = p2 - p1
            v2 = p3 - p1
            
            # 法線ベクトル（外積）
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            
            if norm < 1e-6:  # 3点が一直線上
                continue
            
            normal = normal / norm
            
            # 平面の方程式: ax + by + cz + d = 0
            # normal = (a, b, c), d = -(ax0 + by0 + cz0) where (x0, y0, z0) is a point on plane
            d = -np.dot(normal, p1)
            plane = np.array([normal[0], normal[1], normal[2], d])
            
            # 全ての点から平面までの距離を計算
            distances = np.abs(
                plane[0] * points_3d[:, 0] +
                plane[1] * points_3d[:, 1] +
                plane[2] * points_3d[:, 2] +
                plane[3]
            )
            
            # インライア（平面に近い点）
            inliers = np.where(distances < current_threshold)[0]
            
            if len(inliers) > max_inlier_count:
                max_inlier_count = len(inliers)
                best_plane = plane
                best_inliers = inliers
        
        # 十分なインライアが見つかった場合
        if max_inlier_count >= self.min_inliers and best_plane is not None:
            # インライアのみで平面を再計算（より正確に）
            inlier_points = points_3d[best_inliers]
            
            # 外れ値の除去（インライア内でも外れ値を除去）
            if self.use_outlier_removal and len(inlier_points) > 50:
                inlier_points = self._remove_outliers(inlier_points)
            
            if len(inlier_points) < self.min_inliers:
                return False
            
            # 最小二乗法で最適な平面を計算
            # 平面の方程式: z = ax + by + c の形に変換
            # より安定した方法: 主成分分析（PCA）を使用
            mean_point = np.mean(inlier_points, axis=0)
            centered_points = inlier_points - mean_point
            
            # 共分散行列
            cov_matrix = np.cov(centered_points.T)
            
            # 固有値・固有ベクトル
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # 最小固有値に対応する固有ベクトルが法線（最も分散が小さい方向）
            normal_idx = np.argmin(eigenvalues)
            normal = eigenvectors[:, normal_idx]
            
            # 法線の向きを調整（上向き、つまりY軸が負の方向）
            if normal[1] > 0:
                normal = -normal
            
            # 平面の方程式: ax + by + cz + d = 0
            d = -np.dot(normal, mean_point)
            current_plane = np.array([normal[0], normal[1], normal[2], d])
            current_normal = normal
            current_height = mean_point[1]  # RealSense座標系ではYが下方向
            
            # 平面の品質スコアを計算（インライア率と平面の安定性）
            inlier_ratio = len(inlier_points) / len(points_3d)
            plane_quality = inlier_ratio * (1.0 - eigenvalues[normal_idx] / np.sum(eigenvalues))
            
            # 履歴に追加（安定化のため）
            self.floor_history.append({
                'plane': current_plane.copy(),
                'normal': current_normal.copy(),
                'height': current_height,
                'quality': plane_quality,
                'inlier_count': len(inlier_points)
            })
            
            # 統計情報を更新
            self.inlier_count_history.append(len(inlier_points))
            self.plane_quality_history.append(plane_quality)
            if len(self.inlier_count_history) > self.history_size:
                self.inlier_count_history.pop(0)
            if len(self.plane_quality_history) > self.history_size:
                self.plane_quality_history.pop(0)
            
            # 履歴サイズを制限
            if len(self.floor_history) > self.history_size:
                self.floor_history.pop(0)
            
            # 複数フレームの統合（外れ値を除去してから平均を計算）
            if len(self.floor_history) > 3:
                # 外れ値を除去してから平均を計算
                filtered_history = self._filter_history_outliers(self.floor_history)
                
                if len(filtered_history) > 0:
                    # 重み付き平均（品質スコアを重みとして使用）
                    total_weight = sum(h['quality'] for h in filtered_history)
                    if total_weight > 0:
                        weighted_plane = np.zeros(4)
                        weighted_normal = np.zeros(3)
                        weighted_height = 0.0
                        
                        for h in filtered_history:
                            weight = h['quality'] / total_weight
                            weighted_plane += h['plane'] * weight
                            weighted_normal += h['normal'] * weight
                            weighted_height += h['height'] * weight
                        
                        # 正規化
                        norm = np.linalg.norm(weighted_normal)
                        if norm > 1e-6:
                            weighted_normal = weighted_normal / norm
                            weighted_plane[0:3] = weighted_normal
                            # dを再計算
                            weighted_plane[3] = -np.dot(weighted_normal, np.array([0, weighted_height, 0]))
                        
                        self.floor_plane = weighted_plane
                        self.floor_normal = weighted_normal
                        self.floor_height = weighted_height
                    else:
                        # フォールバック: 通常の平均
                        avg_plane = np.mean([h['plane'] for h in filtered_history], axis=0)
                        avg_normal = np.mean([h['normal'] for h in filtered_history], axis=0)
                        avg_height = np.mean([h['height'] for h in filtered_history])
                
                # 正規化
                norm = np.linalg.norm(avg_normal)
                if norm > 1e-6:
                    avg_normal = avg_normal / norm
                    avg_plane[0:3] = avg_normal
                    avg_plane[3] = -np.dot(avg_normal, np.array([0, avg_height, 0]))
                    
                    self.floor_plane = avg_plane
                    self.floor_normal = avg_normal
                    self.floor_height = avg_height
                else:
                    # 外れ値除去後もデータがない場合は最新の結果を使用
                    self.floor_plane = current_plane
                    self.floor_normal = current_normal
                    self.floor_height = current_height
            else:
                # 履歴が少ない場合は最新の結果を使用
                self.floor_plane = current_plane
                self.floor_normal = current_normal
                self.floor_height = current_height
            
            return True
        
        return False
    
    def get_floor_height_at_point(self, x, z):
        """
        指定したX, Z座標での床の高さ（Y座標）を取得
        
        Args:
            x: X座標（メートル）
            z: Z座標（メートル）
        
        Returns:
            float: 床のY座標（メートル）、検出されていない場合はNone
        """
        if self.floor_plane is None:
            return None
        
        # 平面の方程式: ax + by + cz + d = 0
        # y = -(ax + cz + d) / b (b != 0の場合)
        a, b, c, d = self.floor_plane
        
        if abs(b) < 1e-6:
            return None
        
        y = -(a * x + c * z + d) / b
        return y
    
    def distance_to_floor(self, point_3d):
        """
        3D点から床平面までの距離を計算（床の法線ベクトルを軸にした高さ）
        
        Args:
            point_3d: (x, y, z) タプルまたはリスト
        
        Returns:
            float: 床までの距離（メートル）、検出されていない場合はNone
        """
        if self.floor_plane is None or point_3d is None:
            return None
        
        if point_3d[0] is None or point_3d[1] is None or point_3d[2] is None:
            return None
        
        x, y, z = point_3d
        a, b, c, d = self.floor_plane
        
        # 床の法線ベクトルを使用して点から平面までの距離を計算
        # 法線ベクトルは (a, b, c) で、正規化されている
        # 点から平面への垂線の長さ = |ax + by + cz + d| / ||(a, b, c)||
        # 法線ベクトルが正規化されている場合、||(a, b, c)|| = 1
        if self.floor_normal is not None:
            # 法線ベクトルが利用可能な場合はそれを使用（より正確）
            normal = self.floor_normal
            # 点から床平面上の任意の点へのベクトル
            # 床平面上の点を取得（d = -dot(normal, point_on_plane)）
            # 簡単な方法: 原点から床平面への距離を計算
            point_on_plane = np.array([0, -d / b if abs(b) > 1e-6 else 0, 0]) if abs(b) > 1e-6 else np.array([0, 0, -d / c if abs(c) > 1e-6 else 0])
            point_vec = np.array([x, y, z]) - point_on_plane
            # 法線ベクトル方向への射影（高さ）
            distance = np.abs(np.dot(point_vec, normal))
        else:
            # フォールバック: 従来の方法
            distance = abs(a * x + b * y + c * z + d) / np.sqrt(a**2 + b**2 + c**2)
        
        return distance
    
    def is_point_on_floor(self, point_3d, threshold=0.03):
        """
        点が床面上にあるかどうかを判定
        
        Args:
            point_3d: (x, y, z) タプルまたはリスト
            threshold: 接触判定の閾値（メートル、デフォルト: 3cm）
        
        Returns:
            bool: 床面上にある場合はTrue
        """
        distance = self.distance_to_floor(point_3d)
        if distance is None:
            return False
        
        return distance <= threshold
    
    def get_floor_plane(self):
        """
        検出された床平面のパラメータを取得
        
        Returns:
            tuple: (plane, normal, height) または (None, None, None)
        """
        if self.floor_plane is None:
            return (None, None, None)
        
        return (self.floor_plane.copy(), self.floor_normal.copy(), self.floor_height)
    
    def _remove_outliers(self, points_3d, method='iqr', factor=1.5):
        """
        3D点群から外れ値を除去
        
        Args:
            points_3d: 3D点群の配列 (N, 3)
            method: 外れ値除去の方法 ('iqr' または 'zscore')
            factor: IQR法の場合の係数（デフォルト: 1.5）
        
        Returns:
            np.ndarray: 外れ値を除去した3D点群
        """
        if len(points_3d) < 10:
            return points_3d
        
        if method == 'iqr':
            # IQR（四分位範囲）法を使用
            # 各座標軸について外れ値を検出
            valid_indices = np.ones(len(points_3d), dtype=bool)
            
            for axis in range(3):
                values = points_3d[:, axis]
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                
                if iqr > 1e-6:  # IQRが0でない場合
                    lower_bound = q1 - factor * iqr
                    upper_bound = q3 + factor * iqr
                    valid_indices = valid_indices & (values >= lower_bound) & (values <= upper_bound)
            
            return points_3d[valid_indices]
        
        elif method == 'zscore':
            # Z-score法を使用
            from scipy import stats
            z_scores = np.abs(stats.zscore(points_3d, axis=0))
            valid_indices = np.all(z_scores < 3, axis=1)  # 3シグマ以内
            return points_3d[valid_indices]
        
        return points_3d
    
    def _filter_history_outliers(self, history, method='iqr', factor=1.5):
        """
        履歴から外れ値を除去
        
        Args:
            history: 履歴のリスト
            method: 外れ値除去の方法
            factor: IQR法の場合の係数
        
        Returns:
            list: 外れ値を除去した履歴
        """
        if len(history) < 5:
            return history
        
        # 品質スコアに基づいて外れ値を除去
        qualities = [h['quality'] for h in history]
        q1 = np.percentile(qualities, 25)
        q3 = np.percentile(qualities, 75)
        iqr = q3 - q1
        
        if iqr > 1e-6:
            lower_bound = q1 - factor * iqr
            upper_bound = q3 + factor * iqr
            filtered = [h for h in history if lower_bound <= h['quality'] <= upper_bound]
            
            # 法線ベクトルの類似度も考慮
            if len(filtered) > 0 and self.floor_normal is not None:
                # 法線ベクトルが大きく異なる履歴を除外
                filtered = [
                    h for h in filtered 
                    if np.dot(h['normal'], self.floor_normal) > 0.9  # コサイン類似度 > 0.9
                ]
            
            return filtered if len(filtered) > 0 else history
        
        return history

