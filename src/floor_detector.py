"""
Floor Detection

深度データから床平面を検出するクラス
RANSACアルゴリズムを使用した平面検出を提供
"""

import numpy as np


class FloorDetector:
    """床平面検出クラス"""

    def __init__(self, floor_threshold=0.03, min_inliers=500, max_iterations=500):
        """
        Args:
            floor_threshold: 床平面からの距離閾値（メートル、デフォルト: 3cm）
            min_inliers: 床平面と判定する最小インライア数（デフォルト: 500）
            max_iterations: RANSACの最大反復回数（デフォルト: 500）
        """
        self.floor_threshold = floor_threshold
        self.min_inliers = min_inliers
        self.max_iterations = max_iterations
        
        # 検出された床平面のパラメータ (ax + by + cz + d = 0)
        self.floor_plane = None  # (a, b, c, d) または None
        self.floor_normal = None  # (nx, ny, nz) 法線ベクトル
        self.floor_height = None  # 床の高さ（カメラ座標系のZ値）
        
        # 履歴（安定化のため）
        self.floor_history = []  # 過去の検出結果
        self.history_size = 10
        
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
            inliers = np.where(distances < self.floor_threshold)[0]
            
            if len(inliers) > max_inlier_count:
                max_inlier_count = len(inliers)
                best_plane = plane
                best_inliers = inliers
        
        # 十分なインライアが見つかった場合
        if max_inlier_count >= self.min_inliers and best_plane is not None:
            # インライアのみで平面を再計算（より正確に）
            inlier_points = points_3d[best_inliers]
            
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
            self.floor_plane = np.array([normal[0], normal[1], normal[2], d])
            self.floor_normal = normal
            
            # 床の高さ（カメラ座標系の原点からの距離）
            # 平面: ax + by + cz + d = 0 から原点までの距離
            # ただし、RealSense座標系ではYが下方向なので、注意が必要
            # 簡単のため、床面上の代表点のY座標を床の高さとする
            self.floor_height = mean_point[1]  # RealSense座標系ではYが下方向
            
            # 履歴に追加（安定化のため）
            self.floor_history.append({
                'plane': self.floor_plane.copy(),
                'normal': self.floor_normal.copy(),
                'height': self.floor_height
            })
            
            # 履歴サイズを制限
            if len(self.floor_history) > self.history_size:
                self.floor_history.pop(0)
            
            # 履歴の平均を使用（安定化）
            if len(self.floor_history) > 3:
                avg_plane = np.mean([h['plane'] for h in self.floor_history], axis=0)
                avg_normal = np.mean([h['normal'] for h in self.floor_history], axis=0)
                avg_height = np.mean([h['height'] for h in self.floor_history])
                
                # 正規化
                norm = np.linalg.norm(avg_normal)
                if norm > 1e-6:
                    avg_normal = avg_normal / norm
                    avg_plane[0:3] = avg_normal
                    avg_plane[3] = -np.dot(avg_normal, mean_point)
                
                self.floor_plane = avg_plane
                self.floor_normal = avg_normal
                self.floor_height = avg_height
            
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
        3D点から床平面までの距離を計算
        
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
        
        # 点から平面までの距離
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

