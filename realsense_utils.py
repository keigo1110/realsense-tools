"""
RealSense Utilities

.bagファイルの読み込み、深度データ取得、カメラパラメータ取得、2D→3D変換機能を提供
CUDA高速化対応
"""

import numpy as np
import pyrealsense2 as rs
import cv2

# CuPyの利用可否をチェック（CUDA高速化用）
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    # cpはNoneのまま使用する（条件チェックで使用）


class BagFileReader:
    """RealSense .bagファイルのリーダー"""

    def __init__(self, bag_file_path):
        """
        Args:
            bag_file_path: .bagファイルのパス
        """
        self.bag_file_path = bag_file_path
        self.pipeline = None
        self.config = None
        self.depth_scale = None
        self.intrinsics = None
        self.depth_intrinsics = None
        self.device = None
        self.playback = None
        self.last_timestamp = None
        self.same_timestamp_count = 0
        self.last_frame_number = None
        self.repeated_frames = 0
        self.file_duration = None
        self.start_timestamp = None

    def initialize(self):
        """パイプラインを初期化し、ストリームを有効化"""
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # .bagファイルから再生する設定
        rs.config.enable_device_from_file(self.config, self.bag_file_path)

        try:
            # パイプライン開始
            profile = self.pipeline.start(self.config)

            # デバイスとプレイバックを取得（bagファイル終端検出用）
            self.device = profile.get_device()
            if self.device and self.device.supports(rs.camera_info.name):
                # プレイバックデバイスとして扱う
                try:
                    self.playback = self.device.as_playback()
                    if self.playback:
                        self.file_duration = self.playback.get_duration()
                        print(f"Bag file loaded: {self.bag_file_path}")
                        print(f"Bag file duration: {self.file_duration.total_seconds():.2f} seconds")
                        # 自動リピートを無効化（重要：これがないとループ再生される）
                        self.playback.set_real_time(False)
                except Exception as e:
                    # プレイバック機能が利用できない場合（通常のカメラなど）
                    pass

            # 深度スケールを取得
            depth_sensor = self.device.first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()

            # カメラ内部パラメータを取得
            color_stream = profile.get_stream(rs.stream.color)
            depth_stream = profile.get_stream(rs.stream.depth)

            if color_stream:
                self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

            if depth_stream:
                self.depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

            if not self.playback:
                print(f"Bag file loaded: {self.bag_file_path}")
            print(f"Depth scale: {self.depth_scale}")

            return True

        except RuntimeError as e:
            print(f"Error loading bag file: {e}")
            return False

    def get_frames(self):
        """
        同期されたフレームを取得

        Returns:
            tuple: (color_frame, depth_frame) または (None, None)
        """
        try:
            # タイムアウトを設定してフレームを取得（5秒）
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)

            # フレームの同期
            aligned_frames = rs.align(rs.stream.color).process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                return None, None

            # 最初のフレームで開始タイムスタンプを記録
            is_first_frame = (self.start_timestamp is None)
            if is_first_frame:
                self.start_timestamp = color_frame.get_timestamp()

            current_timestamp = color_frame.get_timestamp()
            
            # プレイバック位置をチェックして終端検出
            if not is_first_frame and self.playback and self.file_duration:
                try:
                    position = self.playback.get_position()
                    duration_sec = self.file_duration.total_seconds()
                    position_sec = position.total_seconds()
                    
                    # 終端に達した場合は即座に終了
                    if position_sec >= duration_sec - 0.05:  # 50msのマージン
                        return None, None
                except Exception as e:
                    # 位置取得に失敗した場合も続行（エラーを無視）
                    pass

            # タイムスタンプによる終端検出
            if not is_first_frame and self.last_timestamp is not None:
                if current_timestamp < self.last_timestamp:
                    # タイムスタンプが後ろに戻った場合はループ再生とみなして終了
                    return None, None
                elif current_timestamp == self.last_timestamp:
                    # 同じタイムスタンプが繰り返されている場合
                    self.same_timestamp_count += 1
                    if self.same_timestamp_count >= 3:
                        return None, None
                else:
                    self.same_timestamp_count = 0
                    
                    # ファイルの長さを超えていないかチェック（相対時間を使用）
                    if self.start_timestamp is not None and self.file_duration:
                        timestamp_diff = abs(current_timestamp - self.start_timestamp)
                        # タイムスタンプが非常に大きい値（10億以上）ならミリ秒単位と判定
                        elapsed_time = timestamp_diff / 1000.0 if self.start_timestamp > 1000000000 else timestamp_diff
                        # ファイルの長さを大きく超えている場合は終了
                        if elapsed_time > self.file_duration.total_seconds() + 1.0:
                            return None, None
            self.last_timestamp = current_timestamp

            # フレーム番号による重複検出（補助的）
            try:
                current_frame_number = color_frame.get_frame_number()
                if self.last_frame_number is not None:
                    if current_frame_number <= self.last_frame_number:
                        self.repeated_frames += 1
                        if self.repeated_frames >= 3:
                            return None, None
                    else:
                        self.repeated_frames = 0
                self.last_frame_number = current_frame_number
            except:
                pass

            return color_frame, depth_frame

        except RuntimeError as e:
            # ファイルの終端に到達またはタイムアウト
            error_msg = str(e).lower()
            if "timeout" in error_msg or "didn't arrive" in error_msg:
                return None, None
            # その他のエラーも終端とみなす
            return None, None

    def frame_to_numpy(self, color_frame, depth_frame):
        """
        フレームをNumPy配列に変換

        Args:
            color_frame: RealSenseカラーフレーム
            depth_frame: RealSense深度フレーム

        Returns:
            tuple: (color_image, depth_image)
        """
        if color_frame is None or depth_frame is None:
            return None, None

        # カラー画像をNumPy配列に変換
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)  # RGB→BGR

        # 深度画像をNumPy配列に変換（メートル単位）
        # astypeと乗算を一度に実行（メモリアクセス削減）
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image = (depth_image.astype(np.float32, copy=False) * self.depth_scale)

        return color_image, depth_image

    def get_depth_at_point(self, depth_frame, x, y):
        """
        指定したピクセル位置の深度値を取得

        Args:
            depth_frame: RealSense深度フレーム
            x: ピクセルのX座標
            y: ピクセルのY座標

        Returns:
            float: 深度値（メートル）、無効な場合はNone
        """
        if depth_frame is None:
            return None

        x = int(round(x))
        y = int(round(y))

        # 深度フレームから深度値を取得
        depth_image = np.asanyarray(depth_frame.get_data())

        if x < 0 or x >= depth_image.shape[1] or y < 0 or y >= depth_image.shape[0]:
            return None

        depth_value = depth_image[y, x]

        if depth_value == 0:
            return None

        # スケールを適用してメートル単位に変換
        depth_m = depth_value * self.depth_scale

        return depth_m

    def pixel_to_3d(self, pixel_x, pixel_y, depth):
        """
        2D画像座標と深度値から3Dカメラ座標に変換

        Args:
            pixel_x: ピクセルのX座標
            pixel_y: ピクセルのY座標
            depth: 深度値（メートル）

        Returns:
            tuple: (X, Y, Z) カメラ座標（メートル）、無効な場合はNone
        """
        if depth is None or depth <= 0:
            return None

        if self.intrinsics is None:
            return None

        # 深度データの内部パラメータを使用（カラーと深度のアライメントを考慮）
        intrinsics = self.intrinsics

        # RealSenseの座標系:
        # X: 右方向
        # Y: 下方向
        # Z: 前方（カメラから見て）

        # 内部パラメータを使用して3D座標を計算
        x = (pixel_x - intrinsics.ppx) / intrinsics.fx
        y = (pixel_y - intrinsics.ppy) / intrinsics.fy

        # 3D座標を計算
        X = x * depth
        Y = y * depth
        Z = depth

        return (X, Y, Z)

    def get_depth_at_points_batch(self, depth_frame, points):
        """
        複数のピクセル位置の深度値を一括取得（バッチ処理、高速化）
        CUDA使用可能な場合はGPUで処理

        Args:
            depth_frame: RealSense深度フレーム
            points: ピクセル座標のリスト [(x, y), ...]

        Returns:
            list: 深度値のリスト（メートル）、無効な場合はNoneを含む
        """
        if depth_frame is None or not points:
            return [None] * len(points)

        # 深度フレームから深度値を取得
        depth_image = np.asanyarray(depth_frame.get_data())

        # CUDA使用可能な場合は、バッチ処理をベクトル化
        if CUPY_AVAILABLE and len(points) > 5:
            try:
                # CuPyで処理（大量の点がある場合のみ）
                # NumPy配列を先に作成してからCuPy配列に変換（高速化）
                points_array = np.array(points, dtype=np.float32)
                x_coords = cp.asarray(np.round(points_array[:, 0]).astype(np.int32))
                y_coords = cp.asarray(np.round(points_array[:, 1]).astype(np.int32))
                
                # 境界チェック
                height, width = depth_image.shape
                valid_mask = (x_coords >= 0) & (x_coords < width) & (y_coords >= 0) & (y_coords < height)
                
                # 深度画像をGPUに転送
                depth_gpu = cp.asarray(depth_image)
                
                # 有効な座標のみ処理
                valid_x = x_coords[valid_mask]
                valid_y = y_coords[valid_mask]
                
                if len(valid_x) > 0:
                    # GPUで深度値を取得（インデックスアクセス）
                    depth_values = depth_gpu[valid_y, valid_x]
                    depth_values = depth_values * self.depth_scale
                    
                    # 無効な深度値（0）をマスク
                    valid_depth_mask = depth_values > 0
                    
                    # 結果をCPUに転送
                    depth_values_cpu = cp.asnumpy(depth_values)
                    valid_mask_cpu = cp.asnumpy(valid_mask)
                    valid_depth_mask_cpu = cp.asnumpy(valid_depth_mask)
                    
                    # 結果をリストに変換
                    depths = []
                    valid_idx = 0
                    for i, is_valid in enumerate(valid_mask_cpu):
                        if is_valid:
                            if valid_depth_mask_cpu[valid_idx]:
                                depths.append(float(depth_values_cpu[valid_idx]))
                            else:
                                depths.append(None)
                            valid_idx += 1
                        else:
                            depths.append(None)
                    
                    return depths
            except Exception:
                # CUDA処理に失敗した場合はCPUで処理（フォールバック）
                pass

        # CPU処理（フォールバックまたは小規模データ）
        depths = []
        for x, y in points:
            x = int(round(x))
            y = int(round(y))

            if x < 0 or x >= depth_image.shape[1] or y < 0 or y >= depth_image.shape[0]:
                depths.append(None)
                continue

            depth_value = depth_image[y, x]

            if depth_value == 0:
                depths.append(None)
                continue

            # スケールを適用してメートル単位に変換
            depth_m = depth_value * self.depth_scale
            depths.append(depth_m)

        return depths

    def pixels_to_3d_batch(self, points, depths):
        """
        複数の2D画像座標と深度値から3Dカメラ座標に一括変換（バッチ処理、高速化）
        CUDA使用可能な場合はGPUで処理

        Args:
            points: ピクセル座標のリスト [(x, y), ...]
            depths: 深度値のリスト（メートル）

        Returns:
            list: 3D座標のリスト [(X, Y, Z), ...]、無効な場合はNoneを含む
        """
        if self.intrinsics is None:
            return [None] * len(points)

        intrinsics = self.intrinsics
        
        # CUDA使用可能な場合はベクトル化処理
        if CUPY_AVAILABLE and len(points) > 5:
            try:
                # 有効なデータのみ抽出
                valid_points = []
                valid_depths = []
                valid_indices = []
                
                for i, ((px, py), d) in enumerate(zip(points, depths)):
                    if d is not None and d > 0:
                        valid_points.append((px, py))
                        valid_depths.append(d)
                        valid_indices.append(i)
                
                if valid_points:
                    # CuPy配列に変換
                    pixel_x = cp.array([p[0] for p in valid_points], dtype=cp.float32)
                    pixel_y = cp.array([p[1] for p in valid_points], dtype=cp.float32)
                    depth = cp.array(valid_depths, dtype=cp.float32)
                    
                    # ベクトル演算で一括計算
                    x = (pixel_x - intrinsics.ppx) / intrinsics.fx
                    y = (pixel_y - intrinsics.ppy) / intrinsics.fy
                    
                    X = x * depth
                    Y = y * depth
                    Z = depth
                    
                    # CPUに転送
                    X_cpu = cp.asnumpy(X)
                    Y_cpu = cp.asnumpy(Y)
                    Z_cpu = cp.asnumpy(Z)
                    
                    # 結果をリストに格納
                    results = [None] * len(points)
                    for idx, (x_val, y_val, z_val) in enumerate(zip(X_cpu, Y_cpu, Z_cpu)):
                        results[valid_indices[idx]] = (float(x_val), float(y_val), float(z_val))
                    
                    return results
            except Exception:
                # CUDA処理に失敗した場合はCPUで処理（フォールバック）
                pass
        
        # CPU処理（フォールバックまたは小規模データ）
        results = []
        for (pixel_x, pixel_y), depth in zip(points, depths):
            if depth is None or depth <= 0:
                results.append(None)
                continue

            # 内部パラメータを使用して3D座標を計算
            x = (pixel_x - intrinsics.ppx) / intrinsics.fx
            y = (pixel_y - intrinsics.ppy) / intrinsics.fy

            # 3D座標を計算
            X = x * depth
            Y = y * depth
            Z = depth

            results.append((X, Y, Z))

        return results

    def stop(self):
        """パイプラインを停止"""
        if self.pipeline:
            self.pipeline.stop()

    def __enter__(self):
        """コンテキストマネージャーとして使用"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャーとして使用"""
        self.stop()

