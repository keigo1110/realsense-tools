"""
RealSense Utilities

.bagファイルの読み込み、深度データ取得、カメラパラメータ取得、2D→3D変換機能を提供
CUDA高速化対応
"""

import numpy as np
import pyrealsense2 as rs
import cv2

# CuPyの利用可否をチェック（CUDA高速化用）
# フォールバック警告フラグ（一度だけ警告を表示）
_cuda_fallback_warned = False

try:
    import cupy as cp
    # CuPyがインストールされていても、実際にCUDAが動作するか確認
    try:
        # CUDAデバイスにアクセスして動作確認
        test_array = cp.array([1.0])
        del test_array
        cp.get_default_memory_pool().free_all_blocks()  # メモリをクリア
        CUPY_AVAILABLE = True
    except Exception as e:
        # CuPyはインストールされているが、CUDAが動作しない場合
        CUPY_AVAILABLE = False
        import sys
        print(f"Warning: CuPy installed but CUDA not available: {e}", file=sys.stderr)
        print("Falling back to NumPy (CPU mode)", file=sys.stderr)
except ImportError:
    CUPY_AVAILABLE = False
    # cpはNoneのまま使用する（条件チェックで使用）


class BagFileReader:
    """RealSense .bagファイルのリーダー"""

    def __init__(self, bag_file_path, start_time=None, end_time=None):
        """
        Args:
            bag_file_path: .bagファイルのパス
            start_time: 開始時間（秒、Noneの場合は最初から）
            end_time: 終了時間（秒、Noneの場合は最後まで）
        """
        self.bag_file_path = bag_file_path
        self.start_time = start_time  # 開始時間（秒）
        self.end_time = end_time  # 終了時間（秒）
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
        self.playback_start_time = None  # 再生開始時のタイムスタンプ
        self._manual_seek = False  # 手動シークフラグ（seek()が使えない場合）
        self._skipped_frames_for_seek = 0  # シークのためにスキップしたフレーム数

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
                        file_duration_sec = self.file_duration.total_seconds()
                        print(f"Bag file loaded: {self.bag_file_path}")
                        print(f"Bag file duration: {file_duration_sec:.2f} seconds")

                        # 開始時間が指定されている場合はシーク（0はNoneとして扱う）
                        if self.start_time is not None and self.start_time > 0:
                            try:
                                # RealSense SDKのseek()はdatetime.timedeltaを受け取る
                                from datetime import timedelta
                                time_delta = timedelta(seconds=self.start_time)
                                self.playback.seek(time_delta)
                                print(f"Seeking to start time: {self.start_time:.2f} seconds")
                            except Exception as e:
                                print(f"Warning: Failed to seek to start time: {e}")
                                print(f"  Will skip frames manually until {self.start_time:.2f} seconds")
                                # シークに失敗した場合は、手動でフレームをスキップする
                                self._manual_seek = True
                                # 初期化をリセット（最初のフレームからタイムスタンプを記録）
                                self.start_timestamp = None
                                self._skipped_frames_for_seek = 0

                        # 終了時間の検証（0はNoneとして扱う）
                        if self.end_time is not None and self.end_time > 0:
                            if self.end_time <= (self.start_time or 0):
                                raise ValueError(f"End time ({self.end_time:.2f}s) must be greater than start time ({self.start_time or 0:.2f}s)")
                            if self.end_time > file_duration_sec:
                                print(f"Warning: End time ({self.end_time:.2f}s) exceeds file duration ({file_duration_sec:.2f}s). Using file duration.")
                                self.end_time = None

                        # 時間範囲の表示
                        if (self.start_time is not None and self.start_time > 0) or (self.end_time is not None and self.end_time > 0):
                            start_str = f"{self.start_time:.2f}" if (self.start_time is not None and self.start_time > 0) else "0.00"
                            end_str = f"{self.end_time:.2f}" if (self.end_time is not None and self.end_time > 0) else f"{file_duration_sec:.2f}"
                            start_val = self.start_time if (self.start_time is not None and self.start_time > 0) else 0
                            end_val = self.end_time if (self.end_time is not None and self.end_time > 0) else file_duration_sec
                            duration_str = f"{end_val - start_val:.2f}"
                            print(f"Playback range: {start_str}s - {end_str}s (duration: {duration_str}s)")

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

            # 最初のフレームで開始タイムスタンプを記録（手動シーク前）
            is_first_frame = (self.start_timestamp is None)
            if is_first_frame:
                self.start_timestamp = color_frame.get_timestamp()
                # 手動シークの場合は、この時点ではplayback_start_timeを設定しない
                if not self._manual_seek:
                    self.playback_start_time = self.start_timestamp

            # 手動シーク処理（seek()が使えない場合、start_time > 0の場合のみ）
            if self._manual_seek and self.start_time is not None and self.start_time > 0:
                current_timestamp = color_frame.get_timestamp()

                # タイムスタンプがミリ秒単位か秒単位かを判定
                timestamp_diff = current_timestamp - self.start_timestamp
                if self.start_timestamp > 1000000000:  # ミリ秒単位と判定
                    elapsed_seconds = timestamp_diff / 1000.0
                else:
                    elapsed_seconds = timestamp_diff

                # デバッグ: 最初の数フレームのみログ出力
                if self._skipped_frames_for_seek < 5 or (self._skipped_frames_for_seek % 30 == 0):
                    print(f"  Frame {self._skipped_frames_for_seek}: elapsed={elapsed_seconds:.2f}s, target={self.start_time:.2f}s")

                # 開始時間に達するまでフレームをスキップ
                if elapsed_seconds < self.start_time:
                    self._skipped_frames_for_seek += 1
                    # 次のフレームを取得（再帰的に呼び出し）
                    return self.get_frames()
                else:
                    # 開始時間に達したので、手動シークを終了
                    self._manual_seek = False
                    self.playback_start_time = current_timestamp  # 再生開始時刻を更新
                    print(f"  Manually seeked to {elapsed_seconds:.2f} seconds (skipped {self._skipped_frames_for_seek} frames)")

            current_timestamp = color_frame.get_timestamp()

            # 終了時間のチェック（開始時間からの経過時間で計算、end_time > 0の場合のみ）
            if self.end_time is not None and self.end_time > 0 and self.playback_start_time is not None:
                # タイムスタンプがミリ秒単位か秒単位かを判定
                timestamp_diff = current_timestamp - self.playback_start_time
                if self.playback_start_time > 1000000000:  # ミリ秒単位と判定
                    elapsed_seconds = timestamp_diff / 1000.0
                else:
                    elapsed_seconds = timestamp_diff

                # 終了時間を超過した場合は終了
                start_val = self.start_time if (self.start_time is not None and self.start_time > 0) else 0
                playback_duration = self.end_time - start_val
                if elapsed_seconds >= playback_duration - 0.05:  # 50msのマージン
                    return None, None

            # プレイバック位置をチェックして終端検出
            if not is_first_frame and self.playback and self.file_duration:
                try:
                    position = self.playback.get_position()
                    duration_sec = self.file_duration.total_seconds()
                    position_sec = position.total_seconds()

                    # 終了時間が指定されている場合は、それを使用（0はNoneとして扱う）
                    max_position = self.end_time if (self.end_time is not None and self.end_time > 0) else duration_sec

                    # 終端に達した場合は即座に終了
                    if position_sec >= max_position - 0.05:  # 50msのマージン
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
            # タイムアウトエラーは正常終了（ファイル終端）
            if "timeout" in str(e).lower() or "frame didn't arrive" in str(e).lower():
                return None, None
            raise

    def poll_for_frames(self):
        """
        非ブロッキングでフレームを取得（デュアルカメラ同期用）
        
        Returns:
            frameset: フレームセット（取得できた場合）、None（取得できない場合）
        """
        try:
            if self.pipeline is None:
                return None
            frames = self.pipeline.poll_for_frames()
            if frames:
                return frames
            return None
        except Exception:
            return None

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

        # 深度データの内部パラメータを使用
        # 深度画像のintrinsicsが利用可能な場合はそれを使用（より正確）
        # そうでない場合はカラー画像のintrinsicsを使用（アライメント後）
        if self.depth_intrinsics is not None:
            intrinsics = self.depth_intrinsics
        else:
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

    def get_depth_at_points_batch(self, depth_frame, points, use_interpolation=True, kernel_size=3, confidences=None):
        """
        複数のピクセル位置の深度値を一括取得（バッチ処理、高速化）
        CUDA使用可能な場合はGPUで処理

        研究用途向けの高精度処理:
        - 空間補間: 周囲ピクセルの中央値を使用（ノイズ低減）
        - 外れ値除去: 統計的手法による異常値検出
        - 信頼度ベース適応的補間: キーポイント信頼度に応じて補間範囲を調整

        Args:
            depth_frame: RealSense深度フレーム
            points: ピクセル座標のリスト [(x, y), ...]
            use_interpolation: Trueの場合、周囲ピクセルからの補間を適用（デフォルト: True）
            kernel_size: 補間カーネルサイズ（奇数推奨、デフォルト: 3）
            confidences: キーポイントの信頼度リスト（オプション、信頼度に基づく適応的処理用）

        Returns:
            list: 深度値のリスト（メートル）、無効な場合はNoneを含む
        """
        if depth_frame is None or not points:
            return [None] * len(points)

        # 深度フレームから深度値を取得
        depth_image = np.asanyarray(depth_frame.get_data())

        # バッチ処理による高速化（複数点を一度に処理）
        # 補間処理はCPUで実行（IQR法による外れ値除去が複雑なため）
        # ただし、NumPyのベクトル化を活用して高速化

        if len(points) > 1:
            # バッチ処理: 複数点を一度に処理（NumPyのベクトル化を活用）
            depths = []
            for i, (x, y) in enumerate(points):
                confidence = confidences[i] if confidences is not None and i < len(confidences) else None
                depth_m = self._get_depth_with_interpolation(
                    depth_image, x, y, use_interpolation, kernel_size, confidence
                )
                depths.append(depth_m)
            return depths
        else:
            # 単一点の場合は直接処理
            if points:
                confidence = confidences[0] if confidences is not None and len(confidences) > 0 else None
                depth_m = self._get_depth_with_interpolation(
                    depth_image, points[0][0], points[0][1], use_interpolation, kernel_size, confidence
                )
                return [depth_m]
            return [None]

    def _get_depth_with_interpolation(self, depth_image, x, y, use_interpolation=True, kernel_size=3, confidence=None):
        """
        深度値を取得（補間オプション付き、研究用途向け高精度処理）

        Args:
            depth_image: 深度画像（NumPy配列）
            x: ピクセルX座標（float）
            y: ピクセルY座標（float）
            use_interpolation: Trueの場合、周囲ピクセルからの補間を適用
            kernel_size: 補間カーネルサイズ（奇数推奨）
            confidence: キーポイントの信頼度（0-1、信頼度が低い場合は補間範囲を拡大）

        Returns:
            float: 深度値（メートル）、無効な場合はNone
        """
        x_int = int(round(x))
        y_int = int(round(y))

        height, width = depth_image.shape

        if x_int < 0 or x_int >= width or y_int < 0 or y_int >= height:
            return None

        if use_interpolation and kernel_size >= 3:
            # 信頼度が低い場合は補間範囲を拡大（研究手法: 信頼度に基づく適応的補間）
            if confidence is not None and confidence < 0.5:
                kernel_size = min(kernel_size + 2, 7)  # 最大7x7

            # 周囲ピクセルから深度値を補間（ノイズ低減のため中央値を使用）
            half_kernel = kernel_size // 2
            x_min = max(0, x_int - half_kernel)
            x_max = min(width, x_int + half_kernel + 1)
            y_min = max(0, y_int - half_kernel)
            y_max = min(height, y_int + half_kernel + 1)

            # 周囲の深度値を取得（NumPyのスライシングを活用）
            neighborhood = depth_image[y_min:y_max, x_min:x_max]
            valid_depths = neighborhood[neighborhood > 0]  # 無効な深度値（0）を除外

            if len(valid_depths) == 0:
                return None

            # 外れ値除去: IQR（四分位範囲）法を使用（NumPyのベクトル演算を活用）
            if len(valid_depths) >= 5:
                # NumPyのpercentile関数を使用（高速化）
                q1 = np.percentile(valid_depths, 25, method='linear')  # 線形補間で高速化
                q3 = np.percentile(valid_depths, 75, method='linear')
                iqr = q3 - q1

                if iqr > 1e-6:  # ゼロ除算を防ぐ
                    # IQR法による外れ値の閾値
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr

                    # 外れ値を除去（NumPyのブールインデックスを使用）
                    mask = (valid_depths >= lower_bound) & (valid_depths <= upper_bound)
                    filtered_depths = valid_depths[mask]

                    if len(filtered_depths) > 0:
                        # 中央値を使用（ノイズに対してロバスト）
                        depth_value = np.median(filtered_depths)
                    else:
                        # 外れ値除去後に有効値がなければ中央値を使用
                        depth_value = np.median(valid_depths)
                else:
                    # IQRが0の場合（すべて同じ値）、中央値を使用
                    depth_value = np.median(valid_depths)
            else:
                # サンプル数が少ない場合は中央値を使用
                depth_value = np.median(valid_depths)
        else:
            # 補間なし: 単一ピクセルの深度値をそのまま使用
            depth_value = depth_image[y_int, x_int]

            if depth_value == 0:
                return None

        # スケールを適用してメートル単位に変換
        depth_m = float(depth_value) * self.depth_scale

        # 現実的な深度範囲をチェック（論文での妥当性確保）
        # RealSense D455の有効範囲: 0.3m～5.0m（仕様書より）
        if depth_m < 0.3 or depth_m > 5.0:
            return None

        return depth_m

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
            except Exception as e:
                # CUDA処理に失敗した場合はCPUで処理（フォールバック）
                # エラーメッセージは最初の一回のみ表示（ログが多くなりすぎるのを防ぐ）
                global _cuda_fallback_warned
                if not _cuda_fallback_warned:
                    print(f"Warning: CuPy CUDA processing failed: {e}")
                    print(f"Falling back to CPU mode for depth calculations")
                    _cuda_fallback_warned = True
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

