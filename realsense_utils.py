"""
RealSense Utilities

.bagファイルの読み込み、深度データ取得、カメラパラメータ取得、2D→3D変換機能を提供
"""

import numpy as np
import pyrealsense2 as rs
import cv2


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

    def initialize(self):
        """パイプラインを初期化し、ストリームを有効化"""
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # .bagファイルから再生する設定
        rs.config.enable_device_from_file(self.config, self.bag_file_path)

        try:
            # パイプライン開始
            profile = self.pipeline.start(self.config)

            # 深度スケールを取得
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()

            # カメラ内部パラメータを取得
            color_stream = profile.get_stream(rs.stream.color)
            depth_stream = profile.get_stream(rs.stream.depth)

            if color_stream:
                self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

            if depth_stream:
                self.depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

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
            frames = self.pipeline.wait_for_frames()

            # フレームの同期
            aligned_frames = rs.align(rs.stream.color).process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                return None, None

            return color_frame, depth_frame

        except RuntimeError:
            # ファイルの終端に到達
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
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image = depth_image.astype(np.float32) * self.depth_scale

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

