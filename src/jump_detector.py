"""
Jump Detector and Measurement

ジャンプ検出アルゴリズム、高さ・距離計算、軌跡計算機能を提供
床検出ベースの高精度ジャンプ検出に対応
"""

import numpy as np
from collections import deque


class JumpDetector:
    """ジャンプ検出・測定クラス"""

    def __init__(
        self,
        threshold_vertical=0.05,
        threshold_horizontal=0.1,
        min_frames=10,
        floor_detector=None,
        use_floor_detection=True,
        min_jump_height=0.20,
        min_air_time=0.20,
        waist_baseline_height=None,
        waist_zero_epsilon=0.01,
    ):
        """
        Args:
            threshold_vertical: 垂直ジャンプ検出の閾値（メートル、床検出不使用時）
            threshold_horizontal: 水平ジャンプ検出の閾値（メートル）
            min_frames: ジャンプと認識する最小フレーム数
            floor_detector: FloorDetectorインスタンス（床検出使用時）
            use_floor_detection: 床検出を使用するかどうか（デフォルト: True）
            min_jump_height: ジャンプとして認識する最小高さ（メートル、デフォルト: 20cm）
            min_air_time: ジャンプとして認識する最小滞空時間（秒、デフォルト: 0.2秒）
        """
        self.threshold_vertical = threshold_vertical
        self.threshold_horizontal = threshold_horizontal
        self.min_frames = min_frames
        self.floor_detector = floor_detector
        self.use_floor_detection = use_floor_detection and (floor_detector is not None)
        self.min_jump_height = min_jump_height  # 最小ジャンプ高さ（20cm、20cm以下は除外）
        self.min_air_time = min_air_time  # 最小滞空時間（0.2秒 = 200ms）
        self.waist_baseline_height = waist_baseline_height
        self.waist_zero_epsilon = waist_zero_epsilon if waist_zero_epsilon is not None else 0.01

        # 履歴データ
        self.height_history = deque(maxlen=30)  # 過去30フレームの高さ
        self.position_history = deque(maxlen=30)  # 過去30フレームの位置
        
        # 全キーポイントの床からの距離履歴（変動性分析用）
        self.keypoint_distance_history = {}  # {keypoint_name: deque(maxlen=1000), ...}
        
        # ジャンプ遷移時（離陸・着地）の変化を記録
        self.takeoff_transitions = []  # [(frame_num, keypoint_distances), ...]
        self.landing_transitions = []  # [(frame_num, keypoint_distances), ...]

        # ジャンプ検出状態
        self.jump_state = "ground"  # "ground", "takeoff", "airborne", "landing"
        self._initial_state = True  # 初期状態フラグ
        self.jump_start_frame = None
        self.jump_start_timestamp = None
        self.jump_start_position = None
        self.jump_takeoff_frame = None
        self.jump_takeoff_timestamp = None
        self.jump_takeoff_position = None
        self.jump_takeoff_height = None  # 離陸時の床からの高さ
        self.jump_end_frame = None
        self.jump_end_timestamp = None
        self.jump_end_position = None
        self.jump_end_height = None  # 着地時の床からの高さ（ゼロクロス点）
        self.jump_max_height = 0.0
        self.jump_max_height_frame = None  # 最大高さが記録されたフレーム
        self.jump_max_height_timestamp = None  # 最大高さが記録されたタイムスタンプ
        self.jump_max_distance = 0.0

        # 検出されたジャンプのリスト
        self.detected_jumps = []

        # 軌跡データ
        self.trajectory = []

        # 腰基準によるゼロクロス検出用
        self._prev_waist_delta = None
        self._prev_height_above_floor = None  # 前フレームの高さ（ゼロクロス補間用）

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

        # 基準点として腰（MidHip）を使用
        # 左右の股関節の中点を計算
        mid_hip = None
        left_hip = keypoints_3d.get("LHip")
        right_hip = keypoints_3d.get("RHip")

        if left_hip and right_hip and left_hip[2] is not None and right_hip[2] is not None:
            mid_hip = (
                (left_hip[0] + right_hip[0]) / 2 if left_hip[0] is not None and right_hip[0] is not None else None,
                (left_hip[1] + right_hip[1]) / 2 if left_hip[1] is not None and right_hip[1] is not None else None,
                (left_hip[2] + right_hip[2]) / 2 if left_hip[2] is not None and right_hip[2] is not None else None
            )

        # 基準点は腰のみを使用
        if mid_hip is None or mid_hip[2] is None:
            return None

        x, y, z = mid_hip

        # 履歴を更新
        self.height_history.append(z)
        self.position_history.append((x, y, z))

        # 床からの距離をすべてのキーポイントについて計算
        keypoint_distances_to_floor = {}
        if self.use_floor_detection and self.floor_detector and self.floor_detector.floor_plane is not None:
            for kp_name, kp_coords in keypoints_3d.items():
                if kp_coords and kp_coords[0] is not None and kp_coords[1] is not None and kp_coords[2] is not None:
                    distance = self.floor_detector.distance_to_floor(kp_coords)
                    keypoint_distances_to_floor[kp_name] = distance
                    
                    # 履歴に追加（変動性分析用）
                    if kp_name not in self.keypoint_distance_history:
                        self.keypoint_distance_history[kp_name] = deque(maxlen=1000)
                    # 距離と現在の状態（ground/jumping）を一緒に保存
                    self.keypoint_distance_history[kp_name].append((distance, self.jump_state))
                else:
                    keypoint_distances_to_floor[kp_name] = None
        
        # 軌跡を記録
        self.trajectory.append({
            "frame": frame_num,
            "timestamp": timestamp,
            "position": (x, y, z),
            "keypoints": keypoints_3d,
            "keypoint_distances_to_floor": keypoint_distances_to_floor if self.use_floor_detection else None
        })

        # ジャンプ検出（腰の高さと基準高さのゼロクロス検出を使用）
        if self.use_floor_detection:
            result = self._detect_jump_with_floor(
                frame_num, x, y, z, timestamp, keypoints_3d
            )
        else:
            result = self._detect_jump(frame_num, x, y, z, timestamp)

        return result

    def _delta_state(self, delta):
        """基準点との差分が正か負か（デッドバンド考慮）を判定"""
        if delta is None:
            return "unknown"
        if delta > self.waist_zero_epsilon:
            return "positive"
        if delta < -self.waist_zero_epsilon:
            return "negative"
        return "zero"

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

            # 水平距離を更新（XZ平面での距離）
            # RealSense座標系: X=左右, Y=上下（垂直）, Z=前後（深度）
            if self.jump_start_position:
                jump_distance = np.sqrt(
                    (x - self.jump_start_position[0])**2 +
                    (z - self.jump_start_position[2])**2
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
    
    def _detect_jump_with_floor(self, frame_num, x, y, z, timestamp, keypoints_3d=None):
        """
        床検出ベースのジャンプ検出ロジック
        腰の高さと基準高さのゼロクロス検出を使用（負から正を離陸、正から負を着陸）
        
        Args:
            frame_num: フレーム番号
            x, y, z: 参照点（腰）の3D座標
            timestamp: タイムスタンプ
            keypoints_3d: キーポイント辞書（離陸・着地時の記録用）
        """
        if len(self.height_history) < self.min_frames:
            return {
                "state": "initializing",
                "frame": frame_num,
                "timestamp": timestamp,
                "height": z,
                "position": (x, y, z)
            }
        
        # 床からの高さを計算（腰の位置から床までの距離）
        height_above_floor = None
        if self.floor_detector and self.floor_detector.floor_plane is not None:
            mid_hip = (x, y, z)
            if mid_hip[0] is not None and mid_hip[1] is not None and mid_hip[2] is not None:
                height_above_floor = self.floor_detector.distance_to_floor(mid_hip)

        if height_above_floor is None:
            return {
                "state": "ground",
                "frame": frame_num,
                "timestamp": timestamp,
                "height": z,
                "position": (x, y, z),
                "height_above_floor": None,
                "jump_type": None,
                "jump_height": 0.0,
                "jump_distance": 0.0,
                "air_time": 0.0
            }
        
        # 基準高さの設定（設定されていない場合は最初の数フレームの平均高さを使用）
        if self.waist_baseline_height is None:
            # 腰の高さ履歴を保持
            if not hasattr(self, 'waist_height_history'):
                self.waist_height_history = deque(maxlen=30)
            self.waist_height_history.append(height_above_floor)
            
            # 最初の10フレームの平均高さを基準とする
            if len(self.waist_height_history) >= 10:
                self.waist_baseline_height = np.mean(list(self.waist_height_history)[:10])
        
        # ゼロクロス検出: delta = 現在の高さ - 基準高さ
        delta = height_above_floor - self.waist_baseline_height
        prev_state = self._delta_state(self._prev_waist_delta)
        current_state = self._delta_state(delta)
        
        # ゼロクロス点の補間（より正確な離着陸タイミングを検出）
        zero_cross_height = None
        if self._prev_waist_delta is not None and self._prev_height_above_floor is not None:
            # 前フレームと現在のフレームで符号が異なる場合、ゼロクロス点を補間
            if (self._prev_waist_delta < 0 and delta > 0) or (self._prev_waist_delta > 0 and delta < 0):
                # 線形補間: ゼロクロス点の高さを計算
                # y = y1 + (y2 - y1) * (0 - x1) / (x2 - x1)
                # ここで、x1 = prev_delta, x2 = current_delta, y1 = prev_height, y2 = current_height
                if abs(delta - self._prev_waist_delta) > 1e-6:  # ゼロ除算を防ぐ
                    t = -self._prev_waist_delta / (delta - self._prev_waist_delta)
                    zero_cross_height = self._prev_height_above_floor + (height_above_floor - self._prev_height_above_floor) * t
        
        self._prev_waist_delta = delta
        self._prev_height_above_floor = height_above_floor
        
        result = {
            "frame": frame_num,
            "timestamp": timestamp,
            "height": z,
            "position": (x, y, z),
            "height_above_floor": height_above_floor,
            "jump_type": None,
            "jump_height": 0.0,
            "jump_distance": 0.0,
            "air_time": 0.0
        }
        
        # 水平距離の更新用関数
        def _update_horizontal_distance():
            if self.jump_start_position:
                distance = np.sqrt(
                    (x - self.jump_start_position[0]) ** 2
                    + (z - self.jump_start_position[2]) ** 2
                )
                if distance > self.jump_max_distance:
                    self.jump_max_distance = distance
        
        # ジャンプ状態の遷移（ゼロクロス検出）
        if self.jump_state == "ground":
            # 離陸検出: 負から正へのゼロクロス（deltaが負またはゼロから正になる）
            if prev_state in ("negative", "zero") and current_state == "positive":
                # 離陸時の全キーポイントの距離を記録
                takeoff_distances = {}
                if keypoints_3d and self.use_floor_detection and self.floor_detector:
                    for kp_name, kp_coords in keypoints_3d.items():
                        if kp_coords and kp_coords[0] is not None and kp_coords[1] is not None and kp_coords[2] is not None:
                            takeoff_distances[kp_name] = self.floor_detector.distance_to_floor(kp_coords)
                self.takeoff_transitions.append((frame_num, takeoff_distances))
                
                # ゼロクロス点の高さを使用（補間された値があればそれを使用、なければ基準高さ）
                if zero_cross_height is not None:
                    takeoff_height = zero_cross_height
                else:
                    # 補間できない場合は基準高さを使用（より正確）
                    takeoff_height = self.waist_baseline_height
                
                self.jump_state = "airborne"
                self.jump_start_frame = frame_num
                self.jump_start_timestamp = timestamp
                self.jump_start_position = (x, y, z)
                self.jump_takeoff_frame = frame_num
                self.jump_takeoff_timestamp = timestamp
                self.jump_takeoff_position = (x, y, z)
                self.jump_takeoff_height = takeoff_height  # ゼロクロス点の高さを使用
                self.jump_max_height = takeoff_height  # 初期値はゼロクロス点の高さ
                self.jump_max_distance = 0.0
                self._initial_state = False
                result["state"] = "jump_start"
            else:
                # 地面状態: 開始位置を記録（準備段階）
                if self.jump_start_position is None:
                    self.jump_start_frame = frame_num
                    self.jump_start_timestamp = timestamp
                    self.jump_start_position = (x, y, z)
                result["state"] = "ground"
        
        elif self.jump_state == "airborne":
            # 浮上中: 最大高さ・距離を更新
            if height_above_floor > self.jump_max_height:
                self.jump_max_height = height_above_floor
                self.jump_max_height_frame = frame_num
                self.jump_max_height_timestamp = timestamp
            _update_horizontal_distance()
            
            # 着地検出: 正から負またはゼロへのゼロクロス（deltaが正から負またはゼロになる）
            if prev_state == "positive" and current_state in ("negative", "zero"):
                # 着地時の全キーポイントの距離を記録
                landing_distances = {}
                if keypoints_3d and self.use_floor_detection and self.floor_detector:
                    for kp_name, kp_coords in keypoints_3d.items():
                        if kp_coords and kp_coords[0] is not None and kp_coords[1] is not None and kp_coords[2] is not None:
                            landing_distances[kp_name] = self.floor_detector.distance_to_floor(kp_coords)
                self.landing_transitions.append((frame_num, landing_distances))
                
                # ゼロクロス点の高さを使用（補間された値があればそれを使用、なければ基準高さ）
                if zero_cross_height is not None:
                    landing_height = zero_cross_height
                else:
                    # 補間できない場合は基準高さを使用（より正確）
                    landing_height = self.waist_baseline_height
                
                self.jump_state = "landing"
                self.jump_end_frame = frame_num
                self.jump_end_timestamp = timestamp
                self.jump_end_position = (x, y, z)
                self.jump_end_height = landing_height  # ゼロクロス点の高さを記録
                result["state"] = "jump_end"
            else:
                # まだ空中
                result["state"] = "jumping"
                if self.jump_takeoff_height is not None:
                    result["jump_height"] = height_above_floor - self.jump_takeoff_height
                result["jump_distance"] = self.jump_max_distance
        
        elif self.jump_state == "landing":
            # 着地: ジャンプ完了を記録
            # ジャンプ高さを計算（最大高さ - 離陸時の高さ）
            if self.jump_takeoff_height is not None:
                jump_height = self.jump_max_height - self.jump_takeoff_height
            elif self.jump_start_position:
                # フォールバック: 開始位置との差
                jump_height = self.jump_max_height - self.jump_start_position[2]
            else:
                jump_height = 0.0
            
            # 滞空時間を計算（秒）
            air_time = 0.0
            ascent_time = 0.0  # 離陸から頂点までの時間
            descent_time = 0.0  # 頂点から着地までの時間
            
            if self.jump_takeoff_timestamp is not None and self.jump_end_timestamp is not None:
                # タイムスタンプがミリ秒単位の可能性があるため、差分を計算
                timestamp_diff = abs(self.jump_end_timestamp - self.jump_takeoff_timestamp)
                # タイムスタンプ自体の値で単位を判定（ミリ秒なら10桁以上の値）
                is_milliseconds = self.jump_takeoff_timestamp > 1000000000
                if is_milliseconds:  # タイムスタンプがミリ秒単位
                    air_time = timestamp_diff / 1000.0  # ミリ秒→秒
                else:
                    # タイムスタンプが秒単位の場合
                    air_time = timestamp_diff
                # 異常値をチェック（最大10秒の滞空時間を想定）
                if air_time > 10.0:
                    # 誤差が大きい可能性があるので、フレーム数から計算を試みる
                    if self.jump_takeoff_frame is not None and self.jump_end_frame is not None:
                        frame_diff = abs(self.jump_end_frame - self.jump_takeoff_frame)
                        # 30fpsを仮定
                        air_time = frame_diff / 30.0
                        is_milliseconds = False
                
                # 上昇時間と下降時間を計算
                if self.jump_max_height_timestamp is not None:
                    # 離陸から頂点までの時間
                    ascent_timestamp_diff = abs(self.jump_max_height_timestamp - self.jump_takeoff_timestamp)
                    if is_milliseconds:
                        ascent_time = ascent_timestamp_diff / 1000.0
                    else:
                        ascent_time = ascent_timestamp_diff
                    
                    # 頂点から着地までの時間
                    descent_timestamp_diff = abs(self.jump_end_timestamp - self.jump_max_height_timestamp)
                    if is_milliseconds:
                        descent_time = descent_timestamp_diff / 1000.0
                    else:
                        descent_time = descent_timestamp_diff
                    
                    # 異常値チェック（フレーム数から計算）
                    if ascent_time > 10.0 and self.jump_takeoff_frame is not None and self.jump_max_height_frame is not None:
                        ascent_frame_diff = abs(self.jump_max_height_frame - self.jump_takeoff_frame)
                        ascent_time = ascent_frame_diff / 30.0
                    
                    if descent_time > 10.0 and self.jump_max_height_frame is not None and self.jump_end_frame is not None:
                        descent_frame_diff = abs(self.jump_end_frame - self.jump_max_height_frame)
                        descent_time = descent_frame_diff / 30.0
            
            # ジャンプの種類を判定
            if self.jump_max_distance < self.threshold_horizontal:
                result["jump_type"] = "vertical"
                result["jump_height"] = jump_height
                result["jump_distance"] = 0.0
            else:
                result["jump_type"] = "horizontal"
                result["jump_height"] = jump_height
                result["jump_distance"] = self.jump_max_distance
            
            result["air_time"] = air_time
            
            # ジャンプとして有効かどうかを判定（最小高さ・滞空時間の閾値チェック）
            is_valid_jump = (
                jump_height >= self.min_jump_height and
                air_time >= self.min_air_time
            )
            
            if not is_valid_jump:
                # 無効なジャンプは記録しない（誤検出を除外）
                print(f"  Skipping invalid jump: height={jump_height*100:.1f}cm (min={self.min_jump_height*100:.1f}cm), "
                      f"air_time={air_time*1000:.1f}ms (min={self.min_air_time*1000:.1f}ms)")
            else:
                # 水平方向のずれを計算（離陸位置と着地位置のXZ平面での距離）
                horizontal_offset = 0.0
                if self.jump_takeoff_position and self.jump_end_position:
                    # XZ平面での距離を計算（Y軸は無視）
                    x_diff = self.jump_end_position[0] - self.jump_takeoff_position[0]
                    z_diff = (
                        self.jump_end_position[2] - self.jump_takeoff_position[2]
                        if len(self.jump_end_position) > 2 and len(self.jump_takeoff_position) > 2
                        else 0.0
                    )
                    horizontal_offset = np.sqrt(x_diff**2 + z_diff**2)
                
                # 初速を計算
                initial_velocity = self._calculate_initial_velocity(
                    result["jump_height"],
                    result["jump_distance"],
                    air_time
                )
                
                # 着陸時の速度を計算
                landing_velocity = self._calculate_landing_velocity(
                    result["jump_height"],
                    horizontal_offset,
                    descent_time
                )
                
                # 着陸時の加速度を計算
                landing_acceleration = self._calculate_landing_acceleration(
                    landing_velocity,
                    descent_time
                )
                
                # 検出されたジャンプを記録
                jump_data = {
                    "frame_start": self.jump_start_frame,
                    "frame_takeoff": self.jump_takeoff_frame,
                    "frame_end": self.jump_end_frame,
                    "timestamp_start": self.jump_start_timestamp,
                    "timestamp_takeoff": self.jump_takeoff_timestamp,
                    "timestamp_end": self.jump_end_timestamp,
                    "jump_type": result["jump_type"],
                    "height": result["jump_height"],
                    "distance": result["jump_distance"],
                    "horizontal_offset": horizontal_offset,  # 水平方向のずれ（着地位置のずれ）
                    "air_time": air_time,
                    "ascent_time": ascent_time,  # 離陸から頂点までの時間
                    "descent_time": descent_time,  # 頂点から着地までの時間
                    "max_height": self.jump_max_height,
                    "peak_frame": self.jump_max_height_frame,  # 頂点のフレーム
                    "peak_timestamp": self.jump_max_height_timestamp,  # 頂点のタイムスタンプ
                    "start_position": self.jump_start_position,
                    "takeoff_position": self.jump_takeoff_position,
                    "end_position": self.jump_end_position,
                    "start_height": self.jump_takeoff_height,  # 離陸時の高さ（ゼロクロス点）
                    "end_height": self.jump_end_height,  # 着地時の高さ（ゼロクロス点）
                    "initial_velocity": initial_velocity,  # 初速情報
                    "landing_velocity": landing_velocity,  # 着陸時の速度
                    "landing_acceleration": landing_acceleration  # 着陸時の加速度（衝撃加速度）
                }
                self.detected_jumps.append(jump_data)
            
            # 状態をリセット
            self.jump_state = "ground"
            self.jump_start_frame = None
            self.jump_start_timestamp = None
            self.jump_start_position = None
            self.jump_takeoff_frame = None
            self.jump_takeoff_timestamp = None
            self.jump_takeoff_position = None
            self.jump_end_frame = None
            self.jump_end_timestamp = None
            self.jump_end_position = None
            self.jump_end_height = None
            # ジャンプ検出関連の変数をリセット
            self.jump_max_height = 0.0
            self.jump_max_distance = 0.0
            self.jump_takeoff_height = None
            self._prev_waist_delta = None  # ゼロクロス検出の履歴をリセット
            self._prev_height_above_floor = None  # 前フレームの高さをリセット
        
        return result

    def _detect_jump_with_baseline(
        self, frame_num, x, y, z, timestamp, height_above_floor
    ):
        """
        腰の床距離と事前に与えられた基準値のゼロクロスでジャンプを検出
        """
        result = {
            "frame": frame_num,
            "timestamp": timestamp,
            "height": z,
            "position": (x, y, z),
            "height_above_floor": height_above_floor,
            "jump_type": None,
            "jump_height": 0.0,
            "jump_distance": 0.0,
            "air_time": 0.0,
            "state": "ground",
        }

        delta = height_above_floor - self.waist_baseline_height
        prev_state = self._delta_state(self._prev_waist_delta)
        current_state = self._delta_state(delta)
        self._prev_waist_delta = delta

        # 水平距離の更新用
        def _update_horizontal_distance():
            if self.jump_start_position:
                distance = np.sqrt(
                    (x - self.jump_start_position[0]) ** 2
                    + (z - self.jump_start_position[2]) ** 2
                )
                if distance > self.jump_max_distance:
                    self.jump_max_distance = distance

        if self.jump_state == "ground":
            if prev_state in ("negative", "zero") and current_state == "positive":
                self.jump_state = "airborne"
                self.jump_start_frame = frame_num
                self.jump_start_timestamp = timestamp
                self.jump_start_position = (x, y, z)
                self.jump_takeoff_frame = frame_num
                self.jump_takeoff_timestamp = timestamp
                self.jump_takeoff_position = (x, y, z)
                self.jump_takeoff_height = height_above_floor
                self.jump_max_height = height_above_floor
                self.jump_max_height_frame = frame_num
                self.jump_max_height_timestamp = timestamp
                self.jump_max_distance = 0.0
                self._initial_state = False
                result["state"] = "jump_start"
            else:
                result["state"] = "ground"
            return result

        if self.jump_state == "airborne":
            if height_above_floor > self.jump_max_height:
                self.jump_max_height = height_above_floor
                self.jump_max_height_frame = frame_num
                self.jump_max_height_timestamp = timestamp
            _update_horizontal_distance()

            if (
                prev_state == "positive"
                and current_state in ("negative", "zero")
                and self.jump_takeoff_height is not None
            ):
                self.jump_end_frame = frame_num
                self.jump_end_timestamp = timestamp
                self.jump_end_position = (x, y, z)
                jump_height = self.jump_max_height - self.jump_takeoff_height

                air_time = 0.0
                ascent_time = 0.0  # 離陸から頂点までの時間
                descent_time = 0.0  # 頂点から着地までの時間
                
                if (
                    self.jump_takeoff_timestamp is not None
                    and timestamp is not None
                ):
                    timestamp_diff = abs(timestamp - self.jump_takeoff_timestamp)
                    is_milliseconds = self.jump_takeoff_timestamp > 1000000000
                    if is_milliseconds:
                        air_time = timestamp_diff / 1000.0
                    else:
                        air_time = timestamp_diff
                    if air_time > 10.0 and self.jump_takeoff_frame is not None:
                        frame_diff = abs(frame_num - self.jump_takeoff_frame)
                        air_time = frame_diff / 30.0
                        is_milliseconds = False
                    
                    # 上昇時間と下降時間を計算
                    if self.jump_max_height_timestamp is not None:
                        # 離陸から頂点までの時間
                        ascent_timestamp_diff = abs(self.jump_max_height_timestamp - self.jump_takeoff_timestamp)
                        if is_milliseconds:
                            ascent_time = ascent_timestamp_diff / 1000.0
                        else:
                            ascent_time = ascent_timestamp_diff
                        
                        # 頂点から着地までの時間
                        descent_timestamp_diff = abs(timestamp - self.jump_max_height_timestamp)
                        if is_milliseconds:
                            descent_time = descent_timestamp_diff / 1000.0
                        else:
                            descent_time = descent_timestamp_diff
                        
                        # 異常値チェック（フレーム数から計算）
                        if ascent_time > 10.0 and self.jump_takeoff_frame is not None and self.jump_max_height_frame is not None:
                            ascent_frame_diff = abs(self.jump_max_height_frame - self.jump_takeoff_frame)
                            ascent_time = ascent_frame_diff / 30.0
                        
                        if descent_time > 10.0 and self.jump_max_height_frame is not None:
                            descent_frame_diff = abs(frame_num - self.jump_max_height_frame)
                            descent_time = descent_frame_diff / 30.0

                jump_distance = (
                    self.jump_max_distance if self.jump_max_distance else 0.0
                )
                if jump_distance < self.threshold_horizontal:
                    jump_type = "vertical"
                    jump_distance_out = 0.0
                else:
                    jump_type = "horizontal"
                    jump_distance_out = jump_distance

                result["state"] = "jump_end"
                result["jump_type"] = jump_type
                result["jump_height"] = jump_height
                result["jump_distance"] = jump_distance_out
                result["air_time"] = air_time

                is_valid_jump = (
                    jump_height >= self.min_jump_height
                    and air_time >= self.min_air_time
                )

                if is_valid_jump:
                    # 水平方向のずれを計算（離陸位置と着地位置のXZ平面での距離）
                    horizontal_offset = 0.0
                    if self.jump_takeoff_position and self.jump_end_position:
                        # XZ平面での距離を計算（Y軸は無視）
                        x_diff = self.jump_end_position[0] - self.jump_takeoff_position[0]
                        z_diff = (
                            self.jump_end_position[2] - self.jump_takeoff_position[2]
                            if len(self.jump_end_position) > 2 and len(self.jump_takeoff_position) > 2
                            else 0.0
                        )
                        horizontal_offset = np.sqrt(x_diff**2 + z_diff**2)
                    
                    # 初速を計算
                    initial_velocity = self._calculate_initial_velocity(
                        jump_height,
                        jump_distance_out,
                        air_time
                    )
                    
                    # 着陸時の速度を計算
                    landing_velocity = self._calculate_landing_velocity(
                        jump_height,
                        horizontal_offset,
                        descent_time
                    )
                    
                    # 着陸時の加速度を計算
                    landing_acceleration = self._calculate_landing_acceleration(
                        landing_velocity,
                        descent_time
                    )
                    
                    jump_data = {
                        "frame_start": self.jump_start_frame,
                        "frame_takeoff": self.jump_takeoff_frame,
                        "frame_end": self.jump_end_frame,
                        "timestamp_start": self.jump_start_timestamp,
                        "timestamp_takeoff": self.jump_takeoff_timestamp,
                        "timestamp_end": self.jump_end_timestamp,
                        "jump_type": jump_type,
                        "height": jump_height,
                        "distance": jump_distance_out,
                        "horizontal_offset": horizontal_offset,  # 水平方向のずれ（着地位置のずれ）
                        "air_time": air_time,
                        "ascent_time": ascent_time,  # 離陸から頂点までの時間
                        "descent_time": descent_time,  # 頂点から着地までの時間
                        "max_height": self.jump_max_height,
                        "peak_frame": self.jump_max_height_frame,  # 頂点のフレーム
                        "peak_timestamp": self.jump_max_height_timestamp,  # 頂点のタイムスタンプ
                        "start_position": self.jump_start_position,
                        "takeoff_position": self.jump_takeoff_position,
                        "end_position": self.jump_end_position,
                        "initial_velocity": initial_velocity,  # 初速情報
                        "landing_velocity": landing_velocity,  # 着陸時の速度
                        "landing_acceleration": landing_acceleration  # 着陸時の加速度（衝撃加速度）
                    }
                    self.detected_jumps.append(jump_data)

                # リセット
                self.jump_state = "ground"
                self.jump_start_frame = None
                self.jump_start_timestamp = None
                self.jump_start_position = None
                self.jump_takeoff_frame = None
                self.jump_takeoff_timestamp = None
                self.jump_takeoff_position = None
                self.jump_end_frame = None
                self.jump_end_timestamp = None
                self.jump_end_position = None
                self.jump_takeoff_height = None
                self.jump_max_height = 0.0
                self.jump_max_height_frame = None
                self.jump_max_height_timestamp = None
                self.jump_max_distance = 0.0
                return result

            # まだ空中
            result["state"] = "jumping"
            if self.jump_takeoff_height is not None:
                result["jump_height"] = self.jump_max_height - self.jump_takeoff_height
            result["jump_distance"] = self.jump_max_distance
            return result

        # フォールバック
        result["state"] = "ground"
        return result

    def _calculate_initial_velocity(self, height, distance, air_time):
        """
        ジャンプの初速を計算
        
        Args:
            height: ジャンプ高さ（m）
            distance: 水平距離（m）
            air_time: 滞空時間（s）
        
        Returns:
            dict: {
                "vertical": 垂直初速（m/s）,
                "horizontal": 水平初速（m/s）,
                "total": 合成初速（m/s）
            }
        """
        g = 9.81  # 重力加速度（m/s²）
        
        # 垂直初速: v_y = sqrt(2 * g * h) または v_y = g * t / 2
        # より正確な計算: v_y = sqrt(2 * g * h) を使用
        if height > 0 and air_time > 0:
            # 方法1: 高さから計算（より正確）
            vertical_velocity = np.sqrt(2 * g * height)
            # 方法2: 滞空時間から計算（検証用）
            vertical_velocity_from_time = g * air_time / 2.0
            # 両方の値が近い場合は高さから計算した値を使用
            if abs(vertical_velocity - vertical_velocity_from_time) / vertical_velocity < 0.2:
                vertical_velocity = vertical_velocity
            else:
                # 値が大きく異なる場合は平均を取る
                vertical_velocity = (vertical_velocity + vertical_velocity_from_time) / 2.0
        elif height > 0:
            # 高さのみから計算
            vertical_velocity = np.sqrt(2 * g * height)
        elif air_time > 0:
            # 滞空時間のみから計算
            vertical_velocity = g * air_time / 2.0
        else:
            vertical_velocity = 0.0
        
        # 水平初速: v_x = d / t
        if air_time > 0:
            horizontal_velocity = distance / air_time
        else:
            horizontal_velocity = 0.0
        
        # 合成初速: v = sqrt(v_x² + v_y²)
        total_velocity = np.sqrt(vertical_velocity**2 + horizontal_velocity**2)
        
        return {
            "vertical": vertical_velocity,
            "horizontal": horizontal_velocity,
            "total": total_velocity
        }

    def _calculate_landing_velocity(self, jump_height, horizontal_offset, descent_time):
        """
        着陸時の速度を計算
        
        Args:
            jump_height: ジャンプ高さ（m）
            horizontal_offset: 水平方向のずれ（m）
            descent_time: 下降時間（s）
        
        Returns:
            dict: {
                "vertical": 着陸時の垂直速度（m/s）,
                "horizontal": 着陸時の水平速度（m/s）,
                "total": 着陸時の合成速度（m/s）
            }
        """
        g = 9.81  # 重力加速度（m/s²）
        
        # 着陸時の垂直速度: v_y = sqrt(2 * g * h) または v_y = g * t_descent
        if jump_height > 0 and descent_time > 0:
            # 方法1: 高さから計算（より正確）
            vertical_velocity = np.sqrt(2 * g * jump_height)
            # 方法2: 下降時間から計算（検証用）
            vertical_velocity_from_time = g * descent_time
            # 両方の値が近い場合は高さから計算した値を使用
            if abs(vertical_velocity - vertical_velocity_from_time) / vertical_velocity < 0.3:
                vertical_velocity = vertical_velocity
            else:
                # 値が大きく異なる場合は平均を取る
                vertical_velocity = (vertical_velocity + vertical_velocity_from_time) / 2.0
        elif jump_height > 0:
            # 高さのみから計算
            vertical_velocity = np.sqrt(2 * g * jump_height)
        elif descent_time > 0:
            # 下降時間のみから計算
            vertical_velocity = g * descent_time
        else:
            vertical_velocity = 0.0
        
        # 着陸時の水平速度: v_x = horizontal_offset / descent_time
        if descent_time > 0:
            horizontal_velocity = horizontal_offset / descent_time
        else:
            horizontal_velocity = 0.0
        
        # 着陸時の合成速度: v = sqrt(v_x² + v_y²)
        total_velocity = np.sqrt(vertical_velocity**2 + horizontal_velocity**2)
        
        return {
            "vertical": vertical_velocity,
            "horizontal": horizontal_velocity,
            "total": total_velocity
        }

    def _calculate_landing_acceleration(self, landing_velocity, descent_time):
        """
        着陸時の加速度（衝撃加速度）を推定
        
        Args:
            landing_velocity: 着陸時の速度（dict with "vertical", "horizontal", "total"）
            descent_time: 下降時間（s）
        
        Returns:
            dict: {
                "vertical": 着陸時の垂直加速度（m/s²）,
                "total": 着陸時の合成加速度（m/s²）
            }
        """
        g = 9.81  # 重力加速度（m/s²）
        
        # 着陸時の垂直加速度
        # 着陸時の減速を仮定（着地から停止まで0.1秒と仮定、実際の値は個人差がある）
        landing_deceleration_time = 0.1  # 秒（着地から停止までの時間）
        
        if landing_velocity["vertical"] > 0 and landing_deceleration_time > 0:
            # 加速度 = 速度変化 / 時間
            # 着陸時の垂直速度から0への減速
            vertical_acceleration = landing_velocity["vertical"] / landing_deceleration_time
            # 重力加速度を考慮（着地時の衝撃 = 減速加速度 + 重力加速度）
            vertical_acceleration_impact = vertical_acceleration + g
        else:
            vertical_acceleration_impact = 0.0
        
        # 合成加速度（主に垂直成分が支配的）
        total_acceleration = vertical_acceleration_impact
        
        return {
            "vertical": vertical_acceleration_impact,
            "total": total_acceleration
        }

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
                "avg_distance": 0.0,
                "max_air_time": 0.0,
                "avg_air_time": 0.0,
                "max_vertical_horizontal_offset": 0.0,
                "avg_vertical_horizontal_offset": 0.0,
                "max_landing_velocity": 0.0,
                "avg_landing_velocity": 0.0,
                "max_landing_vertical_velocity": 0.0,
                "avg_landing_vertical_velocity": 0.0,
                "max_landing_horizontal_velocity": 0.0,
                "avg_landing_horizontal_velocity": 0.0,
                "max_landing_acceleration": 0.0,
                "avg_landing_acceleration": 0.0,
                "max_landing_vertical_acceleration": 0.0,
                "avg_landing_vertical_acceleration": 0.0
            }

        vertical_jumps = [j for j in self.detected_jumps if j["jump_type"] == "vertical"]
        horizontal_jumps = [j for j in self.detected_jumps if j["jump_type"] == "horizontal"]

        all_heights = [j["height"] for j in self.detected_jumps]
        # 距離は全ジャンプから取得（vertical jumpも距離0.0として含める）
        all_distances = [j["distance"] for j in self.detected_jumps]
        # 異常な滞空時間を除外（10秒以上は異常値として扱う）
        all_air_times = [j.get("air_time", 0.0) for j in self.detected_jumps 
                        if "air_time" in j and j.get("air_time", 0.0) < 10.0]
        
        # 初速の統計
        all_initial_velocities = [j.get("initial_velocity", {}).get("total", 0.0) 
                                 for j in self.detected_jumps 
                                 if "initial_velocity" in j]
        all_vertical_velocities = [j.get("initial_velocity", {}).get("vertical", 0.0) 
                                   for j in self.detected_jumps 
                                   if "initial_velocity" in j]
        all_horizontal_velocities = [j.get("initial_velocity", {}).get("horizontal", 0.0) 
                                     for j in self.detected_jumps 
                                     if "initial_velocity" in j]
        
        # 垂直ジャンプの水平方向のずれの統計
        vertical_horizontal_offsets = [j.get("horizontal_offset", 0.0) 
                                       for j in vertical_jumps 
                                       if "horizontal_offset" in j]
        
        # 着陸時の速度の統計
        all_landing_velocities = [j.get("landing_velocity", {}).get("total", 0.0) 
                                  for j in self.detected_jumps 
                                  if "landing_velocity" in j]
        all_landing_vertical_velocities = [j.get("landing_velocity", {}).get("vertical", 0.0) 
                                           for j in self.detected_jumps 
                                           if "landing_velocity" in j]
        all_landing_horizontal_velocities = [j.get("landing_velocity", {}).get("horizontal", 0.0) 
                                            for j in self.detected_jumps 
                                            if "landing_velocity" in j]
        
        # 着陸時の加速度の統計
        all_landing_accelerations = [j.get("landing_acceleration", {}).get("total", 0.0) 
                                    for j in self.detected_jumps 
                                    if "landing_acceleration" in j]
        all_landing_vertical_accelerations = [j.get("landing_acceleration", {}).get("vertical", 0.0) 
                                             for j in self.detected_jumps 
                                             if "landing_acceleration" in j]

        return {
            "total_jumps": len(self.detected_jumps),
            "vertical_jumps": len(vertical_jumps),
            "horizontal_jumps": len(horizontal_jumps),
            "max_height": max(all_heights) if all_heights else 0.0,
            "max_distance": max(all_distances) if all_distances else 0.0,
            "avg_height": np.mean(all_heights) if all_heights else 0.0,
            "avg_distance": np.mean(all_distances) if all_distances and len(all_distances) > 0 else 0.0,
            "max_air_time": max(all_air_times) if all_air_times else 0.0,
            "avg_air_time": np.mean(all_air_times) if all_air_times else 0.0,
            "max_initial_velocity": max(all_initial_velocities) if all_initial_velocities else 0.0,
            "avg_initial_velocity": np.mean(all_initial_velocities) if all_initial_velocities else 0.0,
            "max_vertical_velocity": max(all_vertical_velocities) if all_vertical_velocities else 0.0,
            "avg_vertical_velocity": np.mean(all_vertical_velocities) if all_vertical_velocities else 0.0,
            "max_horizontal_velocity": max(all_horizontal_velocities) if all_horizontal_velocities else 0.0,
            "avg_horizontal_velocity": np.mean(all_horizontal_velocities) if all_horizontal_velocities else 0.0,
            "max_vertical_horizontal_offset": max(vertical_horizontal_offsets) if vertical_horizontal_offsets else 0.0,
            "avg_vertical_horizontal_offset": np.mean(vertical_horizontal_offsets) if vertical_horizontal_offsets else 0.0,
            "max_landing_velocity": max(all_landing_velocities) if all_landing_velocities else 0.0,
            "avg_landing_velocity": np.mean(all_landing_velocities) if all_landing_velocities else 0.0,
            "max_landing_vertical_velocity": max(all_landing_vertical_velocities) if all_landing_vertical_velocities else 0.0,
            "avg_landing_vertical_velocity": np.mean(all_landing_vertical_velocities) if all_landing_vertical_velocities else 0.0,
            "max_landing_horizontal_velocity": max(all_landing_horizontal_velocities) if all_landing_horizontal_velocities else 0.0,
            "avg_landing_horizontal_velocity": np.mean(all_landing_horizontal_velocities) if all_landing_horizontal_velocities else 0.0,
            "max_landing_acceleration": max(all_landing_accelerations) if all_landing_accelerations else 0.0,
            "avg_landing_acceleration": np.mean(all_landing_accelerations) if all_landing_accelerations else 0.0,
            "max_landing_vertical_acceleration": max(all_landing_vertical_accelerations) if all_landing_vertical_accelerations else 0.0,
            "avg_landing_vertical_acceleration": np.mean(all_landing_vertical_accelerations) if all_landing_vertical_accelerations else 0.0,
            "jumps": self.detected_jumps
        }
    
    def analyze_keypoint_variability(self):
        """
        全キーポイントの床からの距離の変動性を分析
        歩行時は上下変化が少なく、ジャンプ時は変位が顕著に現れるキーポイントを特定
        
        Returns:
            dict: 各キーポイントの変動性統計情報
                {
                    'keypoint_name': {
                        'mean': 平均距離,
                        'std': 標準偏差,
                        'min': 最小距離,
                        'max': 最大距離,
                        'range': 最大変位（max - min）,
                        'cv': 変動係数（std/mean）,
                        'valid_samples': 有効サンプル数,
                        'walking_mean': 歩行時平均（ground状態のみ）,
                        'walking_std': 歩行時標準偏差,
                        'jumping_mean': ジャンプ時平均（jumping状態のみ）,
                        'jumping_std': ジャンプ時標準偏差,
                        'jump_walk_ratio': ジャンプ時/歩行時の変動比,
                        'jump_sensitivity': ジャンプ検出感度スコア
                    },
                    ...
                }
        """
        variability_stats = {}
        
        for kp_name, distance_history in self.keypoint_distance_history.items():
            # 履歴から距離と状態を分離
            all_distances = []
            walking_distances = []
            jumping_distances = []
            
            for item in distance_history:
                if isinstance(item, tuple):
                    distance, state = item
                else:
                    # 後方互換性のため（古いデータ形式）
                    distance = item
                    state = 'ground'
                
                if distance is not None:
                    all_distances.append(distance)
                    if state == 'ground' or state == 'initializing':
                        walking_distances.append(distance)
                    elif state in ['takeoff', 'airborne', 'jumping', 'landing']:
                        jumping_distances.append(distance)
            
            if len(all_distances) < 2:
                continue
            
            all_distances = np.array(all_distances)
            
            # 全体の統計量を計算
            mean_dist = np.mean(all_distances)
            std_dist = np.std(all_distances)
            min_dist = np.min(all_distances)
            max_dist = np.max(all_distances)
            range_dist = max_dist - min_dist
            cv = std_dist / mean_dist if mean_dist > 0 else 0.0
            
            # 歩行時とジャンプ時の統計を別々に計算
            walking_mean = np.mean(walking_distances) if len(walking_distances) > 0 else None
            walking_std = np.std(walking_distances) if len(walking_distances) > 1 else None
            
            jumping_mean = np.mean(jumping_distances) if len(jumping_distances) > 0 else None
            jumping_std = np.std(jumping_distances) if len(jumping_distances) > 1 else None
            
            # ジャンプ時と歩行時の変動比を計算
            jump_walk_ratio = None
            jump_sensitivity = 0.0
            
            if walking_std is not None and walking_std > 0 and jumping_std is not None:
                # 標準偏差の比（ジャンプ時の変動が歩行時よりどれだけ大きいか）
                jump_walk_ratio = jumping_std / walking_std
                
                # ジャンプ検出感度スコア: ジャンプ時の変動が大きく、歩行時の変動が小さいほど高い
                # 式: (jumping_std - walking_std) / (jumping_std + walking_std + epsilon)
                # これは-1から1の範囲で、1に近いほどジャンプ検出に適している
                epsilon = 1e-6
                if jumping_std + walking_std > 0:
                    jump_sensitivity = (jumping_std - walking_std) / (jumping_std + walking_std + epsilon)
            
            variability_stats[kp_name] = {
                'mean': float(mean_dist),
                'std': float(std_dist),
                'min': float(min_dist),
                'max': float(max_dist),
                'range': float(range_dist),
                'cv': float(cv),
                'valid_samples': len(all_distances),
                'walking_samples': len(walking_distances),
                'jumping_samples': len(jumping_distances),
                'walking_mean': float(walking_mean) if walking_mean is not None else None,
                'walking_std': float(walking_std) if walking_std is not None else None,
                'jumping_mean': float(jumping_mean) if jumping_mean is not None else None,
                'jumping_std': float(jumping_std) if jumping_std is not None else None,
                'jump_walk_ratio': float(jump_walk_ratio) if jump_walk_ratio is not None else None,
                'jump_sensitivity': float(jump_sensitivity)
            }
        
        return variability_stats
    
    def get_most_jump_sensitive_keypoints(self, min_samples=10, top_n=5):
        """
        ジャンプ検出に最も敏感なキーポイントを特定
        変動係数（CV）と最大変位（range）の両方を考慮
        
        Args:
            min_samples: 最小サンプル数（この数を下回るキーポイントは除外）
            top_n: 返すキーポイントの数
        
        Returns:
            list: タプルのリスト [(keypoint_name, stats_dict), ...]
                  変動係数と最大変位の組み合わせスコアでソート
        """
        stats = self.analyze_keypoint_variability()
        
        # フィルタリング（最小サンプル数以上のもののみ）
        filtered_stats = {
            kp_name: stats_dict
            for kp_name, stats_dict in stats.items()
            if stats_dict['valid_samples'] >= min_samples
        }
        
        if len(filtered_stats) == 0:
            return []
        
        # スコアリング: ジャンプ検出感度（jump_sensitivity）を優先的に使用
        # jump_sensitivityが高いほど、歩行時との違いが明確（ジャンプ検出に適している）
        scored_keypoints = []
        for kp_name, stats_dict in filtered_stats.items():
            # ジャンプ検出感度スコアを主要指標として使用
            # 補助的にCVとrangeも考慮
            jump_sensitivity_score = (stats_dict['jump_sensitivity'] + 1.0) / 2.0  # -1~1を0~1に正規化
            
            # 変動係数と最大変位も考慮（補助スコア）
            cv_score = min(stats_dict['cv'], 2.0) / 2.0 if stats_dict['cv'] > 0 else 0.0
            range_score = min(stats_dict['range'] / 0.5, 1.0)
            
            # ジャンプ検出感度を優先しつつ、全体の変動も考慮
            # jump_sensitivityが利用可能な場合はそれを重視
            if stats_dict['jump_sensitivity'] is not None and stats_dict['jumping_samples'] > 0:
                # ジャンプ検出感度70%、変動係数20%、最大変位10%
                combined_score = jump_sensitivity_score * 0.7 + cv_score * 0.2 + range_score * 0.1
            else:
                # ジャンプ状態のデータがない場合は従来の方法で評価
                combined_score = (cv_score * 0.5 + range_score * 0.5)
            
            scored_keypoints.append((kp_name, stats_dict, combined_score))
        
        # スコアでソート（降順）
        scored_keypoints.sort(key=lambda x: x[2], reverse=True)
        
        # top_n個を返す
        return [(kp_name, stats) for kp_name, stats, score in scored_keypoints[:top_n]]

    def get_trajectory(self):
        """軌跡データを取得"""
        return self.trajectory
    
    def analyze_jump_transitions(self):
        """
        ジャンプの始まり（離陸）と終わり（着地）を捉えるためのキーポイントを特定
        離陸時と着地時の急激な変化を示すキーポイントを分析
        
        Returns:
            dict: {
                'takeoff_keypoints': [(keypoint_name, stats_dict), ...],
                'landing_keypoints': [(keypoint_name, stats_dict), ...]
            }
        """
        takeoff_analysis = {}
        landing_analysis = {}
        
        # 離陸時の変化を分析
        if len(self.takeoff_transitions) > 0:
            # 各キーポイントについて、離陸前後の変化を計算
            for kp_name in self.keypoint_distance_history.keys():
                changes = []
                
                for i, (takeoff_frame, takeoff_dists) in enumerate(self.takeoff_transitions):
                    if kp_name not in takeoff_dists or takeoff_dists[kp_name] is None:
                        continue
                    
                    # 離陸直前（3フレーム前）と離陸時の距離を取得
                    # 履歴から該当フレーム付近の値を探す
                    history = list(self.keypoint_distance_history[kp_name])
                    
                    # 離陸時の値を取得
                    takeoff_value = takeoff_dists[kp_name]
                    
                    # 離陸直前の値を取得（履歴から遡って探す）
                    # 履歴は最近の値が最後にあるので、逆順で探す
                    before_value = None
                    for j in range(len(history) - 1, max(0, len(history) - 10), -1):
                        item = history[j]
                        if isinstance(item, tuple):
                            dist, state = item
                        else:
                            dist = item
                            state = 'ground'
                        
                        if dist is not None and state in ['ground', 'initializing']:
                            before_value = dist
                            break
                    
                    if before_value is not None and takeoff_value is not None:
                        # 変化量を計算（離陸時に上昇するので正の値）
                        change = takeoff_value - before_value
                        changes.append(change)
                
                if len(changes) > 0:
                    changes_array = np.array(changes)
                    avg_change = np.mean(changes_array)
                    std_change = np.std(changes_array)
                    # スコア: 平均変化量の絶対値と一貫性（1/std）を考慮
                    score = abs(avg_change) * (1.0 / (std_change + 0.01))  # 一貫性の高い変化ほど高スコア
                    takeoff_analysis[kp_name] = {
                        'avg_change': float(avg_change),
                        'std_change': float(std_change),
                        'score': float(score),
                        'samples': len(changes)
                    }
        
        # 着地時の変化を分析
        if len(self.landing_transitions) > 0:
            for kp_name in self.keypoint_distance_history.keys():
                changes = []
                
                for i, (landing_frame, landing_dists) in enumerate(self.landing_transitions):
                    if kp_name not in landing_dists or landing_dists[kp_name] is None:
                        continue
                    
                    landing_value = landing_dists[kp_name]
                    
                    # 着地直前（空中の状態）の値を取得
                    history = list(self.keypoint_distance_history[kp_name])
                    before_value = None
                    for j in range(len(history) - 1, max(0, len(history) - 10), -1):
                        item = history[j]
                        if isinstance(item, tuple):
                            dist, state = item
                        else:
                            dist = item
                            state = 'ground'
                        
                        if dist is not None and state in ['airborne', 'jumping', 'takeoff']:
                            before_value = dist
                            break
                    
                    if before_value is not None and landing_value is not None:
                        # 変化量を計算（着地時に下降するので負の値）
                        change = landing_value - before_value
                        changes.append(change)
                
                if len(changes) > 0:
                    changes_array = np.array(changes)
                    avg_change = np.mean(changes_array)
                    std_change = np.std(changes_array)
                    # スコア: 平均変化量の絶対値と一貫性を考慮
                    score = abs(avg_change) * (1.0 / (std_change + 0.01))
                    landing_analysis[kp_name] = {
                        'avg_change': float(avg_change),
                        'std_change': float(std_change),
                        'score': float(score),
                        'samples': len(changes)
                    }
        
        # スコアでソート
        takeoff_sorted = sorted(takeoff_analysis.items(), key=lambda x: x[1]['score'], reverse=True)
        landing_sorted = sorted(landing_analysis.items(), key=lambda x: x[1]['score'], reverse=True)
        
        return {
            'takeoff_keypoints': [(name, stats) for name, stats in takeoff_sorted],
            'landing_keypoints': [(name, stats) for name, stats in landing_sorted]
        }

