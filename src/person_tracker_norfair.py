"""
Person Tracker using NorFair

NorFairライブラリを使用した高精度な人物トラッキング
https://github.com/tryolabs/norfair
"""

import numpy as np
from collections import deque

# NorFairのインポート（オプション）
try:
    from norfair import Tracker, Detection
    NORFAIR_AVAILABLE = True
except ImportError:
    NORFAIR_AVAILABLE = False
    print("Warning: NorFair not available. Install with: pip install norfair")

# 骨格構造の定義（COCO形式）
COCO_PAIRS = [
    [1, 2],   # Neck-RShoulder
    [1, 5],   # Neck-LShoulder
    [2, 3],   # RShoulder-RElbow
    [3, 4],   # RElbow-RWrist
    [5, 6],   # LShoulder-LElbow
    [6, 7],   # LElbow-LWrist
    [1, 8],   # Neck-RHip
    [8, 9],   # RHip-RKnee
    [9, 10],  # RKnee-RAnkle
    [1, 11],  # Neck-LHip
    [11, 12], # LHip-LKnee
    [12, 13], # LKnee-LAnkle
    [1, 0],   # Neck-Nose
    [0, 14],  # Nose-REye
    [14, 16], # REye-REar
    [0, 15],  # Nose-LEye
    [15, 17], # LEye-LEar
]


class PersonTrackerNorFair:
    """NorFairを使用した複数人のトラッキングクラス"""

    def __init__(self, distance_threshold=100, hit_counter_max=120, initialization_delay=5, min_confidence=0.3, min_valid_keypoints=10, 
                 max_3d_distance=0.5, valid_3d_bounds=None):
        """
        Args:
            distance_threshold: トラッキング距離の閾値（ピクセル、デフォルト: 100）
            hit_counter_max: トラッキングが失われるまでの最大フレーム数（デフォルト: 120）
            initialization_delay: 初期化遅延（デフォルト: 5、ノイズ検出を防ぐ）
            min_confidence: 検出の最小信頼度（デフォルト: 0.3）
            min_valid_keypoints: 有効なキーポイントの最小数（デフォルト: 10）
            max_3d_distance: マッチング時の最大3D距離（メートル、デフォルト: 0.5）
            valid_3d_bounds: 有効な3D位置の範囲 [(x_min, x_max), (y_min, y_max), (z_min, z_max)]（デフォルト: None = 自動設定）
        """
        if not NORFAIR_AVAILABLE:
            raise ImportError(
                "NorFair is not installed. Install with: pip install norfair"
            )
        
        self.distance_threshold = distance_threshold
        self.hit_counter_max = hit_counter_max
        self.initialization_delay = initialization_delay
        self.min_confidence = min_confidence
        self.min_valid_keypoints = min_valid_keypoints
        self.max_3d_distance = max_3d_distance
        
        # 有効な3D位置の範囲を設定（デフォルト値）
        if valid_3d_bounds is None:
            # 一般的な室内環境での妥当な範囲
            self.valid_3d_bounds = [
                (-2.0, 2.0),   # X: -2m ~ 2m
                (-1.0, 2.0),   # Y: -1m ~ 2m（床から2mまで）
                (1.0, 5.0)     # Z: 1m ~ 5m（深度）
            ]
        else:
            self.valid_3d_bounds = valid_3d_bounds
        
        # NorFairトラッカーを初期化
        self.tracker = Tracker(
            distance_function="euclidean",
            distance_threshold=distance_threshold,
            hit_counter_max=hit_counter_max,
            initialization_delay=initialization_delay,
        )
        
        # キーポイントの保存: {track_id: keypoints}
        self.tracked_keypoints = {}
        # 3D位置の保存: {track_id: (x, y, z)} - 中心点の3D位置
        self.tracked_3d_positions = {}
        # キーポイントの履歴: {track_id: deque([keypoints_history])} - ポーズの類似度計算用
        self.tracked_keypoints_history = {}
        # 骨格構造の履歴: {track_id: deque([skeleton_features])} - 骨格構造の連続性評価用
        self.tracked_skeleton_history = {}
        # デバッグログ
        self.debug_log = []

    def _calculate_bbox_center(self, keypoints):
        """
        キーポイントからバウンディングボックスの中心を計算
        
        Args:
            keypoints: キーポイントのリスト [(x, y, conf), ...]
        
        Returns:
            (cx, cy) または None
        """
        if not keypoints:
            return None
        
        valid_points = [(kp[0], kp[1]) for kp in keypoints if kp[0] is not None and kp[1] is not None]
        if not valid_points:
            return None
        
        xs = [p[0] for p in valid_points]
        ys = [p[1] for p in valid_points]
        
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        
        return (cx, cy)

    def _calculate_bbox_from_keypoints(self, keypoints):
        """
        キーポイントからバウンディングボックスを計算
        
        Args:
            keypoints: キーポイントのリスト [(x, y, conf), ...]
        
        Returns:
            [x1, y1, x2, y2] または None
        """
        if not keypoints:
            return None
        
        valid_points = [(kp[0], kp[1]) for kp in keypoints if kp[0] is not None and kp[1] is not None]
        if not valid_points:
            return None
        
        xs = [p[0] for p in valid_points]
        ys = [p[1] for p in valid_points]
        
        x1 = min(xs)
        y1 = min(ys)
        x2 = max(xs)
        y2 = max(ys)
        
        # マージンを追加（10%）
        width = x2 - x1
        height = y2 - y1
        margin_x = width * 0.1
        margin_y = height * 0.1
        
        return [x1 - margin_x, y1 - margin_y, x2 + margin_x, y2 + margin_y]

    def _extract_skeleton_features(self, keypoints):
        """
        キーポイントから骨格構造の特徴量を抽出（相対的な位置関係）
        
        Args:
            keypoints: キーポイントリスト [(x, y, conf), ...]
        
        Returns:
            dict: 骨格特徴量 {'bone_lengths': {...}, 'bone_angles': {...}, 'relative_positions': {...}}
        """
        if not keypoints or len(keypoints) < 2:
            return None
        
        features = {
            'bone_lengths': {},  # 骨の長さ
            'bone_angles': {},   # 骨の角度
            'relative_positions': {}  # 相対位置
        }
        
        # 基準点としてNeck（インデックス1）を使用
        neck_idx = 1
        if (neck_idx >= len(keypoints) or 
            keypoints[neck_idx][0] is None or 
            keypoints[neck_idx][1] is None or
            keypoints[neck_idx][2] < 0.3):
            # Neckがない場合は、有効なキーポイントの中心を使用
            valid_points = [(i, kp[0], kp[1], kp[2]) for i, kp in enumerate(keypoints) 
                           if kp[0] is not None and kp[1] is not None and kp[2] > 0.3]
            if not valid_points:
                return None
            # 重心を計算
            cx = np.mean([x for _, x, _, _ in valid_points])
            cy = np.mean([y for _, _, y, _ in valid_points])
            ref_point = (cx, cy)
        else:
            ref_point = (keypoints[neck_idx][0], keypoints[neck_idx][1])
        
        # 骨の長さと角度を計算
        for idx1, idx2 in COCO_PAIRS:
            if idx1 >= len(keypoints) or idx2 >= len(keypoints):
                continue
            
            kp1 = keypoints[idx1]
            kp2 = keypoints[idx2]
            
            if (kp1[0] is not None and kp1[1] is not None and kp1[2] > 0.3 and
                kp2[0] is not None and kp2[1] is not None and kp2[2] > 0.3):
                # 骨の長さ
                bone_length = np.sqrt((kp1[0] - kp2[0])**2 + (kp1[1] - kp2[1])**2)
                features['bone_lengths'][(idx1, idx2)] = bone_length
                
                # 骨の角度（水平方向からの角度）
                dx = kp2[0] - kp1[0]
                dy = kp2[1] - kp1[1]
                angle = np.arctan2(dy, dx)
                features['bone_angles'][(idx1, idx2)] = angle
                
                # 基準点からの相対位置
                rel_pos1 = (kp1[0] - ref_point[0], kp1[1] - ref_point[1])
                rel_pos2 = (kp2[0] - ref_point[0], kp2[1] - ref_point[1])
                features['relative_positions'][idx1] = rel_pos1
                features['relative_positions'][idx2] = rel_pos2
        
        return features

    def _calculate_skeleton_similarity(self, features1, features2):
        """
        2つの骨格特徴量間の類似度を計算
        
        Args:
            features1: 骨格特徴量1
            features2: 骨格特徴量2
        
        Returns:
            類似度スコア (0.0 ~ 1.0、1.0が最も類似)
        """
        if not features1 or not features2:
            return 0.0
        
        # 骨の長さの類似度
        bone_length_scores = []
        common_bones = set(features1['bone_lengths'].keys()) & set(features2['bone_lengths'].keys())
        if common_bones:
            for bone in common_bones:
                len1 = features1['bone_lengths'][bone]
                len2 = features2['bone_lengths'][bone]
                if len1 > 0 and len2 > 0:
                    # 長さの比で類似度を計算
                    ratio = min(len1, len2) / max(len1, len2)
                    bone_length_scores.append(ratio)
        
        # 骨の角度の類似度
        bone_angle_scores = []
        common_angles = set(features1['bone_angles'].keys()) & set(features2['bone_angles'].keys())
        if common_angles:
            for bone in common_angles:
                angle1 = features1['bone_angles'][bone]
                angle2 = features2['bone_angles'][bone]
                # 角度差を計算（-π ~ πに正規化）
                angle_diff = abs(angle1 - angle2)
                angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
                # 角度差が小さいほど類似度が高い
                angle_sim = max(0.0, 1.0 - angle_diff / np.pi)
                bone_angle_scores.append(angle_sim)
        
        # 相対位置の類似度
        relative_pos_scores = []
        common_pos = set(features1['relative_positions'].keys()) & set(features2['relative_positions'].keys())
        if common_pos:
            for idx in common_pos:
                pos1 = features1['relative_positions'][idx]
                pos2 = features2['relative_positions'][idx]
                dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                # 距離が小さいほど類似度が高い（最大50ピクセルで正規化）
                pos_sim = max(0.0, 1.0 - dist / 50.0)
                relative_pos_scores.append(pos_sim)
        
        # 統合スコア: 骨の長さ 30%, 角度 40%, 相対位置 30%
        total_score = 0.0
        total_weight = 0.0
        
        if bone_length_scores:
            avg_length_score = np.mean(bone_length_scores)
            total_score += avg_length_score * 0.3
            total_weight += 0.3
        
        if bone_angle_scores:
            avg_angle_score = np.mean(bone_angle_scores)
            total_score += avg_angle_score * 0.4
            total_weight += 0.4
        
        if relative_pos_scores:
            avg_pos_score = np.mean(relative_pos_scores)
            total_score += avg_pos_score * 0.3
            total_weight += 0.3
        
        if total_weight == 0:
            return 0.0
        
        return total_score / total_weight

    def _calculate_keypoint_similarity(self, kp1, kp2):
        """
        2つのキーポイントセット間の類似度を計算（ポーズの類似度）
        骨格構造の類似度も考慮
        
        Args:
            kp1: キーポイントリスト [(x, y, conf), ...]
            kp2: キーポイントリスト [(x, y, conf), ...]
        
        Returns:
            類似度スコア (0.0 ~ 1.0、1.0が最も類似)
        """
        if not kp1 or not kp2 or len(kp1) != len(kp2):
            return 0.0
        
        # 1. キーポイント位置の直接的な類似度
        valid_pairs = []
        for (x1, y1, c1), (x2, y2, c2) in zip(kp1, kp2):
            if (x1 is not None and y1 is not None and c1 > 0.3 and
                x2 is not None and y2 is not None and c2 > 0.3):
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                conf_weight = (c1 + c2) / 2.0
                valid_pairs.append((dist, conf_weight))
        
        position_similarity = 0.0
        if valid_pairs:
            total_weight = sum(w for _, w in valid_pairs)
            if total_weight > 0:
                weighted_avg_dist = sum(d * w for d, w in valid_pairs) / total_weight
                position_similarity = max(0.0, 1.0 - weighted_avg_dist / 100.0)
        
        # 2. 骨格構造の類似度
        skeleton_features1 = self._extract_skeleton_features(kp1)
        skeleton_features2 = self._extract_skeleton_features(kp2)
        skeleton_similarity = self._calculate_skeleton_similarity(skeleton_features1, skeleton_features2)
        
        # 統合スコア: 位置 40%, 骨格構造 60%
        combined_similarity = position_similarity * 0.4 + skeleton_similarity * 0.6
        
        return combined_similarity

    def _calculate_temporal_consistency(self, track_id, current_kp):
        """
        時系列的な連続性を評価（過去数フレームの履歴と比較）
        
        Args:
            track_id: トラッキングID
            current_kp: 現在のキーポイント
        
        Returns:
            連続性スコア (0.0 ~ 1.0、1.0が最も連続的)
        """
        if track_id not in self.tracked_keypoints_history:
            return 1.0  # 履歴がない場合は完全に一致とみなす
        
        history = self.tracked_keypoints_history[track_id]
        if not history:
            return 1.0
        
        # 過去3フレームの平均類似度を計算
        similarities = []
        for past_kp in list(history)[-3:]:  # 直近3フレーム
            sim = self._calculate_keypoint_similarity(past_kp, current_kp)
            similarities.append(sim)
        
        if not similarities:
            return 1.0
        
        return np.mean(similarities)

    def _calculate_3d_center(self, keypoints_3d):
        """
        3Dキーポイントから中心点の3D位置を計算
        
        Args:
            keypoints_3d: 3Dキーポイント辞書 {keypoint_name: (x, y, z), ...}
        
        Returns:
            (x, y, z) または None
        """
        if not keypoints_3d:
            return None
        
        valid_points = [(kp[0], kp[1], kp[2]) for kp in keypoints_3d.values() 
                       if kp and kp[0] is not None and kp[1] is not None and kp[2] is not None]
        if not valid_points:
            return None
        
        xs = [p[0] for p in valid_points]
        ys = [p[1] for p in valid_points]
        zs = [p[2] for p in valid_points]
        
        cx = np.mean(xs)
        cy = np.mean(ys)
        cz = np.mean(zs)
        
        return (cx, cy, cz)
    
    def _is_valid_3d_position(self, pos_3d):
        """
        3D位置が妥当な範囲内にあるかチェック
        
        Args:
            pos_3d: 3D位置 (x, y, z) または None
        
        Returns:
            bool: 妥当な範囲内であればTrue
        """
        if pos_3d is None:
            return False
        
        x, y, z = pos_3d[0], pos_3d[1], pos_3d[2]
        
        # Noneチェック
        if x is None or y is None or z is None:
            return False
        
        # 範囲チェック
        x_min, x_max = self.valid_3d_bounds[0]
        y_min, y_max = self.valid_3d_bounds[1]
        z_min, z_max = self.valid_3d_bounds[2]
        
        if not (x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max):
            return False
        
        return True
    
    def _evaluate_track_quality(self, track_id):
        """
        トラックの品質を評価（3D位置の安定性をチェック）
        
        Args:
            track_id: トラックID
        
        Returns:
            float: 品質スコア（0.0 ~ 1.0、1.0が最高品質）
        """
        if track_id not in self.tracked_3d_positions:
            return 0.0
        
        current_3d = self.tracked_3d_positions[track_id]
        if not self._is_valid_3d_position(current_3d):
            return 0.0
        
        # 過去の3D位置履歴をチェック（利用可能な場合）
        # ここでは簡易的に、現在の位置が妥当な範囲内にあれば品質が高いと判断
        return 1.0

    def update(self, detected_persons, frame_num, keypoints_3d_list=None):
        """
        検出された人物リストを更新し、トラッキングIDを割り当て
        
        Args:
            detected_persons: 検出された人物のキーポイントリスト [keypoints1, keypoints2, ...]
            frame_num: 現在のフレーム番号
            keypoints_3d_list: 3Dキーポイントのリスト（オプション）[{keypoint_name: (x, y, z), ...}, ...]
        
        Returns:
            list: [(person_id, keypoints), ...] のリスト（トラッキングID付き）
        """
        if not detected_persons:
            # 空の検出でもトラッカーを更新（失われたトラックを処理）
            tracked_objects = self.tracker.update()
            result = []
            for obj in tracked_objects:
                track_id = obj.id
                if track_id in self.tracked_keypoints:
                    result.append((track_id, self.tracked_keypoints[track_id]))
                # 失われたトラックのキーポイントを削除
                elif track_id in self.tracked_keypoints:
                    del self.tracked_keypoints[track_id]
            return result
        
        # 検出をNorFairのDetection形式に変換（信頼度フィルタリング）
        detections = []
        for i, kp in enumerate(detected_persons):
            # キーポイントの平均信頼度を計算
            valid_conf = [pt[2] for pt in kp if pt[0] is not None and pt[2] is not None]
            if not valid_conf:
                continue
            
            score = np.mean(valid_conf)
            
            # 信頼度が低い検出を除外
            if score < self.min_confidence:
                continue
            
            # 有効なキーポイントの数をチェック（少なすぎる場合は除外）
            valid_keypoints = sum(1 for pt in kp if pt[0] is not None and pt[1] is not None and pt[2] is not None)
            if valid_keypoints < self.min_valid_keypoints:  # 最低min_valid_keypoints個のキーポイントが必要
                continue
            
            center = self._calculate_bbox_center(kp)
            if center is not None:
                # 3D位置を取得（利用可能な場合）
                center_3d = None
                if keypoints_3d_list and i < len(keypoints_3d_list):
                    center_3d = self._calculate_3d_center(keypoints_3d_list[i])
                    # 3D位置の妥当性チェック（異常な位置を除外）
                    if not self._is_valid_3d_position(center_3d):
                        continue  # 異常な3D位置を持つ検出を除外
                
                # Detectionオブジェクトを作成（中心点を2D配列として）
                # 3D位置がある場合は、それを考慮した距離計算用に保存
                detection = Detection(
                    points=np.array([[center[0], center[1]]]),
                    scores=np.array([score])
                )
                detections.append((detection, kp, center_3d))
        
        # トラッカーを更新
        if detections:
            detection_objects = [d[0] for d in detections]
            tracked_objects = self.tracker.update(detections=detection_objects)
        else:
            tracked_objects = self.tracker.update()
        
        # トラッキング結果とキーポイントをマッチング（3D位置とキーポイント類似度を考慮）
        result = []
        
        if tracked_objects and detections:
            # トラッキングされたオブジェクトと検出をマッチング
            # 3D位置とキーポイント類似度を考慮したより堅牢なマッチング
            
            # 各トラッキングオブジェクトに対して最適な検出を見つける
            for tracked_obj in tracked_objects:
                track_id = tracked_obj.id
                tracked_center = tracked_obj.estimate[0]  # [x, y]
                tracked_3d = self.tracked_3d_positions.get(track_id)
                tracked_kp_prev = self.tracked_keypoints.get(track_id)
                
                # 最適なマッチを見つける（2D距離、3D距離、キーポイント類似度を統合）
                best_match_idx = None
                best_score = -float('inf')
                
                for i, (detection, kp, center_3d) in enumerate(detections):
                    det_center = detection.points[0]  # [x, y]
                    
                    # 1. 2D距離スコア
                    distance_2d = np.sqrt(
                        (tracked_center[0] - det_center[0])**2 + 
                        (tracked_center[1] - det_center[1])**2
                    )
                    score_2d = max(0.0, 1.0 - distance_2d / 200.0)  # 200ピクセルで正規化
                    
                    # 2. 3D距離スコア（深度距離を考慮）
                    score_3d = 1.0
                    if tracked_3d and center_3d:
                        distance_3d = np.sqrt(
                            (tracked_3d[0] - center_3d[0])**2 + 
                            (tracked_3d[1] - center_3d[1])**2 + 
                            (tracked_3d[2] - center_3d[2])**2
                        )
                        # 深度距離の差が大きい場合は大幅にペナルティ（0.5m以上で大幅減点）
                        if distance_3d > 0.5:
                            score_3d = max(0.0, 1.0 - (distance_3d - 0.5) / 1.0)  # 0.5m以上で減点
                        else:
                            score_3d = 1.0
                    
                    # 3. キーポイント類似度スコア（ポーズの類似度、骨格構造を考慮）
                    score_pose = 1.0
                    if tracked_kp_prev:
                        score_pose = self._calculate_keypoint_similarity(tracked_kp_prev, kp)
                    
                    # 4. 時系列的な連続性スコア（過去数フレームとの一貫性）
                    score_temporal = 1.0
                    if track_id in self.tracked_keypoints_history:
                        score_temporal = self._calculate_temporal_consistency(track_id, kp)
                    
                    # 統合スコア: 2D距離 20%, 3D距離 30%, ポーズ類似度 35%, 時系列連続性 15%
                    combined_score = (score_2d * 0.20 + score_3d * 0.30 + 
                                     score_pose * 0.35 + score_temporal * 0.15)
                    
                    # 3D位置の妥当性チェック
                    if center_3d and not self._is_valid_3d_position(center_3d):
                        continue  # 異常な3D位置を持つ検出を除外
                    
                    if tracked_3d and not self._is_valid_3d_position(tracked_3d):
                        continue  # 異常な3D位置を持つトラックを除外
                    
                    # 3D距離が大きすぎる場合は除外（max_3d_distance以上は別の人と判断）
                    if tracked_3d and center_3d:
                        distance_3d = np.sqrt(
                            (tracked_3d[0] - center_3d[0])**2 + 
                            (tracked_3d[1] - center_3d[1])**2 + 
                            (tracked_3d[2] - center_3d[2])**2
                        )
                        if distance_3d > self.max_3d_distance:  # max_3d_distance以上離れている場合は除外
                            continue
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_match_idx = i
                
                if best_match_idx is not None and best_score > 0.5:  # 最小スコア閾値をさらに上げる（より厳格に）
                    # キーポイントと3D位置を保存
                    _, matched_kp, matched_3d = detections[best_match_idx]
                    
                    # 3D位置の妥当性を再チェック
                    if matched_3d and not self._is_valid_3d_position(matched_3d):
                        # 異常な3D位置の場合は、トラックの品質を評価して削除を検討
                        track_quality = self._evaluate_track_quality(track_id)
                        if track_quality < 0.5:  # 品質が低い場合はスキップ
                            continue
                    
                    self.tracked_keypoints[track_id] = matched_kp
                    if matched_3d and self._is_valid_3d_position(matched_3d):
                        self.tracked_3d_positions[track_id] = matched_3d
                    elif track_id in self.tracked_3d_positions:
                        # 既存の3D位置が妥当な場合は保持、そうでない場合は削除
                        if not self._is_valid_3d_position(self.tracked_3d_positions[track_id]):
                            del self.tracked_3d_positions[track_id]
                    
                    # キーポイント履歴を更新（最大10フレーム、より長い履歴を保持）
                    if track_id not in self.tracked_keypoints_history:
                        self.tracked_keypoints_history[track_id] = deque(maxlen=10)
                    self.tracked_keypoints_history[track_id].append(matched_kp)
                    
                    # 骨格構造の履歴を更新
                    skeleton_features = self._extract_skeleton_features(matched_kp)
                    if track_id not in self.tracked_skeleton_history:
                        self.tracked_skeleton_history[track_id] = deque(maxlen=10)
                    if skeleton_features:
                        self.tracked_skeleton_history[track_id].append(skeleton_features)
                    
                    result.append((track_id, matched_kp))
                elif track_id in self.tracked_keypoints:
                    # マッチしなかったが、以前のキーポイントがある場合はそれを使用
                    result.append((track_id, self.tracked_keypoints[track_id]))
        elif tracked_objects:
            # 検出がないが、トラッキング中のオブジェクトがある場合
            for tracked_obj in tracked_objects:
                track_id = tracked_obj.id
                if track_id in self.tracked_keypoints:
                    result.append((track_id, self.tracked_keypoints[track_id]))
        
        # 失われたトラックのキーポイントと3D位置をクリーンアップ
        active_track_ids = {obj.id for obj in tracked_objects}
        
        # 品質が低いトラックも削除
        for track_id in list(self.tracked_keypoints.keys()):
            if track_id not in active_track_ids:
                # 失われたトラックを削除
                del self.tracked_keypoints[track_id]
                if track_id in self.tracked_3d_positions:
                    del self.tracked_3d_positions[track_id]
                if track_id in self.tracked_keypoints_history:
                    del self.tracked_keypoints_history[track_id]
                if track_id in self.tracked_skeleton_history:
                    del self.tracked_skeleton_history[track_id]
            else:
                # アクティブなトラックでも、品質が低い場合は削除を検討
                track_quality = self._evaluate_track_quality(track_id)
                if track_quality < 0.3:  # 品質が非常に低い場合は削除
                    if track_id in self.tracked_keypoints:
                        del self.tracked_keypoints[track_id]
                    if track_id in self.tracked_3d_positions:
                        del self.tracked_3d_positions[track_id]
                    if track_id in self.tracked_keypoints_history:
                        del self.tracked_keypoints_history[track_id]
                    if track_id in self.tracked_skeleton_history:
                        del self.tracked_skeleton_history[track_id]
        
        # ソート: track_id順
        result.sort(key=lambda x: x[0])
        
        # デバッグ情報を記録（混同の原因分析用）
        if frame_num % 30 == 0:  # 30フレームごとに記録
            self._log_debug_info(frame_num, detections, tracked_objects, result)
        
        return result
    
    def _log_debug_info(self, frame_num, detections, tracked_objects, result):
        """デバッグ情報を記録（混同の原因分析用）"""
        if not hasattr(self, 'debug_log'):
            self.debug_log = []
        
        debug_entry = {
            'frame': frame_num,
            'num_detections': len(detections),
            'num_tracks': len(tracked_objects),
            'num_results': len(result),
            'track_ids': [obj.id for obj in tracked_objects],
            'result_ids': [r[0] for r in result],
            'detection_centers': [],
            'track_centers': [],
            'detection_3d': [],
            'track_3d': [],
        }
        
        # 検出の中心点と3D位置を記録
        for det, kp, center_3d in detections:
            center = self._calculate_bbox_center(kp)
            # numpy配列をリストに変換
            if center is not None:
                center = [float(center[0]), float(center[1])] if isinstance(center, (tuple, list)) else [float(center[0]), float(center[1])]
            debug_entry['detection_centers'].append(center)
            # 3D位置もリストに変換
            if center_3d is not None:
                center_3d = [float(center_3d[0]), float(center_3d[1]), float(center_3d[2])]
            debug_entry['detection_3d'].append(center_3d)
        
        # トラッキングの中心点と3D位置を記録
        for obj in tracked_objects:
            track_id = obj.id
            center = obj.estimate[0]
            # numpy配列をリストに変換
            if isinstance(center, np.ndarray):
                center = center.tolist()
            elif isinstance(center, (tuple, list)):
                center = [float(c) for c in center]
            debug_entry['track_centers'].append(center)
            # 3D位置もリストに変換
            track_3d = self.tracked_3d_positions.get(track_id)
            if track_3d is not None:
                if isinstance(track_3d, (tuple, list)):
                    track_3d = [float(t) for t in track_3d]
                elif isinstance(track_3d, np.ndarray):
                    track_3d = track_3d.tolist()
            debug_entry['track_3d'].append(track_3d)
        
        self.debug_log.append(debug_entry)
        
        # 100エントリを超えたら古いものを削除
        if len(self.debug_log) > 100:
            self.debug_log = self.debug_log[-100:]

    def get_tracked_person_ids(self):
        """現在トラッキング中の人物IDのリストを返す"""
        return list(self.tracked_keypoints.keys())

    def get_person_count(self):
        """現在トラッキング中の人数を返す"""
        return len(self.tracked_keypoints)
    
    def get_debug_info(self):
        """デバッグ情報を取得（トラッキングの状態を分析）"""
        return {
            'active_tracks': len(self.tracked_keypoints),
            'track_ids': list(self.tracked_keypoints.keys()),
            'tracked_3d_positions': {k: v for k, v in self.tracked_3d_positions.items()},
            'keypoint_history_lengths': {k: len(v) for k, v in self.tracked_keypoints_history.items()},
            'debug_log': self.debug_log[-10:] if self.debug_log else [],  # 直近10エントリ
        }
    
    def save_debug_log(self, filepath):
        """デバッグログをJSONファイルに保存"""
        import json
        
        # numpy配列やその他の非シリアライズ可能なオブジェクトを変換
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif obj is None:
                return None
            else:
                try:
                    # 通常のPython型（int, float, str, bool）はそのまま
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    # シリアライズできない場合は文字列に変換
                    return str(obj)
        
        serializable_log = convert_to_serializable(self.debug_log)
        with open(filepath, 'w') as f:
            json.dump(serializable_log, f, indent=2)

