"""
Person Tracker

複数人のトラッキング機能を提供（IoUベースのトラッキング）
各人に一貫したIDを割り当て、フレーム間で追跡
"""

import numpy as np
from collections import deque


class PersonTracker:
    """複数人のトラッキングクラス（IoU + キーポイントベース）"""

    def __init__(self, max_age=60, min_iou=0.15, keypoint_match_threshold=0.5):
        """
        Args:
            max_age: トラッキングが失われるまでの最大フレーム数（デフォルト: 60）
            min_iou: トラッキングを維持するための最小IoU（デフォルト: 0.15）
            keypoint_match_threshold: キーポイントマッチングの閾値（デフォルト: 0.5）
        """
        self.max_age = max_age
        self.min_iou = min_iou
        self.keypoint_match_threshold = keypoint_match_threshold
        
        # トラッキング中の人物データ: {
        #   'bbox': [x1, y1, x2, y2], 
        #   'age': 0, 
        #   'keypoints': [...], 
        #   'last_seen': frame_num,
        #   'bbox_history': deque([...]),  # 過去のbbox履歴（予測用）
        #   'center_history': deque([...])  # 過去の中心点履歴（予測用）
        # }
        self.tracked_persons = {}
        self.next_id = 0  # 次のID

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

    def _calculate_iou(self, bbox1, bbox2):
        """
        2つのバウンディングボックスのIoUを計算
        
        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]
        
        Returns:
            IoU値 (0.0 ~ 1.0)
        """
        if bbox1 is None or bbox2 is None:
            return 0.0
        
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 交差領域
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)
        
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        
        # 各ボックスの面積
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # 和集合
        union_area = area1 + area2 - inter_area
        
        if union_area <= 0:
            return 0.0
        
        return inter_area / union_area

    def _calculate_keypoint_distance(self, kp1, kp2):
        """
        2つのキーポイントセット間の距離を計算（有効なキーポイントのみ）
        
        Args:
            kp1: キーポイントリスト [(x, y, conf), ...]
            kp2: キーポイントリスト [(x, y, conf), ...]
        
        Returns:
            平均距離（ピクセル）または None（有効なキーポイントがない場合）
        """
        if not kp1 or not kp2:
            return None
        
        valid_pairs = []
        for (x1, y1, c1), (x2, y2, c2) in zip(kp1, kp2):
            if (x1 is not None and y1 is not None and c1 > 0.3 and
                x2 is not None and y2 is not None and c2 > 0.3):
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                valid_pairs.append(dist)
        
        if not valid_pairs:
            return None
        
        return np.mean(valid_pairs)

    def _predict_bbox(self, person_data):
        """
        過去のbbox履歴から次のbboxを予測（単純な線形予測）
        
        Args:
            person_data: トラッキング中の人物データ
        
        Returns:
            予測されたbbox [x1, y1, x2, y2] または None
        """
        if 'bbox_history' not in person_data or len(person_data['bbox_history']) < 2:
            return person_data.get('bbox')
        
        history = list(person_data['bbox_history'])
        if len(history) < 2:
            return history[-1] if history else None
        
        # 最後の2つのbboxから速度を計算
        bbox1 = history[-2]
        bbox2 = history[-1]
        
        # 中心点の移動量
        cx1 = (bbox1[0] + bbox1[2]) / 2
        cy1 = (bbox1[1] + bbox1[3]) / 2
        cx2 = (bbox2[0] + bbox2[2]) / 2
        cy2 = (bbox2[1] + bbox2[3]) / 2
        
        vx = cx2 - cx1
        vy = cy2 - cy1
        
        # サイズの変化
        w1 = bbox1[2] - bbox1[0]
        h1 = bbox1[3] - bbox1[1]
        w2 = bbox2[2] - bbox2[0]
        h2 = bbox2[3] - bbox2[1]
        
        dw = w2 - w1
        dh = h2 - h1
        
        # 予測
        pred_cx = cx2 + vx
        pred_cy = cy2 + vy
        pred_w = max(w2 + dw, 10)  # 最小幅
        pred_h = max(h2 + dh, 10)  # 最小高さ
        
        return [
            pred_cx - pred_w / 2,
            pred_cy - pred_h / 2,
            pred_cx + pred_w / 2,
            pred_cy + pred_h / 2
        ]

    def _calculate_combined_score(self, det_bbox, det_kp, tracked_bbox, tracked_kp, predicted_bbox=None):
        """
        bboxとキーポイントの両方を考慮したマッチングスコアを計算
        
        Args:
            det_bbox: 検出されたbbox
            det_kp: 検出されたキーポイント
            tracked_bbox: トラッキング中のbbox
            tracked_kp: トラッキング中のキーポイント
            predicted_bbox: 予測されたbbox（オプション）
        
        Returns:
            スコア (0.0 ~ 1.0)
        """
        if det_bbox is None or tracked_bbox is None:
            return 0.0
        
        # IoUスコア
        iou = self._calculate_iou(det_bbox, tracked_bbox)
        
        # 予測bboxとのIoUも考慮（予測がある場合）
        pred_iou = 0.0
        if predicted_bbox is not None:
            pred_iou = self._calculate_iou(det_bbox, predicted_bbox)
        
        # キーポイント距離スコア
        kp_dist = self._calculate_keypoint_distance(det_kp, tracked_kp)
        kp_score = 0.0
        if kp_dist is not None:
            # 距離が小さいほどスコアが高い（最大100ピクセルで正規化）
            kp_score = max(0.0, 1.0 - kp_dist / 100.0)
        
        # 統合スコア: IoU 60%, 予測IoU 20%, キーポイント 20%
        combined_score = (iou * 0.6 + pred_iou * 0.2 + kp_score * 0.2)
        
        return combined_score

    def update(self, detected_persons, frame_num):
        """
        検出された人物リストを更新し、トラッキングIDを割り当て
        
        Args:
            detected_persons: 検出された人物のキーポイントリスト [keypoints1, keypoints2, ...]
            frame_num: 現在のフレーム番号
        
        Returns:
            list: [(person_id, keypoints), ...] のリスト（トラッキングID付き）
        """
        # 検出された人物のバウンディングボックスを計算
        detected_bboxes = []
        for kp in detected_persons:
            bbox = self._calculate_bbox_from_keypoints(kp)
            detected_bboxes.append(bbox)
        
        # 既存のトラッキングを更新（ageを増やす、bbox履歴を更新）
        for person_id in list(self.tracked_persons.keys()):
            self.tracked_persons[person_id]['age'] += 1
            # bbox履歴を初期化（存在しない場合）
            if 'bbox_history' not in self.tracked_persons[person_id]:
                self.tracked_persons[person_id]['bbox_history'] = deque(maxlen=5)
            if 'center_history' not in self.tracked_persons[person_id]:
                self.tracked_persons[person_id]['center_history'] = deque(maxlen=5)
        
        # 既存のトラッキングと新しい検出をマッチング（ハンガリアンアルゴリズム風）
        matched_indices = set()
        matched_ids = set()
        
        # すべての可能なマッチングを計算
        all_matches = []
        for i, (det_bbox, det_kp) in enumerate(zip(detected_bboxes, detected_persons)):
            if det_bbox is None:
                continue
            
            for person_id, person_data in self.tracked_persons.items():
                if person_id in matched_ids:
                    continue
                
                # 予測bboxを計算
                predicted_bbox = self._predict_bbox(person_data)
                
                # 統合スコアを計算
                score = self._calculate_combined_score(
                    det_bbox, det_kp,
                    person_data['bbox'], person_data.get('keypoints', []),
                    predicted_bbox
                )
                
                # 最小閾値をチェック（IoUベース）
                iou = self._calculate_iou(det_bbox, person_data['bbox'])
                if predicted_bbox is not None:
                    iou = max(iou, self._calculate_iou(det_bbox, predicted_bbox))
                
                if score > 0.0 and iou >= self.min_iou:
                    all_matches.append((i, person_id, score, iou))
        
        # スコアの高い順にソート
        all_matches.sort(key=lambda x: x[2], reverse=True)
        
        # グリーディーにマッチング（各検出と各トラッキングは1対1）
        final_matches = {}
        for i, person_id, score, iou in all_matches:
            if i not in final_matches and person_id not in [m[0] for m in final_matches.values()]:
                final_matches[i] = (person_id, score, iou)
                matched_indices.add(i)
                matched_ids.add(person_id)
        
        # 既存のトラッキングを更新
        for i, (person_id, score, iou) in final_matches.items():
            bbox = detected_bboxes[i]
            keypoints = detected_persons[i]
            
            # bbox履歴を更新
            if 'bbox_history' not in self.tracked_persons[person_id]:
                self.tracked_persons[person_id]['bbox_history'] = deque(maxlen=5)
            self.tracked_persons[person_id]['bbox_history'].append(bbox)
            
            # 中心点履歴を更新
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            if 'center_history' not in self.tracked_persons[person_id]:
                self.tracked_persons[person_id]['center_history'] = deque(maxlen=5)
            self.tracked_persons[person_id]['center_history'].append(center)
            
            self.tracked_persons[person_id].update({
                'bbox': bbox,
                'age': 0,  # リセット
                'keypoints': keypoints,
                'last_seen': frame_num
            })
        
        # 新しい人物にIDを割り当て（マッチしなかった検出のみ）
        for i, kp in enumerate(detected_persons):
            if i not in matched_indices:
                bbox = detected_bboxes[i]
                if bbox is not None:
                    person_id = self.next_id
                    self.next_id += 1
                    
                    # bbox履歴と中心点履歴を初期化
                    bbox_history = deque(maxlen=5)
                    bbox_history.append(bbox)
                    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                    center_history = deque(maxlen=5)
                    center_history.append(center)
                    
                    self.tracked_persons[person_id] = {
                        'bbox': bbox,
                        'age': 0,
                        'keypoints': kp,
                        'last_seen': frame_num,
                        'bbox_history': bbox_history,
                        'center_history': center_history
                    }
                    final_matches[i] = (person_id, 0.0, 0.0)
        
        # 古いトラッキングを削除（max_ageを超えたもの）
        for person_id in list(self.tracked_persons.keys()):
            if self.tracked_persons[person_id]['age'] > self.max_age:
                del self.tracked_persons[person_id]
        
        # 結果を返す: [(person_id, keypoints), ...]
        result = []
        for i, kp in enumerate(detected_persons):
            if i in final_matches:
                person_id = final_matches[i][0]
                result.append((person_id, kp))
        
        # ソート: person_id順
        result.sort(key=lambda x: x[0])
        
        return result

    def get_tracked_person_ids(self):
        """現在トラッキング中の人物IDのリストを返す"""
        return list(self.tracked_persons.keys())

    def get_person_count(self):
        """現在トラッキング中の人数を返す"""
        return len(self.tracked_persons)

