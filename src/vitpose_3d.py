"""
ViTPose 3D Utilities

ViTPoseモデルの読み込み、2D keypoints検出、スケルトン描画機能を提供
YOLOv8PoseDetectorと同様のインターフェースを提供（最高精度のキーポイント推定）
"""

import cv2
import numpy as np
import os
import sys
import shutil
import urllib.request

# ViTPoseを使用（mmpose経由）
try:
    import warnings
    warnings.filterwarnings('ignore')
    import sys
    import types
    
    # mmcv._extモジュールをsys.modulesに追加（mmcv-lite対応）
    if 'mmcv._ext' not in sys.modules:
        import importlib.util
        
        # mmcv._extモジュールを作成
        mmcv_ext = types.ModuleType('mmcv._ext')
        
        # __spec__を設定（pkgutil.find_loader用）
        spec = importlib.util.spec_from_loader('mmcv._ext', None)
        mmcv_ext.__spec__ = spec
        mmcv_ext.__loader__ = None
        mmcv_ext.__file__ = '<dummy>'
        mmcv_ext.__package__ = 'mmcv'
        
        sys.modules['mmcv._ext'] = mmcv_ext
        
        # 必要な関数をダミーとして追加
        def dummy_function(*args, **kwargs):
            return None
        
        # active_rotated_filter関連
        mmcv_ext.active_rotated_filter_forward = dummy_function
        mmcv_ext.active_rotated_filter_backward = dummy_function
        
        # MultiScaleDeformableAttention
        class MultiScaleDeformableAttention:
            def __init__(self, *args, **kwargs):
                pass
        
        multi_scale_deform_attn = types.ModuleType('multi_scale_deform_attn')
        multi_scale_deform_attn.MultiScaleDeformableAttention = MultiScaleDeformableAttention
        mmcv_ext.multi_scale_deform_attn = multi_scale_deform_attn
        sys.modules['mmcv._ext.multi_scale_deform_attn'] = multi_scale_deform_attn
        
        # mmcv.opsをパッチ
        import mmcv
        if not hasattr(mmcv, 'ops'):
            mmcv.ops = types.ModuleType('ops')
        
        # ext_loaderをパッチ
        import mmcv.utils.ext_loader as ext_loader
        original_load_ext = ext_loader.load_ext
        def patched_load_ext(name, funs):
            try:
                return original_load_ext(name, funs)
            except (ModuleNotFoundError, AssertionError):
                # ダミーモジュールを返す
                dummy_mod = types.ModuleType(name)
                for fun in funs:
                    setattr(dummy_mod, fun, dummy_function)
                return dummy_mod
        ext_loader.load_ext = patched_load_ext
        
        # pkgutil.find_loaderをパッチ
        import pkgutil
        original_find_loader = pkgutil.find_loader
        def patched_find_loader(fullname):
            if fullname == 'mmcv._ext':
                return None  # ダミーモジュールとして扱う
            return original_find_loader(fullname)
        pkgutil.find_loader = patched_find_loader
    
    from mmpose.apis import init_model, inference_topdown
    import torch
    try:
        from torch.serialization import add_safe_globals
        add_safe_globals([np.core.multiarray._reconstruct])
    except Exception:
        pass
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
    
    VITPOSE_AVAILABLE = True
except ImportError as e:
    VITPOSE_AVAILABLE = False
    print("Warning: mmpose not available. Install with: pip install mmpose mmcv")
    print("For ViTPose, you may need: pip install openmim && mim install mmengine mmcv mmdet mmpose")
    print(f"Error details: {e}")

# Ultralytics YOLO（人物検出用）
try:
    from ultralytics import YOLO as UltralyticsYOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

# COCO形式のkeypoints（18 points - OpenPose互換）
# yolov8_pose_3d.pyと同じ定義を使用
from src.yolov8_pose_3d import COCO_KEYPOINTS, COCO_PAIRS

# ViTPoseの17 keypoints形式（COCO標準 - YOLOv8と同じ）
VITPOSE_KEYPOINTS = [
    "Nose",
    "LEye",
    "REye",
    "LEar",
    "REar",
    "LShoulder",
    "RShoulder",
    "LElbow",
    "RElbow",
    "LWrist",
    "RWrist",
    "LHip",
    "RHip",
    "LKnee",
    "RKnee",
    "LAnkle",
    "RAnkle",
]


VITPOSE_MODEL_ZOO = {
    "vitpose-tiny": {
        "config": "https://raw.githubusercontent.com/open-mmlab/mmpose/refs/heads/main/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-tiny_8xb64-210e_coco-256x192.py",
        "config_path": os.path.join(
            "configs", "body_2d_keypoint", "topdown_heatmap", "coco",
            "td-hm_ViTPose-tiny_8xb64-210e_coco-256x192.py"),
        "checkpoint": "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-tiny_8xb64-210e_coco-256x192-d0b1eb3b_20230314.pth",
    },
    "vitpose-small": {
        "config": "https://raw.githubusercontent.com/open-mmlab/mmpose/refs/heads/main/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192.py",
        "config_path": os.path.join(
            "configs", "body_2d_keypoint", "topdown_heatmap", "coco",
            "td-hm_ViTPose-small_8xb64-210e_coco-256x192.py"),
        "checkpoint": "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192-5d9c494d_20230314.pth",
    },
    "vitpose-base": {
        "config": "https://raw.githubusercontent.com/open-mmlab/mmpose/refs/heads/main/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192.py",
        "config_path": os.path.join(
            "configs", "body_2d_keypoint", "topdown_heatmap", "coco",
            "td-hm_ViTPose-base_8xb64-210e_coco-256x192.py"),
        "checkpoint": "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192-a9a16113_20230314.pth",
    },
    "vitpose-large": {
        "config": "https://raw.githubusercontent.com/open-mmlab/mmpose/refs/heads/main/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-large_8xb64-210e_coco-256x192.py",
        "config_path": os.path.join(
            "configs", "body_2d_keypoint", "topdown_heatmap", "coco",
            "td-hm_ViTPose-large_8xb64-210e_coco-256x192.py"),
        "checkpoint": "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-large_8xb64-210e_coco-256x192-436c48d8_20230314.pth",
    },
    "vitpose-huge": {
        "config": "https://raw.githubusercontent.com/open-mmlab/mmpose/refs/heads/main/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py",
        "config_path": os.path.join(
            "configs", "body_2d_keypoint", "topdown_heatmap", "coco",
            "td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py"),
        "checkpoint": "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth",
    },
}

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
LOCAL_CONFIG_DIR = os.path.join(PROJECT_ROOT, "models", "vitpose_configs")
TORCH_HUB_DIR = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")

DEFAULT_YOLO_MODEL = "yolov8x.pt"
PERSON_CLASS_ID = 0  # COCO person class


class ViTPoseDetector:
    """ViTPose検出器（mmpose使用）- 最高精度のキーポイント推定"""

    def __init__(self, model_name="vitpose-huge", model_dir="models"):
        """
        Args:
            model_name: 使用するViTPoseモデル名
                - vitpose-tiny: 最小モデル（高速、やや精度低下）
                - vitpose-small: 小型モデル（高速、バランス型）
                - vitpose-base: ベースモデル（中速、高精度）
                - vitpose-large: 大型モデル（やや低速、高精度）
                - vitpose-huge: 最大モデル（最高精度） - **推奨（デフォルト）**
            model_dir: モデルファイルを保存するディレクトリ
        """
        self.model_name = model_name
        self.model_dir = os.path.join(model_dir, "vitpose")
        self.pose_model = None
        self.person_detector = None
        self.detector_model_name = DEFAULT_YOLO_MODEL
        self.device = self._get_device()
        self.target_bbox = None  # 追跡中の人物（映像中央かつ近い人物）を維持

    def _get_device(self):
        """使用可能なデバイスを取得"""
        try:
            import torch

            if torch.cuda.is_available():
                try:
                    device = "cuda:0"
                    test_tensor = torch.zeros(1).to(device)
                    del test_tensor
                    print(f"  CUDA available: {torch.cuda.get_device_name(0)}")
                    print(f"  CUDA version: {torch.version.cuda}")
                    return device
                except Exception as e:
                    print(f"  Warning: CUDA device access failed: {e}")
                    print(f"  Falling back to CPU mode")
                    return "cpu"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                try:
                    device = "mps"
                    test_tensor = torch.zeros(1).to(device)
                    del test_tensor
                    print(f"  Apple Silicon GPU (MPS) available")
                    return device
                except Exception as e:
                    print(f"  Warning: MPS device access failed: {e}")
                    print(f"  Falling back to CPU mode")
                    return "cpu"
            else:
                print(f"  Using CPU mode (no GPU available)")
                return "cpu"
        except Exception as e:
            print(f"  Warning: Device detection failed: {e}")
            print(f"  Falling back to CPU mode")
            return "cpu"

    def load_model(self):
        """ViTPoseモデルを読み込み"""
        if not VITPOSE_AVAILABLE:
            print("Error: mmpose is not available.")
            print("Install with: pip install openmim")
            print("Then: mim install mmengine mmcv mmdet mmpose")
            return False

        try:
            print(f"Loading ViTPose model: {self.model_name}")
            print(f"Using device: {self.device}")

            # モデルディレクトリが存在しない場合は作成
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir, exist_ok=True)
                print(f"Created model directory: {self.model_dir}")

            model_key = self.model_name.lower()
            if model_key not in VITPOSE_MODEL_ZOO:
                print(f"Warning: Unknown ViTPose model '{self.model_name}', falling back to vitpose-huge")
                model_key = "vitpose-huge"
                self.model_name = "vitpose-huge"

            model_info = VITPOSE_MODEL_ZOO[model_key]
            config_filename = os.path.basename(model_info["config"])
            checkpoint_filename = os.path.basename(model_info["checkpoint"])
            local_config_candidate = os.path.join(
                LOCAL_CONFIG_DIR, model_info.get("config_path", config_filename))
            if os.path.exists(local_config_candidate):
                config_path = local_config_candidate
            else:
                config_path = self._ensure_local_file(
                    model_info["config"],
                    os.path.join(self.model_dir, config_filename)
                )
            checkpoint_path = self._ensure_local_file(
                model_info["checkpoint"],
                os.path.join(self.model_dir, checkpoint_filename),
                fallback_paths=[
                    os.path.join(self.model_dir, checkpoint_filename),
                    os.path.join(TORCH_HUB_DIR, checkpoint_filename)
                ]
            )

            # デバイス設定
            device_str = self.device if isinstance(self.device, str) else "cpu"

            # ViTPoseモデル初期化
            self.pose_model = init_model(
                config=config_path,
                checkpoint=checkpoint_path,
                device=device_str
            )
            print(f"ViTPose model loaded successfully from {config_path}")

            # 人物検出器（YOLO）を初期化
            self.person_detector = self._init_person_detector()

            return True

        except Exception as e:
            print(f"Failed to load ViTPose model: {e}")
            print("\nTroubleshooting:")
            print("1. Install mmpose: pip install openmim && mim install mmengine mmcv mmdet mmpose")
            print("2. Make sure you have PyTorch installed: pip install torch torchvision")
            print("3. For GPU support, install CUDA-enabled PyTorch")
            import traceback
            traceback.print_exc()
            return False

    def _ensure_local_file(self, url, local_path, fallback_paths=None):
        """指定URLからファイルをダウンロード（存在しない場合のみ）"""
        if os.path.exists(local_path):
            return local_path

        if fallback_paths:
            for candidate in fallback_paths:
                if candidate and os.path.exists(candidate):
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    shutil.copy(candidate, local_path)
                    print(f"  -> copied cached file from {candidate}")
                    return local_path

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f"Downloading {os.path.basename(local_path)} ...")
        try:
            urllib.request.urlretrieve(url, local_path)
            print(f"  -> saved to {local_path}")
        except Exception as e:
            print(f"Warning: Failed to download {url}: {e}")
            if fallback_paths:
                for candidate in fallback_paths:
                    if candidate and os.path.exists(candidate):
                        shutil.copy(candidate, local_path)
                        print(f"  -> copied cached file from {candidate}")
                        return local_path
            raise
        return local_path

    def _init_person_detector(self):
        """YOLOベースの人物検出器を初期化（ViTPose用の高精度なバウンディングボックス取得）"""
        if not ULTRALYTICS_AVAILABLE:
            print("Warning: Ultralytics YOLO not available. Falling back to full-frame detection.")
            return None

        try:
            detector = UltralyticsYOLO(self.detector_model_name)
            yolo_device = "cuda" if isinstance(self.device, str) and self.device.startswith("cuda") else "cpu"
            detector.to(yolo_device)
            print(f"Person detector ({self.detector_model_name}) initialized on {yolo_device}")
            return detector
        except Exception as e:
            print(f"Warning: Failed to initialize YOLO detector ({self.detector_model_name}): {e}")
            print("  -> Falling back to full-frame detection.")
            return None

    def _detect_person_bboxes(self, image, max_det=5, conf_thr=0.4):
        """YOLOで人物バウンディングボックスを検出（前景かつ中央の人物を優先＆追跡）"""
        h, w = image.shape[:2]

        if self.person_detector is None:
            # 検出器がない場合はフルフレームを返す
            return [[0, 0, w, h]]

        try:
            results = self.person_detector.predict(
                source=image,
                conf=conf_thr,
                verbose=False,
                max_det=max_det,
                classes=[PERSON_CLASS_ID]
            )
            if not results:
                return [[0, 0, w, h]]

            boxes = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    if box.conf is None:
                        continue
                    score = float(box.conf.item())
                    if score < conf_thr:
                        continue
                    xyxy = box.xyxy.cpu().numpy()[0].tolist()
                    # clip to image bounds
                    x1 = float(np.clip(xyxy[0], 0, w - 1))
                    y1 = float(np.clip(xyxy[1], 0, h - 1))
                    x2 = float(np.clip(xyxy[2], 0, w - 1))
                    y2 = float(np.clip(xyxy[3], 0, h - 1))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    boxes.append([x1, y1, x2, y2, score])

            if not boxes:
                return [[0, 0, w, h]]

            prioritized = self._prioritize_bboxes(image.shape, boxes)
            return [box[:4] for box in prioritized[:max_det]]
        except Exception as e:
            print(f"Warning: YOLO detection failed ({e}). Falling back to full-frame detection.")
            return [[0, 0, w, h]]

    def _bbox_iou(self, box_a, box_b):
        """2つのバウンディングボックス（x1,y1,x2,y2,score）間のIoUを計算"""
        if box_a is None or box_b is None:
            return 0.0
        xa1, ya1, xa2, ya2 = box_a[:4]
        xb1, yb1, xb2, yb2 = box_b[:4]
        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)
        inter_w = max(inter_x2 - inter_x1, 0.0)
        inter_h = max(inter_y2 - inter_y1, 0.0)
        inter_area = inter_w * inter_h
        area_a = max((xa2 - xa1) * (ya2 - ya1), 0.0)
        area_b = max((xb2 - xb1) * (yb2 - yb1), 0.0)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    def _prioritize_bboxes(self, image_shape, boxes):
        """中央かつ大きい人物を優先し、過去フレームと追跡し続ける"""
        if not boxes:
            self.target_bbox = None
            return []

        h, w = image_shape[:2]
        image_center = (w / 2.0, h / 2.0)

        # 1) 直前フレームの追跡対象とIoUが高いものを優先
        if self.target_bbox is not None:
            best_iou = 0.0
            best_idx = None
            for idx, box in enumerate(boxes):
                iou = self._bbox_iou(box, self.target_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx is not None and best_iou >= 0.15:
                target_box = boxes.pop(best_idx)
            else:
                target_box = None
        else:
            target_box = None

        # 2) 追跡対象が見つからない場合、中央かつ大きい人物を新しく選定
        if target_box is None:
            def combined_score(box):
                x1, y1, x2, y2, conf = box
                width = max(x2 - x1, 1.0)
                height = max(y2 - y1, 1.0)
                area = width * height
                area_norm = area / (w * h)
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                dx = (cx - image_center[0]) / (w / 2.0)
                dy = (cy - image_center[1]) / (h / 2.0)
                center_penalty = min(dx * dx + dy * dy, 4.0) / 4.0  # 0 (中心) ～1 (端)
                center_score = 1.0 - center_penalty
                return area_norm * 0.6 + center_score * 0.3 + conf * 0.1

            best_idx = max(range(len(boxes)), key=lambda i: combined_score(boxes[i]))
            target_box = boxes.pop(best_idx)

        self.target_bbox = target_box

        # 残りの人物は面積順で並べる（追加情報として保持）
        boxes.sort(key=lambda b: max((b[2] - b[0]) * (b[3] - b[1]), 0.0), reverse=True)

        ordered = [target_box] + boxes
        return ordered

    def _run_pose_inference(self, image, max_det=5):
        """ViTPoseを用いてkeypointsを推定し、18キーポイント形式に変換"""
        if self.pose_model is None:
            print("Error: Pose model not loaded. Call load_model() first.")
            return []

        bboxes = self._detect_person_bboxes(image, max_det=max_det)
        pose_results = inference_topdown(
            self.pose_model,
            image,
            bboxes=bboxes,
            bbox_format="xyxy"
        )

        persons = []
        h, w = image.shape[:2]

        for sample in pose_results:
            if not hasattr(sample, "pred_instances"):
                continue
            pred = sample.pred_instances
            if pred is None or len(pred) == 0:
                continue

            keypoints = pred.keypoints
            keypoint_scores = getattr(pred, "keypoint_scores", None)
            bbox_scores = getattr(pred, "bbox_scores", None)

            if hasattr(keypoints, "cpu"):
                keypoints = keypoints.cpu().numpy()
            keypoints = np.array(keypoints)

            if keypoint_scores is not None and hasattr(keypoint_scores, "cpu"):
                keypoint_scores = keypoint_scores.cpu().numpy()

            # keypoints shape: (1, 17, 2)
            if keypoints.ndim == 3:
                keypoints = keypoints[0]

            if keypoint_scores is not None:
                if keypoint_scores.ndim == 2:
                    keypoint_scores = keypoint_scores[0]
                keypoints_17 = np.concatenate(
                    [keypoints, keypoint_scores[..., None]],
                    axis=-1
                )
                person_score = float(np.mean(keypoint_scores))
            else:
                conf = np.ones((keypoints.shape[0], 1), dtype=np.float32)
                keypoints_17 = np.concatenate([keypoints, conf], axis=-1)
                person_score = float(np.mean(conf))

            coco18 = self._convert_17_to_18(keypoints_17, w, h)
            if bbox_scores is not None and len(bbox_scores) > 0:
                person_score = float(bbox_scores[0])

            persons.append((coco18, person_score))

        # 信頼度でソート
        persons.sort(key=lambda p: p[1], reverse=True)
        return [p[0] for p in persons]

    def detect_keypoints(self, image):
        """
        画像からkeypointsを検出

        Args:
            image: 入力画像（BGR形式）

        Returns:
            list: 検出されたkeypointsのリスト [(x, y, confidence), ...] （18要素）
        """
        if self.pose_model is None:
            print("Error: Model not loaded. Call load_model() first.")
            return None

        try:
            persons = self._run_pose_inference(image, max_det=1)
            return persons[0] if persons else None
        except Exception as e:
            print(f"Error during ViTPose inference: {e}")
            import traceback
            traceback.print_exc()
            return None

    def detect_keypoints_multi(self, image, max_det=5, conf=0.25):
        """
        画像から複数人物のkeypointsを検出（18キーポイント形式）

        Args:
            image: BGR画像
            max_det: 最大検出人数
            conf: 信頼度閾値（ViTPoseでは内部で処理）

        Returns:
            list[list[(x, y, conf)]] または None
        """
        if self.pose_model is None:
            print("Error: Model not loaded. Call load_model() first.")
            return None

        try:
            persons = self._run_pose_inference(image, max_det=max_det)
            if not persons:
                return None

            # 信頼度閾値で絞り込み
            filtered = []
            for kp in persons:
                valid_conf = [pt[2] for pt in kp if pt[0] is not None]
                mean_conf = np.mean(valid_conf) if valid_conf else 0.0
                if mean_conf >= conf:
                    filtered.append(kp)

            if not filtered:
                return None
            return filtered[:max_det]

        except Exception as e:
            print(f"Error during ViTPose multi-person inference: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _convert_17_to_18(self, keypoints_17, w, h):
        """
        ViTPoseの17 keypointsを18 keypoints形式（COCO互換）に変換

        ViTPose (17 keypoints - COCO標準):
        0: Nose, 1: LEye, 2: REye, 3: LEar, 4: REar,
        5: LShoulder, 6: RShoulder,
        7: LElbow, 8: RElbow,
        9: LWrist, 10: RWrist,
        11: LHip, 12: RHip,
        13: LKnee, 14: RKnee,
        15: LAnkle, 16: RAnkle

        COCO (18 keypoints - OpenPose互換):
        0: Nose, 1: Neck, 2: RShoulder, 3: RElbow, 4: RWrist,
        5: LShoulder, 6: LElbow, 7: LWrist,
        8: RHip, 9: RKnee, 10: RAnkle,
        11: LHip, 12: LKnee, 13: LAnkle,
        14: REye, 15: LEye, 16: REar, 17: LEar
        """
        # 18要素のリストを作成（すべてNone）
        keypoints_18 = [(None, None, 0.0) for _ in range(18)]

        # ViTPose 17 -> COCO 18 マッピング（YOLOv8と同じ）
        mapping = [
            (0, 0),  # Nose -> Nose
            (1, 15),  # LEye -> LEye
            (2, 14),  # REye -> REye
            (3, 17),  # LEar -> LEar
            (4, 16),  # REar -> REar
            (5, 5),  # LShoulder -> LShoulder
            (6, 2),  # RShoulder -> RShoulder
            (7, 6),  # LElbow -> LElbow
            (8, 3),  # RElbow -> RElbow
            (9, 7),  # LWrist -> LWrist
            (10, 4),  # RWrist -> RWrist
            (11, 11),  # LHip -> LHip
            (12, 8),  # RHip -> RHip
            (13, 12),  # LKnee -> LKnee
            (14, 9),  # RKnee -> RKnee
            (15, 13),  # LAnkle -> LAnkle
            (16, 10),  # RAnkle -> RAnkle
        ]

        for vitpose_idx, coco_idx in mapping:
            if vitpose_idx < len(keypoints_17):
                kp = keypoints_17[vitpose_idx]
                # keypoints_17は [x, y, confidence] 形式
                x = float(kp[0])
                y = float(kp[1])
                confidence = float(kp[2]) if len(kp) > 2 else 1.0

                # 有効なkeypointかチェック（信頼度 > 0）
                if confidence > 0 and 0 <= x < w and 0 <= y < h:
                    keypoints_18[coco_idx] = (x, y, confidence)

        # Neckを計算（両肩の中点）
        if (
            keypoints_18[5][0] is not None
            and keypoints_18[2][0] is not None
            and keypoints_18[5][1] is not None
            and keypoints_18[2][1] is not None
        ):
            neck_x = (keypoints_18[5][0] + keypoints_18[2][0]) / 2
            neck_y = (keypoints_18[5][1] + keypoints_18[2][1]) / 2
            neck_conf = min(keypoints_18[5][2], keypoints_18[2][2])
            keypoints_18[1] = (neck_x, neck_y, neck_conf)

        return keypoints_18

    def draw_skeleton(self, image, keypoints, threshold=0.1):
        """
        スケルトンを描画

        Args:
            image: 入力画像（BGR形式）
            keypoints: keypointsのリスト [(x, y, confidence), ...]
            threshold: keypointを描画する最小信頼度

        Returns:
            numpy.ndarray: スケルトンが描画された画像
        """
        skeleton_image = image.copy()

        if keypoints is None:
            return skeleton_image

        # 接続ペアを描画
        for pair in COCO_PAIRS:
            idx1, idx2 = pair
            if idx1 < len(keypoints) and idx2 < len(keypoints):
                kp1 = keypoints[idx1]
                kp2 = keypoints[idx2]

                if (
                    kp1[0] is not None
                    and kp1[1] is not None
                    and kp2[0] is not None
                    and kp2[1] is not None
                    and kp1[2] > threshold
                    and kp2[2] > threshold
                ):
                    pt1 = (int(kp1[0]), int(kp1[1]))
                    pt2 = (int(kp2[0]), int(kp2[1]))
                    cv2.line(skeleton_image, pt1, pt2, (0, 255, 0), 2)

        # keypointsを描画
        for i, kp in enumerate(keypoints):
            if kp[0] is not None and kp[1] is not None and kp[2] > threshold:
                pt = (int(kp[0]), int(kp[1]))
                cv2.circle(skeleton_image, pt, 5, (0, 0, 255), -1)
                cv2.circle(skeleton_image, pt, 3, (255, 255, 255), -1)

        return skeleton_image

