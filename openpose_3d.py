"""
OpenPose 3D Utilities

OpenPoseモデルの読み込み、2D keypoints検出、スケルトン描画機能を提供
"""

import cv2
import numpy as np
import os
import urllib.request
import sys


# COCO形式のkeypoints（18 points）
COCO_KEYPOINTS = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder",
    "LElbow", "LWrist", "RHip", "RKnee", "RAnkle", "LHip",
    "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar"
]

COCO_PAIRS = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
    [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
    [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]
]


class OpenPoseDetector:
    """OpenPose検出器（OpenCV DNN使用）"""

    def __init__(self, model_dir="models"):
        """
        Args:
            model_dir: モデルファイルを保存するディレクトリ
        """
        self.model_dir = model_dir
        self.net = None
        self.input_size = (368, 368)  # OpenPoseの推奨入力サイズ

        # モデルファイルのパス
        self.proto_file = os.path.join(model_dir, "pose_deploy_linevec.prototxt")
        self.weights_file = os.path.join(model_dir, "pose_iter_440000.caffemodel")

    def download_models(self):
        """OpenPoseモデルファイルをダウンロード"""
        os.makedirs(self.model_dir, exist_ok=True)

        # モデルのURL（複数のソースを試行）
        # プロトファイル: GitHubのraw URL
        proto_base_url = "https://github.com/CMU-Perceptual-Computing-Lab/openpose/raw/master/models/pose/coco/"
        proto_url = proto_base_url + "pose_deploy_linevec.prototxt"

        # ウェイトファイル: 複数のソースを試行
        weights_urls = [
            "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel",  # 公式サーバー
            "https://github.com/CMU-Perceptual-Computing-Lab/openpose/raw/master/models/pose/coco/pose_iter_440000.caffemodel",  # GitHub raw (通常は動作しない)
        ]

        # プロトファイルのダウンロード
        if not os.path.exists(self.proto_file):
            print(f"Downloading {self.proto_file}...")
            try:
                urllib.request.urlretrieve(proto_url, self.proto_file)
                print(f"Downloaded {self.proto_file}")
            except Exception as e:
                print(f"Error downloading proto file: {e}")
                print(f"\nFailed to download from: {proto_url}")
                print("Please download manually:")
                print("  1. Visit: https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/models/pose/coco")
                print(f"  2. Download 'pose_deploy_linevec.prototxt'")
                print(f"  3. Save to: {self.proto_file}")
                return False

        # ウェイトファイルのダウンロード（大きいファイルなので進捗表示）
        if not os.path.exists(self.weights_file):
            print(f"Downloading {self.weights_file} (this may take a while)...")
            downloaded = False

            for weights_url in weights_urls:
                try:
                    print(f"Trying: {weights_url}")
                    def progress_hook(count, block_size, total_size):
                        if total_size > 0:
                            percent = int(count * block_size * 100 / total_size)
                            sys.stdout.write(f"\rProgress: {percent}%")
                            sys.stdout.flush()

                    urllib.request.urlretrieve(weights_url, self.weights_file, progress_hook)
                    print(f"\nDownloaded {self.weights_file}")
                    downloaded = True
                    break
                except Exception as e:
                    print(f"\nFailed: {e}")
                    continue

            if not downloaded:
                print("\n" + "="*60)
                print("OpenPoseモデルファイルの自動ダウンロードに失敗しました")
                print("すべてのダウンロード元から取得できませんでした")
                print("="*60)
                print("\n【手動ダウンロード手順】")
                print("\n方法1: git lfsを使用（推奨）")
                print("  brew install git-lfs  # macOS")
                print("  git lfs install")
                print("  git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git")
                print("  cd openpose")
                print(f"  mkdir -p {os.path.abspath(self.model_dir)}")
                print("  cp models/pose/coco/pose_deploy_linevec.prototxt models/pose/coco/pose_iter_440000.caffemodel " + os.path.abspath(self.model_dir) + "/")
                print("\n方法2: 直接URLからダウンロード（利用可能な場合）")
                print("  curl -L http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel -o models/pose_iter_440000.caffemodel")
                print("\n方法3: GitHubから手動ダウンロード")
                print("  https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/models/pose/coco")
                print(f"  からファイルをダウンロードして {os.path.abspath(self.model_dir)} に保存")
                print("="*60)
                return False

        return True

    def _setup_backend(self):
        """バックエンド設定: CUDAが利用可能な場合はCUDA、そうでない場合はCPU"""
        # まずCUDAを試行
        try:
            # OpenCVのバージョンによってはgetAvailableBackends()が利用可能
            try:
                backends = cv2.dnn.getAvailableBackends()
                if cv2.dnn.DNN_BACKEND_CUDA in backends:
                    # CUDAバックエンドを設定
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    print("Using CUDA (GPU) for OpenPose")
                    return
            except AttributeError:
                # getAvailableBackends()が利用できない場合は直接設定を試行
                pass

            # 直接CUDA設定を試行
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("Using CUDA (GPU) for OpenPose")

        except Exception as e:
            # CUDAの設定に失敗した場合はCPUにフォールバック
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print(f"CUDA unavailable, using CPU for OpenPose (CUDA error: {type(e).__name__})")
            except Exception as cpu_error:
                print(f"Failed to set backend: {cpu_error}")

    def load_model(self):
        """OpenPoseモデルを読み込み"""
        # モデルファイルが存在しない場合はダウンロード
        if not os.path.exists(self.proto_file) or not os.path.exists(self.weights_file):
            print("Model files not found. Attempting to download...")
            if not self.download_models():
                return False

        try:
            # OpenCV DNNを使用してモデルを読み込み
            self.net = cv2.dnn.readNetFromCaffe(self.proto_file, self.weights_file)

            # バックエンド設定: CUDAが利用可能な場合はCUDA、そうでない場合はCPU
            self._setup_backend()

            print("OpenPose model loaded successfully")
            return True

        except Exception as e:
            print(f"Error loading OpenPose model: {e}")
            return False

    def detect_keypoints(self, image):
        """
        画像からkeypointsを検出

        Args:
            image: 入力画像（BGR形式）

        Returns:
            list: 検出されたkeypointsのリスト [(x, y, confidence), ...] または None
        """
        if self.net is None:
            return None

        h, w = image.shape[:2]

        # 入力画像を前処理
        blob = cv2.dnn.blobFromImage(image, 1.0 / 255, self.input_size, (0, 0, 0), swapRB=False, crop=False)

        # ネットワークに通す
        self.net.setInput(blob)
        output = self.net.forward()

        # 出力の形状: (1, 57, 46, 46) for COCO
        # 57 = 18 keypoints * 3 (x, y, confidence) + 38 PAF (Part Affinity Fields)
        # 実際には、keypointsとPAFが別々に出力される場合もある

        # 出力を解析してkeypointsを抽出
        # OpenCV DNNの出力形式に合わせて処理
        H = output.shape[2]
        W = output.shape[3]

        # keypointsマップを取得（最初の18チャンネルがkeypointsのheatmap）
        keypoints_map = output[:, :18, :, :]

        # 各keypointの最大値位置を見つける
        keypoints = []
        for i in range(18):
            heatmap = keypoints_map[0, i, :, :]

            # 最大値の位置を見つける
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(heatmap)

            if max_val > 0.1:  # 信頼度の閾値
                # 元の画像サイズにスケール
                x = (max_loc[0] / W) * w
                y = (max_loc[1] / H) * h
                confidence = float(max_val)

                keypoints.append((x, y, confidence))
            else:
                keypoints.append((None, None, 0.0))

        return keypoints

    def draw_skeleton(self, image, keypoints, pairs=None):
        """
        検出されたkeypointsとスケルトンを画像に描画

        Args:
            image: 描画対象の画像
            keypoints: keypointsのリスト [(x, y, confidence), ...]
            pairs: 接続するkeypointペアのリスト、Noneの場合はCOCO_PAIRSを使用

        Returns:
            numpy.ndarray: 描画済み画像
        """
        if pairs is None:
            pairs = COCO_PAIRS

        image = image.copy()

        # スケルトンを描画（線）
        for pair in pairs:
            idx1, idx2 = pair[0], pair[1]

            if (keypoints[idx1][2] > 0.1 and keypoints[idx2][2] > 0.1 and
                keypoints[idx1][0] is not None and keypoints[idx2][0] is not None):

                pt1 = (int(keypoints[idx1][0]), int(keypoints[idx1][1]))
                pt2 = (int(keypoints[idx2][0]), int(keypoints[idx2][1]))

                cv2.line(image, pt1, pt2, (0, 255, 0), 2)

        # keypointsを描画（円）
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0.1 and x is not None and y is not None:
                cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
                # keypointの名前を表示（オプション）
                # cv2.putText(image, str(i), (int(x)+5, int(y)),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        return image

