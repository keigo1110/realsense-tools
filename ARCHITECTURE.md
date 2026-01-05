# プロジェクト構造とアーキテクチャ

このドキュメントでは、RealSense Tools プロジェクトの全体構造と各コンポーネントの役割を説明します。

## ディレクトリ構造

```
realsense-tools/
├── src/                          # コアモジュール
│   ├── __init__.py              # パッケージ初期化
│   ├── realsense_utils.py       # RealSenseカメラ・bagファイル操作
│   ├── yolov8_pose_3d.py        # YOLOv8-Poseによる姿勢推定
│   ├── vitpose_3d.py            # ViTPoseによる姿勢推定
│   ├── jump_detector.py         # ジャンプ検出・測定ロジック
│   ├── floor_detector.py        # 床面検出（RANSAC）
│   ├── kalman_filter_3d.py      # カルマンフィルタによる平滑化
│   ├── keypoint_smoother.py     # キーポイント平滑化ユーティリティ
│   ├── person_tracker_norfair.py # 人物トラッキング（NorFair）
│   └── visualizer.py            # 可視化・動画生成
│
├── models/                       # 機械学習モデル
│   ├── yolov8x-pose.pt         # YOLOv8-Poseモデル
│   ├── yolov8n-pose.pt         # YOLOv8-Pose（軽量版）
│   └── vitpose/                # ViTPoseモデルと設定
│
├── bagdata/                      # 録画データ（.bagファイル）
├── results/                      # 分析結果出力先
├── record/                       # 単一カメラ録画データ
│
├── jump_analyzer.py             # メイン分析スクリプト（単一/デュアル対応）
├── jump_analyzer_front_only.py  # フロントカメラ専用分析
├── pose-record.py               # 単一カメラ録画スクリプト
├── record_dual_cameras.py       # デュアルカメラ録画スクリプト
├── calibrate_dual_cameras.py    # デュアルカメラキャリブレーション
├── dual_run.py                  # デュアルカメラ統合ワークフロー
│
├── config.toml                  # 設定ファイル
├── requirements.txt             # 依存パッケージ
└── README.md                    # メインドキュメント
```

## 主要コンポーネント

### 1. 録画スクリプト

#### `pose-record.py`
- **目的**: 単一RealSenseカメラからの映像・深度データ録画
- **機能**:
  - カラーストリーム録画
  - 深度ストリーム録画（オプション）
  - 解像度・FPS設定
  - 自動ファイル名生成（日時ベース）
- **出力**: `.bag`ファイル（`record/`ディレクトリ）

#### `record_dual_cameras.py`
- **目的**: 2台のRealSenseカメラを同期録画
- **機能**:
  - キャリブレーションデータを使用した同期録画
  - フレームペアリング（タイムスタンプベース）
  - 各カメラの個別bagファイル生成
  - メタデータ（キャリブレーション情報）保存
- **出力**: `bagdata/<name>/`ディレクトリ内に各カメラのbagファイル

#### `calibrate_dual_cameras.py`
- **目的**: 2台のカメラ間の外部パラメータ（位置・姿勢）を自動推定
- **手法**: ターゲットレス自動外部標定
  - FPFH特徴量による点群マッチング
  - RANSACによる初期推定
  - ICPによる精密調整
- **出力**: `extrinsics_cam1_to_cam0.json`（変換行列）

#### `dual_run.py`
- **目的**: デュアルカメラワークフローの統合実行
- **処理フロー**:
  1. キャリブレーション実行
  2. キャリブレーション成功時、録画開始
- **用途**: ワンコマンドでデュアルカメラセットアップ

### 2. 分析スクリプト

#### `jump_analyzer.py`
- **目的**: `.bag`ファイルからジャンプ分析を実行
- **機能**:
  - 単一カメラ・デュアルカメラ対応
  - YOLOv8-Pose / ViTPoseによる姿勢推定
  - 3Dキーポイント推定（深度データ統合）
  - ジャンプ検出・測定（高さ・距離・滞空時間）
  - 可視化動画・グラフ生成
  - インタラクティブ3Dアニメーション
- **入力**: `.bag`ファイル（単一）またはディレクトリ（デュアル）
- **出力**: `results/`ディレクトリ内に各種分析結果

#### `jump_analyzer_front_only.py`
- **目的**: フロントカメラのみを使用した簡易分析
- **用途**: デュアルカメラデータの一部のみを分析

### 3. コアモジュール（`src/`）

#### `realsense_utils.py`
- **クラス**: `BagFileReader`
- **機能**:
  - `.bag`ファイルの読み込み
  - フレーム取得（カラー・深度）
  - カメラ内部パラメータ取得
  - 2D→3D座標変換
  - CUDA高速化対応（CuPy）
- **依存**: `pyrealsense2`, `opencv-python`, `numpy`（オプション: `cupy`）

#### `yolov8_pose_3d.py`
- **クラス**: `YOLOv8PoseDetector`
- **機能**:
  - YOLOv8-Poseモデルによる2Dキーポイント検出
  - 深度データとの統合による3D座標推定
  - 複数人物検出・トラッキング
- **依存**: `ultralytics`

#### `vitpose_3d.py`
- **クラス**: `ViTPoseDetector`
- **機能**:
  - ViTPoseモデルによる高精度2Dキーポイント検出
  - 深度データとの統合による3D座標推定
- **依存**: `mmpose`, `mmcv`, `mmdet`, `mmengine`

#### `jump_detector.py`
- **クラス**: `JumpDetector`
- **機能**:
  - ジャンプ検出（状態機械: ground → takeoff → airborne → landing）
  - 高さ・距離・滞空時間の測定
  - 床検出ベースの高精度検出
  - 腰基準ゼロクロス検出
- **依存**: `numpy`

#### `floor_detector.py`
- **クラス**: `FloorDetector`
- **機能**:
  - RANSACによる床面検出
  - 床面からの距離計算
- **依存**: `numpy`

#### `kalman_filter_3d.py`
- **クラス**: `KalmanSmoother`
- **機能**:
  - 3Dキーポイントの時系列平滑化
  - カルマンフィルタによるノイズ除去
- **依存**: `numpy`

#### `keypoint_smoother.py`
- **クラス**: `KeypointSmoother`
- **機能**:
  - 移動平均によるキーポイント平滑化
  - シンプルな平滑化手法
- **依存**: `numpy`

#### `person_tracker_norfair.py`
- **クラス**: `PersonTrackerNorFair`
- **機能**:
  - NorFairライブラリによる高精度人物トラッキング
  - 複数人物のID維持
- **依存**: `norfair`（オプション）

#### `visualizer.py`
- **クラス**: `JumpVisualizer`
- **関数**: `create_3d_keypoint_animation()`
- **機能**:
  - キーポイント・スケルトンの描画
  - 可視化動画生成（MP4）
  - 3Dキーポイントアニメーション（GIF/インタラクティブ）
  - 軌跡グラフ生成
- **依存**: `opencv-python`, `matplotlib`, `pillow`

## データフロー

### 単一カメラ分析フロー

```
.bagファイル
  ↓
BagFileReader (realsense_utils.py)
  ↓
YOLOv8PoseDetector / ViTPoseDetector
  ↓ (2D keypoints + depth)
3D keypoints
  ↓
KalmanSmoother / KeypointSmoother (オプション)
  ↓
FloorDetector (床検出)
  ↓
JumpDetector (ジャンプ検出・測定)
  ↓
JumpVisualizer (可視化)
  ↓
結果出力 (CSV, JSON, MP4, GIF, PNG)
```

### デュアルカメラ分析フロー

```
bagdata/<name>/
  ├── cam0_*.bag
  ├── cam1_*.bag
  └── *_metadata.json
  ↓
FramePairer (タイムスタンプベース同期)
  ↓
各カメラから3D keypoints推定
  ↓
3D座標変換（cam1 → cam0座標系）
  ↓
統合3D keypoints
  ↓
(以降は単一カメラと同じ)
```

## 設定ファイル（`config.toml`）

すべてのパラメータを`config.toml`で一元管理できます。

主要な設定カテゴリ:
- **入力・出力**: ファイルパス、出力オプション
- **モデル設定**: 姿勢推定モデルの選択
- **ジャンプ検出**: 閾値、最小値設定
- **処理オプション**: 平滑化、深度補間、カルマンフィルタ
- **高速化**: フレームスキップ、リサイズ

詳細は`config.toml`のコメントを参照してください。

## 依存関係

### 必須
- `numpy`: 数値計算
- `opencv-python`: 画像処理
- `matplotlib`: グラフ生成
- `pillow`: 画像処理
- `pyrealsense2` / `pyrealsense2-macosx`: RealSense SDK
- `ultralytics`: YOLOv8-Pose
- `toml`: 設定ファイル読み込み

### オプション
- `cupy`: CUDA高速化
- `mmpose`, `mmcv`, `mmdet`, `mmengine`: ViTPose
- `norfair`: 高精度トラッキング
- `open3d`: デュアルカメラキャリブレーション

## 拡張ポイント

### 新しい姿勢推定モデルの追加
1. `src/`に新しいモジュールを作成（例: `src/new_model_3d.py`）
2. 統一インターフェースを実装:
   ```python
   class NewModelDetector:
       def detect(self, image, depth_frame=None):
           # 2D keypoints検出
           # 3D座標推定
           return keypoints_3d
   ```
3. `jump_analyzer.py`に統合

### 新しい検出アルゴリズムの追加
1. `src/jump_detector.py`を拡張、または新しいクラスを作成
2. `JumpDetector`の`update()`メソッドを参考に実装

### 新しい可視化機能の追加
1. `src/visualizer.py`にメソッドを追加
2. `jump_analyzer.py`の可視化セクションで呼び出し

## パフォーマンス最適化

### CPU環境
- 軽量モデル使用（`yolov8n-pose.pt`）
- フレームスキップ（`--frame-skip 2`）
- リサイズ（`--resize-factor 0.7`）

### GPU環境（CUDA）
- CuPyインストール（`cupy-cuda11x`または`cupy-cuda12x`）
- 自動的にGPU処理が有効化

### Apple Silicon（MPS）
- PyTorchが自動的にMPSを使用
- 追加設定不要

## トラブルシューティング

詳細は`README.md`の「トラブルシューティング」セクションを参照してください。

