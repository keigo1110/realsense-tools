# 実装状況レポート

## 概要
RealSense Tools のジャンプ分析システムの現在の実装状況をまとめます。

## 主要機能

### 1. 姿勢推定モデル

#### ✅ YOLOv8-Pose（デフォルト）
- **ファイル**: `src/yolov8_pose_3d.py`
- **クラス**: `YOLOv8PoseDetector`
- **特徴**:
  - 高速な姿勢推定
  - 複数人検出対応
  - COCO形式の18キーポイント出力
- **使用方法**: `--pose-model yolov8`

#### ✅ ViTPose（オプション、最高精度）
- **ファイル**: `src/vitpose_3d.py`
- **クラス**: `ViTPoseDetector`
- **特徴**:
  - 高精度な姿勢推定
  - YOLOv8と組み合わせて人物検出
  - 複数人検出対応
  - COCO形式の18キーポイント出力
- **使用方法**: `--pose-model vitpose --model-name vitpose-huge`
- **依存関係**: `mmpose`, `mmcv`, `mmdet`, `mmengine`, `mmpretrain`

### 2. 複数人トラッキング

#### ✅ カスタムトラッカー（IoU + キーポイントベース）
- **ファイル**: `src/person_tracker.py`
- **クラス**: `PersonTracker`
- **特徴**:
  - IoUベースのマッチング
  - キーポイント類似度を考慮
  - バウンディングボックス予測
  - 統合スコアリング（IoU + 予測IoU + キーポイント距離）
- **パラメータ**:
  - `max_age=60`: トラッキングが失われるまでの最大フレーム数
  - `min_iou=0.15`: 最小IoU閾値
  - `keypoint_match_threshold=0.5`: キーポイントマッチング閾値

#### ✅ NorFairトラッカー（推奨、高精度）
- **ファイル**: `src/person_tracker_norfair.py`
- **クラス**: `PersonTrackerNorFair`
- **特徴**:
  - NorFairライブラリを使用した堅牢なトラッキング
  - 3D位置（深度）を考慮したマッチング
  - キーポイント類似度を考慮したマッチング
  - 骨格構造の類似度を考慮
  - 時系列的な連続性を評価
  - **3D位置の妥当性チェック**（異常な位置を除外）
  - **トラックの品質評価**（低品質トラックを自動削除）
- **パラメータ**（現在の設定）:
  - `distance_threshold=200`: トラッキング距離の閾値（ピクセル）
  - `hit_counter_max=100`: トラッキングが失われるまでの最大フレーム数（約3.3秒）
  - `initialization_delay=50`: 初期化遅延（50フレーム連続検出が必要）
  - `min_confidence=0.45`: 検出の最小信頼度
  - `min_valid_keypoints=10`: 有効なキーポイントの最小数
  - `max_3d_distance=0.5`: マッチング時の最大3D距離（メートル）
  - `valid_3d_bounds=[(-2.0, 2.0), (-1.0, 2.0), (1.0, 5.0)]`: 有効な3D位置の範囲
- **使用方法**: `--use-norfair-tracking`
- **依存関係**: `norfair`

### 3. トラッキングの改善機能

#### ✅ 3D位置の妥当性チェック
- **実装場所**: `PersonTrackerNorFair._is_valid_3d_position()`
- **機能**:
  - 異常な3D位置（例：X=-4.446m、Y=264.620m）を検出して除外
  - 有効な範囲: X: -2.0m ~ 2.0m, Y: -1.0m ~ 2.0m, Z: 1.0m ~ 5.0m
  - 検出とトラックの両方でチェック

#### ✅ トラックの品質評価
- **実装場所**: `PersonTrackerNorFair._evaluate_track_quality()`
- **機能**:
  - トラックの3D位置の安定性を評価
  - 品質スコア < 0.3 のトラックを自動削除
  - 異常な3D位置を持つトラックを早期に削除

#### ✅ 骨格構造の類似度計算
- **実装場所**: `PersonTrackerNorFair._calculate_skeleton_similarity()`
- **機能**:
  - 骨の長さの類似度（30%）
  - 骨の角度の類似度（40%）
  - 相対位置の類似度（30%）
  - 統合スコアでポーズの類似度を評価

#### ✅ 時系列的な連続性評価
- **実装場所**: `PersonTrackerNorFair._calculate_temporal_consistency()`
- **機能**:
  - 過去10フレームの履歴を保持
  - 直近3フレームとの平均類似度を計算
  - 時系列的な一貫性を評価

#### ✅ カスタム距離関数
- **実装場所**: `PersonTrackerNorFair._custom_distance_function()`（NorFairの距離関数として使用）
- **機能**:
  - 2D距離: 20%
  - 3D距離: 30%
  - ポーズ類似度: 35%
  - 時系列連続性: 15%
  - 統合スコアでマッチング

### 4. デバッグ機能

#### ✅ トラッキングデバッグログ
- **実装場所**: `PersonTrackerNorFair._log_debug_info()`, `save_debug_log()`
- **機能**:
  - 30フレームごとにトラッキング状態を記録
  - 検出数、トラック数、距離、類似度などの情報を保存
  - JSON形式で保存（`tracking_debug_log.json`）
  - 混同の原因分析に使用可能

### 5. ジャンプ検出

#### ✅ ジャンプ検出器
- **ファイル**: `src/jump_detector.py`
- **クラス**: `JumpDetector`
- **特徴**:
  - 垂直ジャンプ検出
  - 水平ジャンプ検出
  - 床検出（RANSACアルゴリズム）
  - 腰基準ジャンプ検出（ゼロクロス判定）
- **パラメータ**:
  - `threshold_vertical=0.05`: 垂直ジャンプ検出閾値（メートル）
  - `threshold_horizontal=0.1`: 水平ジャンプ検出閾値（メートル）
  - `min_jump_height=0.10`: 有効なジャンプと認識する最小高さ（メートル）
  - `min_air_time=0.20`: 有効なジャンプと認識する最小滞空時間（秒）

### 6. キーポイント平滑化

#### ✅ 移動平均フィルタ
- **ファイル**: `src/keypoint_smoother.py`
- **クラス**: `KeypointSmoother`
- **特徴**:
  - 移動平均による平滑化
  - ウィンドウサイズを調整可能
- **パラメータ**: `smooth_window_size=5`

#### ✅ Kalmanフィルタ
- **ファイル**: `src/kalman_filter_3d.py`
- **クラス**: `KalmanSmoother`
- **特徴**:
  - 高精度な平滑化
  - 3Dキーポイントに対応
- **パラメータ**:
  - `kalman_process_noise=0.03`: 処理ノイズ
  - `kalman_measurement_noise=0.1`: 測定ノイズ

### 7. 出力機能

#### ✅ 複数人対応の出力
- **実装場所**: `jump_analyzer.py`
- **機能**:
  - 各人ごとに `person_X/` ディレクトリを作成
  - 各人ごとにJSON、CSV、プロット、動画、アニメーションを保存
  - 全体の可視化動画も生成

## 現在の設定値（最適化済み）

### NorFairトラッキング
```python
PersonTrackerNorFair(
    distance_threshold=200,      # 人物の動きに対応
    hit_counter_max=100,        # 約3.3秒まで保持（誤検出を防ぐ）
    initialization_delay=50,     # 50フレーム連続検出が必要（厳格）
    min_confidence=0.45,         # 信頼度の低い検出を除外
    min_valid_keypoints=10,      # 最低10つのキーポイントが必要
    max_3d_distance=0.5,         # マッチング時の最大3D距離（0.5m）
    valid_3d_bounds=[(-2.0, 2.0), (-1.0, 2.0), (1.0, 5.0)]  # 有効な3D位置の範囲
)
```

## 改善効果

### 改善前（初期実装）
- トラック数: 平均1.58
- ユニークなトラックID数: 3個
- 平均3D距離: 1.63m
- 異常な3D位置: 多数存在

### 改善後（現在）
- トラック数: 平均0.85（**46.2%減少**）
- ユニークなトラックID数: 2個
- 平均3D距離: 0.45m（**72.4%改善**）
- 異常な3D位置: **0個（完全に排除）**

## 残存する問題

### ⚠️ Person数の問題
- **現状**: 4人のPersonが検出される（実際は1-2人のはず）
- **原因**: 同一人物が複数のIDで認識されている可能性
- **対策案**:
  - トラッキングの初期化条件をさらに厳格化（initialization_delay: 50 → 70）
  - トラックの統合機能を追加（同一人物の複数IDを統合）

### ⚠️ ジャンプ高さが0.000m
- **現状**: 全Personでジャンプ高さが0.000m
- **原因**: ジャンプ検出の設定や閾値の問題の可能性
- **対策案**: ジャンプ検出のパラメータを確認・調整

## 使用方法

### 基本的な使用方法
```bash
# YOLOv8-Poseを使用（高速）
python jump_analyzer.py --input input.bag --output results/

# ViTPoseを使用（最高精度）
python jump_analyzer.py --input input.bag --output results/ \
  --pose-model vitpose --model-name vitpose-huge

# NorFairトラッキングを使用（推奨）
python jump_analyzer.py --input input.bag --output results/ \
  --use-norfair-tracking

# すべての機能を組み合わせ
python jump_analyzer.py --input input.bag --output results/ \
  --pose-model vitpose --model-name vitpose-huge \
  --use-norfair-tracking \
  --use-kalman-filter \
  --kalman-process-noise 0.01 \
  --kalman-measurement-noise 0.05
```

### 設定ファイルの使用
```bash
# config.tomlを使用
python jump_analyzer.py --config config.toml
```

## ファイル構成

```
realsense-tools/
├── jump_analyzer.py              # メインスクリプト
├── config.toml                   # 設定ファイル
├── requirements.txt              # 依存関係
├── src/
│   ├── yolov8_pose_3d.py         # YOLOv8-Pose検出器
│   ├── vitpose_3d.py             # ViTPose検出器
│   ├── person_tracker.py        # カスタムトラッカー
│   ├── person_tracker_norfair.py # NorFairトラッカー（推奨）
│   ├── jump_detector.py          # ジャンプ検出器
│   ├── keypoint_smoother.py      # 移動平均フィルタ
│   ├── kalman_filter_3d.py       # Kalmanフィルタ
│   ├── floor_detector.py          # 床検出器
│   ├── visualizer.py             # 可視化
│   └── realsense_utils.py        # RealSenseユーティリティ
└── results/                      # 出力ディレクトリ
    └── [output_name]/
        ├── person_1/             # Person 1の結果
        ├── person_2/             # Person 2の結果
        ├── tracking_debug_log.json # デバッグログ
        └── jump_visualization.mp4  # 全体の可視化動画
```

## 依存関係

### 必須
- `pyrealsense2`: RealSenseカメラのサポート
- `opencv-python`: 画像処理
- `numpy`: 数値計算
- `matplotlib`: プロット生成
- `ultralytics`: YOLOv8-Pose

### オプション（ViTPose使用時）
- `mmpose`, `mmcv`, `mmdet`, `mmengine`, `mmpretrain`

### オプション（NorFairトラッキング使用時）
- `norfair`

## 今後の改善案

1. **トラックの統合機能**: 同一人物の複数IDを統合
2. **トラッキングの初期化条件のさらなる厳格化**: initialization_delayを70に増加
3. **ジャンプ検出のパラメータ調整**: ジャンプ高さが0.000mの問題を解決
4. **ByteTrackの統合**: より高度なトラッキング手法の検討

