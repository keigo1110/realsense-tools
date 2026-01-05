# クイックスタートガイド

このガイドでは、RealSense Toolsを最短で使い始める方法を説明します。

## 前提条件

- Python 3.7以上（推奨: 3.10以上）
- RealSenseカメラ（D400シリーズ等）
- インターネット接続（初回モデルダウンロード用）

## インストール（5分）

### 1. 依存パッケージのインストール

```bash
# 基本的なライブラリ
pip install -r requirements.txt

# RealSense SDK（プラットフォーム別）
# macOS
pip install pyrealsense2-macosx

# Linux/Windows
pip install pyrealsense2

# YOLOv8-Pose（姿勢推定モデル）
pip install ultralytics
```

### 2. 動作確認

```bash
# YOLOv8-Poseの確認
python -c "from ultralytics import YOLO; print('OK')"

# RealSense SDKの確認
python -c "import pyrealsense2 as rs; print('OK')"
```

## 基本的な使い方（10分）

### ステップ1: 録画

```bash
# カラー + 深度ストリームを録画（推奨）
python pose-record.py --enable-depth
```

- **ESCキー**: 録画停止・終了
- **スペースキー**: 表示のみ一時停止（録画は継続）

録画ファイルは自動的に`record/`ディレクトリに保存されます（例: `record/20241201_143052.bag`）。

### ステップ2: 分析

```bash
# 設定ファイルを使用（推奨）
# 1. config.tomlを編集してinputパスを設定
# 2. 実行
python jump_analyzer.py --config config.toml
```

または、コマンドライン引数で直接指定:

```bash
python jump_analyzer.py \
  --input record/20241201_143052.bag \
  --output results/
```

### ステップ3: 結果確認

`results/`ディレクトリに以下が生成されます:

- `jump_statistics_*.csv`: ジャンプ統計情報
- `jump_visualization.mp4`: 可視化動画
- `keypoints_3d_animation.gif`: 3Dアニメーション
- `jump_trajectory_*.png`: 軌跡グラフ

## デュアルカメラの使い方（15分）

### ステップ1: キャリブレーション + 録画

```bash
# ワンコマンドでキャリブレーション→録画
python dual_run.py
```

または、個別に実行:

```bash
# 1. キャリブレーション
python calibrate_dual_cameras.py --once

# 2. 録画（'s'キーで開始）
python record_dual_cameras.py --wait-start
```

### ステップ2: 分析

```bash
# config.tomlでdual = trueに設定
# inputにディレクトリを指定（例: "bagdata/my_recording"）
python jump_analyzer.py --config config.toml
```

## よく使う設定

### 高速化（処理が遅い場合）

```bash
python jump_analyzer.py \
  --input record/20241201_143052.bag \
  --output results/ \
  --frame-skip 2 \
  --resize-factor 0.7
```

### 高精度設定（研究用途）

```bash
python jump_analyzer.py \
  --input record/20241201_143052.bag \
  --output results/ \
  --use-kalman-filter \
  --depth-kernel-size 5 \
  --model-name yolov8x-pose.pt
```

### インタラクティブ3D表示

```bash
python jump_analyzer.py \
  --input record/20241201_143052.bag \
  --output results/ \
  --interactive-3d
```

マウスで視点を自由に動かせます。

## トラブルシューティング

### カメラが認識されない

```bash
# デバイス確認
python -c "import pyrealsense2 as rs; ctx = rs.context(); print([d.get_info(rs.camera_info.serial_number) for d in ctx.devices])"
```

### モデルがダウンロードされない

初回実行時に自動ダウンロードされます。インターネット接続を確認してください。

### 処理が遅い

- 軽量モデルを使用: `--model-name yolov8n-pose.pt`
- フレームスキップ: `--frame-skip 2`
- リサイズ: `--resize-factor 0.7`

詳細は`README.md`を参照してください。

## 次のステップ

- **詳細な使い方**: `README.md`
- **プロジェクト構造**: `ARCHITECTURE.md`
- **方法論の詳細**: `METHODOLOGY.md`
- **設定オプション**: `config.toml`のコメント

