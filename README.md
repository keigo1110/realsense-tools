# RealSense 録画・分析ツール

RealSense カメラから映像と深度データを`.bag`ファイルに録画し、YOLOv8-Pose を使用した 3D 姿勢推定によるジャンプ分析を行う Python ツールです。
YOLOv8-Pose は高速（CPU: 10-30fps, GPU: 100-200fps+）で、実用的な精度を実現します。

## システム要件

- Python 3.7 以上
- RealSense カメラ（D400 シリーズ等）
- macOS / Linux / Windows

## インストール

### 基本的なインストール手順

```bash
# 1. 依存ライブラリをインストール
pip install -r requirements.txt

# 2. pyrealsense2をインストール（プラットフォーム別）
# macOS (推奨)
pip install pyrealsense2-macosx

# Linux/Windows
pip install pyrealsense2
```

### 必要なライブラリ

- `numpy`
- `opencv-python`
- `matplotlib`
- `pyrealsense2` または `pyrealsense2-macosx`（macOS の場合）

**注意**: macOS では`pyrealsense2-macosx`を使用してください。公式の`pyrealsense2`は macOS ARM64 に対応していません。

## 基本的な使い方

### 1. カラー + 深度ストリームを録画（推奨）

```bash
python pose-record.py --record my_recording.bag --enable-depth
```

これにより、カラー映像と深度データの両方が`.bag`ファイルに保存されます。
後から詳細な分析が可能になります。

### 2. カラーストリームのみ録画

```bash
python pose-record.py --record my_recording.bag
```

深度データが不要な場合に使用します。ファイルサイズは小さくなりますが、
後から深度データを使った分析はできません。

### 3. 解像度と FPS を指定

```bash
python pose-record.py --record my_recording.bag --enable-depth --resolution 1280 720 --fps 60
```

高解像度・高フレームレートで録画する場合に使用します。

## 操作方法

- **ESC キー**: 録画を停止して終了
- **スペースキー**: 表示のみ一時停止（実際の録画は続行中）

## 出力ファイル

`.bag`ファイルには以下が保存されます：

- ✅ RGB/Color ストリーム（常に保存）
- ✅ Depth ストリーム（`--enable-depth`指定時）
- ✅ タイムスタンプ情報
- ✅ カメラ内部パラメータ（深度計算用）
- ✅ 解像度・FPS 情報

## 録画したファイルのジャンプ分析

録画した`.bag`ファイルから YOLOv8-Pose を使用して 3D 姿勢推定を行い、ジャンプの高さ・距離・軌跡を測定します。
YOLOv8-Pose は高速（CPU: 10-30fps, GPU: 100-200fps+）で、実用的な精度を実現します。

### 基本的な使い方

```bash
# 基本的な分析
python jump_analyzer.py --input bagdata/my_recording.bag --output results/
```

これにより、以下の出力が`results/`ディレクトリに生成されます：

- `keypoints_3d.json`: 全フレームの 3D keypoints データ
- `jump_statistics_statistics.csv`: ジャンプ統計情報
- `jump_statistics_jumps.csv`: 検出されたジャンプの詳細
- `jump_statistics_trajectory.csv`: 軌跡データ
- `jump_visualization.mp4`: 可視化動画（keypoints、軌跡、測定値を描画）

### オプション

```bash
# モデルディレクトリを指定
python jump_analyzer.py --input bagdata/my_recording.bag --output results/ --model-dir models/

# ジャンプ検出の閾値を調整
python jump_analyzer.py --input bagdata/my_recording.bag --output results/ \
  --threshold-vertical 0.1 --threshold-horizontal 0.2

# 可視化動画をスキップ
python jump_analyzer.py --input bagdata/my_recording.bag --output results/ --no-video

# 高速化オプション（処理が遅い場合）
# 2フレームおきに処理（2倍速）
python jump_analyzer.py --input bagdata/my_recording.bag --output results/ --frame-skip 2

# 画像を半分のサイズにリサイズしてから処理（約4倍速、精度はやや低下）
python jump_analyzer.py --input bagdata/my_recording.bag --output results/ --resize-factor 0.5

# 組み合わせて高速化（推奨）
python jump_analyzer.py --input bagdata/my_recording.bag --output results/ \
  --frame-skip 2 --resize-factor 0.7 --minimal-data --no-video
```

### YOLOv8-Pose モデルのインストール

YOLOv8-Pose は `ultralytics` ライブラリを使用します。初回実行時、モデルが自動的にダウンロードされます。

**インストール手順:**

```bash
pip install ultralytics
```

**動作確認:**

```bash
python -c "from ultralytics import YOLO; print('YOLOv8-Poseインストール成功')"
```

初回実行時、YOLOv8-Pose モデルが自動的にダウンロードされます（モデルサイズは選択したモデルにより異なります）。

**モデルの種類:**

- `yolov8n-pose.pt`: nano（超高速、やや精度低下）- **推奨（デフォルト）**
- `yolov8s-pose.pt`: small（高速、バランス型）
- `yolov8m-pose.pt`: medium（中速、高精度）
- `yolov8l-pose.pt`: large（やや低速、高精度）
- `yolov8x-pose.pt`: extra large（低速、最高精度）

**処理速度について:**

- CPU 環境: 約 10-30fps（モデルサイズによる）
- GPU 環境: 100-200fps+（GPU 性能による）

モデルを指定する場合:

```bash
python jump_analyzer.py --input bagdata/my_recording.bag --output results/ \
  --model-name yolov8s-pose.pt  # より高精度なモデルを使用
```

### 分析結果の見方

- **垂直ジャンプ**: 高さ（Z 軸方向）の変化を測定
- **幅跳び**: 水平距離（X, Y 軸方向）を測定
- **軌跡**: 各 keypoint の時系列 3D 座標を記録

## 録画したファイルの再生

録画した`.bag`ファイルは、RealSense SDK の`realsense-viewer`や他の分析ツールで再生・分析できます：

```bash
# RealSense Viewerで再生
realsense-viewer
# ファイルを開くメニューから .bag ファイルを選択
```

## エラーが発生した場合

### pyrealsense2 のインストールエラー

**macOS (ARM64/Intel)**

公式の`pyrealsense2`が利用できないため、以下の代替パッケージを使用してください：

```bash
# 推奨: pyrealsense2-macosx
pip install pyrealsense2-macosx

# 代替案: pyrealsense2-mac
pip install pyrealsense2-mac

# または: realsense-applesilicon (事前に brew install librealsense2 が必要)
brew install librealsense2
pip install realsense-applesilicon
```

**Linux/Windows**

```bash
pip install pyrealsense2
```

**注意**: これらすべてのパッケージは同じインターフェース（`import pyrealsense2 as rs`）を提供するため、コードの変更は不要です。

### その他のトラブルシューティング

- **RealSense カメラが認識されない**: カメラが接続されているか確認し、他のプログラムがカメラを使用していないか確認してください
- **.bag ファイルが読み込めない**: 深度データが記録されているか確認してください（`--enable-depth`オプションで録画）
- **YOLOv8-Pose のインストールエラー**:
  - `ultralytics`のインストールエラーが発生する場合：
    ```bash
    # pipをアップグレードしてから再試行
    pip install --upgrade pip
    pip install ultralytics
    ```
  - モデルのダウンロードエラーが発生する場合：
    - インターネット接続を確認してください
    - モデルは初回実行時に自動ダウンロードされます

## 関連リンク

- [Intel RealSense SDK](https://www.intelrealsense.com/)
- [pyrealsense2 Documentation](https://intelrealsense.github.io/librealsense/python_docs/)
