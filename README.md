# RealSense 録画・分析ツール

RealSense カメラから映像と深度データを`.bag`ファイルに録画し、YOLOv8-Pose を使用した 3D 姿勢推定によるジャンプ分析を行う Python ツールです。
YOLOv8-Pose は高速（CPU: 10-30fps, GPU: 100-200fps+）で、実用的な精度を実現します。

## 目次

- [システム要件](#システム要件)
- [インストール](#インストール)
- [基本的な使い方](#基本的な使い方)
  - [録画](#録画)
  - [ジャンプ分析](#ジャンプ分析)
  - [インタラクティブ3Dアニメーション](#インタラクティブ3dアニメーション)
- [詳細オプション](#詳細オプション)
- [トラブルシューティング](#トラブルシューティング)
- [関連リンク](#関連リンク)

## システム要件

- Python 3.7 以上（推奨: 3.10以上）
- RealSense カメラ（D400 シリーズ等）
- macOS / Linux / Windows

## インストール

### 1. 依存ライブラリのインストール

```bash
# 基本的なライブラリ
pip install -r requirements.txt

# pyrealsense2をインストール（プラットフォーム別）
# macOS (推奨)
pip install pyrealsense2-macosx

# Linux/Windows
pip install pyrealsense2
```

**注意**: macOS では`pyrealsense2-macosx`を使用してください。公式の`pyrealsense2`は macOS ARM64 に対応していません。

### 2. YOLOv8-Poseのインストール

```bash
# ultralyticsをインストール（YOLOv8-Pose含む）
pip install ultralytics
```

初回実行時、モデルが自動的にダウンロードされます。

**動作確認:**
```bash
python -c "from ultralytics import YOLO; print('YOLOv8-Poseインストール成功')"
```

### 3. CUDA高速化（オプション）

CUDA環境では、CuPyをインストールすることで画像処理と深度計算が高速化されます。

```bash
# CUDA 11.x用
pip install cupy-cuda11x

# または CUDA 12.x用
pip install cupy-cuda12x
```

**注意:** 複数のCuPyパッケージがインストールされている場合は警告が表示されます。不要なパッケージを削除してください：

```bash
# 重複しているCuPyパッケージを削除
pip uninstall cupy-cuda11x cupy-cuda12x
# 必要に応じて再インストール
pip install cupy-cuda12x  # お使いのCUDAバージョンに合わせて選択
```

CuPyがインストールされていない場合でも、CPUモードで正常に動作します。

### インストール済みパッケージ一覧（参考）

以下のバージョンで動作確認済み：

```
numpy
opencv-python
matplotlib
pillow  # 3Dキーポイントアニメーション（GIF出力）用
ultralytics  # 8.0.0以上
torch  # PyTorchはultralyticsと一緒にインストールされます
pyrealsense2 または pyrealsense2-macosx
```

## 基本的な使い方

### 録画

#### 1. カラー + 深度ストリームを録画（推奨）

```bash
python pose-record.py --record my_recording.bag --enable-depth
```

これにより、カラー映像と深度データの両方が`.bag`ファイルに保存されます。
後から詳細な分析が可能になります。

#### 2. カラーストリームのみ録画

```bash
python pose-record.py --record my_recording.bag
```

深度データが不要な場合に使用します。ファイルサイズは小さくなりますが、
後から深度データを使った分析はできません。

#### 3. 解像度とFPSを指定

```bash
python pose-record.py --record my_recording.bag --enable-depth --resolution 1280 720 --fps 60
```

高解像度・高フレームレートで録画する場合に使用します。

#### 操作方法

- **ESCキー**: 録画を停止して終了
- **スペースキー**: 表示のみ一時停止（実際の録画は続行中）

### ジャンプ分析

録画した`.bag`ファイルから YOLOv8-Pose を使用して 3D 姿勢推定を行い、ジャンプの高さ・距離・軌跡を測定します。

#### 基本的な分析

```bash
python jump_analyzer.py --input bagdata/my_recording.bag --output results/
```

これにより、以下の出力が`results/`ディレクトリに生成されます：

- `keypoints_3d.json`: 全フレームの 3D keypoints データ
- `jump_statistics_statistics.csv`: ジャンプ統計情報
- `jump_statistics_jumps.csv`: 検出されたジャンプの詳細
- `jump_statistics_trajectory.csv`: 軌跡データ
- `jump_visualization.mp4`: 可視化動画（keypoints、軌跡、測定値を描画）
- `keypoints_3d_animation.gif`: 3Dキーポイントアニメーション（スケルトンの3D動画）

### インタラクティブ3Dアニメーション

**よく使う機能**: マウスで視点を自由に動かせるインタラクティブな3Dアニメーションを表示できます。

```bash
# インタラクティブモードで表示（推奨）
python jump_analyzer.py --input bagdata/my_recording.bag --output results/ --interactive-3d
```

**操作方法:**
- **マウスドラッグ**: 視点を回転
- **マウスホイール**: ズームイン/アウト
- **ウィンドウを閉じる**: 終了

インタラクティブモードでは、アニメーション中に視点を自由に変更できます。
通常のGIF生成と併用することも可能です（`--interactive-3d`を指定するとインタラクティブ表示が優先されます）。

## 詳細オプション

### 研究用途向け高精度設定

```bash
# Kalmanフィルタを使用した高精度平滑化
python jump_analyzer.py --input bagdata/my_recording.bag --output results/ \
  --use-kalman-filter \
  --kalman-process-noise 0.01 \
  --kalman-measurement-noise 0.1

# 深度補間とキーポイント平滑化を組み合わせた最高精度設定
python jump_analyzer.py --input bagdata/my_recording.bag --output results/ \
  --use-kalman-filter \
  --depth-kernel-size 5 \
  --smooth-keypoints 0  # Kalman使用時は不要
```

### ジャンプ検出の閾値調整

```bash
# 垂直ジャンプと水平ジャンプの検出閾値を調整
python jump_analyzer.py --input bagdata/my_recording.bag --output results/ \
  --threshold-vertical 0.1 \
  --threshold-horizontal 0.2
```

### YOLOv8-Poseモデルの選択

**モデルの種類:**

- `yolov8n-pose.pt`: nano（超高速、やや精度低下）
- `yolov8s-pose.pt`: small（高速、バランス型）
- `yolov8m-pose.pt`: medium（中速、高精度）
- `yolov8l-pose.pt`: large（やや低速、高精度）
- `yolov8x-pose.pt`: extra large（最高精度）- **推奨（デフォルト）**

```bash
# より高速なモデルを使用する場合
python jump_analyzer.py --input bagdata/my_recording.bag --output results/ \
  --model-name yolov8n-pose.pt

# 最高精度モデルを明示的に指定
python jump_analyzer.py --input bagdata/my_recording.bag --output results/ \
  --model-name yolov8x-pose.pt
```

**処理速度について:**
- CPU 環境: 約 10-30fps（モデルサイズによる）
- GPU 環境: 70-200fps+（GPU 性能とモデルサイズによる）
- CUDA環境では最高精度モデル（yolov8x-pose.pt）でも十分な速度が得られます

### 高速化オプション（処理が遅い場合）

```bash
# 2フレームおきに処理（2倍速）
python jump_analyzer.py --input bagdata/my_recording.bag --output results/ --frame-skip 2

# 画像を半分のサイズにリサイズしてから処理（約4倍速、精度はやや低下）
python jump_analyzer.py --input bagdata/my_recording.bag --output results/ --resize-factor 0.5

# 組み合わせて高速化（推奨）
python jump_analyzer.py --input bagdata/my_recording.bag --output results/ \
  --frame-skip 2 --resize-factor 0.7 --minimal-data --no-video
```

### 出力のカスタマイズ

```bash
# 可視化動画をスキップ
python jump_analyzer.py --input bagdata/my_recording.bag --output results/ --no-video

# 3Dキーポイントアニメーションをスキップ
python jump_analyzer.py --input bagdata/my_recording.bag --output results/ --no-3d-animation

# 最小限のデータのみ保存（ジャンプ検出時のみ）
python jump_analyzer.py --input bagdata/my_recording.bag --output results/ --minimal-data
```

### 全オプション一覧

```bash
python jump_analyzer.py --help
```

主要なオプション：

- `--input`: 入力.bagファイルのパス（必須）
- `--output`: 出力ディレクトリ（必須）
- `--model-dir`: モデルディレクトリ（デフォルト: `models/`）
- `--model-name`: 使用するYOLOv8-Poseモデル（デフォルト: `yolov8x-pose.pt`）
- `--threshold-vertical`: 垂直ジャンプ検出閾値（メートル、デフォルト: 0.05）
- `--threshold-horizontal`: 水平ジャンプ検出閾値（メートル、デフォルト: 0.1）
- `--interactive-3d`: インタラクティブ3Dアニメーションを表示
- `--use-kalman-filter`: Kalmanフィルタによる時系列平滑化を使用
- `--smooth-keypoints N`: キーポイント平滑化のウィンドウサイズ（デフォルト: 5、0で無効化）
- `--depth-kernel-size N`: 深度補間のカーネルサイズ（デフォルト: 3）
- `--no-depth-interpolation`: 深度補間を無効化（高速だが精度低下）
- `--no-video`: 可視化動画をスキップ
- `--no-3d-animation`: 3Dキーポイントアニメーションをスキップ
- `--frame-skip N`: Nフレームおきに処理（デフォルト: 1）
- `--resize-factor F`: 画像リサイズ率（0.0-1.0、デフォルト: 1.0）
- `--minimal-data`: ジャンプ検出時のみデータを保存

## 分析結果の見方

- **垂直ジャンプ**: 高さ（Z軸方向）の変化を測定
- **幅跳び**: 水平距離（X, Y軸方向）を測定
- **軌跡**: 各keypointの時系列3D座標を記録

出力ファイル：
- `keypoints_3d.json`: 全フレームの3Dキーポイントデータ（JSON形式）
- `jump_statistics_*.csv`: 統計情報、ジャンプ詳細、軌跡データ（CSV形式）
- `jump_visualization.mp4`: 2D可視化動画（キーポイント、軌跡、測定値を描画）
- `keypoints_3d_animation.gif`: 3Dキーポイントアニメーション（またはインタラクティブ表示）

## トラブルシューティング

### pyrealsense2のインストールエラー

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

### ultralyticsのインストールエラー

```bash
# pipをアップグレードしてから再試行
pip install --upgrade pip
pip install ultralytics
```

### モデルのダウンロードエラー

- インターネット接続を確認してください
- モデルは初回実行時に自動ダウンロードされます
- 手動でダウンロードする場合：https://github.com/ultralytics/assets/releases からダウンロードして`models/`ディレクトリに配置

### PyTorchのCUDAサポートエラー（GPU使用時）

CUDA対応版のPyTorchをインストール：

```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

ただし、YOLOv8-PoseはCPUでも高速に動作します。

### RealSenseカメラが認識されない

- カメラが接続されているか確認
- 他のプログラムがカメラを使用していないか確認
- デバイスドライバーが正しくインストールされているか確認

### .bagファイルが読み込めない

- 深度データが記録されているか確認（`--enable-depth`オプションで録画）
- ファイルパスが正しいか確認
- ファイルが破損していないか確認

### パフォーマンス関連

**CPU環境:**
- `yolov8n-pose`: 約20-30fps
- `yolov8s-pose`: 約10-20fps
- `yolov8m-pose`: 約5-10fps

**GPU環境（CUDA）:**
- `yolov8n-pose`: 約150-200fps
- `yolov8s-pose`: 約100-150fps
- `yolov8m-pose`: 約50-100fps

**Apple Silicon（MPS）:**
- 自動的にMPSが使用されます（利用可能な場合）
- 速度はCPUとGPUの中間程度

## 録画したファイルの再生

録画した`.bag`ファイルは、RealSense SDK の`realsense-viewer`や他の分析ツールで再生・分析できます：

```bash
# RealSense Viewerで再生
realsense-viewer
# ファイルを開くメニューから .bag ファイルを選択
```

## 関連リンク

- [Intel RealSense SDK](https://www.intelrealsense.com/)
- [pyrealsense2 Documentation](https://intelrealsense.github.io/librealsense/python_docs/)
- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
