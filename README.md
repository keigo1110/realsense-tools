# RealSense 録画・分析ツール

RealSense カメラから映像と深度データを`.bag`ファイルに録画し、YOLOv8-Pose または ViTPose を使用した 3D 姿勢推定によるジャンプ分析を行う Python ツールです。

- **YOLOv8-Pose**: 高速（CPU: 10-30fps, GPU: 100-200fps+）で実用的な精度を実現
- **ViTPose**: 最高精度のキーポイント推定（CPU: 5-15fps, GPU: 30-60fps）

## 📚 ドキュメント

- **[クイックスタートガイド](QUICKSTART.md)**: 最短で使い始める方法
- **[アーキテクチャドキュメント](ARCHITECTURE.md)**: プロジェクト構造とコンポーネントの詳細
- **[方法論の詳細](METHODOLOGY.md)**: ジャンプ計測方法論（論文実験章向け）

## 目次

- [システム要件](#システム要件)
- [インストール](#インストール)
- [基本的な使い方](#基本的な使い方)
  - [録画](#録画)
  - [ジャンプ分析](#ジャンプ分析)
  - [インタラクティブ 3D アニメーション](#インタラクティブ3dアニメーション)
- [詳細オプション](#詳細オプション)
- [トラブルシューティング](#トラブルシューティング)
- [関連リンク](#関連リンク)

## システム要件

- Python 3.7 以上（推奨: 3.10 以上）
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

### 2. 姿勢推定モデルのインストール

#### YOLOv8-Pose（デフォルト、高速）

```bash
# ultralyticsをインストール（YOLOv8-Pose含む）
pip install ultralytics
```

初回実行時、モデルが自動的にダウンロードされます。

**動作確認:**

```bash
python -c "from ultralytics import YOLO; print('YOLOv8-Poseインストール成功')"
```

#### ViTPose（オプション、最高精度）

ViTPoseは最高精度のキーポイント推定を実現しますが、YOLOv8-Poseよりも推論速度が遅くなります。

```bash
# openmimをインストール
pip install openmim

# mmposeとその依存関係をインストール（時間がかかる場合があります）
mim install mmengine mmcv mmdet mmpose
```

**動作確認:**

```bash
python -c "from mmpose.apis import MMPoseInferencer; print('ViTPoseインストール成功')"
```

**注意:** ViTPoseを使用しない場合は、このセクションをスキップできます。デフォルトのYOLOv8-Poseでも十分な精度が得られます。

#### NorFairトラッキング（オプション、より堅牢な人物トラッキング）

複数人のトラッキングが不安定な場合（一瞬の検出漏れで別人として認識されるなど）、NorFairライブラリを使用することでより堅牢なトラッキングが可能です。

```bash
# NorFairをインストール
pip install norfair
```

**使用方法:**

```bash
# --use-norfair-tracking フラグを追加
python jump_analyzer.py --input bagdata/my_recording.bag --output results/ --use-norfair-tracking
```

**注意:** NorFairを使用しない場合、カスタム実装のトラッキング（IoU + キーポイントベース）が使用されます。通常はカスタム実装で十分ですが、より堅牢なトラッキングが必要な場合はNorFairを推奨します。

### 3. CUDA 高速化（オプション）

CUDA 環境では、CuPy をインストールすることで画像処理と深度計算が高速化されます。

```bash
# CUDA 11.x用
pip install cupy-cuda11x

# または CUDA 12.x用
pip install cupy-cuda12x
```

**注意:** 複数の CuPy パッケージがインストールされている場合は警告が表示されます。不要なパッケージを削除してください：

```bash
# 重複しているCuPyパッケージを削除
pip uninstall cupy-cuda11x cupy-cuda12x
# 必要に応じて再インストール
pip install cupy-cuda12x  # お使いのCUDAバージョンに合わせて選択
```

CuPy がインストールされていない場合でも、CPU モードで正常に動作します。

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

録画したファイルは自動的に`record/`ディレクトリに保存されます。ディレクトリが存在しない場合は自動的に作成されます。

#### 1. 実行日時で自動ファイル名生成（推奨）

**カラー + 深度ストリームを録画:**

```bash
python pose-record.py --enable-depth
```

**カラーストリームのみ録画:**

```bash
python pose-record.py
```

実行日時（`YYYYMMDD_HHMMSS.bag`形式）で自動的にファイル名が生成されます。
例: `record/20241201_143052.bag`

連続して録画を実行しても、毎回新しいファイル名が自動生成されるため、ファイル名の重複を気にする必要がありません。

#### 2. カスタムファイル名を指定

カスタムファイル名を指定する場合も、`record/`ディレクトリに保存されます：

```bash
python pose-record.py --record my_recording.bag --enable-depth
```

この場合、`record/my_recording.bag`に保存されます。

#### 3. 解像度と FPS を指定

```bash
python pose-record.py --enable-depth --resolution 1280 720 --fps 60
```

高解像度・高フレームレートで録画する場合に使用します。ファイル名は実行日時で自動生成されます。

#### 操作方法

- **ESC キー**: 録画を停止して終了
- **スペースキー**: 表示のみ一時停止（実際の録画は続行中）

### ジャンプ分析

録画した`.bag`ファイルから YOLOv8-Pose を使用して 3D 姿勢推定を行い、ジャンプの高さ・距離・軌跡を測定します。

#### 設定ファイルを使用する方法（推奨）

設定ファイル（`config.toml`）を使用することで、すべてのパラメータを一元管理できます。

1. **設定ファイルを準備**

   `config.toml`ファイルを作成し、パラメータを設定します：

   ```toml
   input = "record/20241201_143052.bag"  # 録画した.bagファイルのパス
   output = "results/"
   model_name = "yolov8x-pose.pt"
   interactive_3d = true
   # その他の設定...
   ```

   **注意**: 録画したファイルは`record/`ディレクトリに保存されます。実行日時で自動生成されたファイル名（例: `20241201_143052.bag`）を使用するか、カスタムファイル名を指定した場合はそのファイル名を指定してください。

2. **実行**

   ```bash
   python jump_analyzer.py --config config.toml
   ```

   コマンドライン引数で一部のパラメータを上書きすることも可能です：

   ```bash
   python jump_analyzer.py --config config.toml --interactive-3d
   ```

#### コマンドライン引数を使用する方法

設定ファイルを使わない場合、コマンドライン引数で直接指定できます：

```bash
python jump_analyzer.py --input bagdata/my_recording.bag --output results/
```

これにより、以下の出力が`results/`ディレクトリに生成されます：

- `keypoints_3d.json`: 全フレームの 3D keypoints データ
- `jump_statistics_statistics.csv`: ジャンプ統計情報
- `jump_statistics_jumps.csv`: 検出されたジャンプの詳細
- `jump_statistics_trajectory.csv`: 軌跡データ
- `jump_visualization.mp4`: 可視化動画（keypoints、軌跡、測定値を描画）
- `keypoints_3d_animation.gif`: 3D キーポイントアニメーション（スケルトンの 3D 動画）
- `jump_trajectory_horizontal.png`: ジャンプ軌跡（水平面：XZ 平面での移動経路）
- `jump_trajectory_height.png`: ジャンプ軌跡（高さ-時間：ジャンプの高さ変化）
- `keypoint_x_timeline.png`: 全キーポイントの X 座標時系列グラフ
- `keypoint_y_timeline.png`: 全キーポイントの Y 座標時系列グラフ
- `keypoint_z_timeline.png`: 全キーポイントの Z 座標時系列グラフ

### インタラクティブ 3D アニメーション

**よく使う機能**: マウスで視点を自由に動かせるインタラクティブな 3D アニメーションを表示できます。

**設定ファイルを使用する場合：**

```bash
# config.tomlで interactive_3d = true に設定
python jump_analyzer.py --config config.toml
```

**コマンドライン引数を使用する場合：**

```bash
# インタラクティブモードで表示
python jump_analyzer.py --input bagdata/my_recording.bag --output results/ --interactive-3d
```

**操作方法:**

- **マウスドラッグ**: 視点を回転
- **マウスホイール**: ズームイン/アウト
- **ウィンドウを閉じる**: 終了

インタラクティブモードでは、アニメーション中に視点を自由に変更できます。
通常の GIF 生成と併用することも可能です（`--interactive-3d`を指定するとインタラクティブ表示が優先されます）。

## 詳細オプション

### 設定ファイル（config.toml）

すべてのパラメータを`config.toml`で管理できます。`config.toml`の例：

```toml
# 入力・出力パス
input = "record/20241201_143052.bag"  # 録画した.bagファイルのパス（record/ディレクトリに保存されます）
output = "results/"

# モデル設定
model_dir = "models"
model_name = "yolov8x-pose.pt"

# ジャンプ検出閾値
threshold_vertical = 0.05
threshold_horizontal = 0.1
min_jump_height = 0.10
min_air_time = 0.20

# 出力オプション
enable_video = true  # ビデオ出力を有効化（true: 出力する、false: スキップ）
enable_3d_animation = true  # 3Dアニメーションを有効化（true: 生成する、false: スキップ）
interactive_3d = true

# キーポイントスムージング
smooth_keypoints = true
smooth_window_size = 5

# 深度データ処理
enable_depth_interpolation = true  # 深度補間を有効化（true: 補間する、false: 単一ピクセルを使用）
depth_kernel_size = 3

# カルマンフィルタ
use_kalman_filter = false
kalman_process_noise = 0.03
kalman_measurement_noise = 0.1

# 床検出
enable_floor_detection = true  # 床検出を有効化（true: 床検出を使用、false: 従来の高さベース検出を使用）

# 腰基準ジャンプ検出
waist_baseline_height = 0.85  # 床から腰までの基準高さ（m）
waist_zero_epsilon = 0.01     # 基準付近のデッドバンド（m）

# 人物トラッキング（複数人対応）
use_norfair_tracking = false  # NorFairトラッキングを使用（true: NorFair、false: カスタム実装）

# 再生時間範囲（秒、0の場合は最初から/最後まで）
start_time = 5.0
end_time = 13.0

# 高速化オプション
frame_skip = 1
resize_factor = 1.0
minimal_data = false
```

設定ファイルを使用する場合：

```bash
python jump_analyzer.py --config config.toml
```

コマンドライン引数は設定ファイルの値を上書きします。

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

### 姿勢推定モデルの選択

#### YOLOv8-Pose（デフォルト、高速）

**モデルの種類:**

- `yolov8n-pose.pt`: nano（超高速、やや精度低下）
- `yolov8s-pose.pt`: small（高速、バランス型）
- `yolov8m-pose.pt`: medium（中速、高精度）
- `yolov8l-pose.pt`: large（やや低速、高精度）
- `yolov8x-pose.pt`: extra large（最高精度）- **推奨（デフォルト）**

```bash
# より高速なモデルを使用する場合
python jump_analyzer.py --input bagdata/my_recording.bag --output results/ \
  --pose-model yolov8 --model-name yolov8n-pose.pt

# 最高精度モデルを明示的に指定
python jump_analyzer.py --input bagdata/my_recording.bag --output results/ \
  --pose-model yolov8 --model-name yolov8x-pose.pt
```

**処理速度について:**

- CPU 環境: 約 10-30fps（モデルサイズによる）
- GPU 環境: 70-200fps+（GPU 性能とモデルサイズによる）
- CUDA 環境では最高精度モデル（yolov8x-pose.pt）でも十分な速度が得られます

#### ViTPose（オプション、最高精度）

ViTPoseは最高精度のキーポイント推定を実現します。特に複雑なポーズやオクルージョン（遮蔽）に対して優れた性能を発揮します。

**モデルの種類:**

- `vitpose-tiny`: 最小モデル（高速、やや精度低下）
- `vitpose-small`: 小型モデル（高速、バランス型）
- `vitpose-base`: ベースモデル（中速、高精度）
- `vitpose-large`: 大型モデル（やや低速、高精度）
- `vitpose-huge`: 最大モデル（最高精度）- **推奨**

```bash
# ViTPoseを使用（最高精度）
python jump_analyzer.py --input bagdata/my_recording.bag --output results/ \
  --pose-model vitpose --model-name vitpose-huge

# より高速なViTPoseモデルを使用
python jump_analyzer.py --input bagdata/my_recording.bag --output results/ \
  --pose-model vitpose --model-name vitpose-base
```

**処理速度について:**

- CPU 環境: 約 5-15fps（モデルサイズによる）
- GPU 環境: 約 30-60fps（GPU 性能とモデルサイズによる）
- YOLOv8-Poseと比較して推論速度は遅くなりますが、精度は向上します

**推奨用途:**

- 研究用途で最高精度が必要な場合
- 複雑なポーズやオクルージョンが多い場合
- 小さい人物でも高精度が必要な場合

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

- `--input`: 入力.bag ファイルのパス（必須）
- `--output`: 出力ディレクトリ（必須）
- `--pose-model`: 姿勢推定モデルタイプ（`yolov8` または `vitpose`、デフォルト: `yolov8`）
- `--model-dir`: モデルディレクトリ（デフォルト: `models/`）
- `--model-name`: 使用するモデル名（YOLOv8: `yolov8x-pose.pt` など、ViTPose: `vitpose-huge` など、デフォルト: `yolov8x-pose.pt`）
- `--threshold-vertical`: 垂直ジャンプ検出閾値（メートル、デフォルト: 0.05）
- `--threshold-horizontal`: 水平ジャンプ検出閾値（メートル、デフォルト: 0.1）
- `--interactive-3d`: インタラクティブ 3D アニメーションを表示
- `--use-kalman-filter`: Kalman フィルタによる時系列平滑化を使用
- `--smooth-keypoints N`: キーポイント平滑化のウィンドウサイズ（デフォルト: 5、0 で無効化）
- `--depth-kernel-size N`: 深度補間のカーネルサイズ（デフォルト: 3）
- `--no-depth-interpolation`: 深度補間を無効化（高速だが精度低下）
- `--no-video`: 可視化動画をスキップ
- `--no-3d-animation`: 3D キーポイントアニメーションをスキップ
- `--frame-skip N`: N フレームおきに処理（デフォルト: 1）
- `--resize-factor F`: 画像リサイズ率（0.0-1.0、デフォルト: 1.0）
- `--minimal-data`: ジャンプ検出時のみデータを保存
- `--waist-baseline-height`: 床から腰までの基準値（メートル）。指定すると腰基準のゼロクロスでジャンプを検出
- `--waist-zero-epsilon`: 基準値付近のデッドバンド（メートル、デフォルト: 0.01）

## 分析結果の見方

- **垂直ジャンプ**: 高さ（Z 軸方向）の変化を測定
- **幅跳び**: 水平距離（X, Y 軸方向）を測定
- **軌跡**: 各 keypoint の時系列 3D 座標を記録

出力ファイル：

- `keypoints_3d.json`: 全フレームの 3D キーポイントデータ（JSON 形式）
- `jump_statistics_*.csv`: 統計情報、ジャンプ詳細、軌跡データ（CSV 形式）
- `jump_visualization.mp4`: 2D 可視化動画（キーポイント、軌跡、測定値を描画）
- `keypoints_3d_animation.gif`: 3D キーポイントアニメーション（またはインタラクティブ表示）
- `jump_trajectory_horizontal.png`: ジャンプ軌跡（水平面：XZ 平面での移動経路、開始点・離陸点・着地点をマーク）
- `jump_trajectory_height.png`: ジャンプ軌跡（高さ-時間：ジャンプの高さ変化を時系列で表示）
- `keypoint_*_timeline.png`: 全キーポイントの各座標軸の時系列グラフ

## 関連ドキュメント

- **[METHODOLOGY.md](METHODOLOGY.md)**: ジャンプ計測方法論の詳細説明（論文実験章向け）
  - 座標系の定義
  - 床検出アルゴリズム
  - ジャンプ検出アルゴリズム
  - 測定値（高さ、距離、滞空時間）の計算方法
  - データ処理手法

## トラブルシューティング

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

### ultralytics のインストールエラー

```bash
# pipをアップグレードしてから再試行
pip install --upgrade pip
pip install ultralytics
```

### モデルのダウンロードエラー

- インターネット接続を確認してください
- モデルは初回実行時に自動ダウンロードされます
- 手動でダウンロードする場合：https://github.com/ultralytics/assets/releases からダウンロードして`models/`ディレクトリに配置

### PyTorch の CUDA サポートエラー（GPU 使用時）

CUDA 対応版の PyTorch をインストール：

```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

ただし、YOLOv8-Pose は CPU でも高速に動作します。

### RealSense カメラが認識されない

- カメラが接続されているか確認
- 他のプログラムがカメラを使用していないか確認
- デバイスドライバーが正しくインストールされているか確認

### .bag ファイルが読み込めない

- 深度データが記録されているか確認（`--enable-depth`オプションで録画）
- ファイルパスが正しいか確認
- ファイルが破損していないか確認

### パフォーマンス関連

**CPU 環境:**

- `yolov8n-pose`: 約 20-30fps
- `yolov8s-pose`: 約 10-20fps
- `yolov8m-pose`: 約 5-10fps

**GPU 環境（CUDA）:**

- `yolov8n-pose`: 約 150-200fps
- `yolov8s-pose`: 約 100-150fps
- `yolov8m-pose`: 約 50-100fps

**Apple Silicon（MPS）:**

- 自動的に MPS が使用されます（利用可能な場合）
- 速度は CPU と GPU の中間程度

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
