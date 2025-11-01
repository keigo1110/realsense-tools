# YOLOv8-Pose インストールガイド

このドキュメントは、`conda activate bullet`環境で動作確認済みのインストール手順です。

## 前提条件

- Python 3.10以上
- pip 23.3.1以上
- conda環境（任意ですが推奨）

## インストール手順

### ステップ1: ultralyticsのインストール

```bash
conda activate bullet
pip install ultralytics
```

**注意**: YOLOv8-Poseは`ultralytics`ライブラリのみで動作します。複雑な依存関係は不要です。

### ステップ2: 動作確認

```bash
python -c "from ultralytics import YOLO; print('YOLOv8-Poseインストール成功')"
```

成功すれば以下が表示されます：
```
YOLOv8-Poseインストール成功
```

### ステップ3: モデルのダウンロード確認（初回実行時）

初回実行時に、モデルが自動的にダウンロードされます：

```bash
python jump_analyzer.py --input bagdata/my_recording.bag --output results/
```

以下のようなモデルが自動的にダウンロードされます：
- `yolov8n-pose.pt`（デフォルト）- 約6MB
- その他のモデル（`yolov8s-pose.pt`、`yolov8m-pose.pt`など）は使用時に自動ダウンロード

## 検証済みのインストール済みパッケージ一覧

以下のバージョンで動作確認済み：

```
ultralytics       8.0.0以上
torch             2.9.0（PyTorchはultralyticsと一緒にインストールされます）
```

## モデルの種類

以下のモデルサイズから選択できます：

- `yolov8n-pose.pt`: nano（超高速、やや精度低下）- **推奨（デフォルト）**
- `yolov8s-pose.pt`: small（高速、バランス型）
- `yolov8m-pose.pt`: medium（中速、高精度）
- `yolov8l-pose.pt`: large（やや低速、高精度）
- `yolov8x-pose.pt`: extra large（低速、最高精度）

モデルを指定する場合：

```bash
python jump_analyzer.py --input bagdata/my_recording.bag --output results/ \
  --model-name yolov8s-pose.pt
```

## トラブルシューティング

### ultralyticsのインストールエラー

**エラー内容:**
```
ERROR: Could not find a version that satisfies the requirement ultralytics
```

**解決方法:**
```bash
# pipをアップグレード
pip install --upgrade pip
pip install ultralytics
```

### モデルのダウンロードエラー

**エラー内容:**
```
Failed to download model
```

**解決方法:**
- インターネット接続を確認してください
- モデルは初回実行時に自動ダウンロードされます
- 手動でダウンロードする場合：https://github.com/ultralytics/assets/releases からダウンロードして`models/`ディレクトリに配置

### PyTorchのCUDAサポートエラー（GPU使用時）

**エラー内容:**
```
CUDA not available
```

**解決方法:**
CUDA対応版のPyTorchをインストール：

```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

ただし、YOLOv8-PoseはCPUでも高速に動作します。

## ワンライナーインストール

以下のコマンドで一括インストールできます：

```bash
conda activate bullet && \
pip install --upgrade pip && \
pip install ultralytics && \
python -c "from ultralytics import YOLO; print('YOLOv8-Poseインストール成功')"
```

## requirements.txtからのインストール

`requirements.txt`からインストールすることもできます：

```bash
pip install -r requirements.txt
```

ただし、`ultralytics`のみで動作するため、単純に以下で十分です：

```bash
pip install ultralytics
```

## パフォーマンス

### CPU環境

- `yolov8n-pose`: 約20-30fps
- `yolov8s-pose`: 約10-20fps
- `yolov8m-pose`: 約5-10fps

### GPU環境（CUDA）

- `yolov8n-pose`: 約150-200fps
- `yolov8s-pose`: 約100-150fps
- `yolov8m-pose`: 約50-100fps

### Apple Silicon（MPS）

- 自動的にMPSが使用されます（利用可能な場合）
- 速度はCPUとGPUの中間程度
