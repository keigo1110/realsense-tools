# RealSense 録画ツール

RealSense カメラから映像と深度データを`.bag`ファイルに録画するシンプルな Python ツールです。

## 特徴

- **軽量**: MediaPipe やその他の分析ライブラリに依存しません
- **シンプル**: 録画機能のみに特化
- **使いやすい**: 最小限のコマンドで録画開始

## システム要件

- Python 3.7 以上
- RealSense カメラ（D400 シリーズ等）
- macOS / Linux / Windows

## インストール

```bash
# 1. リポジトリをクローン
git clone https://github.com/keigo1110/realsense-tools.git
cd realsense-tools

# 2. 依存ライブラリをインストール
pip install -r requirements.txt
```

必要なライブラリ：

- `numpy`
- `opencv-python`
- `pyrealsense2`

MediaPipe や python-osc は不要です。

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

## 録画したファイルの分析

録画した`.bag`ファイルは、RealSense SDK の`realsense-viewer`や他の分析ツールで再生・分析できます：

```bash
# RealSense Viewerで再生
realsense-viewer
# ファイルを開くメニューから .bag ファイルを選択
```

## エラーが発生した場合

- RealSense カメラが接続されているか確認
- 他のプログラムがカメラを使用していないか確認
- 必要なライブラリがインストールされているか確認（`pip install -r requirements.txt`）

## 関連リンク

- [Intel RealSense SDK](https://www.intelrealsense.com/)
- [pyrealsense2 Documentation](https://intelrealsense.github.io/librealsense/python_docs/)
