# ジャンプ計測方法論

## 概要

本研究では、Intel RealSense深度カメラ（D455）とYOLOv8-Pose姿勢推定モデルを用いて、3次元空間におけるジャンプ動作の高精度計測を実現している。本ドキュメントでは、ジャンプ検出から各種測定値の算出方法まで、実験で使用した計測手法の詳細を記載する。

## システム構成

- **深度カメラ**: Intel RealSense D455
- **姿勢推定モデル**: YOLOv8-Pose (yolov8x-pose.pt)
- **床検出**: RANSACアルゴリズムによる平面検出
- **キーポイント平滑化**: 移動平均フィルタ（ウィンドウサイズ: 5フレーム）

## 座標系の定義

### RealSense座標系

- **X軸**: 左右方向（カメラから見て右が正）
- **Y軸**: 垂直方向（カメラから見て下が正）
- **Z軸**: 前後方向（カメラから見て奥が正、深度方向）

### 基準キーポイント

すべての高さ計測は**腰（Mid-Hip）**の位置を基準とする。
- Mid-Hipは左股関節（LHip）と右股関節（RHip）の中点として計算される
- 床検出が有効な場合、Mid-Hipから検出された床平面への垂直距離を使用

## 床検出

### 床平面の検出

RANSAC（Random Sample Consensus）アルゴリズムを使用して深度データから床平面を検出する。

- **サンプリング**: 深度画像を10ピクセル間隔でサンプリング
- **有効深度範囲**: 0.3m ～ 5.0m
- **床判定閾値**: 3cm（平面からの距離が3cm以内の点をインライアとして判定）
- **最小インライア数**: 500点
- **最大反復回数**: 500回

検出された床平面のパラメータは、過去10フレームの平均値を使用して安定化を図る。

### 床からの距離計算

任意の3D点 `(x, y, z)` から床平面までの垂直距離は、平面方程式 `ax + by + cz + d = 0` を用いて以下のように計算される：

```
distance = |ax + by + cz + d| / √(a² + b² + c²)
```

## ジャンプ検出アルゴリズム

### 状態遷移

ジャンプ検出は以下の4つの状態で管理される：

1. **ground（接地状態）**: 両足が床に接触している、またはジャンプ準備段階
2. **takeoff（離陸）**: 両足が床から離れた瞬間
3. **airborne（浮上中）**: 空中にいる状態
4. **landing（着地）**: 両足が床に再び着地した状態

### 離陸検出

離陸は以下の条件で検出される：

1. **基準足首高さの設定**
   - 最初の5フレームの足首（LAnkle, RAnkle）の床からの距離の最低値を基準値として設定
   - 基準値は常に更新され、より低い値が見つかった場合は更新される

2. **離陸判定条件**
   - 足首の平均高さが基準値から5cm以上上昇した場合
   - または、過去3フレームの足首の平均高さが基準値から5cm以上上昇した場合
   - 注: 腰の高さ上昇は補助判定として使用しない（ジャンプ前のかがむ動作で誤検出の可能性があるため）

3. **離陸時の記録**
   - `jump_takeoff_frame`: 離陸フレーム番号
   - `jump_takeoff_timestamp`: 離陸タイムスタンプ
   - `jump_takeoff_position`: 離陸時の3D位置 `(x, y, z)`
   - `jump_takeoff_height`: 離陸時の床からの高さ（ジャンプ高さ計算の基準）

### 着地検出

着地は以下の条件で検出される：

1. **着地判定条件**
   - 左足首と右足首がそれぞれ基準値から5cm以内の高さに戻る
   - この状態が5フレーム連続で継続する（安定性の確保）

2. **着地時の記録**
   - `jump_end_frame`: 着地フレーム番号
   - `jump_end_timestamp`: 着地タイムスタンプ
   - `jump_end_position`: 着地時の3D位置 `(x, y, z)`

## 測定値の定義と計算方法

### 1. ジャンプ高さ（Jump Height）

**定義**: 離陸時の床からの高さを基準とした、ジャンプ中の最大上昇距離

**計算式**:
```
jump_height = jump_max_height - jump_takeoff_height
```

- `jump_max_height`: ジャンプ中（takeoff → landing）に記録された最大の床からの高さ（Mid-Hip）
- `jump_takeoff_height`: 離陸時（takeoff）の床からの高さ（Mid-Hip）

**単位**: メートル（m）またはセンチメートル（cm）

**注意点**:
- 床からの高さは床検出が有効な場合のみ使用される
- 床検出が無効な場合は、カメラ座標系のY座標差を使用（精度が低下する可能性がある）

### 2. ジャンプ距離（Jump Distance）

**定義**: ジャンプ開始位置から着地位置までの水平移動距離

**計算式**:
```
jump_distance = √((x_current - x_start)² + (z_current - z_start)²)
```

- `(x_start, z_start)`: ジャンプ開始位置のXZ座標
- `(x_current, z_current)`: ジャンプ中の現在位置のXZ座標
- ジャンプ中に記録された最大値を使用

**単位**: メートル（m）またはセンチメートル（cm）

**注意点**:
- 水平距離は**XZ平面**での距離である（Y軸は垂直方向のため含めない）
- RealSense座標系では、X=左右、Z=前後（深度）方向
- 開始位置は `ground` 状態からジャンプ検出が始まった最初のフレームで記録される

### 3. 滞空時間（Air Time）

**定義**: 離陸から着地までの時間

**計算式**:
```
air_time = (jump_end_timestamp - jump_takeoff_timestamp) / 1000.0
```

- `jump_takeoff_timestamp`: 離陸時のタイムスタンプ（ミリ秒）
- `jump_end_timestamp`: 着地時のタイムスタンプ（ミリ秒）
- タイムスタンプがミリ秒単位の場合、1000で除算して秒に変換

**単位**: 秒（s）またはミリ秒（ms）

**異常値処理**:
- 計算された滞空時間が10秒を超える場合、フレーム数から再計算する
- フレームベースの計算: `air_time = (end_frame - takeoff_frame) / fps`
- フレームレート（fps）は30fpsを仮定

### 4. ジャンプの種類判定

ジャンプは水平移動距離に基づいて分類される：

- **垂直ジャンプ（Vertical Jump）**: `jump_distance < threshold_horizontal`（デフォルト: 10cm）
  - 距離は0.0mとして記録される
- **水平ジャンプ（Horizontal Jump）**: `jump_distance ≥ threshold_horizontal`（デフォルト: 10cm）
  - 実際のXZ平面距離が記録される

## 有効性フィルタリング

誤検出を防ぐため、以下の最小閾値を設定している：

- **最小ジャンプ高さ**: 10cm（`min_jump_height = 0.10m`）
- **最小滞空時間**: 200ms（`min_air_time = 0.20s`）

これらの閾値を満たさない検出結果は、ジャンプとして記録されない。

## フレームと位置の記録

各ジャンプについて、以下の情報が記録される：

| 変数名 | 説明 | 記録タイミング |
|--------|------|---------------|
| `frame_start` | ジャンプ開始フレーム番号 | ジャンプ検出が開始された時点 |
| `frame_takeoff` | 離陸フレーム番号 | 両足が床から離れた時点 |
| `frame_end` | 着地フレーム番号 | 両足が床に安定して着地した時点 |
| `start_position` | 開始位置 `(x, y, z)` | `frame_start` 時点 |
| `takeoff_position` | 離陸位置 `(x, y, z)` | `frame_takeoff` 時点 |
| `end_position` | 着地位置 `(x, y, z)` | `frame_end` 時点 |

## キーポイント平滑化

姿勢推定のノイズを低減するため、移動平均フィルタを適用：

- **ウィンドウサイズ**: 5フレーム（デフォルト）
- **適用範囲**: 全17キーポイントの2D座標（X, Y）および3D座標（X, Y, Z）
- **実装**: 過去Nフレームの平均値を計算

より高度な平滑化として、カルマンフィルタも利用可能（オプション）。

## 深度データ処理

### 空間補間

キーポイント位置の深度値を取得する際、周辺ピクセルの深度値を考慮した補間を行う：

- **カーネルサイズ**: 3×3（デフォルト）
- **方法**: メディアンフィルタによる補間
- **外れ値除去**: IQR（四分位範囲）法による異常値検出と除去
- **適応的カーネルサイズ**: キーポイントの信頼度に応じて調整

### キーポイント除外条件

以下の条件に該当するキーポイントは、深度計算から除外される：

- 画像端から10ピクセル以内の位置（境界効果を避けるため）
- 信頼度が閾値未満のキーポイント

## データ出力

### CSVファイル

各ジャンプの詳細情報が以下の形式で保存される：

- **jump_statistics_jumps.csv**: 各ジャンプの詳細データ
  - Jump #, Type, Height (cm), Distance (cm), Start Frame, Takeoff Frame, End Frame, Duration (frames), Air Time (s)
- **jump_statistics_statistics.csv**: 全体統計データ
  - Total Jumps, Vertical Jumps, Horizontal Jumps, Max Height, Max Distance, Average Height, Average Distance, Max Air Time, Average Air Time

### JSONファイル

全フレームのキーポイントデータとジャンプ検出結果が保存される（`minimal_data`モードの場合はジャンプ検出フレームのみ）。

## 検証方法

測定精度の検証には以下を推奨：

1. **手動測定との比較**: 既知の高さ・距離のジャンプを実施し、測定値との一致度を評価
2. **再現性検証**: 同じジャンプ動作を複数回実施し、測定値の変動係数を評価
3. **床検出精度**: 検出された床平面の精度を検証（既知の水平面との比較）

## 参考文献

- YOLOv8-Pose: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- RealSense SDK: [Intel RealSense SDK 2.0](https://www.intelrealsense.com/sdk-2/)
- RANSAC: Fischler, M. A., & Bolles, R. C. (1981). Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. Communications of the ACM, 24(6), 381-395.