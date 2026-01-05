# ジャンプ計測システムの方法論（論文用）

## Abstract

本研究では、RGB-D深度カメラと深層学習ベースの姿勢推定モデルを統合した非接触型ジャンプ計測システムを開発した。本システムは、3次元空間におけるジャンプ動作の高精度計測を実現し、ジャンプ高さ、水平移動距離、滞空時間を自動的に測定する。本ドキュメントでは、システムの構成要素、データ処理アルゴリズム、測定手法について詳述する。

---

## 1. Introduction

従来のジャンプ計測では、マーカーベースのモーションキャプチャシステムやフォースプレートが使用されてきたが、これらは高コストかつセットアップが複雑である。本研究では、Intel RealSense深度カメラとYOLOv8-Pose姿勢推定モデルを用いた低コストで簡便な非接触型計測システムを提案する。

---

## 2. System Configuration

### 2.1 Hardware

**深度カメラ**: Intel RealSense D455
- **解像度**: カラー映像 1920×1080 pixels、深度映像 1280×720 pixels
- **フレームレート**: 30 fps
- **深度範囲**: 0.3 m ～ 5.0 m
- **深度精度**: ±2% (2 m距離時)
- **カメラ内部パラメータ**: 焦点距離 (fx, fy)、主点座標 (ppx, ppy) を使用

**計測環境**:
- カメラ設置高さ: 約1.5 m（被験者の腰の高さ）
- 被験者とカメラの距離: 2.0 m ～ 4.0 m
- 照明条件: 室内照明（自然光または蛍光灯）

### 2.2 Software

**オペレーティングシステム**: Ubuntu 20.04 LTS / macOS / Windows 10

**開発環境**:
- Python 3.10
- PyRealSense2 2.54.2（RealSense SDK）
- OpenCV 4.8.0（画像処理）
- NumPy 1.24.0（数値計算）
- Matplotlib 3.7.0（可視化）

**姿勢推定モデル**:
- **YOLOv8-Pose** (Ultralytics, 2023): yolov8x-pose モデル（最高精度版）
  - 入力解像度: 1920×1080 pixels
  - 推論速度: GPU環境で約70-100 fps、CPU環境で約10-20 fps
  - 出力: COCO形式17キーポイント（鼻、両目、両耳、両肩、両肘、両手首、両股関節、両膝、両足首）

**代替モデル（オプション）**:
- **ViTPose** (Xu et al., 2022): vitpose-huge モデル
  - より高精度なキーポイント推定（特に複雑なポーズやオクルージョンに対して優位）
  - 推論速度: GPU環境で約30-60 fps、CPU環境で約5-15 fps

**複数人トラッキング**:
- **NorFair** (Tryolabs, 2021): 距離ベースのトラッキングライブラリ
  - 2D画像距離、3D空間距離、キーポイント類似度、時系列連続性を統合したカスタム距離関数を使用
  - トラック初期化条件: 50フレーム連続検出
  - トラック消失条件: 100フレーム（約3.3秒）未検出

---

## 3. Coordinate System

### 3.1 RealSense Camera Coordinate System

本システムでは、RealSense SDKが定義するカメラ座標系を使用する：

- **X軸**: 水平方向（カメラから見て右方向が正）
- **Y軸**: 垂直方向（カメラから見て下方向が正）
- **Z軸**: 深度方向（カメラから見て奥方向が正）

### 3.2 Reference Keypoint

すべての高さ計測において、**腰（Mid-Hip）**を基準キーポイントとして使用する。Mid-Hipは左股関節（Left Hip）と右股関節（Right Hip）の中点として以下のように計算される：

```
Mid-Hip = (Left Hip + Right Hip) / 2
```

腰を基準とする理由：
1. 身体の重心に近い位置であり、ジャンプ動作全体を代表する
2. 肩や頭部と比較して、腕や首の動きの影響を受けにくい
3. 下肢（脚部）と比較して、オクルージョン（隠れ）が少ない

---

## 4. Data Acquisition and Preprocessing

### 4.1 Depth Data Acquisition

各フレームにおいて、RealSense D455カメラから以下のデータを取得する：
- カラー映像（RGB、1920×1080 pixels）
- 深度映像（16-bit、1280×720 pixels）
- タイムスタンプ（ミリ秒精度）

### 4.2 Pose Estimation

YOLOv8-Poseモデルを使用して、カラー映像から2D姿勢推定を実行する：

1. **人物検出**: YOLOv8の物体検出機能により、画像内の人物を検出
2. **キーポイント推定**: 検出された各人物に対して、17個のキーポイント座標を推定
3. **信頼度評価**: 各キーポイントの検出信頼度（0.0～1.0）を取得

**出力形式**:
```
keypoint_2d = {
    "Nose": (x, y, confidence),
    "Left Hip": (x, y, confidence),
    "Right Hip": (x, y, confidence),
    ...
}
```

### 4.3 Depth Data Preprocessing

深度カメラのノイズを低減するため、以下の前処理を適用する。

#### 4.3.1 Spatial Interpolation

各2Dキーポイント位置に対応する深度値を取得する際、単一ピクセルではなく周囲ピクセルから補間を行う。

**処理手順**:
1. キーポイント位置を中心とした 5×5 ピクセル領域から深度値を取得
2. 有効深度範囲（0.3 m ～ 5.0 m）外の値を除外
3. IQR（Interquartile Range）法による外れ値除去
4. 残った深度値の中央値（median）を採用

**IQR法による外れ値除去**:
```
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1
Valid range = [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
```

**適応的カーネルサイズ**:
- キーポイント信頼度 ≥ 0.5: カーネルサイズ 5×5
- キーポイント信頼度 < 0.5: カーネルサイズ 7×7（補間範囲を拡大）

#### 4.3.2 3D Coordinate Transformation

カメラ内部パラメータを使用して、2D画像座標と深度値から3D空間座標を計算する。

**変換式**（ピンホールカメラモデル）:
```
x_3d = (pixel_x - ppx) × depth / fx
y_3d = (pixel_y - ppy) × depth / fy
z_3d = depth
```

ここで、
- `(pixel_x, pixel_y)`: 2D画像座標（pixels）
- `depth`: 深度値（meters）
- `(fx, fy)`: 焦点距離（pixels）
- `(ppx, ppy)`: 主点座標（pixels）

**有効性チェック**:
- 深度値が None または 0 以下の場合、そのキーポイントは無効として扱う
- 画像端から10ピクセル以内のキーポイントは境界効果を避けるため除外

### 4.4 Keypoint Temporal Smoothing

姿勢推定のフレーム間変動を低減するため、時系列平滑化を適用する。

#### 4.4.1 Kalman Filter (推奨)

3次元Kalmanフィルタを各キーポイントに適用し、位置推定の精度を向上させる。

**状態ベクトル**:
```
x = [position_x, position_y, position_z, velocity_x, velocity_y, velocity_z]^T
```

**システムモデル**:
```
x_k = F·x_{k-1} + w_k
z_k = H·x_k + v_k
```

ここで、
- F: 状態遷移行列（等速度モデル）
- H: 観測行列
- w_k: プロセスノイズ（Q = 0.005 I）
- v_k: 測定ノイズ（R = 0.15 I）

**適応的ノイズ調整**:
- 観測値が予測値から大きく外れる場合（マハラノビス距離 > 閾値）、測定ノイズを一時的に増加させ、外れ値の影響を抑制

#### 4.4.2 Moving Average Filter (代替手法)

重み付き移動平均フィルタによる平滑化：

**ウィンドウサイズ**: 10フレーム

**重み付け**: ガウス型重み（最新フレームほど重みが大きい）

```
smoothed_position = Σ(w_i × position_i) / Σ(w_i)
```

ここで、i = 0, 1, ..., 9（0が最新フレーム）

---

## 5. Floor Detection

床平面の検出には、RANSAC（Random Sample Consensus）アルゴリズムを使用する。

### 5.1 RANSAC Algorithm

**処理手順**:

1. **点群サンプリング**: 深度画像を10ピクセル間隔でサンプリングし、3D点群を生成
2. **有効点のフィルタリング**: 深度範囲（0.3 m ～ 5.0 m）内の点のみを使用
3. **平面推定**: 
   - ランダムに3点を選択し、平面方程式 `ax + by + cz + d = 0` を計算
   - 各点から平面までの距離を計算し、閾値（3 cm）以内の点をインライアとして判定
   - インライア数が最大となる平面を最適解として採用
4. **反復**: 最大500回反復し、最小インライア数（500点）以上の平面が見つかった場合に成功

### 5.2 Temporal Stabilization

床平面パラメータの時系列安定化：

**処理手順**:
1. 過去10フレームの平面パラメータ（a, b, c, d）を保存
2. IQR法により外れ値を除去
3. 品質スコア（インライア数）を重みとして、重み付き平均を計算

**品質スコア**:
```
quality_score = number_of_inliers / total_number_of_points
```

### 5.3 Distance from Floor

任意の3D点 `P = (x, y, z)` から床平面までの垂直距離：

```
distance = |ax + by + cz + d| / √(a² + b² + c²)
```

**符号の解釈**:
- 正の値: 床より上（被験者の位置）
- 負の値: 床より下（異常値）

---

## 6. Jump Detection Algorithm

ジャンプ検出は、有限状態機械（Finite State Machine）により実装される。

### 6.1 State Definitions

**4つの状態**:

1. **ground**: 接地状態（両足が床に接触）
2. **takeoff**: 離陸状態（両足が床から離れた瞬間）
3. **airborne**: 浮上状態（空中にいる状態）
4. **landing**: 着地状態（両足が床に再び接触）

### 6.2 State Transitions

#### 6.2.1 Ground → Takeoff (離陸検出)

**基準足首高さの設定**:
- 初期5フレームの足首（Left Ankle, Right Ankle）の床からの距離の最小値を基準値として設定
- 基準値は動的に更新され、より低い値が観測された場合は更新

**離陸判定条件**（以下のいずれかを満たす場合）:
1. 足首の平均高さが基準値から 5 cm 以上上昇
2. 過去3フレームの足首平均高さが基準値から 5 cm 以上上昇

**離陸時の記録**:
- `frame_takeoff`: 離陸フレーム番号
- `timestamp_takeoff`: 離陸タイムスタンプ（ms）
- `position_takeoff`: 離陸時の3D位置（Mid-Hip）
- `height_takeoff`: 離陸時の床からの高さ（meters）

#### 6.2.2 Airborne (浮上中)

**浮上状態での測定**:
- 各フレームで床からの高さを測定
- 最大高さ（`height_max`）を更新
- 水平移動距離を計算

**水平移動距離**（XZ平面での距離）:
```
distance_horizontal = √((x - x_takeoff)² + (z - z_takeoff)²)
```

注: Y軸は垂直方向のため、水平距離計算には含めない

#### 6.2.3 Airborne → Landing (着地検出)

**着地判定条件**（すべてを満たす必要がある）:
1. 左足首と右足首がそれぞれ基準値から 5 cm 以内の高さに戻る
2. この状態が5フレーム連続で継続（安定性の確保）

**着地時の記録**:
- `frame_landing`: 着地フレーム番号
- `timestamp_landing`: 着地タイムスタンプ（ms）
- `position_landing`: 着地時の3D位置（Mid-Hip）

### 6.3 Jump Validation

誤検出を防ぐため、以下の最小閾値を設定：

- **最小ジャンプ高さ**: 0.20 m（20 cm）
- **最小滞空時間**: 0.20 s（200 ms）

これらの閾値を満たさない検出結果は、ジャンプとして記録されない。

---

## 7. Measurement Metrics

### 7.1 Jump Height

**定義**: 離陸時の高さを基準とした、ジャンプ中の最大上昇距離

**計算式**:
```
Jump Height (h) = h_max - h_takeoff
```

ここで、
- `h_max`: ジャンプ中に記録された最大の床からの高さ（Mid-Hip）
- `h_takeoff`: 離陸時の床からの高さ（Mid-Hip）

**単位**: meters (m) または centimeters (cm)

**測定精度に影響する要因**:
1. 深度カメラの測定精度（±2% @ 2m）
2. 床検出の精度（RANSAC閾値: 3 cm）
3. キーポイント推定の精度（YOLOv8-Poseの信頼度）

### 7.2 Jump Distance (Horizontal)

**定義**: ジャンプ開始位置から着地位置までの水平移動距離

**計算式**:
```
Jump Distance (d) = √((x_landing - x_takeoff)² + (z_landing - z_takeoff)²)
```

ここで、
- `(x_takeoff, z_takeoff)`: 離陸時のXZ座標（Mid-Hip）
- `(x_landing, z_landing)`: 着地時のXZ座標（Mid-Hip）

**単位**: meters (m) または centimeters (cm)

**ジャンプタイプの分類**:
- **垂直ジャンプ（Vertical Jump）**: d < 0.10 m（10 cm）
- **水平ジャンプ（Horizontal Jump）**: d ≥ 0.10 m（10 cm）

### 7.3 Air Time

**定義**: 離陸から着地までの滞空時間

**計算式**:
```
Air Time (t) = (timestamp_landing - timestamp_takeoff) / 1000.0
```

**単位**: seconds (s)

**異常値処理**:
- 計算された滞空時間が10秒を超える場合、フレーム数から再計算：
```
Air Time (t) = (frame_landing - frame_takeoff) / fps
```

ここで、`fps` はフレームレート（30 fps）

### 7.4 Trajectory

**定義**: ジャンプ中の3D軌跡

**記録データ**:
- 各フレームのMid-Hip位置 `(x, y, z)`
- 各フレームのタイムスタンプ
- 床からの高さ

**可視化**:
1. **水平軌跡（XZ平面）**: 開始点、離陸点、最高点、着地点をマーク
2. **垂直軌跡（高さ-時間）**: ジャンプの高さ変化を時系列でプロット

---

## 8. Multiple Person Tracking

複数人の同時計測には、NorFairライブラリを使用したトラッキングを実装した。

### 8.1 Custom Distance Function

各フレームでの人物マッチングには、以下の統合距離関数を使用：

**統合スコア**:
```
Score = w1×D_2D + w2×D_3D + w3×(1-S_pose) + w4×(1-C_temporal)
```

ここで、
- `D_2D`: 2D画像距離（正規化）、重み w1 = 0.20
- `D_3D`: 3D空間距離（正規化）、重み w2 = 0.30
- `S_pose`: 骨格構造の類似度（0～1）、重み w3 = 0.35
- `C_temporal`: 時系列連続性（0～1）、重み w4 = 0.15

### 8.2 3D Position Validation

異常な3D位置を除外するため、有効範囲を設定：

- X軸（左右）: -2.0 m ～ 2.0 m
- Y軸（上下）: -1.0 m ～ 2.0 m
- Z軸（深度）: 1.0 m ～ 5.0 m

範囲外の検出は無効として処理される。

### 8.3 Track Quality Evaluation

トラックの品質を評価し、低品質トラックを自動削除：

**品質評価基準**:
1. 過去10フレームの3D位置の標準偏差
2. 有効キーポイント数（最低10個必要）
3. 平均信頼度（最低0.45必要）

**品質スコア < 0.3** のトラックは削除される。

---

## 9. Data Output

### 9.1 CSV Files

**jump_statistics_jumps.csv**: 各ジャンプの詳細データ

| 列名 | 説明 | 単位 |
|------|------|------|
| Jump # | ジャンプ番号 | - |
| Type | ジャンプタイプ（Vertical/Horizontal） | - |
| Height | ジャンプ高さ | cm |
| Distance | 水平移動距離 | cm |
| Start Frame | 開始フレーム番号 | - |
| Takeoff Frame | 離陸フレーム番号 | - |
| End Frame | 着地フレーム番号 | - |
| Duration | ジャンプ継続フレーム数 | frames |
| Air Time | 滞空時間 | s |

**jump_statistics_statistics.csv**: 全体統計データ

| 列名 | 説明 |
|------|------|
| Total Jumps | 総ジャンプ数 |
| Vertical Jumps | 垂直ジャンプ数 |
| Horizontal Jumps | 水平ジャンプ数 |
| Max Height | 最大ジャンプ高さ (cm) |
| Max Distance | 最大水平距離 (cm) |
| Average Height | 平均ジャンプ高さ (cm) |
| Average Distance | 平均水平距離 (cm) |
| Max Air Time | 最大滞空時間 (s) |
| Average Air Time | 平均滞空時間 (s) |

### 9.2 JSON Files

**keypoints_3d.json**: 全フレームのキーポイントデータ

構造：
```json
{
  "frame_1": {
    "timestamp": 1000,
    "keypoints_3d": {
      "Nose": [x, y, z],
      "Left Hip": [x, y, z],
      ...
    },
    "floor_height": 0.0,
    "jump_state": "ground"
  },
  ...
}
```

---

## 10. Accuracy Validation

### 10.1 Validation Methods

測定精度の検証には以下の方法を推奨する：

#### 10.1.1 Ground Truth Comparison

**方法**:
1. 既知の高さ（例: 20 cm, 30 cm, 40 cm）のステップ台を使用
2. 本システムで計測した高さと真値を比較
3. 誤差率を算出

**評価指標**:
- 平均絶対誤差（MAE）: `Σ|y_pred - y_true| / n`
- 平均二乗誤差（RMSE）: `√(Σ(y_pred - y_true)² / n)`
- 相対誤差: `|(y_pred - y_true) / y_true| × 100%`

#### 10.1.2 Repeatability Test

**方法**:
1. 同一被験者に同じジャンプ動作を複数回（例: 10回）実施
2. 測定値の標準偏差と変動係数を算出

**評価指標**:
- 標準偏差（SD）: `√(Σ(x_i - μ)² / (n-1))`
- 変動係数（CV）: `(SD / μ) × 100%`

#### 10.1.3 Comparison with Reference System

**方法**:
1. フォースプレートまたは高速度カメラシステムとの同時計測
2. 相関係数（Pearson's r）およびBland-Altman分析を実施

**評価指標**:
- 相関係数（r）: 0.90以上を目標
- Bland-Altman分析: 95%信頼区間内の一致度を評価

### 10.2 Error Sources and Mitigation

**主な誤差要因**:

1. **深度測定誤差**: ±2% @ 2m（RealSenseカメラの仕様）
   - **対策**: 空間補間とIQR法による外れ値除去

2. **キーポイント推定誤差**: 姿勢推定モデルの精度限界
   - **対策**: Kalmanフィルタによる時系列平滑化

3. **床検出誤差**: RANSAC閾値（±3 cm）
   - **対策**: 複数フレームの統合と重み付き平均

4. **タイムスタンプ誤差**: カメラのタイムスタンプ精度（±数ms）
   - **対策**: フレーム番号による滞空時間の再計算

5. **人物トラッキング誤差**: 複数人時のID混同
   - **対策**: 統合距離関数と品質評価による誤トラック除去

---

## 11. Experimental Protocol (推奨)

### 11.1 Setup

1. **カメラ配置**: 被験者から2.0～4.0 m、高さ約1.5 m
2. **計測エリア**: 2.0 m × 2.0 m（水平ジャンプ用）
3. **床面**: 平坦で反射の少ない表面（マットを推奨）
4. **照明**: 一定の室内照明（自然光の変動を避ける）

### 11.2 Calibration

1. **床検出の確認**: 初期30フレームで床平面を検出し、安定性を確認
2. **基準高さの設定**: 立位姿勢で腰の高さを記録
3. **カメラ内部パラメータ**: RealSense SDKから自動取得

### 11.3 Data Collection

1. **ウォーミングアップ**: 3-5回の練習ジャンプ
2. **本試行**: 各条件で10回以上のジャンプを実施
3. **休憩**: 各試行間に十分な休憩時間を設ける

### 11.4 Data Processing

1. **録画データ**: `.bag`ファイルとして保存
2. **オフライン解析**: `jump_analyzer.py`で後処理
3. **品質管理**: 
   - 床検出の成功率を確認（90%以上を推奨）
   - キーポイント検出信頼度を確認（平均0.5以上を推奨）
   - トラッキングの安定性を確認（ID変更が少ないこと）

---

## 12. Limitations and Future Work

### 12.1 Current Limitations

1. **深度範囲**: RealSenseカメラの測定範囲（0.3～5.0 m）に制限
2. **フレームレート**: 30 fpsのため、高速動作の詳細解析には限界
3. **オクルージョン**: 体の一部が隠れる場合、キーポイント推定精度が低下
4. **複数人トラッキング**: 人物が重なる場合、ID混同の可能性
5. **照明条件**: 逆光や極端な暗環境では精度が低下

### 12.2 Future Improvements

1. **高速度カメラ**: 60 fps以上のカメラを使用し、より詳細な動作解析
2. **複数カメラ**: 異なる角度から同時撮影し、オクルージョンを低減
3. **深層学習の改善**: ViTPoseやHRNetなど最新モデルの導入
4. **リアルタイム処理**: GPUアクセラレーションによるリアルタイムフィードバック
5. **バイオメカニクス指標**: 関節角度、力推定などの追加指標

---

## 13. Conclusion

本研究で開発した非接触型ジャンプ計測システムは、低コストかつ簡便なセットアップで、3次元空間におけるジャンプ動作の高精度計測を実現した。深度カメラと深層学習ベースの姿勢推定モデルを統合することで、ジャンプ高さ、水平移動距離、滞空時間を自動的に測定可能である。本システムは、スポーツ科学、リハビリテーション、人間工学など、幅広い分野での応用が期待される。

---

## References

### Primary References

1. **YOLOv8**:
   - Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics

2. **ViTPose**:
   - Xu, Y., Zhang, J., Zhang, Q., & Tao, D. (2022). ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation. In NeurIPS 2022.

3. **RANSAC**:
   - Fischler, M. A., & Bolles, R. C. (1981). Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. Communications of the ACM, 24(6), 381-395.

4. **Kalman Filter**:
   - Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. Journal of Basic Engineering, 82(1), 35-45.

5. **RealSense**:
   - Intel Corporation. (2023). Intel RealSense SDK 2.0. https://www.intelrealsense.com/sdk-2/

6. **NorFair**:
   - Tryolabs. (2021). Norfair: Lightweight Python library for adding real-time multi-object tracking to any detector. https://github.com/tryolabs/norfair

### Related Work

7. **Human Pose Estimation**:
   - Cao, Z., Simon, T., Wei, S. E., & Sheikh, Y. (2017). Realtime multi-person 2D pose estimation using part affinity fields. In CVPR 2017.
   - Sun, K., Xiao, B., Liu, D., & Wang, J. (2019). Deep high-resolution representation learning for human pose estimation. In CVPR 2019.

8. **3D Motion Analysis**:
   - Colyer, S. L., Evans, M., Cosker, D. P., & Salo, A. I. (2018). A review of the evolution of vision-based motion analysis and the integration of advanced computer vision methods towards developing a markerless system. Sports Medicine-Open, 4(1), 1-15.

9. **Jump Height Estimation**:
   - Bosquet, L., Berryman, N., & Dupuy, O. (2009). A comparison of 2 optical timing systems designed to measure flight time and contact time during jumping and hopping. Journal of Strength and Conditioning Research, 23(9), 2660-2665.

---

## Appendix A: Configuration Parameters

本システムのデフォルト設定値（`config.toml`）：

```toml
# ジャンプ検出閾値
threshold_vertical = 0.05        # 垂直ジャンプ検出閾値 (m)
threshold_horizontal = 0.1       # 水平ジャンプ検出閾値 (m)
min_jump_height = 0.20          # 最小ジャンプ高さ (m)
min_air_time = 0.20             # 最小滞空時間 (s)

# キーポイント平滑化
smooth_window_size = 10         # 移動平均ウィンドウサイズ
use_kalman_filter = true        # Kalmanフィルタ使用
kalman_process_noise = 0.005    # Kalman処理ノイズ
kalman_measurement_noise = 0.15 # Kalman測定ノイズ

# 深度データ処理
enable_depth_interpolation = true  # 深度補間有効化
depth_kernel_size = 5              # 補間カーネルサイズ

# 床検出
enable_floor_detection = true   # 床検出有効化

# トラッキング（NorFair）
use_norfair_tracking = true     # NorFairトラッキング使用
distance_threshold = 200        # トラッキング距離閾値 (pixels)
hit_counter_max = 100          # トラック保持フレーム数
initialization_delay = 50       # トラック初期化遅延
min_confidence = 0.45          # 最小検出信頼度
min_valid_keypoints = 10       # 最小有効キーポイント数
```

---

## Appendix B: COCO Keypoint Format

YOLOv8-PoseおよびViTPoseが出力するCOCO形式17キーポイント：

| Index | Keypoint Name | 説明 |
|-------|---------------|------|
| 0 | Nose | 鼻 |
| 1 | Left Eye | 左目 |
| 2 | Right Eye | 右目 |
| 3 | Left Ear | 左耳 |
| 4 | Right Ear | 右耳 |
| 5 | Left Shoulder | 左肩 |
| 6 | Right Shoulder | 右肩 |
| 7 | Left Elbow | 左肘 |
| 8 | Right Elbow | 右肘 |
| 9 | Left Wrist | 左手首 |
| 10 | Right Wrist | 右手首 |
| 11 | Left Hip | 左股関節 |
| 12 | Right Hip | 右股関節 |
| 13 | Left Knee | 左膝 |
| 14 | Right Knee | 右膝 |
| 15 | Left Ankle | 左足首 |
| 16 | Right Ankle | 右足首 |

**基準キーポイント（Mid-Hip）の計算**:
```
Mid-Hip = (Left Hip + Right Hip) / 2
        = (Keypoint[11] + Keypoint[12]) / 2
```

---

## Appendix C: Code Availability

本研究で使用したコードは、以下のGitHubリポジトリで公開予定：

```
[リポジトリURL]
```

**主要なファイル**:
- `jump_analyzer.py`: メイン解析スクリプト
- `src/jump_detector.py`: ジャンプ検出アルゴリズム
- `src/floor_detector.py`: 床検出アルゴリズム
- `src/keypoint_smoother.py`: キーポイント平滑化
- `src/kalman_filter_3d.py`: 3D Kalmanフィルタ
- `src/person_tracker_norfair.py`: 複数人トラッキング
- `src/yolov8_pose_3d.py`: YOLOv8-Pose検出器
- `src/vitpose_3d.py`: ViTPose検出器
- `config.toml`: 設定ファイル

**ライセンス**: MIT License

---

## Document Information

- **作成日**: 2024年12月
- **バージョン**: 1.0
- **対象**: 学術論文のMethodsセクション
- **言語**: 日本語（英語版は別途作成予定）

---

**注意事項**: 
本ドキュメントは論文執筆用の技術的詳細を提供するものであり、実際の論文では、対象とする学術誌のフォーマットに合わせて適宜編集・要約する必要があります。特に、詳細なパラメータ値や設定は本文ではなく補足資料（Supplementary Materials）に記載することを推奨します。

