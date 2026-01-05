# ジャンプ区間検出ロジックの現状

## 概要

現在のジャンプ検出ロジックは、床検出ベースの高精度な検出を実装しています。

## 状態遷移マシン

### 状態定義

1. **`ground`（接地状態）**
   - 両足が床に接触している、またはジャンプ準備段階
   - 初期状態

2. **`takeoff`（離陸）**
   - 両足が床から離れた瞬間
   - 離陸フレーム・タイムスタンプ・位置・高さを記録

3. **`airborne`（浮上中）**
   - 空中にいる状態
   - 最大高さ・距離を更新

4. **`landing`（着地）**
   - 両足が床に再び着地した状態
   - ジャンプ完了を記録

### 状態遷移図

```
ground → takeoff → airborne → landing → ground
  ↑                                    ↓
  └────────────────────────────────────┘
```

## 離陸検出ロジック

### 基準足首高さの設定

```python
# 最初の5フレームの足首の床からの距離の最低値を基準値として設定
baseline_ankle_height = min(最初の5フレームの足首高さ)

# 基準値は常に更新（より低い値が見つかった場合のみ）
if recent_min < baseline_ankle_height:
    baseline_ankle_height = recent_min
```

### 離陸判定条件

```python
# 条件1: 足首の平均高さが基準値から5cm以上上昇
if avg_ankle_height > baseline_ankle_height + 0.05:
    ankle_height_increased = True

# 条件2: 過去3フレームの足首の平均高さが基準値から5cm以上上昇
elif len(ankle_height_history) >= 3:
    avg_recent = mean(過去3フレームの足首高さ)
    if avg_recent > baseline_ankle_height + 0.05:
        ankle_height_increased = True
```

### 離陸時の記録

- `jump_takeoff_frame`: 離陸フレーム番号
- `jump_takeoff_timestamp`: 離陸タイムスタンプ
- `jump_takeoff_position`: 離陸時の3D位置 `(x, y, z)`
- `jump_takeoff_height`: 離陸時の床からの高さ（ジャンプ高さ計算の基準）

## 着地検出ロジック

### 着地判定条件

```python
# 条件1: 両足が基準値から5cm以内の高さに戻る
left_on_floor = abs(left_ankle_height - baseline_ankle_height) <= 0.05
right_on_floor = abs(right_ankle_height - baseline_ankle_height) <= 0.05

# 条件2: この状態が5フレーム連続で継続
both_feet_stable_on_floor = (
    len(left_foot_on_floor) >= 5 and
    len(right_foot_on_floor) >= 5 and
    all(過去5フレームのleft_foot_on_floor) and
    all(過去5フレームのright_foot_on_floor)
)
```

### 着地時の記録

- `jump_end_frame`: 着地フレーム番号
- `jump_end_timestamp`: 着地タイムスタンプ
- `jump_end_position`: 着地時の3D位置 `(x, y, z)`

## ジャンプ高さ計算

### 計算式

```python
jump_height = jump_max_height - jump_takeoff_height
```

- `jump_max_height`: ジャンプ中（takeoff → landing）に記録された最大の床からの高さ（Mid-Hip）
- `jump_takeoff_height`: 離陸時（takeoff）の床からの高さ（Mid-Hip）

### 最大高さの更新

```python
# airborne状態中に最大高さを更新
if height_above_floor > jump_max_height:
    jump_max_height = height_above_floor
```

## ジャンプ有効性チェック

### 最小高さチェック

```python
min_jump_height = 0.10  # 10cm
if jump_height < min_jump_height:
    # 無効なジャンプとして記録しない
    is_valid_jump = False
```

### 最小滞空時間チェック

```python
min_air_time = 0.20  # 0.2秒 = 200ms
if air_time < min_air_time:
    # 無効なジャンプとして記録しない
    is_valid_jump = False
```

### 有効性判定

```python
is_valid_jump = (
    jump_height >= min_jump_height and
    air_time >= min_air_time
)
```

## 滞空時間計算

### 計算方法

```python
# タイムスタンプの差分から計算
timestamp_diff = abs(jump_end_timestamp - jump_takeoff_timestamp)

# タイムスタンプの単位を判定（ミリ秒なら10桁以上の値）
if jump_takeoff_timestamp > 1000000000:
    air_time = timestamp_diff / 1000.0  # ミリ秒→秒
else:
    air_time = timestamp_diff  # 秒単位

# 異常値チェック（最大10秒の滞空時間を想定）
if air_time > 10.0:
    # フレーム数から計算（30fpsを仮定）
    frame_diff = abs(jump_end_frame - jump_takeoff_frame)
    air_time = frame_diff / 30.0
```

## ジャンプ距離計算

### 計算式

```python
# XZ平面での距離（水平距離）
jump_distance = sqrt(
    (x - jump_start_position[0])^2 +
    (z - jump_start_position[2])^2
)
```

- `jump_start_position`: ジャンプ開始位置（ground状態から記録）
- `(x, z)`: 現在位置のXZ座標

### ジャンプタイプ判定

```python
threshold_horizontal = 0.1  # 10cm

if jump_max_distance < threshold_horizontal:
    jump_type = "vertical"  # 垂直ジャンプ
    jump_distance = 0.0
else:
    jump_type = "horizontal"  # 水平ジャンプ
    jump_distance = jump_max_distance
```

## 実装の詳細

### 3つの検出モード

1. **`_detect_jump()`**: 床検出なし（従来方式）
   - Z座標の変化を基準に検出
   - 精度が低い

2. **`_detect_jump_with_floor()`**: 床検出あり（推奨）
   - 足首の高さ変化を基準に検出
   - より正確な離陸・着地検出

3. **`_detect_jump_with_baseline()`**: 腰基準ゼロクロス検出
   - `waist_baseline_height`が設定されている場合に使用
   - 腰の床距離と基準値のゼロクロスで検出

### 初期状態の処理

```python
_initial_state = True  # 初期状態フラグ

# 初期状態では離陸検出をスキップ
if _initial_state:
    # 開始位置を記録
    jump_start_position = (x, y, z)
    
    # 5フレーム経過後に初期状態を解除
    if len(height_history) >= 5:
        _initial_state = False
```

### 着地後の処理

```python
# 着地後は一定フレーム間、新しいジャンプを検出しない（誤検出を防ぐ）
# 履歴をクリアして、安定した床接触を再確立
left_foot_on_floor.clear()
right_foot_on_floor.clear()
_initial_state = True  # 次のジャンプ検出のために初期状態に戻す
```

## 現在のパラメータ

### 離陸検出

- **基準足首高さの初期化**: 最初の5フレームの最低値
- **離陸判定閾値**: 基準値から5cm以上上昇（`0.05m`）
- **過去フレーム数**: 3フレーム

### 着地検出

- **着地判定閾値**: 基準値から5cm以内（`0.05m`）
- **安定性チェック**: 5フレーム連続

### ジャンプ有効性

- **最小高さ**: 10cm（`0.10m`）
- **最小滞空時間**: 0.2秒（`200ms`）

### ジャンプタイプ判定

- **水平ジャンプ閾値**: 10cm（`0.1m`）

## 潜在的な問題点

### 1. 離陸検出の感度

- **問題**: 5cmの閾値が適切かどうか
- **影響**: 小さな動き（つま先立ちなど）を誤検出する可能性
- **改善案**: 閾値を調整可能にする、または動的に調整

### 2. 着地検出の安定性

- **問題**: 5フレーム連続の条件が厳しすぎる可能性
- **影響**: 着地が検出されない、または遅延する可能性
- **改善案**: フレーム数を調整可能にする

### 3. 基準足首高さの更新

- **問題**: より低い値が見つかった場合に常に更新される
- **影響**: 誤検出やノイズにより基準値が不正確になる可能性
- **改善案**: 更新条件を厳格化（例: 一定期間の最低値のみ更新）

### 4. 初期状態の処理

- **問題**: 初期状態では離陸検出をスキップするが、最初のジャンプを見逃す可能性
- **影響**: 最初のジャンプが検出されない
- **改善案**: 初期状態の判定条件を改善

### 5. ジャンプ高さが0.000mになる問題

- **問題**: ジャンプ高さが常に0.000mになる
- **原因**: 
  - 離陸時の高さと最大高さが同じ
  - 最大高さの更新が正しく行われていない
  - 床検出の精度の問題
- **改善案**: 最大高さの更新ロジックを確認・改善

## 改善提案

### 1. パラメータの調整可能性

- 離陸判定閾値、着地判定閾値、安定性チェックのフレーム数を設定可能にする

### 2. 最大高さの更新ロジック改善

- 最大高さの更新タイミングを確認
- 離陸後の一定期間のみ最大高さを更新（下降開始後は更新しない）

### 3. 基準足首高さの更新条件改善

- 一定期間（例: 10フレーム）の最低値のみ更新
- 外れ値除去を適用

### 4. 初期状態の改善

- 初期状態の判定条件を改善
- 最初のジャンプを見逃さないようにする

### 5. デバッグ情報の追加

- 離陸・着地検出の詳細ログを出力
- 基準足首高さ、現在の足首高さ、判定結果を記録

## まとめ

現在のジャンプ検出ロジックは、床検出ベースの高精度な検出を実装していますが、以下の点で改善の余地があります：

1. **パラメータの調整可能性**: 閾値やフレーム数を設定可能にする
2. **最大高さの更新ロジック**: ジャンプ高さが0.000mになる問題を解決
3. **基準足首高さの更新条件**: より安定した基準値の設定
4. **初期状態の処理**: 最初のジャンプを見逃さないようにする
5. **デバッグ情報の追加**: 検出ロジックの動作を確認できるようにする

