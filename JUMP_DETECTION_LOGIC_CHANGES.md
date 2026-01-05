# ジャンプ検出ロジックの変更

## 変更内容

### 1. 足首評価の削除

**変更前**:
- 両足首の高さ変化を基準に離着陸を検出
- 基準足首高さから5cm以上上昇した場合に離陸と判定
- 両足が基準値から5cm以内に戻り、5フレーム連続で継続した場合に着地と判定

**変更後**:
- 足首評価を完全に削除
- 腰の高さと基準高さのゼロクロス検出を使用

### 2. ゼロクロス検出の実装

**離陸検出**:
- 負から正へのゼロクロス（deltaが負またはゼロから正になる）
- `delta = 現在の高さ - 基準高さ`
- `prev_state in ("negative", "zero") and current_state == "positive"`

**着地検出**:
- 正から負またはゼロへのゼロクロス（deltaが正から負またはゼロになる）
- `prev_state == "positive" and current_state in ("negative", "zero")`

### 3. 基準高さの設定

**設定方法**:
- `waist_baseline_height`が設定されている場合: その値を使用
- 設定されていない場合: 最初の10フレームの平均高さを基準とする

### 4. 状態遷移の簡略化

**変更前**:
```
ground → takeoff → airborne → landing → ground
```

**変更後**:
```
ground → airborne → landing → ground
```

- `takeoff`状態を削除し、`ground`から直接`airborne`に移行

## 実装の詳細

### ゼロクロス検出のロジック

```python
# ゼロクロス検出: delta = 現在の高さ - 基準高さ
delta = height_above_floor - self.waist_baseline_height
prev_state = self._delta_state(self._prev_waist_delta)
current_state = self._delta_state(delta)
self._prev_waist_delta = delta

# 離陸検出: 負から正へのゼロクロス
if prev_state in ("negative", "zero") and current_state == "positive":
    # 離陸処理

# 着地検出: 正から負またはゼロへのゼロクロス
if prev_state == "positive" and current_state in ("negative", "zero"):
    # 着地処理
```

### `_delta_state()`メソッド

```python
def _delta_state(self, delta):
    """基準点との差分が正か負か（デッドバンド考慮）を判定"""
    if delta is None:
        return "unknown"
    if delta > self.waist_zero_epsilon:
        return "positive"
    if delta < -self.waist_zero_epsilon:
        return "negative"
    return "zero"
```

- `waist_zero_epsilon`: デッドバンド幅（デフォルト: 0.01m = 1cm）
- この範囲内は「zero」として扱い、ノイズによる誤検出を防止

## メリット

1. **シンプルなロジック**: 足首評価の複雑な条件判定が不要
2. **明確な判定基準**: ゼロクロス検出により、離着陸のタイミングが明確
3. **腰の高さを使用**: ジャンプ動作と直接関係があり、足首よりも安定
4. **ノイズ耐性**: デッドバンド（`waist_zero_epsilon`）により、小さな振動を無視

## 注意点

1. **基準高さの設定**: 最初の10フレームの平均高さを基準とするため、立位姿勢で開始する必要がある
2. **デッドバンド**: `waist_zero_epsilon`の値により、検出感度が変わる
3. **初期状態**: 基準高さが設定されるまで（10フレーム）は離陸検出が行われない

## パラメータ

- `waist_baseline_height`: 腰の基準高さ（メートル、設定されていない場合は自動計算）
- `waist_zero_epsilon`: ゼロ判定のデッドバンド幅（メートル、デフォルト: 0.01m = 1cm）

## テスト推奨事項

1. **基準高さの自動計算**: 最初の10フレームが立位姿勢であることを確認
2. **ゼロクロス検出**: 離着陸のタイミングが正確に検出されることを確認
3. **デッドバンド**: 小さな振動が誤検出されないことを確認
4. **ジャンプ高さ計算**: 最大高さが正しく更新されることを確認

