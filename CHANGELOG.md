# 変更履歴

## 2024-11-22: カスタムトラッカーの削除と高さ計算の改善

### 削除
- ✅ `src/person_tracker.py` - カスタムトラッカー（IoU + キーポイントベース）を完全に削除
- ✅ `--use-norfair-tracking` フラグ - 常にNorFairトラッキングを使用するように変更

### 改善
- ✅ **高さ計算の改善**: `src/floor_detector.py` の `distance_to_floor()` メソッドを改善
  - 床の法線ベクトルを明示的に使用して高さを計算
  - 点から床平面への垂線の長さを正確に計算
  - 法線ベクトルが利用可能な場合はそれを使用（より正確）

### 変更内容

#### `src/floor_detector.py`
- `distance_to_floor()` メソッドを改善
  - 床の法線ベクトル (`self.floor_normal`) を明示的に使用
  - 点から床平面への垂線の長さを計算（法線ベクトル方向への射影）
  - フォールバック: 法線ベクトルが利用できない場合は従来の方法を使用

#### `jump_analyzer.py`
- `PersonTracker` のインポートを削除
- `--use-norfair-tracking` フラグを削除
- 常にNorFairトラッキングを使用するように変更
- `args.use_norfair_tracking` の参照をすべて削除

### 高さ計算の改善詳細

**改善前**:
```python
distance = abs(a * x + b * y + c * z + d) / np.sqrt(a**2 + b**2 + c**2)
```

**改善後**:
```python
if self.floor_normal is not None:
    # 法線ベクトルが利用可能な場合はそれを使用（より正確）
    normal = self.floor_normal
    point_on_plane = np.array([0, -d / b if abs(b) > 1e-6 else 0, 0]) if abs(b) > 1e-6 else np.array([0, 0, -d / c if abs(c) > 1e-6 else 0])
    point_vec = np.array([x, y, z]) - point_on_plane
    # 法線ベクトル方向への射影（高さ）
    distance = np.abs(np.dot(point_vec, normal))
else:
    # フォールバック: 従来の方法
    distance = abs(a * x + b * y + c * z + d) / np.sqrt(a**2 + b**2 + c**2)
```

これにより、床の法線ベクトルを明示的に使用して高さを計算するようになりました。

### 今後の改善案

1. **より精度の高いトラッカーの実装**
   - BoT-SORTやByteTrackのアルゴリズムを参考にした改善
   - または、より高度なトラッキング手法の統合

2. **トラッキングのさらなる改善**
   - トラックの統合機能（同一人物の複数IDを統合）
   - より厳格な初期化条件

