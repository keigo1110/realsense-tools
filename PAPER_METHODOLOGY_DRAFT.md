# Proposed Methodology Section for Paper

本研究で開発したジャンプ計測システムの計測手法について記述するためのドラフトです。論文の「Methodology（手法）」セクションとして使用することを想定しています。

---

## 3. Proposed Method

### 3.1 System Overview
本システムは、深度カメラ（Intel RealSense D455）と深層学習ベースの姿勢推定モデル（YOLOv8-Pose）を統合し、非接触で三次元的なジャンプ動作解析を行う。システムは、(1) 深度画像とカラー画像の同期取得、(2) 2次元骨格検出、(3) 深度情報の統合による3次元骨格再構成、(4) 床平面の推定、(5) ジャンプイベントの検出と運動学的パラメータの算出、の5段階で構成される。

### 3.2 3D Skeleton Reconstruction
人体の3次元骨格座標 $P_{3D}^{(i)} = (x_i, y_i, z_i)$ （$i$は関節インデックス）は、カラー画像上の2次元キーポイント座標 $(u_i, v_i)$ と、対応する深度値 $d_i$ を用いてピンホールカメラモデルにより算出される。

$$
\begin{cases}
x_i = \frac{(u_i - c_x) \cdot d_i}{f_x} \\
y_i = \frac{(v_i - c_y) \cdot d_i}{f_y} \\
z_i = d_i
\end{cases}
$$

ここで、$(f_x, f_y)$ はカメラの焦点距離、$(c_x, c_y)$ は主点座標である。
深度値 $d_i$ の取得において、単一ピクセルのノイズや欠損の影響を軽減するため、キーポイント周辺 $5 \times 5$ ピクセル領域の深度値をサンプリングし、四分位範囲（IQR）法による外れ値除去を行った後、中央値（Median）を採用する空間フィルタリング処理を適用している。さらに、時系列的なノイズに対しては、等速運動モデルを仮定したKalmanフィルタを各関節の3次元座標に適用し、平滑化を行っている。

### 3.3 Floor Plane Detection
ジャンプ高さを正確に計測するためには、基準となる床平面の頑健な推定が不可欠である。本手法では、RANSAC (Random Sample Consensus) アルゴリズムを用いて深度点群から床平面を推定する。
床平面の方程式を $ax + by + cz + d = 0$ と定義し、以下の手順でパラメータ $(a, b, c, d)$ を決定する。

1. 深度画像から $0.3\text{m} < z < 5.0\text{m}$ の範囲にある点群をサンプリングする。
2. ランダムに3点を選択して仮の平面を生成し、平面からの距離が閾値 $\tau_{floor}$ (3cm) 以内にある点（インライア）の数を数える。
3. この試行を $N$ 回繰り返し、インライア数が最大となる平面モデルを採用する。
4. 推定の安定性を向上させるため、過去 $K$ フレーム ($K=20$) の平面パラメータの加重移動平均を用いて、最終的な床平面を決定する。

任意の3次元点 $P(x, y, z)$ の床からの高さ $H_{floor}$ は、点と平面の距離公式により算出される。

$$
H_{floor} = \frac{|ax + by + cz + d|}{\sqrt{a^2 + b^2 + c^2}}
$$

### 3.4 Jump Event Detection Algorithm
ジャンプ動作の検出には、腰（Mid-Hip: 左右股関節の中点）の床からの高さ $h_{hip}(t)$ の時間変化に基づく状態遷移モデルを採用した。動作状態は **Ground (接地)**, **Takeoff (離陸)**, **Airborne (空中)**, **Landing (着地)** の4状態として定義される。

離陸および着地のタイミング検出には、ゼロクロス検出法を拡張したアルゴリズムを用いる。
まず、静止立位時の腰の高さの平均値を基準高さ $h_{base}$ として初期化する。各時刻 $t$ における基準高さからの変位 $\Delta h(t) = h_{hip}(t) - h_{base}$ を監視し、以下の条件でイベントを判定する。

*   **Takeoff Detection**: $\Delta h(t)$ が負またはゼロから正に転じ、かつその上昇トレンドが閾値 $\tau_{vert}$ (5cm) を超えた時点を離陸フレーム $t_{takeoff}$ とする。正確な離陸時刻は、前後のフレーム間の線形補間によりサブフレーム精度で算出する。
*   **Landing Detection**: 空中状態において、$\Delta h(t)$ が正から負またはゼロに転じた時点を着地フレーム $t_{landing}$ とする。

### 3.5 Calculation of Kinematic Parameters
検出されたジャンプイベントに基づき、以下の指標を算出する。

1.  **Jump Height ($H_{jump}$)**:
    離陸時の腰の高さ $h_{hip}(t_{takeoff})$ を基準とした、空中フェーズにおける腰の最大到達高さとして定義する。
    $$
    H_{jump} = \max_{t \in [t_{takeoff}, t_{landing}]} (h_{hip}(t)) - h_{hip}(t_{takeoff})
    $$

2.  **Jump Distance ($D_{jump}$)**:
    離陸位置 $(x_{to}, z_{to})$ から着地位置 $(x_{land}, z_{land})$ までの水平面（XZ平面）上のユークリッド距離。
    $$
    D_{jump} = \sqrt{(x_{land} - x_{to})^2 + (z_{land} - z_{to})^2}
    $$

3.  **Air Time ($T_{air}$)**:
    着地時刻と離陸時刻の差分。
    $$
    T_{air} = t_{landing} - t_{takeoff}
    $$

誤検出を排除するため、$H_{jump} < 20\text{cm}$ または $T_{air} < 200\text{ms}$ のイベントは解析から除外する処理を行っている。

---

## 補足：実装パラメータ設定 (Implementation Details)

論文の実験設定（Experimental Setup）などに記載する場合の参考値です。

| Parameter | Value | Description |
|-----------|-------|-------------|
| Camera Resolution | 1280 x 720 | Depth & Color resolution |
| Frame Rate | 30 fps | Capture frame rate |
| Depth Interpolation Kernel | 5 x 5 | Kernel size for spatial median filter |
| Floor RANSAC Threshold | 3 cm | Distance threshold for inlier detection |
| RANSAC Iterations | 500 | Max iterations for plane estimation |
| Jump Height Threshold | 20 cm | Minimum height to be considered a jump |
| Air Time Threshold | 200 ms | Minimum duration to be considered a jump |
| Smoothing | Kalman Filter | Constant velocity model applied to 3D points |


