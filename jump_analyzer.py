"""
Jump Analyzer

.bagファイルからYOLOv8-Poseを使用して3D姿勢推定を行い、
ジャンプの高さ・距離・軌跡を測定するメインスクリプト
"""

import argparse
import json
import csv
import os
import sys
import time
from pathlib import Path

try:
    import toml

    TOML_AVAILABLE = True
except ImportError:
    TOML_AVAILABLE = False

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")  # GUI不要のバックエンドを使用
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 日本語フォントの設定を試行
try:
    # よく使われる日本語フォントを探す
    jp_fonts = [
        "Noto Sans CJK JP",
        "TakaoGothic",
        "IPAexGothic",
        "IPAPGothic",
        "VL PGothic",
        "Yu Gothic",
    ]
    for font_name in jp_fonts:
        try:
            font = fm.findfont(fm.FontProperties(family=font_name))
            if font and "DejaVu" not in font:  # デフォルトフォント以外が見つかった場合
                plt.rcParams["font.family"] = font_name
                break
        except:
            continue
    # フォントが見つからない場合は、英語ラベルを使用する設定にする
    current_font = plt.rcParams.get("font.family", ["DejaVu Sans"])
    if isinstance(current_font, list):
        current_font = current_font[0] if current_font else "DejaVu Sans"
    if "DejaVu" in current_font:
        USE_JAPANESE_LABELS = False
    else:
        USE_JAPANESE_LABELS = True
except:
    USE_JAPANESE_LABELS = False

from src.realsense_utils import BagFileReader, CUPY_AVAILABLE
from src.yolov8_pose_3d import YOLOv8PoseDetector, COCO_KEYPOINTS
from src.jump_detector import JumpDetector
from src.visualizer import JumpVisualizer, create_3d_keypoint_animation
from src.keypoint_smoother import KeypointSmoother
from src.kalman_filter_3d import KalmanSmoother
from src.floor_detector import FloorDetector

# CuPyのインポート（CUDA高速化用）
if CUPY_AVAILABLE:
    import cupy as cp

    print("CuPy available: Using CUDA acceleration for image processing")
else:
    print("CuPy not available: Using NumPy (CPU mode)")
    print(
        "  Note: Install CuPy with 'pip install cupy-cuda11x' or 'pip install cupy-cuda12x' for GPU acceleration"
    )


def resize_image(image, new_width, new_height):
    """
    画像をリサイズ

    Args:
        image: 入力画像（NumPy配列）
        new_width: 新しい幅
        new_height: 新しい高さ

    Returns:
        リサイズされた画像（NumPy配列）
    """
    return cv2.resize(image, (new_width, new_height))


def convert_keypoints_to_dict(keypoints_2d, keypoints_3d):
    """
    keypointsを辞書形式に変換

    Args:
        keypoints_2d: 2D keypointsのリスト [(x, y, confidence), ...]
        keypoints_3d: 3D keypointsの辞書 {keypoint_name: (x, y, z), ...}

    Returns:
        dict: keypointsデータの辞書
    """
    result = {}

    for i, keypoint_name in enumerate(COCO_KEYPOINTS):
        if i < len(keypoints_2d):
            kp_2d = keypoints_2d[i]
            kp_3d = keypoints_3d.get(keypoint_name)

            result[keypoint_name] = {
                "2d": {
                    "x": float(kp_2d[0]) if kp_2d[0] is not None else None,
                    "y": float(kp_2d[1]) if kp_2d[1] is not None else None,
                    "confidence": float(kp_2d[2]),
                },
                "3d": {
                    "x": float(kp_3d[0]) if kp_3d and kp_3d[0] is not None else None,
                    "y": float(kp_3d[1]) if kp_3d and kp_3d[1] is not None else None,
                    "z": float(kp_3d[2]) if kp_3d and kp_3d[2] is not None else None,
                },
            }
        else:
            result[keypoint_name] = {
                "2d": {"x": None, "y": None, "confidence": 0.0},
                "3d": {"x": None, "y": None, "z": None},
            }

    return result


def save_json(data, output_path):
    """JSONファイルに保存"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"JSON saved to: {output_path}")


def save_csv(statistics, trajectory, output_path):
    """CSVファイルに保存"""
    # 統計情報をCSVに保存
    stats_path = output_path.replace(".csv", "_statistics.csv")
    with open(stats_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Total Jumps", statistics.get("total_jumps", 0)])
        writer.writerow(["Vertical Jumps", statistics.get("vertical_jumps", 0)])
        writer.writerow(["Horizontal Jumps", statistics.get("horizontal_jumps", 0)])
        writer.writerow(["Max Height (cm)", statistics.get("max_height", 0) * 100])
        writer.writerow(["Max Distance (cm)", statistics.get("max_distance", 0) * 100])
        writer.writerow(["Avg Height (cm)", statistics.get("avg_height", 0) * 100])
        writer.writerow(["Avg Distance (cm)", statistics.get("avg_distance", 0) * 100])
    print(f"Statistics CSV saved to: {stats_path}")

    # ジャンプ詳細をCSVに保存
    if statistics.get("jumps"):
        jumps_path = output_path.replace(".csv", "_jumps.csv")
        with open(jumps_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # 床検出使用時は滞空時間も出力
            headers = [
                "Jump #",
                "Type",
                "Height (cm)",
                "Distance (cm)",
                "Start Frame",
                "Takeoff Frame",
                "End Frame",
                "Duration (frames)",
            ]
            if any("air_time" in jump for jump in statistics["jumps"]):
                headers.append("Air Time (s)")
            writer.writerow(headers)

            for i, jump in enumerate(statistics["jumps"], 1):
                row = [
                    i,
                    jump["jump_type"],
                    jump["height"] * 100,
                    jump["distance"] * 100,
                    jump["frame_start"],
                    jump.get("frame_takeoff", jump["frame_start"]),
                    jump["frame_end"],
                    jump["frame_end"] - jump["frame_start"],
                ]
                if "air_time" in jump:
                    row.append(jump["air_time"])
                writer.writerow(row)
        print(f"Jumps CSV saved to: {jumps_path}")

    # 軌跡データをCSVに保存
    if trajectory:
        trajectory_path = output_path.replace(".csv", "_trajectory.csv")
        with open(trajectory_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Frame", "Timestamp", "X (m)", "Y (m)", "Z (m)"])
            for point in trajectory:
                pos = point.get("position", (None, None, None))
                writer.writerow(
                    [
                        point.get("frame", ""),
                        point.get("timestamp", ""),
                        pos[0] if pos[0] is not None else "",
                        pos[1] if pos[1] is not None else "",
                        pos[2] if pos[2] is not None else "",
                    ]
                )
        print(f"Trajectory CSV saved to: {trajectory_path}")


def plot_keypoint_coordinate_timeline(all_frames_data, output_dir, floor_detector=None):
    """
    全キーポイントのX, Y, Z座標を時系列でプロット（3つのグラフを生成）

    Args:
        all_frames_data: 全フレームデータのリスト
        output_dir: 出力ディレクトリ
        floor_detector: 床検出器（Noneの場合、カメラ座標系のYを使用）
    """
    if not all_frames_data:
        print("Warning: No frame data available for coordinate timeline plot")
        return

    # タイムスタンプと各キーポイントのX, Y, Z座標を収集
    timestamps = []
    keypoint_x = {kp_name: [] for kp_name in COCO_KEYPOINTS}
    keypoint_y = {kp_name: [] for kp_name in COCO_KEYPOINTS}
    keypoint_z = {kp_name: [] for kp_name in COCO_KEYPOINTS}

    # 最初のタイムスタンプを基準に（秒単位に変換）
    first_timestamp = None

    for frame_data in all_frames_data:
        timestamp = frame_data.get("timestamp")
        if timestamp is None:
            continue

        # 最初のタイムスタンプを記録
        if first_timestamp is None:
            first_timestamp = timestamp

        # 経過時間を計算（ミリ秒か秒かを判定）
        if first_timestamp > 1000000000:  # ミリ秒単位と判定
            elapsed_time = (timestamp - first_timestamp) / 1000.0  # 秒に変換
        else:
            elapsed_time = timestamp - first_timestamp

        timestamps.append(elapsed_time)

        # 各キーポイントのX, Y, Z座標を取得
        keypoints = frame_data.get("keypoints", {})
        for kp_name in COCO_KEYPOINTS:
            kp_data = keypoints.get(kp_name, {})
            kp_3d = kp_data.get("3d", {})

            # X座標
            x = kp_3d.get("x") if kp_3d.get("x") is not None else None
            keypoint_x[kp_name].append(x)

            # Y座標（床からの距離が利用可能な場合はそれを使用）
            if (
                floor_detector
                and "distance_to_floor" in kp_data
                and kp_data["distance_to_floor"] is not None
            ):
                y = kp_data["distance_to_floor"]
            elif kp_3d.get("y") is not None:
                y = kp_3d["y"]
            else:
                y = None
            keypoint_y[kp_name].append(y)

            # Z座標
            z = kp_3d.get("z") if kp_3d.get("z") is not None else None
            keypoint_z[kp_name].append(z)

    if not timestamps:
        print("Warning: No valid timestamps found for coordinate timeline plot")
        return

    # カラーマップを準備（キーポイントごとに異なる色）
    try:
        # matplotlib 3.7以降の新しい方法
        from matplotlib import colormaps

        colors = colormaps.get_cmap("tab20")
    except (AttributeError, ImportError):
        # フォールバック
        try:
            colors = plt.get_cmap("tab20")
        except:
            # さらにフォールバック（古い方法）
            from matplotlib import cm

            colors = cm.get_cmap("tab20")

    # X座標のグラフ
    fig, ax = plt.subplots(figsize=(14, 8))
    for i, kp_name in enumerate(COCO_KEYPOINTS):
        x_values = keypoint_x[kp_name]
        valid_data = [(t, x) for t, x in zip(timestamps, x_values) if x is not None]
        if valid_data:
            valid_times, valid_x = zip(*valid_data)
            ax.plot(
                valid_times,
                valid_x,
                label=kp_name,
                color=colors(i),
                alpha=0.7,
                linewidth=1.5,
            )
    if USE_JAPANESE_LABELS:
        ax.set_xlabel("時間 (秒)", fontsize=12, fontweight="bold")
        ax.set_ylabel("X座標 (m)", fontsize=12, fontweight="bold")
        ax.set_title("全キーポイントのX座標（時系列）", fontsize=14, fontweight="bold")
    else:
        ax.set_xlabel("Time (seconds)", fontsize=12, fontweight="bold")
        ax.set_ylabel("X coordinate (m)", fontsize=12, fontweight="bold")
        ax.set_title(
            "All Keypoints X Coordinate Timeline", fontsize=14, fontweight="bold"
        )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=2)
    plt.tight_layout()
    x_path = output_dir / "keypoint_x_timeline.png"
    plt.savefig(str(x_path), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Keypoint X coordinate timeline plot saved to: {x_path}")

    # Y座標のグラフ（高さ）
    fig, ax = plt.subplots(figsize=(14, 8))
    for i, kp_name in enumerate(COCO_KEYPOINTS):
        y_values = keypoint_y[kp_name]
        valid_data = [(t, y) for t, y in zip(timestamps, y_values) if y is not None]
        if valid_data:
            valid_times, valid_y = zip(*valid_data)
            ax.plot(
                valid_times,
                valid_y,
                label=kp_name,
                color=colors(i),
                alpha=0.7,
                linewidth=1.5,
            )
    if USE_JAPANESE_LABELS:
        ax.set_xlabel("時間 (秒)", fontsize=12, fontweight="bold")
        if floor_detector:
            ax.set_ylabel("床からの距離 (m)", fontsize=12, fontweight="bold")
            ax.set_title(
                "全キーポイントの床からの距離（時系列）", fontsize=14, fontweight="bold"
            )
        else:
            ax.set_ylabel("Y座標 (m)", fontsize=12, fontweight="bold")
            ax.set_title(
                "全キーポイントのY座標（時系列）", fontsize=14, fontweight="bold"
            )
    else:
        ax.set_xlabel("Time (seconds)", fontsize=12, fontweight="bold")
        if floor_detector:
            ax.set_ylabel("Distance from Floor (m)", fontsize=12, fontweight="bold")
            ax.set_title(
                "All Keypoints Distance from Floor Timeline",
                fontsize=14,
                fontweight="bold",
            )
        else:
            ax.set_ylabel("Y coordinate (m)", fontsize=12, fontweight="bold")
            ax.set_title(
                "All Keypoints Y Coordinate Timeline", fontsize=14, fontweight="bold"
            )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=2)
    plt.tight_layout()
    y_path = output_dir / "keypoint_y_timeline.png"
    plt.savefig(str(y_path), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Keypoint Y coordinate timeline plot saved to: {y_path}")

    # Z座標のグラフ
    fig, ax = plt.subplots(figsize=(14, 8))
    for i, kp_name in enumerate(COCO_KEYPOINTS):
        z_values = keypoint_z[kp_name]
        valid_data = [(t, z) for t, z in zip(timestamps, z_values) if z is not None]
        if valid_data:
            valid_times, valid_z = zip(*valid_data)
            ax.plot(
                valid_times,
                valid_z,
                label=kp_name,
                color=colors(i),
                alpha=0.7,
                linewidth=1.5,
            )
    if USE_JAPANESE_LABELS:
        ax.set_xlabel("時間 (秒)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Z座標 (m)", fontsize=12, fontweight="bold")
        ax.set_title("全キーポイントのZ座標（時系列）", fontsize=14, fontweight="bold")
    else:
        ax.set_xlabel("Time (seconds)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Z coordinate (m)", fontsize=12, fontweight="bold")
        ax.set_title(
            "All Keypoints Z Coordinate Timeline", fontsize=14, fontweight="bold"
        )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=2)
    plt.tight_layout()
    z_path = output_dir / "keypoint_z_timeline.png"
    plt.savefig(str(z_path), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Keypoint Z coordinate timeline plot saved to: {z_path}")


def plot_jump_trajectory(trajectory, statistics, output_dir, floor_detector=None):
    """
    ジャンプ軌跡を可視化（水平面と高さ-時間グラフ）

    Args:
        trajectory: 軌跡データのリスト
        statistics: 統計情報（ジャンプ検出結果を含む）
        output_dir: 出力ディレクトリ
        floor_detector: 床検出器（オプション）
    """
    if not trajectory:
        print("Warning: No trajectory data available for jump trajectory plot")
        return

    # タイムスタンプを取得（秒単位に変換）
    timestamps = []
    positions_x = []
    positions_y = []
    positions_z = []
    frames = []

    for point in trajectory:
        timestamp = point.get("timestamp")
        pos = point.get("position", (None, None, None))
        frame = point.get("frame")

        if pos[0] is not None and pos[1] is not None and pos[2] is not None:
            # タイムスタンプを秒に変換（ミリ秒単位の可能性がある）
            if timestamp is not None:
                if timestamp > 1000000000:  # ミリ秒単位と判定
                    timestamp_sec = timestamp / 1000.0
                else:
                    timestamp_sec = timestamp
                # 最初のタイムスタンプを0秒に基準化
                if not timestamps:
                    base_timestamp = timestamp_sec
                    timestamps.append(0.0)
                else:
                    timestamps.append(timestamp_sec - base_timestamp)
            else:
                timestamps.append(None)

            positions_x.append(pos[0])
            positions_y.append(pos[1])  # Y軸は高さ（RealSense座標系では下が正）
            positions_z.append(pos[2])
            frames.append(frame)

    if not positions_x:
        print("Warning: No valid trajectory positions found for jump trajectory plot")
        return

    # 日本語ラベル使用可否を確認
    USE_JAPANESE_LABELS = True
    try:
        import matplotlib.font_manager as fm

        japanese_fonts = [
            f.name
            for f in fm.fontManager.ttflist
            if "japan" in f.name.lower()
            or "noto" in f.name.lower()
            or "gothic" in f.name.lower()
        ]
        if not japanese_fonts:
            USE_JAPANESE_LABELS = False
    except:
        USE_JAPANESE_LABELS = False

    # カラーマップを準備（複数のグラフで使用）
    try:
        from matplotlib import colormaps

        colors = colormaps.get_cmap("tab10")
    except (AttributeError, ImportError):
        try:
            colors = plt.get_cmap("tab10")
        except:
            from matplotlib import cm

            colors = cm.get_cmap("tab10")

    # 1. 水平面（XZ平面）での軌跡を描画
    fig, ax = plt.subplots(figsize=(12, 10))

    # 全軌跡を描画（薄いグレー）
    ax.plot(
        positions_x,
        positions_z,
        "gray",
        alpha=0.3,
        linewidth=1,
        label="Full trajectory" if not USE_JAPANESE_LABELS else "全軌跡",
    )

    # ジャンプ中の軌跡を強調
    jumps = statistics.get("jumps", [])
    if jumps:
        for i, jump in enumerate(jumps):
            frame_start = jump.get("frame_start")
            frame_takeoff = jump.get("frame_takeoff", frame_start)
            frame_end = jump.get("frame_end")

            # ジャンプ範囲のインデックスを取得
            jump_indices = []
            for j, frame in enumerate(frames):
                if frame_start is not None and frame_end is not None:
                    if frame_start <= frame <= frame_end:
                        jump_indices.append(j)

            if jump_indices:
                jump_x = [positions_x[idx] for idx in jump_indices]
                jump_z = [positions_z[idx] for idx in jump_indices]
                color = colors(i % 10)

                # ジャンプ軌跡を描画
                ax.plot(
                    jump_x,
                    jump_z,
                    color=color,
                    linewidth=2.5,
                    alpha=0.8,
                    label=(
                        f"Jump {i+1}" if not USE_JAPANESE_LABELS else f"ジャンプ {i+1}"
                    ),
                )

                # 開始点、離陸点、着地点をマーク
                if jump_indices:
                    start_idx = jump_indices[0]
                    takeoff_idx = None
                    end_idx = jump_indices[-1]

                    # 離陸点を探す
                    for idx in jump_indices:
                        if frames[idx] == frame_takeoff:
                            takeoff_idx = idx
                            break

                    # 開始点を描画（緑の円）- 最初のジャンプのみ凡例に追加
                    ax.scatter(
                        [positions_x[start_idx]],
                        [positions_z[start_idx]],
                        c="green",
                        s=100,
                        marker="o",
                        edgecolors="black",
                        linewidths=1.5,
                        zorder=5,
                        label=(
                            "Start"
                            if not USE_JAPANESE_LABELS
                            else "開始" if i == 0 else ""
                        ),
                    )

                    # 離陸点を描画（オレンジの三角）- 最初のジャンプのみ凡例に追加
                    if takeoff_idx is not None:
                        ax.scatter(
                            [positions_x[takeoff_idx]],
                            [positions_z[takeoff_idx]],
                            c="orange",
                            s=100,
                            marker="^",
                            edgecolors="black",
                            linewidths=1.5,
                            zorder=5,
                            label=(
                                "Takeoff"
                                if not USE_JAPANESE_LABELS
                                else "離陸" if i == 0 else ""
                            ),
                        )

                    # 着地点を描画（赤の四角）- 最初のジャンプのみ凡例に追加
                    ax.scatter(
                        [positions_x[end_idx]],
                        [positions_z[end_idx]],
                        c="red",
                        s=100,
                        marker="s",
                        edgecolors="black",
                        linewidths=1.5,
                        zorder=5,
                        label=(
                            "Landing"
                            if not USE_JAPANESE_LABELS
                            else "着地" if i == 0 else ""
                        ),
                    )

    # 軸ラベルとタイトル
    if USE_JAPANESE_LABELS:
        ax.set_xlabel("X座標 (m) - 左右方向", fontsize=12, fontweight="bold")
        ax.set_ylabel("Z座標 (m) - 前後方向", fontsize=12, fontweight="bold")
        ax.set_title("ジャンプ軌跡（水平面）", fontsize=14, fontweight="bold")
    else:
        ax.set_xlabel("X coordinate (m) - Right", fontsize=12, fontweight="bold")
        ax.set_ylabel("Z coordinate (m) - Forward", fontsize=12, fontweight="bold")
        ax.set_title(
            "Jump Trajectory (Horizontal Plane)", fontsize=14, fontweight="bold"
        )

    ax.grid(True, alpha=0.3, linestyle="--")
    # 凡例をグラフの外側（右側）に配置して重なりを避ける
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10, framealpha=0.9)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()

    trajectory_horizontal_path = output_dir / "jump_trajectory_horizontal.png"
    plt.savefig(str(trajectory_horizontal_path), dpi=300, bbox_inches="tight")
    plt.close()
    print(
        f"Jump trajectory (horizontal plane) plot saved to: {trajectory_horizontal_path}"
    )

    # 2. 高さ（Y座標）と時間の関係を描画
    # RealSense座標系ではY軸が下向きが正なので、反転して上向きが正になるようにする
    fig, ax = plt.subplots(figsize=(14, 8))

    # Y座標を反転（RealSense座標系: 下向きが正 → 表示座標系: 上向きが正）
    # 基準値を求める（床面の高さ = 最大Y値）
    if positions_y:
        y_min = min(positions_y)
        y_max = max(positions_y)
        # RealSense座標系ではYが大きいほど下にあるので、最大Y値が床面
        # 床検出が有効な場合は、より正確な床面の高さを使用することも可能
        # ここでは最大Y値を床面として使用（軌跡データの中で最も低い位置）
        y_floor = y_max  # 最大Y値を床面の基準とする（RealSense座標系では下が正）
        # Y座標を反転: 床からの距離として表示（上向きが正、0以上）
        # y_floor - y で計算: yが大きい（下）→0に近い、yが小さい（上）→大きな正の値
        heights = [y_floor - y for y in positions_y]
    else:
        heights = []
        y_floor = 0

    # 有効なタイムスタンプがあるかチェック
    valid_timestamps = [t for t in timestamps if t is not None]
    if valid_timestamps:
        # タイムスタンプを使用
        plot_x = timestamps
        x_label = "Time (seconds)" if not USE_JAPANESE_LABELS else "時間 (秒)"
    else:
        # フレーム番号を使用
        plot_x = frames
        x_label = "Frame number" if not USE_JAPANESE_LABELS else "フレーム番号"

    # 全軌跡を描画（薄いグレー）
    ax.plot(
        plot_x,
        heights,
        "gray",
        alpha=0.3,
        linewidth=1,
        label="Full trajectory" if not USE_JAPANESE_LABELS else "全軌跡",
    )

    # ジャンプ中の軌跡を強調
    if jumps:
        for i, jump in enumerate(jumps):
            frame_start = jump.get("frame_start")
            frame_takeoff = jump.get("frame_takeoff", frame_start)
            frame_end = jump.get("frame_end")
            jump_height = jump.get("height", 0) * 100  # cmに変換

            # ジャンプ範囲のインデックスを取得
            jump_indices = []
            for j, frame in enumerate(frames):
                if frame_start is not None and frame_end is not None:
                    if frame_start <= frame <= frame_end:
                        jump_indices.append(j)

            if jump_indices:
                jump_x = [plot_x[idx] for idx in jump_indices if idx < len(plot_x)]
                jump_heights = [
                    heights[idx] for idx in jump_indices
                ]  # 反転済みの高さを使用
                color = colors(i % 10)

                # ジャンプ軌跡を描画
                ax.plot(
                    jump_x,
                    jump_heights,
                    color=color,
                    linewidth=2.5,
                    alpha=0.8,
                    label=(
                        f"Jump {i+1} ({jump_height:.1f}cm)"
                        if not USE_JAPANESE_LABELS
                        else f"ジャンプ {i+1} ({jump_height:.1f}cm)"
                    ),
                )

                # 開始点、離陸点、着地点をマーク
                if jump_indices:
                    start_idx = jump_indices[0]
                    takeoff_idx = None
                    end_idx = jump_indices[-1]

                    # 離陸点を探す
                    for idx in jump_indices:
                        if frames[idx] == frame_takeoff:
                            takeoff_idx = idx
                            break

                    # 開始点を描画（反転済みの高さを使用）- 最初のジャンプのみ凡例に追加
                    if start_idx < len(plot_x) and start_idx < len(heights):
                        ax.scatter(
                            [plot_x[start_idx]],
                            [heights[start_idx]],
                            c="green",
                            s=100,
                            marker="o",
                            edgecolors="black",
                            linewidths=1.5,
                            zorder=5,
                            label=(
                                "Start"
                                if not USE_JAPANESE_LABELS
                                else "開始" if i == 0 else ""
                            ),
                        )

                    # 離陸点を描画（反転済みの高さを使用）- 最初のジャンプのみ凡例に追加
                    if (
                        takeoff_idx is not None
                        and takeoff_idx < len(plot_x)
                        and takeoff_idx < len(heights)
                    ):
                        ax.scatter(
                            [plot_x[takeoff_idx]],
                            [heights[takeoff_idx]],
                            c="orange",
                            s=100,
                            marker="^",
                            edgecolors="black",
                            linewidths=1.5,
                            zorder=5,
                            label=(
                                "Takeoff"
                                if not USE_JAPANESE_LABELS
                                else "離陸" if i == 0 else ""
                            ),
                        )

                    # 着地点を描画（反転済みの高さを使用）- 最初のジャンプのみ凡例に追加
                    if end_idx < len(plot_x) and end_idx < len(heights):
                        ax.scatter(
                            [plot_x[end_idx]],
                            [heights[end_idx]],
                            c="red",
                            s=100,
                            marker="s",
                            edgecolors="black",
                            linewidths=1.5,
                            zorder=5,
                            label=(
                                "Landing"
                                if not USE_JAPANESE_LABELS
                                else "着地" if i == 0 else ""
                            ),
                        )

    # 軸ラベルとタイトル（反転済みなので上向きが正）
    if USE_JAPANESE_LABELS:
        ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
        ax.set_ylabel("高さ (m) - 床からの距離", fontsize=12, fontweight="bold")
        ax.set_title("ジャンプ軌跡（高さ-時間）", fontsize=14, fontweight="bold")
    else:
        ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
        ax.set_ylabel(
            "Height (m) - Distance from floor", fontsize=12, fontweight="bold"
        )
        ax.set_title("Jump Trajectory (Height-Time)", fontsize=14, fontweight="bold")

    ax.grid(True, alpha=0.3, linestyle="--")
    # 凡例をグラフの外側（右側）に配置して重なりを避ける
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10, framealpha=0.9)
    plt.tight_layout()

    trajectory_height_path = output_dir / "jump_trajectory_height.png"
    plt.savefig(str(trajectory_height_path), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Jump trajectory (height-time) plot saved to: {trajectory_height_path}")


def load_config(config_path):
    """
    設定ファイルを読み込む

    Args:
        config_path: 設定ファイルのパス（TOML形式）

    Returns:
        dict: 設定辞書、読み込み失敗時はNone
    """
    if not TOML_AVAILABLE:
        print("Warning: toml not installed. Install with: pip install toml")
        return None

    config_file = Path(config_path)
    if not config_file.exists():
        return None

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = toml.load(f)
        return config
    except Exception as e:
        print(f"Warning: Failed to load config file {config_path}: {e}")
        return None


def merge_config_with_args(config, args):
    """
    設定ファイルの値をコマンドライン引数で上書き

    Args:
        config: 設定ファイルの辞書
        args: argparse.Namespaceオブジェクト

    Returns:
        argparse.Namespace: マージされた引数オブジェクト
    """
    if config is None:
        return args

    # 設定ファイルの値をデフォルトとして使用（コマンドライン引数が指定されていない場合）
    # 新旧両方の形式に対応（enable_* 形式を優先、後方互換性のため no_* 形式もサポート）

    # 新しいenable_*形式をno_*形式に変換（コマンドライン引数との互換性のため）
    if "enable_video" in config:
        config["no_video"] = not config["enable_video"]
    if "enable_3d_animation" in config:
        config["no_3d_animation"] = not config["enable_3d_animation"]
    if "enable_depth_interpolation" in config:
        config["no_depth_interpolation"] = not config["enable_depth_interpolation"]
    if "enable_floor_detection" in config:
        config["no_floor_detection"] = not config["enable_floor_detection"]

    config_to_args_map = {
        "input": "input",
        "output": "output",
        "model_dir": "model_dir",
        "model_name": "model_name",
        "threshold_vertical": "threshold_vertical",
        "threshold_horizontal": "threshold_horizontal",
        "min_jump_height": "min_jump_height",
        "min_air_time": "min_air_time",
        "no_video": "no_video",
        "no_3d_animation": "no_3d_animation",
        "interactive_3d": "interactive_3d",
        "smooth_keypoints": "smooth_keypoints",
        "smooth_window_size": "smooth_window_size",
        "no_depth_interpolation": "no_depth_interpolation",
        "depth_kernel_size": "depth_kernel_size",
        "use_kalman_filter": "use_kalman_filter",
        "kalman_process_noise": "kalman_process_noise",
        "kalman_measurement_noise": "kalman_measurement_noise",
        "no_floor_detection": "no_floor_detection",
        "start_time": "start_time",
        "end_time": "end_time",
        "frame_skip": "frame_skip",
        "resize_factor": "resize_factor",
        "minimal_data": "minimal_data",
    }

    for config_key, arg_name in config_to_args_map.items():
        if config_key in config and config[config_key] is not None:
            # コマンドライン引数が指定されていない（デフォルト値）場合のみ設定ファイルの値を使用
            # argparseでは、明示的に指定された引数とデフォルト値を区別するのが難しいため、
            # 設定ファイルの値が存在し、かつコマンドライン引数の値がNoneまたはデフォルト値の場合は上書き
            current_value = getattr(args, arg_name, None)

            # smooth_keypointsは特別扱い（設定ファイルではboolean、コマンドラインではwindow_size）
            if config_key == "smooth_keypoints":
                if isinstance(config[config_key], bool):
                    # 設定ファイルでbooleanの場合は、window_sizeを設定
                    if config[config_key]:
                        # smooth_window_sizeが設定されていればそれを使用、なければデフォルト
                        window_size = config.get("smooth_window_size", 5)
                        setattr(args, "smooth_keypoints", window_size)
                    else:
                        setattr(args, "smooth_keypoints", 0)
                else:
                    # 既に整数値の場合はそのまま使用
                    setattr(args, arg_name, config[config_key])
            # start_timeとend_time: 0はNoneとして扱う（最初から/最後まで）
            elif config_key in ["start_time", "end_time"]:
                if config[config_key] == 0:
                    setattr(args, arg_name, None)
                else:
                    setattr(args, arg_name, config[config_key])
            # ブール値の場合は、設定ファイルの値を優先（明示的にFalseでも有効）
            elif config_key in [
                "no_video",
                "no_3d_animation",
                "interactive_3d",
                "no_depth_interpolation",
                "use_kalman_filter",
                "no_floor_detection",
                "minimal_data",
            ]:
                setattr(args, arg_name, config[config_key])
            # Noneがデフォルトの値
            elif current_value is None and config[config_key] is not None:
                setattr(args, arg_name, config[config_key])
            # その他の値は、デフォルト値の可能性がある場合に上書き
            # （簡単のため、設定ファイルの値で上書き。明示的に指定したい場合はコマンドライン引数を使用）
            elif current_value is None or (
                isinstance(current_value, (int, float))
                and current_value == getattr(argparse.Namespace(), arg_name, None)
            ):
                setattr(args, arg_name, config[config_key])

    return args


def main():
    parser = argparse.ArgumentParser(
        description="Jump Analyzer: Analyze jump height, distance, and trajectory from RealSense .bag file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python jump_analyzer.py --input bagdata/my_recording.bag --output results/

  # Use config file
  python jump_analyzer.py --config config.toml
  python jump_analyzer.py --config config.toml --input other_file.bag  # Override input from config

  # Analyze specific time range (seconds)
  python jump_analyzer.py --input bagdata/my_recording.bag --output results/ --start-time 5.0 --end-time 10.0

  # Specify model directory
  python jump_analyzer.py --input bagdata/my_recording.bag --output results/ --model-dir models/

  # Adjust jump detection thresholds
  python jump_analyzer.py --input bagdata/my_recording.bag --output results/ --threshold-vertical 0.1 --threshold-horizontal 0.2
        """,
    )

    # 設定ファイル
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (TOML format). If specified, config values will be used as defaults (command line args override config).",
    )

    # 必須引数（設定ファイルで指定可能なためrequired=False、後でチェック）
    parser.add_argument(
        "--input", type=str, required=False, default=None, help="Input .bag file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        default=None,
        help="Output directory for results",
    )

    # オプション引数
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory for YOLOv8-Pose model files (default: models)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="yolov8x-pose.pt",
        help="YOLOv8-Pose model name (yolov8n-pose.pt, yolov8s-pose.pt, etc.) (default: yolov8x-pose.pt)",
    )
    parser.add_argument(
        "--threshold-vertical",
        type=float,
        default=0.05,
        help="Vertical jump detection threshold in meters (default: 0.05)",
    )
    parser.add_argument(
        "--threshold-horizontal",
        type=float,
        default=0.1,
        help="Horizontal jump detection threshold in meters (default: 0.1)",
    )
    parser.add_argument(
        "--min-jump-height",
        type=float,
        default=0.10,
        help="Minimum jump height in meters to be considered a valid jump (default: 0.10m = 10cm)",
    )
    parser.add_argument(
        "--min-air-time",
        type=float,
        default=0.20,
        help="Minimum air time in seconds to be considered a valid jump (default: 0.20s = 200ms)",
    )
    parser.add_argument(
        "--no-video", action="store_true", help="Skip video visualization output"
    )
    parser.add_argument(
        "--no-3d-animation",
        action="store_true",
        help="Skip 3D keypoint animation generation",
    )
    parser.add_argument(
        "--interactive-3d",
        action="store_true",
        help="Display interactive 3D keypoint animation (can rotate view with mouse)",
    )
    parser.add_argument(
        "--smooth-keypoints",
        type=int,
        default=5,
        metavar="N",
        help="Enable keypoint smoothing with window size N (default: 5, set to 0 to disable)",
    )
    parser.add_argument(
        "--no-depth-interpolation",
        action="store_true",
        help="Disable depth interpolation (use single pixel depth, faster but less accurate)",
    )
    parser.add_argument(
        "--depth-kernel-size",
        type=int,
        default=3,
        metavar="N",
        help="Depth interpolation kernel size (default: 3, must be odd, larger = more smoothing)",
    )
    parser.add_argument(
        "--use-kalman-filter",
        action="store_true",
        help="Use Kalman filter for temporal smoothing (research-grade, more accurate than moving average)",
    )
    parser.add_argument(
        "--kalman-process-noise",
        type=float,
        default=0.01,
        help="Kalman filter process noise (default: 0.01, smaller = smoother but slower response)",
    )
    parser.add_argument(
        "--kalman-measurement-noise",
        type=float,
        default=0.1,
        help="Kalman filter measurement noise (default: 0.1, larger = more trust in predictions)",
    )
    parser.add_argument(
        "--no-floor-detection",
        action="store_true",
        help="Disable floor detection (use traditional height-based detection)",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="Start time for playback in seconds (default: start from beginning)",
    )
    parser.add_argument(
        "--end-time",
        type=float,
        default=None,
        help="End time for playback in seconds (default: play until end)",
    )

    # 高速化オプション
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Process every N frames (1=all frames, 2=every other frame, etc.) (default: 1)",
    )
    parser.add_argument(
        "--resize-factor",
        type=float,
        default=1.0,
        help="Resize image before YOLOv8-Pose inference (1.0=no resize, 0.5=half size) (default: 1.0)",
    )
    parser.add_argument(
        "--minimal-data",
        action="store_true",
        help="Save only jump detection frames in JSON (faster, smaller file)",
    )

    args = parser.parse_args()

    # 設定ファイルを読み込む
    config = None
    if args.config:
        config = load_config(args.config)
        if config:
            print(f"Loaded config from: {args.config}")
            # 設定ファイルの値をマージ（コマンドライン引数が優先）
            args = merge_config_with_args(config, args)
    elif Path("config.toml").exists():
        # デフォルトで config.toml を探す
        config = load_config("config.toml")
        if config:
            print("Loaded config from: config.toml")
            args = merge_config_with_args(config, args)

    # 必須引数のチェック（設定ファイルでも指定可能）
    if args.input is None:
        print(
            "Error: --input is required (can be specified in config file or command line)",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.output is None:
        print(
            "Error: --output is required (can be specified in config file or command line)",
            file=sys.stderr,
        )
        sys.exit(1)

    # 出力ディレクトリを作成
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # メタデータJSONファイルの確認（2台のカメラ録画の場合）
    bag_file_path = args.input
    metadata = None
    if args.input.endswith("_metadata.json"):
        # メタデータJSONが指定された場合
        if not os.path.exists(args.input):
            print(f"Error: Metadata file not found: {args.input}", file=sys.stderr)
            sys.exit(1)

        try:
            with open(args.input, "r") as f:
                metadata = json.load(f)

            # cam0のbagファイルパスを取得
            metadata_dir = os.path.dirname(args.input)
            if not metadata_dir:
                metadata_dir = "."
            bag_file_path = os.path.join(metadata_dir, metadata["cam0_file"])

            if not os.path.exists(bag_file_path):
                print(
                    f"Error: Bag file not found: {bag_file_path}",
                    file=sys.stderr,
                )
                print(
                    f"  Metadata file: {args.input}",
                    file=sys.stderr,
                )
                sys.exit(1)

            print(f"[INFO] Dual camera recording detected")
            print(f"[INFO] Metadata file: {args.input}")
            print(f"[INFO] Using camera 0 bag file: {bag_file_path}")
            print(
                f"[INFO] Calibration fitness: {metadata.get('calibration', {}).get('fitness', 'N/A'):.3f}, RMSE: {metadata.get('calibration', {}).get('inlier_rmse', 'N/A'):.4f}m"
            )

        except Exception as e:
            print(
                f"Error: Failed to load metadata file: {e}",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        # 通常のbagファイルの場合
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}", file=sys.stderr)
            sys.exit(1)

    print("=" * 60)
    print("Jump Analyzer - YOLOv8-Pose 3D Analysis")
    print("=" * 60)
    print(f"Input file: {bag_file_path}")
    print(f"Output directory: {args.output}")
    print()

    # モデルディレクトリを作成（存在しない場合）
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # YOLOv8-Poseモデルの読み込み
    print("Loading YOLOv8-Pose model...")
    yolo_pose = YOLOv8PoseDetector(model_name=args.model_name, model_dir=str(model_dir))
    if not yolo_pose.load_model():
        print("Error: Failed to load YOLOv8-Pose model", file=sys.stderr)
        sys.exit(1)

    # バグファイルの読み込み
    print("Loading .bag file...")
    bag_reader = BagFileReader(
        bag_file_path, start_time=args.start_time, end_time=args.end_time
    )
    if not bag_reader.initialize():
        print("Error: Failed to load .bag file", file=sys.stderr)
        sys.exit(1)

    # メタデータがあれば保存（後で参照可能に）
    if metadata:
        metadata_save_path = output_dir / "recording_metadata.json"
        with open(metadata_save_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[INFO] Metadata saved to: {metadata_save_path}")

    # 床検出器の初期化（オプション）
    floor_detector = None
    if not args.no_floor_detection:
        floor_detector = FloorDetector(
            floor_threshold=0.03,  # 3cm（より緩い閾値で検出しやすく）
            min_inliers=500,  # より少ない点数で検出可能に
            max_iterations=500,
        )
        print(
            "Floor detection enabled: Using foot-floor contact for precise jump detection"
        )
    else:
        print("Floor detection disabled: Using traditional height-based detection")

    # ジャンプ検出器の初期化
    jump_detector = JumpDetector(
        threshold_vertical=args.threshold_vertical,
        threshold_horizontal=args.threshold_horizontal,
        min_jump_height=args.min_jump_height,
        min_air_time=args.min_air_time,
        floor_detector=floor_detector,
        use_floor_detection=not args.no_floor_detection,
    )

    # 可視化器の初期化
    visualizer = JumpVisualizer()

    # キーポイント平滑化器の初期化（Kalmanフィルタまたは移動平均）
    if args.use_kalman_filter:
        smoother = KalmanSmoother(
            process_noise=args.kalman_process_noise,
            measurement_noise=args.kalman_measurement_noise,
        )
        print(
            f"Kalman filter smoothing enabled: process_noise={args.kalman_process_noise}, measurement_noise={args.kalman_measurement_noise}"
        )
    elif args.smooth_keypoints > 0:
        smoother = KeypointSmoother(
            window_size=args.smooth_keypoints, smoothing_type="moving_average"
        )
        print(f"Moving average smoothing enabled: window_size={args.smooth_keypoints}")
    else:
        smoother = None
        print("Keypoint smoothing disabled")

    # 深度補間の設定を表示
    if not args.no_depth_interpolation:
        print(f"Depth interpolation enabled: kernel_size={args.depth_kernel_size}")
    else:
        print("Depth interpolation disabled (using single pixel depth)")

    # データ保存用のリスト
    all_frames_data = []

    # 可視化動画ライター（メモリ効率化のため逐次書き込み）
    video_writer = None
    video_path = None
    video_fps = 30

    print("\nProcessing frames...")
    if args.frame_skip > 1:
        print(f"  Frame skipping: Processing every {args.frame_skip} frame(s)")
    if args.resize_factor < 1.0:
        print(f"  Image resize: {args.resize_factor*100:.0f}% of original size")
    if args.minimal_data:
        print("  Minimal data mode: Saving only jump detection frames")
    print()
    print("Note: First frame processing may take longer (model warm-up)...")
    print()

    frame_num = 0
    processed_frame_count = 0
    skipped_frame_count = 0
    start_time = time.time()
    last_progress_time = start_time
    floor_detection_done = False  # 床検出が完了したかどうか

    try:
        while True:
            # フレームを取得
            color_frame, depth_frame = bag_reader.get_frames()

            if color_frame is None or depth_frame is None:
                break

            color_image, depth_image = bag_reader.frame_to_numpy(
                color_frame, depth_frame
            )

            if color_image is None or depth_image is None:
                continue

            frame_num += 1

            # 床検出（最初の数フレームで実行、フレームスキッピング前）
            if floor_detector and not floor_detection_done:
                # フレームスキッピングを考慮せず、実際の読み込みフレーム数で判定
                if frame_num <= 100:  # 最初の100フレームで床検出を試行
                    # 床検出を試行（毎フレーム試行、成功したら終了）
                    if floor_detector.detect_floor_from_depth(
                        depth_frame,
                        bag_reader.depth_scale,
                        bag_reader.depth_intrinsics or bag_reader.intrinsics,
                    ):
                        floor_plane, floor_normal, floor_height = (
                            floor_detector.get_floor_plane()
                        )
                        if floor_plane is not None:
                            floor_detection_done = True
                            print(f"\n✓ Floor detected at frame {frame_num}!")
                            print(f"  Floor height (Y): {floor_height:.3f}m")
                            print(
                                f"  Floor normal: ({floor_normal[0]:.3f}, {floor_normal[1]:.3f}, {floor_normal[2]:.3f})"
                            )
                elif frame_num > 100:
                    # 100フレーム試しても検出できない場合は警告し、従来方式に切り替え
                    print("\n⚠ Warning: Floor detection failed after 100 frames.")
                    print("  Continuing with traditional height-based detection.")
                    floor_detection_done = True
                    # 床検出を無効化
                    if jump_detector:
                        jump_detector.use_floor_detection = False
                    floor_detector = None

            # フレームスキッピング
            if frame_num % args.frame_skip != 0:
                skipped_frame_count += 1
                continue

            processed_frame_count += 1

            # 画像リサイズ（YOLOv8-Pose推論前）
            original_shape = color_image.shape[:2]
            inference_image = color_image
            resize_scale_x = 1.0
            resize_scale_y = 1.0

            if args.resize_factor < 1.0:
                new_width = int(color_image.shape[1] * args.resize_factor)
                new_height = int(color_image.shape[0] * args.resize_factor)
                inference_image = resize_image(color_image, new_width, new_height)
                resize_scale_x = color_image.shape[1] / new_width
                resize_scale_y = color_image.shape[0] / new_height

            # 2D keypointsを検出（リサイズ済み画像を使用）
            if processed_frame_count == 1:
                print(f"Processing first frame (frame {frame_num})...")

            # ポーズ推定の時間を測定
            pose_start = time.time()
            keypoints_2d = yolo_pose.detect_keypoints(inference_image)
            pose_time = time.time() - pose_start

            # keypoints座標を元の画像サイズにスケール
            if args.resize_factor < 1.0 and keypoints_2d:
                keypoints_2d = [
                    (
                        (kp[0] * resize_scale_x, kp[1] * resize_scale_y, kp[2])
                        if kp[0] is not None and kp[1] is not None
                        else kp
                    )
                    for kp in keypoints_2d
                ]

            if keypoints_2d is None:
                continue

            # 3D keypointsを計算（バッチ処理で高速化）
            depth_start = time.time()

            # 有効なkeypointsの座標を収集
            # 研究用途: 画像端のキーポイントは信頼度が低い（OpenPoseの手法を参考）
            image_height, image_width = color_image.shape[:2]
            border_threshold = 8  # 画像端から8ピクセル以内は除外

            valid_data = [
                (kp_name, kp_2d[0], kp_2d[1], kp_2d[2])  # confidenceも含める
                for kp_name, kp_2d in zip(COCO_KEYPOINTS, keypoints_2d)
                if (
                    kp_2d[0] is not None
                    and kp_2d[1] is not None
                    and kp_2d[2] > 0.1  # 信頼度閾値
                    and border_threshold <= kp_2d[0] < image_width - border_threshold
                    and border_threshold <= kp_2d[1] < image_height - border_threshold
                )
            ]

            # バッチ処理で深度値を一括取得（研究用途向け高精度処理）
            # 信頼度に基づく適応的補間のため、信頼度も渡す
            if valid_data:
                valid_points = [(x, y) for _, x, y, _ in valid_data]
                # キーポイントの信頼度を取得
                kp_confidences = [conf for _, _, _, conf in valid_data]
                depths = bag_reader.get_depth_at_points_batch(
                    depth_frame,
                    valid_points,
                    use_interpolation=not args.no_depth_interpolation,
                    kernel_size=(
                        args.depth_kernel_size if args.depth_kernel_size % 2 == 1 else 3
                    ),
                    confidences=kp_confidences,
                )
                coords_3d = bag_reader.pixels_to_3d_batch(valid_points, depths)

                # 結果を辞書に格納
                valid_coords_dict = {
                    kp_name: (
                        coords_3d[i] if coords_3d[i] is not None else (None, None, None)
                    )
                    for i, (kp_name, _, _, _) in enumerate(valid_data)
                }
                keypoints_3d = {
                    kp_name: valid_coords_dict.get(kp_name, (None, None, None))
                    for kp_name in COCO_KEYPOINTS
                }
            else:
                keypoints_3d = {
                    kp_name: (None, None, None) for kp_name in COCO_KEYPOINTS
                }

            # キーポイント平滑化を適用
            if smoother is not None:
                keypoints_3d = smoother.smooth(keypoints_3d)

            depth_time = time.time() - depth_start

            # 最初の数フレームで時間を表示
            if processed_frame_count <= 5:
                total_time = pose_time + depth_time
                print(
                    f"Frame {processed_frame_count}: "
                    f"Pose={pose_time:.3f}s ({pose_time/total_time*100:.1f}%), "
                    f"3D={depth_time:.3f}s ({depth_time/total_time*100:.1f}%)"
                )

            # ジャンプ検出
            timestamp = color_frame.get_timestamp()
            jump_result = jump_detector.update(
                processed_frame_count, keypoints_3d, timestamp
            )

            # フレームデータを保存（minimal_dataモードではジャンプ検出時のみ）
            should_save = not args.minimal_data or (
                jump_result is not None
                and jump_result.get("state") in ["jump_start", "jump_end", "jumping"]
            )

            if should_save:
                # キーポイントの辞書を作成
                keypoints_dict = convert_keypoints_to_dict(keypoints_2d, keypoints_3d)

                # 床からの距離をすべてのキーポイントについて追加（床検出が有効な場合）
                if floor_detector and floor_detector.floor_plane is not None:
                    for kp_name, kp_data in keypoints_dict.items():
                        if kp_data.get("3d") and kp_data["3d"]["x"] is not None:
                            kp_coords = (
                                kp_data["3d"]["x"],
                                kp_data["3d"]["y"],
                                kp_data["3d"]["z"],
                            )
                            distance = floor_detector.distance_to_floor(kp_coords)
                            kp_data["distance_to_floor"] = distance
                        else:
                            kp_data["distance_to_floor"] = None

                frame_data = {
                    "frame": frame_num,
                    "processed_frame": processed_frame_count,
                    "timestamp": timestamp,
                    "keypoints": keypoints_dict,
                }

                if jump_result:
                    frame_data["jump_result"] = {
                        "state": jump_result.get("state", "unknown"),
                        "height": jump_result.get("height"),
                        "position": jump_result.get("position"),
                        "jump_type": jump_result.get("jump_type"),
                        "jump_height": jump_result.get("jump_height"),
                        "jump_distance": jump_result.get("jump_distance"),
                    }

                all_frames_data.append(frame_data)

            # 可視化フレームを生成
            if not args.no_video:
                # 統計情報を取得
                statistics = jump_detector.get_statistics()

                # 可視化フレームを生成（キーポイントとスケルトンのみ）
                vis_frame = visualizer.draw_frame(
                    color_image, keypoints_2d, jump_result, statistics
                )

                # 動画ライターを初期化
                if video_writer is None:
                    video_path = output_dir / "jump_visualization.mp4"
                    height, width = vis_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        str(video_path), fourcc, video_fps, (width, height)
                    )
                    print(f"Initialized video writer: {video_path}")

                video_writer.write(vis_frame)

            # 進捗表示
            current_time = time.time()
            if (
                processed_frame_count == 1
                or processed_frame_count % 10 == 0
                or (current_time - last_progress_time) >= 5.0
            ):
                elapsed_time = current_time - start_time
                if processed_frame_count > 0 and elapsed_time > 0:
                    fps = processed_frame_count / elapsed_time
                    print(
                        f"Processed {processed_frame_count} frames (total frames read: {frame_num}, skipped: {skipped_frame_count}) | "
                        f"Speed: {fps:.1f} fps | Elapsed: {elapsed_time:.0f}s"
                    )
                    last_progress_time = current_time

        elapsed_time = time.time() - start_time
        print(
            f"\nFinished processing {processed_frame_count} frames (total: {frame_num}, skipped: {skipped_frame_count})"
        )
        print(f"Total time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
        if processed_frame_count > 0:
            print(f"Average speed: {processed_frame_count/elapsed_time:.1f} fps")

        # 統計情報を取得
        statistics = jump_detector.get_statistics()
        trajectory = jump_detector.get_trajectory()

        print("\n" + "=" * 60)
        print("Analysis Results")
        print("=" * 60)
        print(f"Total jumps detected: {statistics['total_jumps']}")
        print(f"  - Vertical jumps: {statistics['vertical_jumps']}")
        print(f"  - Horizontal jumps: {statistics['horizontal_jumps']}")
        print(f"Max height: {statistics['max_height'] * 100:.1f} cm")
        print(f"Max distance: {statistics['max_distance'] * 100:.1f} cm")
        print(f"Average height: {statistics['avg_height'] * 100:.1f} cm")
        print(f"Average distance: {statistics['avg_distance'] * 100:.1f} cm")
        if statistics.get("max_air_time", 0) > 0:
            print(
                f"Max air time: {statistics['max_air_time'] * 1000:.1f} ms ({statistics['max_air_time']:.3f} s)"
            )
        if statistics.get("avg_air_time", 0) > 0:
            print(
                f"Average air time: {statistics.get('avg_air_time', 0) * 1000:.1f} ms ({statistics.get('avg_air_time', 0):.3f} s)"
            )
        print()

        # 検出されたジャンプの詳細情報を表示
        if statistics.get("jumps"):
            print("=" * 60)
            print("Detected Jump Details:")
            print("=" * 60)
            for i, jump in enumerate(statistics["jumps"], 1):
                print(f"\nJump #{i}:")
                print(f"  Type: {jump['jump_type']}")
                print(f"  Height: {jump['height'] * 100:.1f} cm")
                print(f"  Distance: {jump['distance'] * 100:.1f} cm")
                if jump.get("air_time"):
                    print(
                        f"  Air time: {jump['air_time'] * 1000:.1f} ms ({jump['air_time']:.3f} s)"
                    )
                print(
                    f"  Frames: {jump.get('frame_start', 'N/A')} → {jump.get('frame_takeoff', 'N/A')} → {jump.get('frame_end', 'N/A')}"
                )
                if jump.get("start_position") and jump.get("end_position"):
                    start_pos = jump["start_position"]
                    end_pos = jump["end_position"]
                    if (
                        start_pos
                        and end_pos
                        and len(start_pos) >= 2
                        and len(end_pos) >= 2
                    ):
                        # XZ平面での距離を計算
                        x_diff = end_pos[0] - start_pos[0]
                        z_diff = (
                            end_pos[2] - start_pos[2]
                            if len(end_pos) > 2 and len(start_pos) > 2
                            else 0
                        )
                        actual_distance = np.sqrt(x_diff**2 + z_diff**2) * 100
                        print(
                            f"  Start position: ({start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2] if len(start_pos) > 2 else 'N/A'})"
                        )
                        print(
                            f"  End position: ({end_pos[0]:.3f}, {end_pos[1]:.3f}, {end_pos[2] if len(end_pos) > 2 else 'N/A'})"
                        )
                        print(f"  Actual XZ distance: {actual_distance:.1f} cm")
                print(f"  Reported distance: {jump['distance'] * 100:.1f} cm")
            print("=" * 60)
            print()

        # キーポイント変動性分析（床検出が有効な場合）
        if floor_detector and not args.no_floor_detection:
            print("=" * 60)
            print("キーポイント変動性分析")
            print("=" * 60)

            # 最もジャンプ検出に敏感なキーポイントを特定
            sensitive_keypoints = jump_detector.get_most_jump_sensitive_keypoints(
                min_samples=10, top_n=10
            )

            if sensitive_keypoints:
                print(
                    "\nジャンプ検出に最も敏感なキーポイント（歩行時との違いが明確な順）:"
                )
                print(
                    f"{'キーポイント':<15} {'歩行時SD':<10} {'ジャンプ時SD':<12} {'変動比':<10} "
                    f"{'検出感度':<10} {'最大変位(m)':<12} {'サンプル数':<10}"
                )
                print("-" * 95)
                for kp_name, stats in sensitive_keypoints:
                    walking_std_str = (
                        f"{stats['walking_std']:.3f}"
                        if stats["walking_std"] is not None
                        else "N/A"
                    )
                    jumping_std_str = (
                        f"{stats['jumping_std']:.3f}"
                        if stats["jumping_std"] is not None
                        else "N/A"
                    )
                    ratio_str = (
                        f"{stats['jump_walk_ratio']:.2f}"
                        if stats["jump_walk_ratio"] is not None
                        else "N/A"
                    )
                    sensitivity_str = (
                        f"{stats['jump_sensitivity']:.3f}"
                        if stats["jump_sensitivity"] is not None
                        else "N/A"
                    )

                    print(
                        f"{kp_name:<15} {walking_std_str:<10} {jumping_std_str:<12} {ratio_str:<10} "
                        f"{sensitivity_str:<10} {stats['range']:<12.3f} "
                        f"({stats['walking_samples']}/{stats['jumping_samples']})"
                    )

                # CSV出力
                import csv

                variability_csv_path = output_dir / "keypoint_variability.csv"
                with open(variability_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "Keypoint",
                            "Mean (m)",
                            "Std (m)",
                            "Min (m)",
                            "Max (m)",
                            "Range (m)",
                            "CV",
                            "Valid Samples",
                            "Walking Mean (m)",
                            "Walking Std (m)",
                            "Walking Samples",
                            "Jumping Mean (m)",
                            "Jumping Std (m)",
                            "Jumping Samples",
                            "Jump/Walk Ratio",
                            "Jump Sensitivity",
                        ]
                    )
                    all_stats = jump_detector.analyze_keypoint_variability()
                    for kp_name in sorted(all_stats.keys()):
                        stats = all_stats[kp_name]
                        writer.writerow(
                            [
                                kp_name,
                                f"{stats['mean']:.4f}",
                                f"{stats['std']:.4f}",
                                f"{stats['min']:.4f}",
                                f"{stats['max']:.4f}",
                                f"{stats['range']:.4f}",
                                f"{stats['cv']:.4f}",
                                stats["valid_samples"],
                                (
                                    f"{stats['walking_mean']:.4f}"
                                    if stats["walking_mean"] is not None
                                    else ""
                                ),
                                (
                                    f"{stats['walking_std']:.4f}"
                                    if stats["walking_std"] is not None
                                    else ""
                                ),
                                stats["walking_samples"],
                                (
                                    f"{stats['jumping_mean']:.4f}"
                                    if stats["jumping_mean"] is not None
                                    else ""
                                ),
                                (
                                    f"{stats['jumping_std']:.4f}"
                                    if stats["jumping_std"] is not None
                                    else ""
                                ),
                                stats["jumping_samples"],
                                (
                                    f"{stats['jump_walk_ratio']:.4f}"
                                    if stats["jump_walk_ratio"] is not None
                                    else ""
                                ),
                                f"{stats['jump_sensitivity']:.4f}",
                            ]
                        )
                print(f"\nキーポイント変動性CSV saved to: {variability_csv_path}")

                # ジャンプ遷移分析（始まりと終わりを捉えるキーポイント）
                print("=" * 60)
                print("ジャンプ遷移分析（始まりと終わりを捉えるキーポイント）")
                print("=" * 60)

                transitions = jump_detector.analyze_jump_transitions()

                if transitions["takeoff_keypoints"]:
                    print("\n【離陸（ジャンプ開始）時に急激に変化するキーポイント】:")
                    print(
                        f"{'キーポイント':<15} {'平均変化量(m)':<15} {'標準偏差(m)':<15} {'スコア':<12} {'サンプル数':<10}"
                    )
                    print("-" * 75)
                    for kp_name, stats in transitions["takeoff_keypoints"][:10]:
                        print(
                            f"{kp_name:<15} {stats['avg_change']:<15.4f} {stats['std_change']:<15.4f} "
                            f"{stats['score']:<12.2f} {stats['samples']:<10}"
                        )
                else:
                    print("\n【離陸時の分析】: データが不足しています")

                if transitions["landing_keypoints"]:
                    print("\n【着地（ジャンプ終了）時に急激に変化するキーポイント】:")
                    print(
                        f"{'キーポイント':<15} {'平均変化量(m)':<15} {'標準偏差(m)':<15} {'スコア':<12} {'サンプル数':<10}"
                    )
                    print("-" * 75)
                    for kp_name, stats in transitions["landing_keypoints"][:10]:
                        print(
                            f"{kp_name:<15} {stats['avg_change']:<15.4f} {stats['std_change']:<15.4f} "
                            f"{stats['score']:<12.2f} {stats['samples']:<10}"
                        )
                else:
                    print("\n【着地時の分析】: データが不足しています")

                print()
            else:
                print("キーポイント変動性分析: 十分なデータがありません")
            print()

        # JSONファイルに保存
        json_output = {"frames": all_frames_data, "statistics": statistics}
        json_path = output_dir / "keypoints_3d.json"
        save_json(json_output, str(json_path))

        # キーポイントのX, Y, Z座標時系列グラフを作成
        plot_keypoint_coordinate_timeline(all_frames_data, output_dir, floor_detector)

        # ジャンプ軌跡の可視化画像を作成
        plot_jump_trajectory(trajectory, statistics, output_dir, floor_detector)

        # CSVファイルに保存
        csv_path = output_dir / "jump_statistics.csv"
        save_csv(statistics, trajectory, str(csv_path))

        # 可視化動画を完成
        if not args.no_video and video_writer is not None:
            video_writer.release()
            print(f"Video saved to: {video_path}")

        # 3Dキーポイントアニメーションを生成
        if not args.no_3d_animation:
            print("\nGenerating 3D keypoint animation...")
            animation_path = output_dir / "keypoints_3d_animation.gif"
            if args.interactive_3d:
                # インタラクティブモードで表示 + ファイルも保存
                if create_3d_keypoint_animation(
                    str(json_path), str(animation_path), fps=30, interactive=True
                ):
                    if Path(animation_path).exists():
                        print(f"3D keypoint animation saved to: {animation_path}")
                else:
                    print("Warning: Failed to create 3D keypoint animation")
            else:
                # ファイルとして保存
                if create_3d_keypoint_animation(
                    str(json_path), str(animation_path), fps=30, interactive=False
                ):
                    print(f"3D keypoint animation saved to: {animation_path}")
                else:
                    print("Warning: Failed to create 3D keypoint animation")

        print("\n" + "=" * 60)
        print("Analysis complete!")
        print(f"Results saved to: {args.output}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during analysis: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        bag_reader.stop()


if __name__ == "__main__":
    main()
