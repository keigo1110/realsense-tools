"""
Dual Camera Integrated Runner

1) Run calibration once (non-interactive). If it fails, print error and exit.
2) If calibration succeeds, start dual recording with preview waiting for 's' to start.

Usage:
    python dual_run.py
    python dual_run.py --calibration-file extrinsics_cam1_to_cam0.json --output my_record
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime


def run(cmd):
    proc = subprocess.run(cmd, stdout=None, stderr=None)
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Dual Camera Integrated Workflow: calibrate then record",
    )
    parser.add_argument(
        "--calibration-file",
        type=str,
        default="extrinsics_cam1_to_cam0.json",
        help="Path to save/load calibration file",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=None,
        help="Override voxel size for calibration (optional)",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=None,
        help="Override max depth for calibration (optional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Base name for recording output (passed to recorder)",
    )

    args = parser.parse_args()

    # Decide base_name and recording dir (bagdata/<base_name>)
    if args.output:
        base_name = args.output[:-4] if args.output.endswith(".bag") else args.output
    else:
        base_name = f"dual_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    record_root = os.path.join(os.path.dirname(__file__), "bagdata")
    record_dir = os.path.join(record_root, base_name)
    os.makedirs(record_dir, exist_ok=True)

    # 1) Calibration (once)
    print("=" * 60)
    print("Dual Workflow: Calibration phase")
    print("=" * 60)

    calib_cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "calibrate_dual_cameras.py"),
        "--once",
        "--calibration-file",
        os.path.join(record_dir, os.path.basename(args.calibration_file)),
    ]
    if args.voxel_size is not None:
        calib_cmd += ["--voxel-size", str(args.voxel_size)]
    if args.max_depth is not None:
        calib_cmd += ["--max-depth", str(args.max_depth)]

    rc = run(calib_cmd)
    if rc != 0:
        print("\n[ERROR] Calibration failed. See above log.")
        sys.exit(rc)

    print("\n[INFO] Calibration complete. Proceeding to recording.")

    # 2) Recording with preview wait
    print("=" * 60)
    print("Dual Workflow: Recording phase")
    print("=" * 60)

    rec_cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "record_dual_cameras.py"),
        "--calibration-file",
        os.path.join(record_dir, os.path.basename(args.calibration_file)),
        "--wait-start",
    ]
    # Recorder will derive the same base_name and dir when provided
    rec_cmd += ["--output", base_name]

    rc = run(rec_cmd)
    sys.exit(rc)


if __name__ == "__main__":
    main()


