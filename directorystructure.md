# Directory Structure

- `src/`
  - `jump_detector.py`: Jump detection logic, state machine, and metrics calculation
  - `floor_detector.py`: Floor plane detection using RANSAC
  - `kalman_filter_3d.py`: 3D Keypoint smoothing using Kalman Filter
  - `realsense_utils.py`: Intel RealSense camera interface and depth processing utilities
  - `keypoint_smoother.py`: General purpose keypoint smoothing utilities
  - `person_tracker.py`: (Deprecated/Deleted) Person tracking logic
- `config.toml`: Configuration file for parameters
- `METHODOLOGY.md`: Detailed documentation of the measurement methodology
- `DEPTH_INTEGRATION_ANALYSIS.md`: Analysis of depth integration strategies
- `JUMP_HEIGHT_IMPROVEMENTS.md`: Log of improvements in jump height estimation
- `JUMP_DETECTION_LOGIC_STATUS.md`: Status of jump detection logic implementation
- `jump_analyzer.py`: Main entry point for offline video analysis
- `jump_analyzer_front_only.py`: Simplified analyzer for frontal view only

