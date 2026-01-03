# Human Body Reconstruction Pipeline - Technical Report

## Overview

This pipeline reconstructs 3D human body models from 2D videos and extracts precise body measurements (circumferences in cm). It uses computer vision, pose estimation, and parametric body modeling to create accurate digital representations of human subjects.

---

## What It Does

**Input**: Video file of a person (front/side view recommended)
**Output**:
- 10 body shape parameters (beta coefficients)
- Body measurements: chest, waist, hips, thigh, calf, bicep circumferences
- 3D body mesh visualization
- Processing takes 2-5 minutes on GPU

---

## Technical Architecture

### Pipeline Stages

```
Video Input
    ↓
1. Pose Detection (MediaPipe)
    ↓
2. Camera Estimation (Geometric Method)
    ↓
3. Body Model Optimization (SMPL-X + PyTorch)
    ↓
4. Measurement Extraction (PCA-based)
    ↓
Body Measurements + 3D Model
```

### Stage Details

#### 1. **Pose Detection**
- **Technology**: MediaPipe Pose Landmarker (Tasks API v0.10.14+)
- **Process**: Extracts 33 2D keypoints per frame (shoulders, hips, knees, etc.)
- **Model**: `pose_landmarker_heavy.task` (auto-downloaded)
- **Output**: 2D pixel coordinates for each body joint

#### 2. **Camera Estimation**
- **Method**: Geometric estimation using similar triangles
- **Formula**: `distance = focal_length × real_height / pixel_height`
- **Why Not PnP**: SMPL-X neutral pose has coplanar joints (all Z ≈ 0), which causes PnP to fail
- **Output**: Camera position, rotation, and intrinsic parameters for each frame

#### 3. **Body Model Optimization**
- **Model**: SMPL-X (parametric human body model)
- **Parameters Optimized**:
  - `beta[10]`: Body shape coefficients (height, weight distribution, proportions)
  - `body_pose[63]`: Joint rotations for pose
- **Optimization**: PyTorch gradient descent with multiple loss functions:
  - **2D Keypoint Reprojection Loss**: Minimizes difference between detected and projected keypoints
  - **Beta Regularization**: Keeps shape parameters realistic
  - **Proportion Loss**: Enforces anatomically correct shoulder/hip ratios
- **Iterations**: 150 steps with Adam optimizer

#### 4. **Measurement Extraction**
- **Method**: PCA-based circumference calculation
- **Process**:
  1. Extract body vertices at specific heights (chest, waist, hips, etc.)
  2. Project vertices onto 2D plane (handles torso tilt)
  3. Compute convex hull perimeter
  4. Convert from meters to centimeters
- **Measurements**: 6 circumferences (chest, waist, hips, left/right thigh, left/right calf, left/right bicep)

---

## Technologies Used

### Core Libraries

| Technology | Version | Purpose |
|------------|---------|---------|
| **MediaPipe** | 0.10.14+ | Pose detection and 2D keypoint extraction |
| **SMPL-X** | Latest | Parametric 3D body model (10,475 vertices) |
| **PyTorch** | Latest | Optimization and gradient descent |
| **OpenCV** | Latest | Video processing and frame extraction |
| **NumPy** | Latest | Matrix operations and linear algebra |
| **SciPy** | Latest | Convex hull computation |
| **scikit-learn** | Latest | PCA for measurement plane projection |
| **Gradio** | Latest | Web UI for file uploads and results |

### Key Models

1. **MediaPipe Pose Landmarker**
   - Pre-trained neural network for human pose estimation
   - 33 keypoints including face, torso, arms, legs
   - Auto-downloads: `pose_landmarker_heavy.task` (50MB)

2. **SMPL-X Neutral Model**
   - Parametric body mesh: 10,475 vertices, 20,908 faces
   - User must upload: `SMPLX_NEUTRAL.npz` (download from [SMPL-X website](https://smpl-x.is.tue.mpg.de/))
   - License: Academic use only, requires registration
   - Shape parameters: 10 PCA coefficients controlling body proportions

---

## Critical Bug Fixed (v2.4)

### The Problem
- Different videos produced **identical measurements**
- Processing took only 5 seconds instead of 2-5 minutes
- Camera estimation was failing completely (0/16 frames successful)

### Root Cause
- **PnP (Perspective-n-Point)** algorithm requires non-coplanar 3D points
- SMPL-X neutral pose has all joints with Z ≈ 0 (flat/coplanar)
- PnP mathematically cannot solve this configuration
- Optimization ran without valid camera data → generic results

### Solution (v2.4)
Replaced PnP with **geometric camera estimation**:

```python
# Estimate distance using similar triangles
body_height_3d = 1.7  # meters (SMPL-X approximate height)
body_height_px = max_y - min_y  # pixels in image

distance = focal_length × body_height_3d / body_height_px

# Compute camera translation from body center
dx = (center_x - image_cx) / focal × distance
dy = (center_y - image_cy) / focal × distance
t = [-dx, -dy, distance]
```

**Benefits**:
- ✅ Works with coplanar points
- ✅ Directly estimates from body proportions
- ✅ More robust for human body reconstruction
- ✅ Produces different results for different videos

---

## System Requirements

### Hardware
- **GPU**: 4-6GB VRAM minimum, 8GB+ recommended
- **RAM**: 8GB minimum
- **Storage**: 500MB for models

### Software
- Python 3.8+
- CUDA-enabled GPU (for PyTorch)
- Linux/macOS/Windows with GPU drivers

### Deployment
- **Current**: Google Colab (free T4 GPU)
- **Interface**: Gradio web UI with public share link
- **Usage**: Upload video + SMPL-X model → get measurements

---

## Pipeline Files

### Main Application
- `gradio_app_v24.py` - Latest version with geometric camera estimation
- `gradio_app_v22.py` - Previous version (PnP-based, deprecated)

### Notebooks
- `human_reconstruction_v22_new_api.ipynb` - Jupyter notebook with MediaPipe Tasks API
- `human_reconstruction_v21_fixed_3.ipynb` - Legacy notebook (old MediaPipe API)

---

## Usage in Colab

```python
# Clone repository
!git clone https://github.com/kvksatish/averse_py_server.git
%cd averse_py_server

# Install dependencies
!pip install -q gradio opencv-python mediapipe torch smplx scipy scikit-learn

# Run Gradio app (creates public URL)
!python gradio_app_v24.py

# Interface will provide:
# - Upload box for video file
# - Upload box for SMPLX_NEUTRAL.npz
# - "Process" button
# - Results: beta parameters + measurements
```

---

## Future Improvements

### Potential Enhancements
1. **Multi-view fusion**: Use multiple camera angles for better accuracy
2. **Automatic SMPL-X download**: Streamline setup (license permitting)
3. **Real-time preview**: Show detected poses during upload
4. **Measurement validation**: Flag unrealistic outputs
5. **Export formats**: 3D mesh export (OBJ, PLY, FBX)

### Known Limitations
1. Requires clear view of full body
2. SMPL-X license restricts commercial use
3. Single-person videos only
4. Processing time varies with video length (2-5 min for 16 frames)

---

## References

- **SMPL-X**: [https://smpl-x.is.tue.mpg.de/](https://smpl-x.is.tue.mpg.de/)
- **MediaPipe Pose**: [https://developers.google.com/mediapipe/solutions/vision/pose_landmarker](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
- **PyTorch**: [https://pytorch.org/](https://pytorch.org/)
- **Gradio**: [https://gradio.app/](https://gradio.app/)

---

**Last Updated**: January 2026
**Version**: 2.4 (Geometric Camera Estimation)
**Status**: Production-ready for testing
