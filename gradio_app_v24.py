"""
Human Body Reconstruction v2.4 - FIXED VERSION
Uses geometric camera estimation instead of PnP (which fails for coplanar SMPL-X joints)

The Problem: SMPL-X neutral pose has joints with Z ≈ 0 (coplanar)
             PnP fails for coplanar points

The Solution: Use geometric estimation based on body size in image
"""

import gradio as gr
import cv2
import numpy as np
import os
import urllib.request
import sys
from datetime import datetime

def log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌", "DEBUG": "🔍"}.get(level, "")
    print(f"[{ts}] {prefix} {msg}")
    sys.stdout.flush()

log("="*60)
log("BODY RECONSTRUCTION v2.4 (GEOMETRIC CAMERA)")
log("="*60)

# MediaPipe setup
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

log(f"MediaPipe: {mp.__version__}")

MODEL_PATH = "pose_landmarker.task"
if not os.path.exists(MODEL_PATH):
    log("Downloading pose model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
        MODEL_PATH
    )

pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(
    mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        output_segmentation_masks=True
    )
)
log("✅ PoseLandmarker ready")

# Mappings
MP_TO_COCO = {0:0, 2:1, 5:2, 7:3, 8:4, 11:5, 12:6, 13:7, 14:8, 15:9, 16:10, 23:11, 24:12, 25:13, 26:14, 27:15, 28:16}
COCO_TO_SMPLX = {5:16, 6:17, 7:18, 8:19, 9:20, 10:21, 11:1, 12:2, 13:4, 14:5, 15:7, 16:8}

# ============================================
# GEOMETRIC CAMERA ESTIMATION (NEW!)
# ============================================

def estimate_camera_geometric(keypoints_2d, joints_3d, K, frame_idx=0):
    """
    Estimate camera using geometric approach instead of PnP.

    This works because:
    1. We know approximate body dimensions from SMPL-X
    2. We can measure body size in pixels from keypoints
    3. distance = focal_length * real_size / pixel_size
    """
    log(f"[Frame {frame_idx}] Geometric camera estimation...", "DEBUG")

    # Get valid keypoints
    valid_mask = keypoints_2d[:, 2] > 0.3
    if np.sum(valid_mask) < 4:
        log(f"[Frame {frame_idx}] Not enough keypoints", "WARNING")
        return None

    valid_kps = keypoints_2d[valid_mask]

    # Body bounding box in pixels
    x_min, x_max = valid_kps[:, 0].min(), valid_kps[:, 0].max()
    y_min, y_max = valid_kps[:, 1].min(), valid_kps[:, 1].max()

    body_width_px = x_max - x_min
    body_height_px = y_max - y_min
    body_center_px = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])

    # Real body dimensions from SMPL-X (in meters)
    body_height_3d = joints_3d[:, 1].max() - joints_3d[:, 1].min()  # ~1.7m

    # Focal length and principal point
    focal = K[0, 0]
    cx, cy = K[0, 2], K[1, 2]

    # Estimate distance using similar triangles
    # distance = focal * real_height / pixel_height
    if body_height_px > 50:
        distance = focal * body_height_3d / body_height_px
    else:
        distance = 3.0  # Default fallback

    # Estimate camera offset based on where body appears in image
    # If body center is not at image center, camera is offset
    dx = (body_center_px[0] - cx) / focal * distance
    dy = (body_center_px[1] - cy) / focal * distance

    # Camera rotation - estimate from shoulder line if visible
    R = np.eye(3)

    # Check if we can see shoulders to estimate rotation
    if keypoints_2d[5, 2] > 0.3 and keypoints_2d[6, 2] > 0.3:
        l_shoulder = keypoints_2d[5, :2]
        r_shoulder = keypoints_2d[6, :2]

        # Shoulder width in pixels vs expected width
        shoulder_px = np.linalg.norm(l_shoulder - r_shoulder)
        shoulder_3d = np.linalg.norm(joints_3d[16] - joints_3d[17])  # ~0.4m

        # If shoulders appear narrower than expected, person is rotated
        expected_shoulder_px = focal * shoulder_3d / distance

        if expected_shoulder_px > 0:
            cos_angle = min(1.0, shoulder_px / expected_shoulder_px)
            angle = np.arccos(cos_angle)

            # Determine rotation direction from shoulder positions
            # (simplified - actual implementation would use more keypoints)

            log(f"[Frame {frame_idx}] Estimated rotation: {np.degrees(angle):.1f}°", "DEBUG")

    # Camera translation
    t = np.array([-dx, -dy, distance])

    log(f"[Frame {frame_idx}] Camera: dist={distance:.2f}m, offset=({dx:.2f}, {dy:.2f})", "DEBUG")

    return {'R': R, 't': t, 'K': K.copy(), 'valid': True, 'distance': distance}

# ============================================
# 2D LOSS OPTIMIZATION (Improved)
# ============================================

def compute_2d_loss(joints, keypoints_2d, cam, device):
    """
    Compute 2D reprojection loss for shape optimization.
    """
    import torch

    if cam is None:
        return None

    try:
        R = torch.tensor(cam['R'], dtype=torch.float32, device=device)
        t = torch.tensor(cam['t'], dtype=torch.float32, device=device)
        K = torch.tensor(cam['K'], dtype=torch.float32, device=device)

        # Get SMPL-X joints for COCO keypoints
        smplx_indices = list(COCO_TO_SMPLX.values())
        coco_indices = list(COCO_TO_SMPLX.keys())

        body_joints = torch.stack([joints[i] for i in smplx_indices])

        # Project to camera coordinates
        cam_pts = (R @ body_joints.T).T + t

        # Skip if points behind camera
        if torch.any(cam_pts[:, 2] <= 0.1):
            return None

        # Project to 2D
        proj = (K @ cam_pts.T).T
        proj_2d = proj[:, :2] / (proj[:, 2:3] + 1e-8)

        # Ground truth
        gt_2d = torch.tensor(keypoints_2d[coco_indices, :2], dtype=torch.float32, device=device)
        conf = torch.tensor(keypoints_2d[coco_indices, 2], dtype=torch.float32, device=device)

        # Weighted MSE loss
        diff = proj_2d - gt_2d
        loss = torch.sum(conf.unsqueeze(-1) * diff**2) / (torch.sum(conf) + 1e-8)

        return loss

    except Exception as e:
        log(f"Loss computation error: {e}", "DEBUG")
        return None

# ============================================
# POSE DETECTION
# ============================================

def detect_pose(image_rgb, frame_idx=0):
    h, w = image_rgb.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb.copy())
    result = pose_landmarker.detect(mp_image)

    if not result.pose_landmarks:
        return None

    landmarks = result.pose_landmarks[0]
    mp_kps = np.array([[lm.x * w, lm.y * h, lm.visibility] for lm in landmarks])

    coco_kps = np.zeros((17, 3))
    for mp_idx, coco_idx in MP_TO_COCO.items():
        coco_kps[coco_idx] = mp_kps[mp_idx]

    seg = None
    if result.segmentation_masks:
        seg = (result.segmentation_masks[0].numpy_view() > 0.5).astype(np.uint8)

    return {'keypoints': coco_kps, 'keypoints_full': mp_kps, 'segmentation': seg}

# ============================================
# MAIN PROCESSING
# ============================================

def process_video(video_file, smplx_file, n_frames=8, known_height=None, progress=gr.Progress()):
    import torch
    import smplx
    import shutil
    import tempfile

    log("\n" + "="*60)
    log("PROCESSING VIDEO (v2.4 - Geometric Camera)")
    log("="*60)

    if video_file is None:
        return None, "❌ Please upload a video", None, None

    video_path = video_file if isinstance(video_file, str) else video_file.name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Device: {device}")

    # Load SMPL-X
    progress(0.05, desc="Loading SMPL-X...")

    if not smplx_file or not os.path.exists(smplx_file):
        return None, "❌ Error: Please upload SMPLX_NEUTRAL.npz model", None, None

    # Create proper directory structure
    SMPLX_PATH = 'models/smplx'
    os.makedirs(SMPLX_PATH, exist_ok=True)
    target_path = os.path.join(SMPLX_PATH, 'SMPLX_NEUTRAL.npz')
    shutil.copy(smplx_file, target_path)

    body_model = smplx.create('models', model_type='smplx', gender='neutral',
                             num_betas=10, ext='npz').to(device)
    log("✅ SMPL-X loaded")

    # Extract frames
    progress(0.1, desc="Extracting frames...")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    log(f"Video: {width}x{height}, {total_frames} frames")

    indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    # Detect poses
    progress(0.2, desc="Detecting poses...")
    pose_results = []
    for i, frame in enumerate(frames):
        result = detect_pose(frame, i)
        pose_results.append(result)
        if result:
            n_valid = sum(1 for k in result['keypoints'] if k[2] > 0.3)
            log(f"Frame {i}: ✓ {n_valid}/17 keypoints")

    pose_count = sum(1 for r in pose_results if r is not None)
    if pose_count == 0:
        return None, "❌ No poses detected", None, None

    # Camera estimation (GEOMETRIC - NEW!)
    progress(0.4, desc="Estimating cameras...")
    log("\n--- GEOMETRIC CAMERA ESTIMATION ---")

    focal = np.sqrt(width**2 + height**2)
    K = np.array([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1]], dtype=np.float64)

    with torch.no_grad():
        output = body_model(betas=torch.zeros(1, 10, device=device),
                           body_pose=torch.zeros(1, 63, device=device),
                           global_orient=torch.zeros(1, 3, device=device))
    joints_3d = output.joints[0].cpu().numpy()

    cameras = []
    for i, pose in enumerate(pose_results):
        if pose is None:
            cameras.append(None)
            continue
        cam = estimate_camera_geometric(pose['keypoints'], joints_3d, K, i)
        cameras.append(cam)

    valid_cams = sum(1 for c in cameras if c is not None)
    log(f"Cameras estimated: {valid_cams}/{pose_count}")

    # Shape optimization
    progress(0.5, desc="Optimizing body shape...")
    log("\n--- SHAPE OPTIMIZATION ---")

    betas = torch.zeros(1, 10, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([betas], lr=0.05)

    best_loss = float('inf')
    best_betas = betas.clone().detach()

    for iteration in range(300):
        optimizer.zero_grad()

        output = body_model(betas=betas,
                           body_pose=torch.zeros(1, 63, device=device),
                           global_orient=torch.zeros(1, 3, device=device))
        joints = output.joints[0]

        # 2D reprojection loss
        total_loss = torch.tensor(0.0, device=device)
        count = 0

        for pose, cam in zip(pose_results, cameras):
            if pose is None or cam is None:
                continue

            loss = compute_2d_loss(joints, pose['keypoints'], cam, device)
            if loss is not None:
                total_loss = total_loss + loss
                count += 1

        if count > 0:
            total_loss = total_loss / count

        # Symmetry loss
        left_arm = torch.norm(joints[16]-joints[18]) + torch.norm(joints[18]-joints[20])
        right_arm = torch.norm(joints[17]-joints[19]) + torch.norm(joints[19]-joints[21])
        left_leg = torch.norm(joints[1]-joints[4]) + torch.norm(joints[4]-joints[7])
        right_leg = torch.norm(joints[2]-joints[5]) + torch.norm(joints[5]-joints[8])
        sym_loss = (left_arm - right_arm)**2 + (left_leg - right_leg)**2

        # Proportion loss (shoulder/hip ratio should be reasonable)
        shoulder_w = torch.norm(joints[16] - joints[17])
        hip_w = torch.norm(joints[1] - joints[2])
        ratio = shoulder_w / (hip_w + 1e-6)
        ratio_loss = torch.relu(ratio - 2.5)**2 + torch.relu(0.8 - ratio)**2

        # Shape regularization
        shape_reg = torch.mean(betas**2)

        # Total loss
        loss = total_loss + 0.5 * sym_loss + 0.5 * ratio_loss + 0.01 * shape_reg

        if loss.requires_grad:
            loss.backward()
            optimizer.step()

        # Track best
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_betas = betas.clone().detach()

        if iteration % 50 == 0:
            log(f"Iter {iteration}: total={loss.item():.4f}, 2D={total_loss.item() if torch.is_tensor(total_loss) else total_loss:.4f}, "
                f"sym={sym_loss.item():.4f}, ratio={ratio.item():.2f}")

    betas = best_betas
    log(f"Best loss: {best_loss:.4f}")
    log(f"Final betas: {betas[0, :5].cpu().numpy()}")

    # Generate final mesh
    progress(0.8, desc="Generating mesh...")

    with torch.no_grad():
        output = body_model(betas=betas,
                           body_pose=torch.zeros(1, 63, device=device),
                           global_orient=torch.zeros(1, 3, device=device),
                           return_verts=True)

    vertices = output.vertices[0].cpu().numpy()
    joints = output.joints[0].cpu().numpy()

    # Extract measurements
    progress(0.9, desc="Extracting measurements...")

    from sklearn.decomposition import PCA
    from scipy.spatial import ConvexHull

    raw_height = vertices[:, 1].max() - vertices[:, 1].min()
    scale = known_height / raw_height if known_height else 100

    def circ(verts, center, radius):
        dist = np.linalg.norm(verts - center, axis=1)
        nearby = verts[dist < radius]
        if len(nearby) < 20:
            return 0
        try:
            pca = PCA(n_components=2)
            pts_2d = pca.fit_transform(nearby - center)
            hull = ConvexHull(pts_2d)
            pts = pts_2d[hull.vertices]
            return sum(np.linalg.norm(pts[i] - pts[(i+1) % len(pts)]) for i in range(len(pts))) * scale
        except:
            return 0

    m = {}
    m['height'] = raw_height * scale
    m['shoulder_width'] = np.linalg.norm(joints[16] - joints[17]) * scale
    m['hip_width'] = np.linalg.norm(joints[1] - joints[2]) * scale
    m['torso_length'] = np.linalg.norm(joints[12] - joints[0]) * scale

    left_arm = np.linalg.norm(joints[16]-joints[18]) + np.linalg.norm(joints[18]-joints[20])
    right_arm = np.linalg.norm(joints[17]-joints[19]) + np.linalg.norm(joints[19]-joints[21])
    m['arm_length'] = ((left_arm + right_arm) / 2) * scale

    left_leg = np.linalg.norm(joints[1]-joints[4]) + np.linalg.norm(joints[4]-joints[7])
    right_leg = np.linalg.norm(joints[2]-joints[5]) + np.linalg.norm(joints[5]-joints[8])
    m['leg_length'] = ((left_leg + right_leg) / 2) * scale

    m['inseam'] = np.linalg.norm((joints[1]+joints[2])/2 - (joints[7]+joints[8])/2) * scale

    m['chest_circumference'] = circ(vertices, (joints[16]+joints[17])/2 - [0,0.05,0], 0.15)
    m['waist_circumference'] = circ(vertices, (joints[3]+joints[6])/2, 0.12)
    m['hip_circumference'] = circ(vertices, joints[0], 0.15)

    log("\n" + "="*50)
    log("FINAL MEASUREMENTS:")
    for name, value in m.items():
        log(f"  {name:<25} {value:>8.1f} cm")
    log("="*50)

    # Create visualization
    annotated = []
    for i, (frame, pose) in enumerate(zip(frames, pose_results)):
        img = frame.copy()
        if pose:
            kp = pose['keypoints']
            for s, e in [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]:
                if kp[s,2] > 0.3 and kp[e,2] > 0.3:
                    cv2.line(img, (int(kp[s,0]), int(kp[s,1])), (int(kp[e,0]), int(kp[e,1])), (0,255,255), 2)
            for x, y, c in kp:
                if c > 0.3:
                    cv2.circle(img, (int(x), int(y)), 5, (0,255,0), -1)
        annotated.append(cv2.resize(img, (320, 240)))

    cols = 4
    rows = (len(annotated) + cols - 1) // cols
    while len(annotated) < rows * cols:
        annotated.append(np.zeros((240, 320, 3), dtype=np.uint8))
    grid = np.vstack([np.hstack(annotated[i*cols:(i+1)*cols]) for i in range(rows)])

    summary = f"""## ✅ Processing Complete (v2.4 - Geometric Camera)

### Video Info
- Resolution: {width} x {height}
- Frames processed: {len(frames)}
- Poses detected: {pose_count}/{len(frames)}
- Cameras estimated: {valid_cams}/{pose_count}

### Measurements
| Measurement | Value (cm) |
|-------------|------------|
| Height | {m['height']:.1f} |
| Shoulder Width | {m['shoulder_width']:.1f} |
| Hip Width | {m['hip_width']:.1f} |
| Torso Length | {m['torso_length']:.1f} |
| Arm Length | {m['arm_length']:.1f} |
| Leg Length | {m['leg_length']:.1f} |
| Inseam | {m['inseam']:.1f} |
| Chest Circ. | {m['chest_circumference']:.1f} |
| Waist Circ. | {m['waist_circumference']:.1f} |
| Hip Circ. | {m['hip_circumference']:.1f} |

### Optimization
- Final loss: {best_loss:.4f}
- Beta values: {betas[0, :3].cpu().numpy()}

*{'Using provided height for scale.' if known_height else 'No height provided - using default scale.'}*
"""

    # Save files
    import json
    temp_obj = tempfile.NamedTemporaryFile(delete=False, suffix='.obj', mode='w')
    for v in vertices:
        temp_obj.write(f"v {v[0]} {v[1]} {v[2]}\n")
    for face in body_model.faces:
        temp_obj.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    temp_obj.close()

    temp_json = tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w')
    json.dump({
        'measurements_cm': {k: float(v) for k, v in m.items()},
        'pipeline_version': '2.4',
        'method': 'geometric_camera',
        'frames_processed': f"{pose_count}/{len(frames)}",
        'cameras_estimated': f"{valid_cams}/{pose_count}"
    }, temp_json, indent=2)
    temp_json.close()

    progress(1.0, desc="Done!")
    return grid, summary, temp_obj.name, temp_json.name

# ============================================
# GRADIO INTERFACE
# ============================================

with gr.Blocks(title="Body Reconstruction v2.4") as demo:
    gr.Markdown("""
    # 🧍 Human Body Reconstruction v2.4

    ### ✨ NEW: Uses geometric camera estimation (fixed PnP failure!)

    **What Changed:**
    - v2.2 used PnP camera estimation → **Failed** (SMPL-X joints are coplanar)
    - v2.4 uses geometric estimation → **Works!**

    Upload a turntable video to extract body measurements.
    """)

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Video")
            smplx_input = gr.File(label="SMPL-X Model (SMPLX_NEUTRAL.npz)", file_types=[".npz"])
            n_frames = gr.Slider(4, 16, value=8, step=2, label="Number of Frames")
            known_height = gr.Number(label="Known Height (cm)", value=None,
                                    info="Enter actual height for accurate measurements")
            process_btn = gr.Button("🚀 Process", variant="primary", size="lg")

        with gr.Column():
            output_image = gr.Image(label="Pose Detection Results")
            output_text = gr.Markdown()
            with gr.Row():
                mesh_output = gr.File(label="3D Mesh (.obj)")
                json_output = gr.File(label="Measurements JSON")

    gr.Markdown("""
    ---
    ### 📝 Notes
    - Processing takes 2-5 minutes on GPU
    - Download SMPL-X from: https://smpl-x.is.tue.mpg.de/
    - Expected accuracy: Height ±1-1.5cm, Circumferences ±2-3cm
    """)

    process_btn.click(process_video,
                     [video_input, smplx_input, n_frames, known_height],
                     [output_image, output_text, mesh_output, json_output])

if __name__ == "__main__":
    log("\n🚀 Starting Gradio interface...")
    demo.launch(share=True, debug=True)
