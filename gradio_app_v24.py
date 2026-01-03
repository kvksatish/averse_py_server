"""
Human Body Reconstruction v2.5 - CONSTRAINED OPTIMIZATION
Fixes the exploding betas problem from v2.4

Key Changes:
1. Strong beta regularization (keep betas in [-3, +3])
2. Normalized loss (divide by image size)
3. Beta clamping after each step
4. Lower learning rate
5. Better loss weighting
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
log("BODY RECONSTRUCTION v2.5 (CONSTRAINED BETAS)")
log("="*60)

# MediaPipe setup
import mediapipe as mp

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
# GEOMETRIC CAMERA ESTIMATION
# ============================================

def estimate_camera_geometric(keypoints_2d, joints_3d, K, frame_idx=0):
    """Estimate camera using geometric approach"""
    
    valid_mask = keypoints_2d[:, 2] > 0.3
    if np.sum(valid_mask) < 4:
        return None
    
    valid_kps = keypoints_2d[valid_mask]
    
    x_min, x_max = valid_kps[:, 0].min(), valid_kps[:, 0].max()
    y_min, y_max = valid_kps[:, 1].min(), valid_kps[:, 1].max()
    
    body_height_px = y_max - y_min
    body_center_px = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
    
    body_height_3d = joints_3d[:, 1].max() - joints_3d[:, 1].min()
    
    focal = K[0, 0]
    cx, cy = K[0, 2], K[1, 2]
    
    if body_height_px > 50:
        distance = focal * body_height_3d / body_height_px
    else:
        distance = 3.0
    
    dx = (body_center_px[0] - cx) / focal * distance
    dy = (body_center_px[1] - cy) / focal * distance
    
    # Estimate rotation from shoulder width
    R = np.eye(3)
    rotation_angle = 0.0
    
    if keypoints_2d[5, 2] > 0.3 and keypoints_2d[6, 2] > 0.3:
        l_shoulder = keypoints_2d[5, :2]
        r_shoulder = keypoints_2d[6, :2]
        shoulder_px = np.linalg.norm(l_shoulder - r_shoulder)
        shoulder_3d = np.linalg.norm(joints_3d[16] - joints_3d[17])
        expected_shoulder_px = focal * shoulder_3d / distance
        
        if expected_shoulder_px > 0:
            cos_angle = min(1.0, shoulder_px / expected_shoulder_px)
            rotation_angle = np.arccos(cos_angle)
            
            # Apply rotation around Y axis
            c, s = np.cos(rotation_angle), np.sin(rotation_angle)
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    
    t = np.array([-dx, -dy, distance])
    
    log(f"[Frame {frame_idx}] dist={distance:.2f}m, rot={np.degrees(rotation_angle):.0f}°", "DEBUG")
    
    return {'R': R, 't': t, 'K': K.copy(), 'valid': True, 'distance': distance}

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
    
    return {'keypoints': coco_kps, 'keypoints_full': mp_kps}

# ============================================
# CONSTRAINED SHAPE OPTIMIZATION (KEY FIX!)
# ============================================

def optimize_shape_constrained(pose_results, cameras, body_model, device, 
                                image_width, image_height, n_iterations=300):
    """
    Optimize body shape with STRONG constraints on beta values.
    
    Key differences from v2.4:
    1. Normalize loss by image diagonal (not raw pixel^2)
    2. Strong beta regularization (weight=10.0)
    3. Clamp betas to [-3, +3] after each step
    4. Lower learning rate (0.01 instead of 0.05)
    """
    import torch
    
    log("Starting CONSTRAINED shape optimization...", "INFO")
    log(f"  - Beta range: [-3, +3]", "DEBUG")
    log(f"  - Strong regularization: weight=10.0", "DEBUG")
    
    # Normalization factor (image diagonal)
    img_diag = np.sqrt(image_width**2 + image_height**2)
    
    # Initialize betas (small random values)
    betas = torch.zeros(1, 10, device=device, requires_grad=True)
    
    # Lower learning rate!
    optimizer = torch.optim.Adam([betas], lr=0.01)
    
    best_loss = float('inf')
    best_betas = betas.clone().detach()
    
    smplx_indices = list(COCO_TO_SMPLX.values())
    coco_indices = list(COCO_TO_SMPLX.keys())
    
    for iteration in range(n_iterations):
        optimizer.zero_grad()
        
        output = body_model(
            betas=betas,
            body_pose=torch.zeros(1, 63, device=device),
            global_orient=torch.zeros(1, 3, device=device)
        )
        joints = output.joints[0]
        
        # ============================================
        # 2D REPROJECTION LOSS (NORMALIZED!)
        # ============================================
        kp_loss = torch.tensor(0.0, device=device)
        count = 0
        
        for pose, cam in zip(pose_results, cameras):
            if pose is None or cam is None:
                continue
            
            try:
                R = torch.tensor(cam['R'], dtype=torch.float32, device=device)
                t = torch.tensor(cam['t'], dtype=torch.float32, device=device)
                K = torch.tensor(cam['K'], dtype=torch.float32, device=device)
                
                body_joints = torch.stack([joints[i] for i in smplx_indices])
                cam_pts = (R @ body_joints.T).T + t
                
                if torch.any(cam_pts[:, 2] <= 0.1):
                    continue
                
                proj = (K @ cam_pts.T).T
                proj_2d = proj[:, :2] / (proj[:, 2:3] + 1e-8)
                
                gt_2d = torch.tensor(pose['keypoints'][coco_indices, :2], 
                                    dtype=torch.float32, device=device)
                conf = torch.tensor(pose['keypoints'][coco_indices, 2],
                                   dtype=torch.float32, device=device)
                
                # NORMALIZE by image diagonal!
                diff = (proj_2d - gt_2d) / img_diag
                frame_loss = torch.sum(conf.unsqueeze(-1) * diff**2) / (torch.sum(conf) + 1e-8)
                
                kp_loss = kp_loss + frame_loss
                count += 1
            except:
                pass
        
        if count > 0:
            kp_loss = kp_loss / count
        
        # ============================================
        # SYMMETRY LOSS
        # ============================================
        left_arm = torch.norm(joints[16]-joints[18]) + torch.norm(joints[18]-joints[20])
        right_arm = torch.norm(joints[17]-joints[19]) + torch.norm(joints[19]-joints[21])
        left_leg = torch.norm(joints[1]-joints[4]) + torch.norm(joints[4]-joints[7])
        right_leg = torch.norm(joints[2]-joints[5]) + torch.norm(joints[5]-joints[8])
        sym_loss = (left_arm - right_arm)**2 + (left_leg - right_leg)**2
        
        # ============================================
        # PROPORTION LOSS (shoulder/hip ratio)
        # ============================================
        shoulder_w = torch.norm(joints[16] - joints[17])
        hip_w = torch.norm(joints[1] - joints[2])
        ratio = shoulder_w / (hip_w + 1e-6)
        # Ratio should be between 1.0 and 2.0 for humans
        ratio_loss = torch.relu(ratio - 2.0)**2 + torch.relu(1.0 - ratio)**2
        
        # ============================================
        # STRONG BETA REGULARIZATION (KEY!)
        # ============================================
        # Penalize betas that go beyond reasonable range
        # L2 regularization + soft boundary at ±3
        beta_l2 = torch.mean(betas**2)
        beta_boundary = torch.sum(torch.relu(torch.abs(betas) - 3.0)**2)
        shape_reg = beta_l2 + 10.0 * beta_boundary  # Heavy penalty for going beyond ±3
        
        # ============================================
        # TOTAL LOSS
        # ============================================
        # Weights: kp=1.0, sym=1.0, ratio=1.0, shape=10.0 (STRONG!)
        total_loss = kp_loss + 1.0 * sym_loss + 1.0 * ratio_loss + 10.0 * shape_reg
        
        if total_loss.requires_grad:
            total_loss.backward()
            optimizer.step()
        
        # ============================================
        # CLAMP BETAS AFTER EACH STEP (HARD CONSTRAINT!)
        # ============================================
        with torch.no_grad():
            betas.data = torch.clamp(betas.data, -3.0, 3.0)
        
        # Track best
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_betas = betas.clone().detach()
        
        if iteration % 50 == 0:
            log(f"Iter {iteration}: loss={total_loss.item():.6f}, kp={kp_loss.item():.6f}, "
                f"ratio={ratio.item():.2f}, betas=[{betas[0,0].item():.2f}, {betas[0,1].item():.2f}, ...]", "DEBUG")
    
    log(f"Best loss: {best_loss:.6f}", "SUCCESS")
    log(f"Final betas: {best_betas[0, :5].cpu().numpy()}", "SUCCESS")
    
    # Verify betas are in range
    beta_max = torch.max(torch.abs(best_betas)).item()
    if beta_max > 3.0:
        log(f"WARNING: Beta max={beta_max:.2f} exceeds limit!", "WARNING")
    else:
        log(f"Betas in valid range (max={beta_max:.2f})", "SUCCESS")
    
    return best_betas

# ============================================
# MEASUREMENT EXTRACTION
# ============================================

def extract_measurements(vertices, joints, known_height_cm=None):
    """Extract body measurements from mesh"""
    from sklearn.decomposition import PCA
    from scipy.spatial import ConvexHull
    
    raw_height = vertices[:, 1].max() - vertices[:, 1].min()
    scale = known_height_cm / raw_height if known_height_cm else 100
    
    m = {}
    m['height'] = raw_height * scale
    m['shoulder_width'] = np.linalg.norm(joints[16] - joints[17]) * scale
    m['hip_width'] = np.linalg.norm(joints[1] - joints[2]) * scale
    m['torso_length'] = np.linalg.norm(joints[12] - joints[0]) * scale
    
    # Arm: shoulder -> elbow -> wrist
    left_arm = np.linalg.norm(joints[16]-joints[18]) + np.linalg.norm(joints[18]-joints[20])
    right_arm = np.linalg.norm(joints[17]-joints[19]) + np.linalg.norm(joints[19]-joints[21])
    m['arm_length'] = ((left_arm + right_arm) / 2) * scale
    
    # Leg: hip -> knee -> ankle
    left_leg = np.linalg.norm(joints[1]-joints[4]) + np.linalg.norm(joints[4]-joints[7])
    right_leg = np.linalg.norm(joints[2]-joints[5]) + np.linalg.norm(joints[5]-joints[8])
    m['leg_length'] = ((left_leg + right_leg) / 2) * scale
    
    # Inseam: crotch to ankle
    crotch = (joints[1] + joints[2]) / 2
    ankle = (joints[7] + joints[8]) / 2
    m['inseam'] = np.linalg.norm(crotch - ankle) * scale
    
    # Circumferences using PCA + ConvexHull
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
    
    # Chest (at armpit level)
    chest_center = (joints[16] + joints[17]) / 2
    chest_center[1] -= 0.05  # Slightly below shoulders
    m['chest_circumference'] = circ(vertices, chest_center, 0.15)
    
    # Waist (narrowest point of torso)
    waist_center = (joints[3] + joints[6]) / 2
    m['waist_circumference'] = circ(vertices, waist_center, 0.12)
    
    # Hip (at pelvis level)
    m['hip_circumference'] = circ(vertices, joints[0], 0.15)
    
    return m

# ============================================
# MAIN PROCESSING
# ============================================

def process_video(video_file, smplx_file, n_frames=8, known_height=None, progress=gr.Progress()):
    import torch
    import smplx
    import shutil
    import tempfile
    
    log("\n" + "="*60)
    log("PROCESSING VIDEO (v2.5 - CONSTRAINED BETAS)")
    log("="*60)
    
    if video_file is None:
        return None, "❌ Please upload a video", None, None
    
    video_path = video_file if isinstance(video_file, str) else video_file.name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Device: {device}")
    
    # Load SMPL-X
    progress(0.05, desc="Loading SMPL-X...")
    
    if not smplx_file or not os.path.exists(smplx_file):
        return None, "❌ Please upload SMPLX_NEUTRAL.npz", None, None
    
    SMPLX_PATH = 'models/smplx'
    os.makedirs(SMPLX_PATH, exist_ok=True)
    shutil.copy(smplx_file, os.path.join(SMPLX_PATH, 'SMPLX_NEUTRAL.npz'))
    
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
    
    log(f"Video: {width}x{height}, {total_frames} frames, {fps:.0f} fps")
    
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
    
    # Camera estimation
    progress(0.3, desc="Estimating cameras...")
    
    focal = np.sqrt(width**2 + height**2)
    K = np.array([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1]], dtype=np.float64)
    
    with torch.no_grad():
        output = body_model(
            betas=torch.zeros(1, 10, device=device),
            body_pose=torch.zeros(1, 63, device=device),
            global_orient=torch.zeros(1, 3, device=device)
        )
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
    
    # CONSTRAINED shape optimization
    progress(0.4, desc="Optimizing shape (constrained)...")
    
    betas = optimize_shape_constrained(
        pose_results, cameras, body_model, device,
        width, height, n_iterations=300
    )
    
    # Generate mesh
    progress(0.8, desc="Generating mesh...")
    
    with torch.no_grad():
        output = body_model(
            betas=betas,
            body_pose=torch.zeros(1, 63, device=device),
            global_orient=torch.zeros(1, 3, device=device),
            return_verts=True
        )
    
    vertices = output.vertices[0].cpu().numpy()
    joints = output.joints[0].cpu().numpy()
    
    # Extract measurements
    progress(0.9, desc="Extracting measurements...")
    measurements = extract_measurements(vertices, joints, known_height)
    
    # Validate measurements
    log("\n" + "="*50)
    log("FINAL MEASUREMENTS:")
    warnings = []
    for name, value in measurements.items():
        flag = ""
        # Sanity checks
        if name == 'arm_length' and (value < 40 or value > 80):
            flag = " ⚠️"
            warnings.append(f"Arm length {value:.1f}cm outside normal range (40-80cm)")
        if name == 'leg_length' and (value < 60 or value > 100):
            flag = " ⚠️"
            warnings.append(f"Leg length {value:.1f}cm outside normal range (60-100cm)")
        if name == 'chest_circumference' and (value < 60 or value > 150):
            flag = " ⚠️"
            warnings.append(f"Chest circumference {value:.1f}cm outside normal range (60-150cm)")
        log(f"  {name:<25} {value:>8.1f} cm{flag}")
    log("="*50)
    
    if warnings:
        log("Measurement warnings:", "WARNING")
        for w in warnings:
            log(f"  - {w}", "WARNING")
    
    # Visualization
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
    
    # Summary
    m = measurements
    warning_text = "\n".join([f"- ⚠️ {w}" for w in warnings]) if warnings else ""
    
    summary = f"""## ✅ Processing Complete (v2.5 - Constrained)

### Video Info
- Resolution: {width} x {height}
- Frames processed: {len(frames)}
- Poses detected: {pose_count}/{len(frames)}
- Cameras estimated: {valid_cams}/{pose_count}

### Beta Values (should be in [-3, +3])
`{betas[0, :5].cpu().numpy().round(2)}`

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

{warning_text}

*{'Using provided height for scale.' if known_height else 'No height provided - using default 100cm/m scale.'}*
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
        'betas': betas[0].cpu().numpy().tolist(),
        'pipeline_version': '2.5',
        'method': 'constrained_geometric',
        'frames_processed': f"{pose_count}/{len(frames)}",
        'cameras_estimated': f"{valid_cams}/{pose_count}"
    }, temp_json, indent=2)
    temp_json.close()
    
    progress(1.0, desc="Done!")
    return grid, summary, temp_obj.name, temp_json.name

# ============================================
# GRADIO INTERFACE
# ============================================

with gr.Blocks(title="Body Reconstruction v2.5") as demo:
    gr.Markdown("""
    # 🧍 Human Body Reconstruction v2.5 (CONSTRAINED)
    
    ### Changes from v2.4:
    - ✅ **Constrained betas** to [-3, +3] range
    - ✅ **Normalized loss** (prevents explosion)
    - ✅ **Strong regularization** (weight=10.0)
    - ✅ **Validation warnings** for unrealistic measurements
    
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
    
    process_btn.click(
        process_video,
        [video_input, smplx_input, n_frames, known_height],
        [output_image, output_text, mesh_output, json_output]
    )
    
    gr.Markdown("""
    ---
    ### Expected Ranges
    | Measurement | Normal Range |
    |-------------|--------------|
    | Shoulder Width | 35-50 cm |
    | Arm Length | 50-70 cm |
    | Leg Length | 75-95 cm |
    | Chest Circ. | 75-120 cm |
    | Waist Circ. | 60-100 cm |
    | Hip Circ. | 85-120 cm |
    """)

if __name__ == "__main__":
    demo.launch(share=True, debug=True)