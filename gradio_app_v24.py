"""
Human Body Reconstruction v2.6 - BALANCED OPTIMIZATION + SILHOUETTE
Fixes: v2.5 was too constrained, betas couldn't change enough

Key Changes:
1. Reduced regularization (10.0 → 0.1-0.5)
2. Higher learning rate (0.01 → 0.05)
3. Proper camera rotation in projection
4. Two-phase optimization: explore then refine
5. **SILHOUETTE LOSS** - matches body width from projected mesh bounds
   (keypoints alone can't capture body width changes!)
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
log("BODY RECONSTRUCTION v2.6 (BALANCED OPTIMIZATION)")
log("="*60)

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

MP_TO_COCO = {0:0, 2:1, 5:2, 7:3, 8:4, 11:5, 12:6, 13:7, 14:8, 15:9, 16:10, 23:11, 24:12, 25:13, 26:14, 27:15, 28:16}
COCO_TO_SMPLX = {5:16, 6:17, 7:18, 8:19, 9:20, 10:21, 11:1, 12:2, 13:4, 14:5, 15:7, 16:8}

# ============================================
# IMPROVED CAMERA ESTIMATION
# ============================================

def estimate_camera_with_rotation(keypoints_2d, joints_3d, K, frame_idx=0):
    """
    Estimate camera with proper rotation handling.
    Returns camera that includes body rotation.
    """
    valid_mask = keypoints_2d[:, 2] > 0.3
    if np.sum(valid_mask) < 4:
        return None
    
    valid_kps = keypoints_2d[valid_mask]
    
    # Body bounds
    x_min, x_max = valid_kps[:, 0].min(), valid_kps[:, 0].max()
    y_min, y_max = valid_kps[:, 1].min(), valid_kps[:, 1].max()
    
    body_height_px = y_max - y_min
    body_center_px = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
    
    # SMPL-X body height
    body_height_3d = joints_3d[:, 1].max() - joints_3d[:, 1].min()
    
    focal = K[0, 0]
    cx, cy = K[0, 2], K[1, 2]
    
    # Distance estimation
    if body_height_px > 50:
        distance = focal * body_height_3d / body_height_px
    else:
        distance = 3.0
    
    # Camera offset
    dx = (body_center_px[0] - cx) / focal * distance
    dy = (body_center_px[1] - cy) / focal * distance
    
    # Rotation estimation from shoulder width
    R = np.eye(3)
    rotation_angle = 0.0
    rotation_sign = 1.0
    
    if keypoints_2d[5, 2] > 0.3 and keypoints_2d[6, 2] > 0.3:
        l_shoulder = keypoints_2d[5, :2]
        r_shoulder = keypoints_2d[6, :2]
        
        shoulder_px = np.linalg.norm(l_shoulder - r_shoulder)
        shoulder_3d = np.linalg.norm(joints_3d[16] - joints_3d[17])
        expected_shoulder_px = focal * shoulder_3d / distance
        
        if expected_shoulder_px > 10:
            ratio = shoulder_px / expected_shoulder_px
            cos_angle = min(1.0, ratio)
            rotation_angle = np.arccos(cos_angle)
            
            # Determine rotation direction based on shoulder Z offset
            # If left shoulder is closer (lower X in image when facing right), rotate positive
            shoulder_diff = r_shoulder[0] - l_shoulder[0]
            if shoulder_diff < 0:
                rotation_sign = -1.0
            
            # Apply Y-axis rotation
            angle = rotation_angle * rotation_sign
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    
    t = np.array([-dx, -dy, distance])
    
    return {
        'R': R, 't': t, 'K': K.copy(), 
        'valid': True, 'distance': distance,
        'rotation_deg': np.degrees(rotation_angle)
    }

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
# SILHOUETTE PROJECTION (KEY FOR BODY WIDTH!)
# ============================================

def compute_silhouette_loss(vertices, pose_data, cam, device, image_width, image_height):
    """
    Compute silhouette-based loss for body width estimation.
    
    The problem with keypoint-only loss:
    - Joint positions barely change with body shape (betas)
    - Betas mainly affect body WIDTH and SURFACE
    - Need silhouette to capture width changes!
    """
    import torch
    
    if pose_data is None or cam is None:
        return None
    
    # Get valid keypoints to estimate body bounds
    kp = pose_data['keypoints']
    valid_mask = kp[:, 2] > 0.3
    if np.sum(valid_mask) < 4:
        return None
    
    valid_kps = kp[valid_mask]
    
    # Body bounds from 2D keypoints
    kp_x_min, kp_x_max = valid_kps[:, 0].min(), valid_kps[:, 0].max()
    kp_y_min, kp_y_max = valid_kps[:, 1].min(), valid_kps[:, 1].max()
    
    # Estimate body width from keypoints
    # Use shoulder-to-shoulder or hip-to-hip distance
    observed_width = kp_x_max - kp_x_min
    observed_height = kp_y_max - kp_y_min
    
    try:
        R = torch.tensor(cam['R'], dtype=torch.float32, device=device)
        t = torch.tensor(cam['t'], dtype=torch.float32, device=device)
        K = torch.tensor(cam['K'], dtype=torch.float32, device=device)
        
        # Project all vertices (sample for speed)
        sample_idx = torch.randperm(vertices.shape[0])[:1000]
        sampled_verts = vertices[sample_idx]
        
        cam_pts = (R @ sampled_verts.T).T + t
        
        # Only use points in front of camera
        valid_depth = cam_pts[:, 2] > 0.1
        if torch.sum(valid_depth) < 100:
            return None
        
        cam_pts = cam_pts[valid_depth]
        
        # Project to 2D
        proj = (K @ cam_pts.T).T
        proj_2d = proj[:, :2] / (proj[:, 2:3] + 1e-8)
        
        # Compute projected bounds
        proj_x_min, proj_x_max = proj_2d[:, 0].min(), proj_2d[:, 0].max()
        proj_y_min, proj_y_max = proj_2d[:, 1].min(), proj_2d[:, 1].max()
        
        proj_width = proj_x_max - proj_x_min
        proj_height = proj_y_max - proj_y_min
        
        # Loss: match observed width/height
        # Normalize by image diagonal
        img_diag = np.sqrt(image_width**2 + image_height**2)
        
        width_diff = (proj_width - observed_width) / img_diag
        height_diff = (proj_height - observed_height) / img_diag
        
        # Width is more important for body shape
        silhouette_loss = 2.0 * width_diff**2 + height_diff**2
        
        return silhouette_loss
        
    except Exception as e:
        return None

# ============================================
# TWO-PHASE OPTIMIZATION
# ============================================

def optimize_shape_balanced(pose_results, cameras, body_model, device,
                           image_width, image_height, n_iterations=400):
    """
    Two-phase optimization:
    Phase 1 (0-200): Explore - low regularization, higher LR
    Phase 2 (200-400): Refine - moderate regularization, lower LR
    """
    import torch
    
    log("Starting BALANCED shape optimization...", "INFO")
    
    img_diag = np.sqrt(image_width**2 + image_height**2)
    
    betas = torch.zeros(1, 10, device=device, requires_grad=True)
    
    # Phase 1: Exploration
    optimizer = torch.optim.Adam([betas], lr=0.05)
    
    best_loss = float('inf')
    best_betas = betas.clone().detach()
    
    smplx_indices = list(COCO_TO_SMPLX.values())
    coco_indices = list(COCO_TO_SMPLX.keys())
    
    for iteration in range(n_iterations):
        
        # Phase transition at iteration 200
        if iteration == 200:
            log("Switching to Phase 2 (refinement)...", "INFO")
            optimizer = torch.optim.Adam([betas], lr=0.01)
        
        # Regularization weight: lower in phase 1, higher in phase 2
        if iteration < 200:
            reg_weight = 0.1  # Explore freely
        else:
            reg_weight = 0.5  # Moderate constraint
        
        optimizer.zero_grad()
        
        output = body_model(
            betas=betas,
            body_pose=torch.zeros(1, 63, device=device),
            global_orient=torch.zeros(1, 3, device=device)
        )
        joints = output.joints[0]
        
        # 2D Keypoint Loss
        kp_loss = torch.tensor(0.0, device=device)
        sil_loss = torch.tensor(0.0, device=device)
        count = 0
        sil_count = 0
        
        # Get current vertices for silhouette loss
        current_verts = output.vertices[0]
        
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
                
                # Normalized loss
                diff = (proj_2d - gt_2d) / img_diag
                frame_loss = torch.sum(conf.unsqueeze(-1) * diff**2) / (torch.sum(conf) + 1e-8)
                
                kp_loss = kp_loss + frame_loss
                count += 1
                
                # Silhouette loss for body width
                sil = compute_silhouette_loss(current_verts, pose, cam, device, image_width, image_height)
                if sil is not None:
                    sil_loss = sil_loss + sil
                    sil_count += 1
                    
            except Exception as e:
                pass
        
        if count > 0:
            kp_loss = kp_loss / count
        if sil_count > 0:
            sil_loss = sil_loss / sil_count
        
        # Symmetry Loss
        left_arm = torch.norm(joints[16]-joints[18]) + torch.norm(joints[18]-joints[20])
        right_arm = torch.norm(joints[17]-joints[19]) + torch.norm(joints[19]-joints[21])
        left_leg = torch.norm(joints[1]-joints[4]) + torch.norm(joints[4]-joints[7])
        right_leg = torch.norm(joints[2]-joints[5]) + torch.norm(joints[5]-joints[8])
        sym_loss = (left_arm - right_arm)**2 + (left_leg - right_leg)**2
        
        # Proportion Loss
        shoulder_w = torch.norm(joints[16] - joints[17])
        hip_w = torch.norm(joints[1] - joints[2])
        ratio = shoulder_w / (hip_w + 1e-6)
        ratio_loss = torch.relu(ratio - 2.0)**2 + torch.relu(1.0 - ratio)**2
        
        # Beta Regularization (with soft boundary)
        beta_l2 = torch.mean(betas**2)
        beta_boundary = torch.sum(torch.relu(torch.abs(betas) - 2.5)**2)  # Soft limit at ±2.5
        shape_reg = beta_l2 + 5.0 * beta_boundary
        
        # Total Loss - NOW INCLUDING SILHOUETTE!
        total_loss = kp_loss + 2.0 * sil_loss + 0.5 * sym_loss + 0.5 * ratio_loss + reg_weight * shape_reg
        
        if total_loss.requires_grad:
            total_loss.backward()
            optimizer.step()
        
        # Hard clamp at ±3
        with torch.no_grad():
            betas.data = torch.clamp(betas.data, -3.0, 3.0)
        
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_betas = betas.clone().detach()
        
        if iteration % 50 == 0:
            beta_str = ", ".join([f"{b:.2f}" for b in betas[0, :5].cpu().numpy()])
            log(f"Iter {iteration}: loss={total_loss.item():.4f}, kp={kp_loss.item():.4f}, "
                f"sil={sil_loss.item():.4f}, ratio={ratio.item():.2f}, betas=[{beta_str}]", "DEBUG")
    
    log(f"Best loss: {best_loss:.4f}", "SUCCESS")
    log(f"Final betas: {best_betas[0].cpu().numpy().round(3)}", "SUCCESS")
    
    beta_max = torch.max(torch.abs(best_betas)).item()
    log(f"Beta range: [{best_betas.min().item():.2f}, {best_betas.max().item():.2f}]", "SUCCESS")
    
    return best_betas

# ============================================
# MEASUREMENTS
# ============================================

def extract_measurements(vertices, joints, known_height_cm=None):
    from sklearn.decomposition import PCA
    from scipy.spatial import ConvexHull
    
    raw_height = vertices[:, 1].max() - vertices[:, 1].min()
    scale = known_height_cm / raw_height if known_height_cm else 100
    
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
    
    crotch = (joints[1] + joints[2]) / 2
    ankle = (joints[7] + joints[8]) / 2
    m['inseam'] = np.linalg.norm(crotch - ankle) * scale
    
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
    
    chest_center = (joints[16] + joints[17]) / 2
    chest_center[1] -= 0.05
    m['chest_circumference'] = circ(vertices, chest_center, 0.15)
    
    waist_center = (joints[3] + joints[6]) / 2
    m['waist_circumference'] = circ(vertices, waist_center, 0.12)
    
    m['hip_circumference'] = circ(vertices, joints[0], 0.15)
    
    return m

# ============================================
# MAIN PROCESSING
# ============================================

def process_video(video_file, smplx_file, n_frames=12, known_height=None, progress=gr.Progress()):
    import torch
    import smplx
    import shutil
    import tempfile
    
    log("\n" + "="*60)
    log("PROCESSING VIDEO (v2.6 - BALANCED)")
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
    
    # Use more frames for better coverage
    indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    log(f"Extracted {len(frames)} frames")
    
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
        cam = estimate_camera_with_rotation(pose['keypoints'], joints_3d, K, i)
        cameras.append(cam)
        if cam:
            log(f"Frame {i}: dist={cam['distance']:.2f}m, rot={cam['rotation_deg']:.0f}°", "DEBUG")
    
    valid_cams = sum(1 for c in cameras if c is not None)
    log(f"Cameras estimated: {valid_cams}/{pose_count}")
    
    # Optimization
    progress(0.4, desc="Optimizing shape...")
    
    betas = optimize_shape_balanced(
        pose_results, cameras, body_model, device,
        width, height, n_iterations=400
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
    
    log("\n" + "="*50)
    log("FINAL MEASUREMENTS:")
    for name, value in measurements.items():
        log(f"  {name:<25} {value:>8.1f} cm")
    log("="*50)
    
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
        annotated.append(cv2.resize(img, (240, 180)))
    
    cols = 4
    rows = (len(annotated) + cols - 1) // cols
    while len(annotated) < rows * cols:
        annotated.append(np.zeros((180, 240, 3), dtype=np.uint8))
    grid = np.vstack([np.hstack(annotated[i*cols:(i+1)*cols]) for i in range(rows)])
    
    m = measurements
    summary = f"""## ✅ Processing Complete (v2.6)

### Pipeline Info
- Resolution: {width} x {height}
- Frames: {len(frames)} | Poses: {pose_count} | Cameras: {valid_cams}
- Device: {device}

### Beta Values
`{betas[0].cpu().numpy().round(2)}`

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

*{'Scaled to provided height.' if known_height else 'Using default scale (no height provided).'}*
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
        'pipeline_version': '2.6',
    }, temp_json, indent=2)
    temp_json.close()
    
    progress(1.0, desc="Done!")
    return grid, summary, temp_obj.name, temp_json.name

# ============================================
# GRADIO INTERFACE
# ============================================

with gr.Blocks(title="Body Reconstruction v2.6") as demo:
    gr.Markdown("""
    # 🧍 Human Body Reconstruction v2.6
    
    ### Changes from v2.5:
    - ✅ **Silhouette loss** (matches body width - critical fix!)
    - ✅ **Two-phase optimization** (explore → refine)
    - ✅ **Reduced regularization** (allows betas to change)
    - ✅ **More frames** (12 default for better coverage)
    
    **Why silhouette?** 2D keypoints are at joint positions which barely 
    change with body shape. Body WIDTH changes are captured by silhouette!
    
    Upload a turntable video to extract body measurements.
    """)
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Video")
            smplx_input = gr.File(label="SMPL-X Model (SMPLX_NEUTRAL.npz)", file_types=[".npz"])
            n_frames = gr.Slider(8, 20, value=12, step=2, label="Number of Frames")
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

if __name__ == "__main__":
    demo.launch(share=True, debug=True)