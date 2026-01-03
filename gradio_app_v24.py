"""
Human Body Reconstruction v2.7 - SEGMENTATION-BASED FITTING
Uses actual body silhouette from segmentation mask, not just keypoints

Key Insight: Keypoints are at JOINT positions which barely change with body shape.
Body WIDTH changes are only visible in the SILHOUETTE between joints.

Solution: Compare projected mesh silhouette with segmentation mask.
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
log("BODY RECONSTRUCTION v2.7 (SEGMENTATION-BASED)")
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
# POSE DETECTION WITH SEGMENTATION
# ============================================

def detect_pose_with_segmentation(image_rgb, frame_idx=0):
    """Detect pose AND extract segmentation mask"""
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
    
    # Get segmentation mask
    seg_mask = None
    if result.segmentation_masks and len(result.segmentation_masks) > 0:
        seg_mask = result.segmentation_masks[0].numpy_view()
        seg_mask = (seg_mask > 0.5).astype(np.uint8)
    
    # Compute width profile from segmentation
    width_profile = None
    if seg_mask is not None:
        width_profile = compute_width_profile(seg_mask)
    
    return {
        'keypoints': coco_kps,
        'keypoints_full': mp_kps,
        'segmentation': seg_mask,
        'width_profile': width_profile
    }

def compute_width_profile(seg_mask, n_samples=20):
    """
    Compute body width at different heights from segmentation mask.
    Returns: array of (y_position, width) pairs normalized to [0,1]
    """
    h, w = seg_mask.shape
    
    # Find body bounding box
    rows = np.any(seg_mask, axis=1)
    cols = np.any(seg_mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return None
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    body_height = y_max - y_min
    if body_height < 50:
        return None
    
    # Sample width at different heights
    profile = []
    for i in range(n_samples):
        y = int(y_min + (body_height * i / (n_samples - 1)))
        row = seg_mask[y, :]
        
        if np.any(row):
            x_indices = np.where(row)[0]
            width = x_indices[-1] - x_indices[0]
            # Normalize: y_rel is 0 at top, 1 at bottom; width is in pixels
            y_rel = i / (n_samples - 1)
            profile.append({
                'y_rel': y_rel,
                'y_px': y,
                'width_px': width,
                'x_center': (x_indices[0] + x_indices[-1]) / 2
            })
    
    return profile

# ============================================
# GEOMETRIC CAMERA ESTIMATION
# ============================================

def estimate_camera_geometric(keypoints_2d, joints_3d, K, frame_idx=0):
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
    
    R = np.eye(3)
    rotation_angle = 0.0
    
    if keypoints_2d[5, 2] > 0.3 and keypoints_2d[6, 2] > 0.3:
        l_shoulder = keypoints_2d[5, :2]
        r_shoulder = keypoints_2d[6, :2]
        shoulder_px = np.linalg.norm(l_shoulder - r_shoulder)
        shoulder_3d = np.linalg.norm(joints_3d[16] - joints_3d[17])
        expected_shoulder_px = focal * shoulder_3d / distance
        
        if expected_shoulder_px > 10:
            ratio = min(1.0, shoulder_px / expected_shoulder_px)
            rotation_angle = np.arccos(ratio)
            
            angle = rotation_angle
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    
    t = np.array([-dx, -dy, distance])
    
    return {
        'R': R, 't': t, 'K': K.copy(),
        'valid': True, 'distance': distance,
        'rotation_deg': np.degrees(rotation_angle)
    }

# ============================================
# SEGMENTATION-BASED SILHOUETTE LOSS
# ============================================

def compute_silhouette_width_loss(vertices, pose_data, cam, device, image_height):
    """
    Compare body width from segmentation with projected mesh width.
    
    This is the KEY function that captures body THICKNESS!
    """
    import torch
    
    if pose_data is None or cam is None:
        return None
    
    width_profile = pose_data.get('width_profile')
    if width_profile is None or len(width_profile) < 5:
        return None
    
    try:
        R = torch.tensor(cam['R'], dtype=torch.float32, device=device)
        t = torch.tensor(cam['t'], dtype=torch.float32, device=device)
        K = torch.tensor(cam['K'], dtype=torch.float32, device=device)
        
        # Project all vertices
        cam_pts = (R @ vertices.T).T + t
        
        valid_depth = cam_pts[:, 2] > 0.1
        if torch.sum(valid_depth) < 100:
            return None
        
        proj = (K @ cam_pts.T).T
        proj_2d = proj[:, :2] / (proj[:, 2:3] + 1e-8)
        
        # Get projected mesh bounds
        proj_y_min = proj_2d[:, 1].min()
        proj_y_max = proj_2d[:, 1].max()
        proj_height = proj_y_max - proj_y_min
        
        if proj_height < 10:
            return None
        
        # Compare widths at different heights
        total_loss = torch.tensor(0.0, device=device)
        count = 0
        
        for wp in width_profile:
            y_rel = wp['y_rel']
            observed_width = wp['width_px']
            
            # Find projected vertices at this relative height
            target_y = proj_y_min + proj_height * y_rel
            
            # Find vertices within a small band around target_y
            band = proj_height * 0.05  # 5% band
            in_band = (proj_2d[:, 1] >= target_y - band) & (proj_2d[:, 1] <= target_y + band)
            
            if torch.sum(in_band) > 10:
                band_pts = proj_2d[in_band]
                proj_width = band_pts[:, 0].max() - band_pts[:, 0].min()
                
                # Loss: squared difference in width
                width_diff = (proj_width - observed_width) / image_height
                total_loss = total_loss + width_diff ** 2
                count += 1
        
        if count > 0:
            return total_loss / count
        
        return None
        
    except Exception as e:
        return None

# ============================================
# SHAPE OPTIMIZATION WITH SEGMENTATION
# ============================================

def optimize_shape_with_segmentation(pose_results, cameras, body_model, device,
                                     image_width, image_height, n_iterations=400):
    """
    Optimize body shape using:
    1. Keypoint loss (joint positions)
    2. Silhouette width loss (body thickness at different heights)
    """
    import torch
    
    log("Starting SEGMENTATION-BASED optimization...", "INFO")
    
    # Count frames with segmentation
    seg_count = sum(1 for p in pose_results if p and p.get('width_profile'))
    log(f"Frames with width profile: {seg_count}/{len(pose_results)}", "INFO")
    
    img_diag = np.sqrt(image_width**2 + image_height**2)
    
    betas = torch.zeros(1, 10, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([betas], lr=0.05)
    
    best_loss = float('inf')
    best_betas = betas.clone().detach()
    
    smplx_indices = list(COCO_TO_SMPLX.values())
    coco_indices = list(COCO_TO_SMPLX.keys())
    
    for iteration in range(n_iterations):
        
        if iteration == 200:
            log("Switching to Phase 2 (refinement)...", "INFO")
            optimizer = torch.optim.Adam([betas], lr=0.01)
        
        reg_weight = 0.1 if iteration < 200 else 0.3
        
        optimizer.zero_grad()
        
        output = body_model(
            betas=betas,
            body_pose=torch.zeros(1, 63, device=device),
            global_orient=torch.zeros(1, 3, device=device)
        )
        joints = output.joints[0]
        vertices = output.vertices[0]
        
        # Keypoint Loss
        kp_loss = torch.tensor(0.0, device=device)
        kp_count = 0
        
        # Silhouette Width Loss (NEW!)
        sil_loss = torch.tensor(0.0, device=device)
        sil_count = 0
        
        for pose, cam in zip(pose_results, cameras):
            if pose is None or cam is None:
                continue
            
            try:
                R = torch.tensor(cam['R'], dtype=torch.float32, device=device)
                t = torch.tensor(cam['t'], dtype=torch.float32, device=device)
                K = torch.tensor(cam['K'], dtype=torch.float32, device=device)
                
                # Keypoint loss
                body_joints = torch.stack([joints[i] for i in smplx_indices])
                cam_pts = (R @ body_joints.T).T + t
                
                if not torch.any(cam_pts[:, 2] <= 0.1):
                    proj = (K @ cam_pts.T).T
                    proj_2d = proj[:, :2] / (proj[:, 2:3] + 1e-8)
                    
                    gt_2d = torch.tensor(pose['keypoints'][coco_indices, :2],
                                        dtype=torch.float32, device=device)
                    conf = torch.tensor(pose['keypoints'][coco_indices, 2],
                                       dtype=torch.float32, device=device)
                    
                    diff = (proj_2d - gt_2d) / img_diag
                    kp_loss = kp_loss + torch.sum(conf.unsqueeze(-1) * diff**2) / (torch.sum(conf) + 1e-8)
                    kp_count += 1
                
                # Silhouette width loss
                sil = compute_silhouette_width_loss(vertices, pose, cam, device, image_height)
                if sil is not None:
                    sil_loss = sil_loss + sil
                    sil_count += 1
                    
            except:
                pass
        
        if kp_count > 0:
            kp_loss = kp_loss / kp_count
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
        
        # Beta Regularization
        beta_l2 = torch.mean(betas**2)
        beta_boundary = torch.sum(torch.relu(torch.abs(betas) - 2.5)**2)
        shape_reg = beta_l2 + 5.0 * beta_boundary
        
        # Total Loss - SILHOUETTE WEIGHT IS HIGH!
        total_loss = kp_loss + 5.0 * sil_loss + 0.5 * sym_loss + 0.5 * ratio_loss + reg_weight * shape_reg
        
        if total_loss.requires_grad:
            total_loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            betas.data = torch.clamp(betas.data, -3.0, 3.0)
        
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_betas = betas.clone().detach()
        
        if iteration % 50 == 0:
            beta_str = ", ".join([f"{b:.2f}" for b in betas[0, :5].detach().cpu().numpy()])
            log(f"Iter {iteration}: loss={total_loss.item():.4f}, kp={kp_loss.item():.4f}, "
                f"sil={sil_loss.item():.4f}, betas=[{beta_str}]", "DEBUG")
    
    log(f"Best loss: {best_loss:.4f}", "SUCCESS")
    log(f"Final betas: {best_betas[0].cpu().numpy().round(3)}", "SUCCESS")
    
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

def process_video(video_file, smplx_file, n_frames=16, known_height=None, progress=gr.Progress()):
    import torch
    import smplx
    import shutil
    import tempfile
    
    log("\n" + "="*60)
    log("PROCESSING VIDEO (v2.7 - SEGMENTATION)")
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
    
    log(f"Extracted {len(frames)} frames")
    
    # Detect poses WITH segmentation
    progress(0.2, desc="Detecting poses & segmentation...")
    pose_results = []
    for i, frame in enumerate(frames):
        result = detect_pose_with_segmentation(frame, i)
        pose_results.append(result)
        if result:
            n_kp = sum(1 for k in result['keypoints'] if k[2] > 0.3)
            has_seg = "✓" if result.get('width_profile') else "✗"
            log(f"Frame {i}: {n_kp}/17 kp, seg={has_seg}")
    
    pose_count = sum(1 for r in pose_results if r is not None)
    seg_count = sum(1 for r in pose_results if r and r.get('width_profile'))
    
    if pose_count == 0:
        return None, "❌ No poses detected", None, None
    
    log(f"Poses: {pose_count}/{len(frames)}, Segmentations: {seg_count}/{len(frames)}")
    
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
        if cam:
            log(f"Frame {i}: dist={cam['distance']:.2f}m, rot={cam['rotation_deg']:.0f}°", "DEBUG")
    
    valid_cams = sum(1 for c in cameras if c is not None)
    log(f"Cameras: {valid_cams}/{pose_count}")
    
    # Optimization with segmentation
    progress(0.4, desc="Optimizing shape (with segmentation)...")
    
    betas = optimize_shape_with_segmentation(
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
    
    # Visualization with segmentation overlay
    annotated = []
    for i, (frame, pose) in enumerate(zip(frames, pose_results)):
        img = frame.copy()
        
        # Draw segmentation if available
        if pose and pose.get('segmentation') is not None:
            seg = pose['segmentation']
            overlay = np.zeros_like(img)
            overlay[:, :, 1] = seg * 100  # Green overlay
            img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
        
        # Draw keypoints
        if pose:
            kp = pose['keypoints']
            for s, e in [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]:
                if kp[s,2] > 0.3 and kp[e,2] > 0.3:
                    cv2.line(img, (int(kp[s,0]), int(kp[s,1])), (int(kp[e,0]), int(kp[e,1])), (0,255,255), 2)
            for x, y, c in kp:
                if c > 0.3:
                    cv2.circle(img, (int(x), int(y)), 5, (255,0,0), -1)
        
        annotated.append(cv2.resize(img, (240, 180)))
    
    cols = 4
    rows = (len(annotated) + cols - 1) // cols
    while len(annotated) < rows * cols:
        annotated.append(np.zeros((180, 240, 3), dtype=np.uint8))
    grid = np.vstack([np.hstack(annotated[i*cols:(i+1)*cols]) for i in range(rows)])
    
    m = measurements
    summary = f"""## ✅ Processing Complete (v2.7 - Segmentation)

### Pipeline Info
- Resolution: {width} x {height}
- Frames: {len(frames)} | Poses: {pose_count} | Segmentations: {seg_count}
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

*{'Scaled to provided height.' if known_height else 'Using default scale.'}*
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
        'pipeline_version': '2.7',
    }, temp_json, indent=2)
    temp_json.close()
    
    progress(1.0, desc="Done!")
    return grid, summary, temp_obj.name, temp_json.name

# ============================================
# GRADIO INTERFACE
# ============================================

with gr.Blocks(title="Body Reconstruction v2.7") as demo:
    gr.Markdown("""
    # 🧍 Human Body Reconstruction v2.7 (SEGMENTATION)
    
    ### Key Innovation:
    Uses **segmentation masks** to measure body width at different heights!
    
    Previous versions only used keypoints (joint positions), which don't 
    capture body THICKNESS. Now we compare actual body silhouette width.
    
    **Green overlay** in results shows detected body silhouette.
    """)
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Video")
            smplx_input = gr.File(label="SMPL-X Model (SMPLX_NEUTRAL.npz)", file_types=[".npz"])
            n_frames = gr.Slider(8, 24, value=16, step=2, label="Number of Frames")
            known_height = gr.Number(label="Known Height (cm)", value=None,
                                    info="Enter actual height for accurate measurements")
            process_btn = gr.Button("🚀 Process", variant="primary", size="lg")
        
        with gr.Column():
            output_image = gr.Image(label="Pose & Segmentation Results")
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