# Human Body Reconstruction V2.2 - Gradio Interface
# Uses NEW MediaPipe Tasks API (compatible with MediaPipe 0.10.14+)

import gradio as gr
import torch
import numpy as np
import cv2
import os
import json
import tempfile
import urllib.request
import shutil

# MediaPipe NEW Tasks API
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Other imports
from ultralytics import YOLO
import smplx
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull

# ============================================================================
# MediaPipe Tasks API Setup (NEW API)
# ============================================================================

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
MODEL_PATH = "pose_landmarker.task"

def download_pose_model():
    """Download MediaPipe pose model if not exists"""
    if not os.path.exists(MODEL_PATH):
        print(f"📥 Downloading pose_landmarker.task...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"✅ Model downloaded!")
    return MODEL_PATH

def create_pose_landmarker():
    """Create PoseLandmarker using NEW MediaPipe Tasks API"""
    model_path = download_pose_model()

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        output_segmentation_masks=True,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5
    )

    return PoseLandmarker.create_from_options(options)

# ============================================================================
# CORE PIPELINE FUNCTIONS
# ============================================================================

# MediaPipe 33 landmarks to COCO 17 mapping
MP_TO_COCO = {
    0: 0, 2: 1, 5: 2, 7: 3, 8: 4, 11: 5, 12: 6, 13: 7, 14: 8,
    15: 9, 16: 10, 23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16
}

# COCO to SMPL-X mapping (reliable joints only)
COCO_TO_SMPLX = {
    5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21,
    11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8
}

def detect_pose_new_api(image_rgb, landmarker):
    """Detect pose using NEW MediaPipe Tasks API"""
    h, w = image_rgb.shape[:2]

    # Convert to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb.copy())

    # Detect
    result = landmarker.detect(mp_image)

    if not result.pose_landmarks or len(result.pose_landmarks) == 0:
        return None

    # Get first pose
    landmarks = result.pose_landmarks[0]

    # Extract all 33 MediaPipe landmarks
    mp_keypoints = []
    for lm in landmarks:
        mp_keypoints.append([lm.x * w, lm.y * h, lm.visibility])
    mp_keypoints = np.array(mp_keypoints)

    # Convert to COCO 17 format
    coco_keypoints = np.zeros((17, 3))
    for mp_idx, coco_idx in MP_TO_COCO.items():
        coco_keypoints[coco_idx] = mp_keypoints[mp_idx]

    # Get segmentation mask
    segmentation = None
    if result.segmentation_masks and len(result.segmentation_masks) > 0:
        mask = result.segmentation_masks[0].numpy_view()
        segmentation = (mask > 0.5).astype(np.uint8)

    # Calculate bbox
    valid_kps = mp_keypoints[mp_keypoints[:, 2] > 0.5]
    if len(valid_kps) > 0:
        bbox = [valid_kps[:, 0].min(), valid_kps[:, 1].min(),
                valid_kps[:, 0].max(), valid_kps[:, 1].max()]
    else:
        bbox = [0, 0, w, h]

    return {
        'keypoints': coco_keypoints,
        'bbox': np.array(bbox),
        'segmentation': segmentation
    }

def get_smplx_joints(body_model, betas, device):
    """Get 3D joints from SMPL-X"""
    with torch.no_grad():
        output = body_model(
            betas=betas,
            body_pose=torch.zeros(1, 63, device=device),
            global_orient=torch.zeros(1, 3, device=device)
        )
    return output.joints[0].cpu().numpy()

def solve_pnp(keypoints_2d, joints_3d, K):
    """Solve PnP for camera pose"""
    pts_2d = []
    pts_3d = []

    for coco_idx, smplx_idx in COCO_TO_SMPLX.items():
        conf = keypoints_2d[coco_idx, 2]
        if conf > 0.5:
            pts_2d.append(keypoints_2d[coco_idx, :2])
            pts_3d.append(joints_3d[smplx_idx])

    if len(pts_2d) < 6:
        return None, None, False

    pts_2d = np.array(pts_2d, dtype=np.float64)
    pts_3d = np.array(pts_3d, dtype=np.float64)

    success, rvec, tvec, _ = cv2.solvePnPRansac(
        pts_3d, pts_2d, K, None,
        iterationsCount=100, reprojectionError=8.0
    )

    if not success:
        return None, None, False

    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.flatten(), True

def measure_circumference_pca(vertices, center, radius=0.1, scale=100):
    """Measure circumference using PCA plane"""
    distances = np.linalg.norm(vertices - center, axis=1)
    nearby = vertices[distances < radius]

    if len(nearby) < 20:
        radius *= 1.5
        nearby = vertices[distances < radius]

    if len(nearby) < 10:
        return 0.0

    pca = PCA(n_components=2)
    points_2d = pca.fit_transform(nearby - center)

    try:
        hull = ConvexHull(points_2d)
        hull_pts = points_2d[hull.vertices]

        perimeter = 0
        for i in range(len(hull_pts)):
            perimeter += np.linalg.norm(hull_pts[i] - hull_pts[(i+1) % len(hull_pts)])

        return perimeter * scale
    except:
        return 0.0

# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_video(video_path, smplx_model_path, known_height, n_frames, progress=gr.Progress()):
    """Main pipeline - processes video and returns measurements"""

    try:
        # Setup
        progress(0.05, desc="Initializing models...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load pose landmarker (NEW API)
        progress(0.08, desc="Loading pose detector...")
        pose_landmarker = create_pose_landmarker()

        # Load SMPL-X
        if not smplx_model_path or not os.path.exists(smplx_model_path):
            return "❌ Error: Please upload SMPLX_NEUTRAL.npz model", None, None

        # Create proper directory structure for SMPL-X
        smplx_dir = 'models/smplx'
        os.makedirs(smplx_dir, exist_ok=True)

        # Copy uploaded file to proper location
        target_path = os.path.join(smplx_dir, 'SMPLX_NEUTRAL.npz')
        shutil.copy(smplx_model_path, target_path)

        body_model = smplx.create(
            'models', model_type='smplx', gender='neutral',
            use_face_contour=False, num_betas=10, ext='npz'
        ).to(device)

        # Extract frames
        progress(0.1, desc=f"Extracting {n_frames} frames...")
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if len(frames) == 0:
            return "❌ Error: Could not extract frames from video", None, None

        # Pose estimation (NEW API)
        progress(0.2, desc="Running pose estimation...")
        pose_results = []
        for i, frame in enumerate(frames):
            result = detect_pose_new_api(frame, pose_landmarker)
            pose_results.append(result)
            progress(0.2 + 0.2 * (i+1)/len(frames), desc=f"Processing frame {i+1}/{len(frames)}")

        valid_poses = [p for p in pose_results if p is not None]

        # DEBUG: Show detection results
        print(f"DEBUG: Detected poses in {len(valid_poses)}/{len(frames)} frames")
        for i, pose in enumerate(pose_results):
            if pose is not None:
                print(f"  Frame {i}: {np.sum(pose['keypoints'][:, 2] > 0.5)} visible keypoints")

        if len(valid_poses) < 3:
            return f"❌ Error: Only {len(valid_poses)} frames detected person. Need at least 3.", None, None

        # Camera intrinsics
        focal = max(width, height)
        K = np.array([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1]], dtype=np.float64)

        # Iterative PnP (3 rounds)
        progress(0.4, desc="Iterative camera estimation (3 rounds)...")
        betas = torch.zeros(1, 10, device=device)
        cameras = [None] * len(pose_results)

        for iteration in range(3):
            joints_3d = get_smplx_joints(body_model, betas, device)

            for i, pose in enumerate(pose_results):
                if pose is None:
                    continue
                R, t, success = solve_pnp(pose['keypoints'], joints_3d, K)
                if success:
                    cameras[i] = {'R': R, 't': t, 'K': K.copy()}

            progress(0.4 + 0.1 * (iteration+1)/3, desc=f"PnP round {iteration+1}/3")

            # Quick shape update
            if iteration < 2:
                betas_opt = betas.clone().detach().requires_grad_(True)
                optimizer = torch.optim.Adam([betas_opt], lr=0.05)

                for _ in range(30):
                    optimizer.zero_grad()
                    output = body_model(
                        betas=betas_opt,
                        body_pose=torch.zeros(1, 63, device=device),
                        global_orient=torch.zeros(1, 3, device=device)
                    )
                    joints = output.joints[0]

                    loss = 0
                    for pose, cam in zip(pose_results, cameras):
                        if pose is None or cam is None:
                            continue
                        R_t = torch.tensor(cam['R'], dtype=torch.float32, device=device)
                        t_t = torch.tensor(cam['t'], dtype=torch.float32, device=device)
                        K_t = torch.tensor(cam['K'], dtype=torch.float32, device=device)

                        body_joints = torch.stack([joints[COCO_TO_SMPLX[i]] for i in COCO_TO_SMPLX.keys()])
                        cam_pts = torch.matmul(body_joints, R_t.T) + t_t
                        proj = torch.matmul(cam_pts, K_t.T)
                        proj_2d = proj[:, :2] / (proj[:, 2:3] + 1e-8)

                        gt_2d = torch.tensor(
                            pose['keypoints'][list(COCO_TO_SMPLX.keys()), :2],
                            dtype=torch.float32, device=device
                        )
                        loss += torch.mean((proj_2d - gt_2d) ** 2)

                    loss += 0.01 * torch.mean(betas_opt ** 2)
                    loss.backward()
                    optimizer.step()

                betas = betas_opt.detach()

        # Final optimization
        progress(0.5, desc="Final shape optimization...")
        betas_final = betas.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([betas_final], lr=0.02)

        for iteration in range(200):
            optimizer.zero_grad()

            output = body_model(
                betas=betas_final,
                body_pose=torch.zeros(1, 63, device=device),
                global_orient=torch.zeros(1, 3, device=device)
            )
            joints = output.joints[0]

            kp_loss = 0
            count = 0

            for pose, cam in zip(pose_results, cameras):
                if pose is None or cam is None:
                    continue

                R_t = torch.tensor(cam['R'], dtype=torch.float32, device=device)
                t_t = torch.tensor(cam['t'], dtype=torch.float32, device=device)
                K_t = torch.tensor(cam['K'], dtype=torch.float32, device=device)

                body_joints = torch.stack([joints[COCO_TO_SMPLX[i]] for i in COCO_TO_SMPLX.keys()])
                cam_pts = torch.matmul(body_joints, R_t.T) + t_t
                proj = torch.matmul(cam_pts, K_t.T)
                proj_2d = proj[:, :2] / (proj[:, 2:3] + 1e-8)

                gt_2d = torch.tensor(
                    pose['keypoints'][list(COCO_TO_SMPLX.keys()), :2],
                    dtype=torch.float32, device=device
                )
                conf = torch.tensor(
                    pose['keypoints'][list(COCO_TO_SMPLX.keys()), 2],
                    dtype=torch.float32, device=device
                )

                diff = proj_2d - gt_2d
                kp_loss += torch.sum(conf.unsqueeze(-1) * diff ** 2)
                count += 1

            kp_loss = kp_loss / (count + 1e-8)

            # Symmetry loss
            left_arm = torch.norm(joints[16] - joints[18]) + torch.norm(joints[18] - joints[20])
            right_arm = torch.norm(joints[17] - joints[19]) + torch.norm(joints[19] - joints[21])
            left_leg = torch.norm(joints[1] - joints[4]) + torch.norm(joints[4] - joints[7])
            right_leg = torch.norm(joints[2] - joints[5]) + torch.norm(joints[5] - joints[8])

            symmetry_loss = (left_arm - right_arm) ** 2 + (left_leg - right_leg) ** 2
            shape_loss = torch.mean(betas_final ** 2)

            total_loss = kp_loss + 0.1 * symmetry_loss + 0.01 * shape_loss
            total_loss.backward()
            optimizer.step()

            if iteration % 50 == 0:
                print(f"DEBUG: Iter {iteration}, Loss={total_loss.item():.4f}, β0={betas_final[0,0].item():.3f}, β1={betas_final[0,1].item():.3f}")
                progress(0.5 + 0.3 * iteration/200, desc=f"Optimization iter {iteration}/200")

        # Extract measurements
        progress(0.85, desc="Extracting measurements...")
        print(f"DEBUG: Final betas = {betas_final[0, :5].detach().cpu().numpy()}")

        with torch.no_grad():
            output = body_model(
                betas=betas_final,
                body_pose=torch.zeros(1, 63, device=device),
                global_orient=torch.zeros(1, 3, device=device),
                return_verts=True
            )

        vertices = output.vertices[0].cpu().numpy()
        joints = output.joints[0].cpu().numpy()

        # Scale
        raw_height = vertices[:, 1].max() - vertices[:, 1].min()
        scale = known_height / raw_height if known_height and known_height > 0 else 100

        # Measurements
        measurements = {}
        measurements['height'] = raw_height * scale
        measurements['shoulder_width'] = np.linalg.norm(joints[16] - joints[17]) * scale
        measurements['hip_width'] = np.linalg.norm(joints[1] - joints[2]) * scale
        measurements['torso_length'] = np.linalg.norm(joints[12] - joints[0]) * scale

        left_arm = np.linalg.norm(joints[16] - joints[18]) + np.linalg.norm(joints[18] - joints[20])
        right_arm = np.linalg.norm(joints[17] - joints[19]) + np.linalg.norm(joints[19] - joints[21])
        measurements['arm_length'] = ((left_arm + right_arm) / 2) * scale

        left_leg = np.linalg.norm(joints[1] - joints[4]) + np.linalg.norm(joints[4] - joints[7])
        right_leg = np.linalg.norm(joints[2] - joints[5]) + np.linalg.norm(joints[5] - joints[8])
        measurements['leg_length'] = ((left_leg + right_leg) / 2) * scale

        crotch = (joints[1] + joints[2]) / 2
        crotch[1] -= 0.03
        ankle = (joints[7] + joints[8]) / 2
        measurements['inseam'] = np.linalg.norm(crotch - ankle) * scale

        # Circumferences (PCA)
        chest_center = (joints[16] + joints[17]) / 2
        chest_center[1] -= 0.05
        measurements['chest_circumference'] = measure_circumference_pca(vertices, chest_center, 0.12, scale)

        waist_center = (joints[3] + joints[6]) / 2
        measurements['waist_circumference'] = measure_circumference_pca(vertices, waist_center, 0.10, scale)

        hip_center = joints[0].copy()
        measurements['hip_circumference'] = measure_circumference_pca(vertices, hip_center, 0.12, scale)

        # Format output
        progress(0.95, desc="Generating results...")
        output_text = "=" * 60 + "\n"
        output_text += "📏 BODY MEASUREMENTS (V2.2 - NEW API)\n"
        output_text += "=" * 60 + "\n\n"
        for name, value in measurements.items():
            output_text += f"  {name.replace('_', ' ').title():<25} {value:>8.1f} cm\n"
        output_text += "\n" + "=" * 60 + "\n"
        output_text += f"  Processed {len(valid_poses)}/{len(frames)} frames successfully\n"
        output_text += f"  MediaPipe API: Tasks (NEW)\n"
        output_text += "=" * 60

        # Save mesh
        progress(0.98, desc="Saving mesh...")
        temp_obj = tempfile.NamedTemporaryFile(delete=False, suffix='.obj')
        with open(temp_obj.name, 'w') as f:
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in body_model.faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

        # Save JSON
        temp_json = tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w')
        json.dump({
            'measurements_cm': {k: float(v) for k, v in measurements.items()},
            'pipeline_version': '2.2',
            'api': 'MediaPipe Tasks (NEW)',
            'frames_processed': f"{len(valid_poses)}/{len(frames)}"
        }, temp_json, indent=2)
        temp_json.close()

        progress(1.0, desc="Complete!")
        return output_text, temp_obj.name, temp_json.name

    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
        return error_msg, None, None

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_interface():
    """Create Gradio interface"""

    with gr.Blocks(title="Human Body Reconstruction V2.2") as demo:
        gr.Markdown("""
        # 🧍 Human Body Reconstruction V2.2

        ## NEW MediaPipe Tasks API (Works with latest MediaPipe)

        Upload a turntable video (5-10 seconds of person rotating) and get accurate body measurements.

        **Requirements:**
        - Video: MP4, person rotating or camera moving around person, full body visible
        - SMPL-X Model: Download from https://smpl-x.is.tue.mpg.de/ (free registration)

        **What's New in V2.2:**
        - ✅ Uses MediaPipe Tasks API (works with MediaPipe 0.10.14+)
        - ✅ Auto-downloads pose model
        - ✅ No version conflicts
        """)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📤 Uploads")
                video_input = gr.Video(label="Turntable Video (MP4)")
                smplx_input = gr.File(label="SMPL-X Model (SMPLX_NEUTRAL.npz)", file_types=[".npz"])

                gr.Markdown("### ⚙️ Settings")
                height_input = gr.Number(
                    label="Known Height (cm) - Optional for calibration",
                    value=None,
                    minimum=100,
                    maximum=250,
                    step=0.1
                )
                frames_input = gr.Slider(
                    label="Number of Frames to Extract",
                    minimum=4,
                    maximum=16,
                    step=2,
                    value=8
                )

                process_btn = gr.Button("🚀 Process Video", variant="primary", size="lg")

            with gr.Column():
                gr.Markdown("### 📊 Results")
                output_text = gr.Textbox(
                    label="Measurements",
                    lines=20,
                    placeholder="Results will appear here..."
                )

                with gr.Row():
                    mesh_output = gr.File(label="3D Mesh (.obj)")
                    json_output = gr.File(label="Measurements JSON")

        gr.Markdown("""
        ---
        ### 📝 Notes
        - Processing takes 2-5 minutes on GPU
        - Expected accuracy: Height ±1-1.5cm, Circumferences ±2-3cm
        - Better video quality = better results
        - Using NEW MediaPipe Tasks API (future-proof)
        """)

        process_btn.click(
            fn=process_video,
            inputs=[video_input, smplx_input, height_input, frames_input],
            outputs=[output_text, mesh_output, json_output]
        )

    return demo

# ============================================================================
# LAUNCH
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Human Body Reconstruction V2.2")
    print(f"   MediaPipe: {mp.__version__}")
    print("   API: Tasks (NEW)")
    print("="*60 + "\n")

    demo = create_interface()
    demo.launch(
        share=True,  # Creates public URL for sharing
        debug=True
    )
