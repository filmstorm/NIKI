#!/usr/bin/env python3
# """Video demo script, modified to save NIKI+HybrIK output for BVH conversion."""
import argparse
import os
import pickle as pk
import sys

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict

# --- NIKI / HybrIK Imports ---
from niki.utils.hybrik_utils import builder
from niki.utils.config import update_config
from niki.utils.hybrik_utils.simple_transform_3d_smpl_cam import SimpleTransform3DSMPLCam
from niki.utils.render_pytorch3d import render_mesh
from niki.models.NIKI_1stage import FlowIK_camnet

# --- Import specific functions directly ---
from niki.utils.demo_utils import xyxy_to_center_scale_batch, center_scale_to_box, get_one_box, reproject_uv

# --- REMOVED Faulty Import ---
# from niki.utils.transforms import aa_to_rotmat

# --- ADDED Scipy for rotation conversion fallback ---
from scipy.spatial.transform import Rotation as SciPyRotation

# Other standard imports
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm

# --- (Rest of the script setup remains the same) ---

det_transform = T.Compose([T.ToTensor()])

# --- Helper Functions (Keep as is) ---
def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox; cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
    w = x2 - x1; h = y2 - y1; return [cx, cy, w, h]

def get_video_info(in_file):
    stream = cv2.VideoCapture(in_file); assert stream.isOpened(), f'Cannot capture source: {in_file}'
    datalen = int(stream.get(cv2.CAP_PROP_FRAME_COUNT)); fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))
    fps = stream.get(cv2.CAP_PROP_FPS); frameSize = (int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoinfo = {'fourcc': fourcc, 'fps': fps, 'frameSize': frameSize}; stream.release()
    return stream, videoinfo, datalen

def recognize_video_ext(ext=''):
    if ext == 'mp4': return cv2.VideoWriter_fourcc(*'mp4v'), '.' + ext
    elif ext == 'avi': return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
    elif ext == 'mov': return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
    else: print(f"Unknown video format {ext}, will use .mp4"); return cv2.VideoWriter_fourcc(*'mp4v'), '.mp4'

# --- Argument Parsing (Keep as is) ---
parser = argparse.ArgumentParser(description='NIKI Enhanced HybrIK Demo with BVH Export V5 (Scipy Fix)') # V5
parser.add_argument('--video-name', help='video name', required=True, type=str)
parser.add_argument('--out-dir', help='output folder', required=True, type=str)
parser.add_argument('--gpu', help='gpu ID', default=0, type=int)
parser.add_argument('--flip-test', default=False, help='flip test', action='store_true')
parser.add_argument('--save-pk', default=True, dest='save_pk', help='Save final prediction parameters for BVH as pickle', action=argparse.BooleanOptionalAction)
parser.add_argument('--not-vis', default=False, dest='not_vis', help='do not visualize', action='store_true')
parser.add_argument('--hybrik_cam', default=True, help='use camera parameter predict by HybrIK', type=bool)
parser.add_argument('--verbose', default=False, dest='verbose', help='Print verbose output', action=argparse.BooleanOptionalAction)
opt = parser.parse_args()


# --- Config and Model Loading (Keep as is, including file checks) ---
cfg_file = 'configs/hybrik_config.yaml'
CKPT = 'exp/checkpoint_49_cocoeft.pth' # HybrIK weights
v_cfg_file = 'configs/NIKI-1stage.yaml'
V_CKPT = 'exp/niki_model_28.pth' # NIKI weights
camnet_dict_path = 'exp/niki_model_28.pth'

if not os.path.isfile(cfg_file): print(f"Error: Config file not found: {cfg_file}"); sys.exit(1)
if not os.path.isfile(CKPT): print(f"Error: HybrIK checkpoint not found: {CKPT}"); sys.exit(1)
if not os.path.isfile(v_cfg_file): print(f"Error: NIKI config file not found: {v_cfg_file}"); sys.exit(1)
if not os.path.isfile(V_CKPT): print(f"Error: NIKI checkpoint not found: {V_CKPT}"); sys.exit(1)
if not os.path.isfile(camnet_dict_path): print(f"Error: CamNet weights file not found: {camnet_dict_path}"); sys.exit(1)

cfg = update_config(cfg_file)
v_cfg = update_config(v_cfg_file)
bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2200, 2200, 2200))
bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]
dummpy_set = edict({'joint_pairs_17': None, 'joint_pairs_24': None, 'joint_pairs_29': None, 'bbox_3d_shape': bbox_3d_shape})

res_keys = ['pred_uvd', 'pred_xyz_29', 'pred_scores', 'pred_sigma', 'f', 'pred_betas',
            'pred_phi', 'pred_cam_root', 'bbox', 'height', 'width',
            'img_path', 'img_sizes']
res_db = {k: [] for k in res_keys}
bvh_keys = ['pred_thetas', 'pred_betas', 'transl', 'transl_camsys', 'frame_idx',
            'bbox', 'height', 'width', 'f']
bvh_save_db = {k: [] for k in bvh_keys}

transformation = SimpleTransform3DSMPLCam(
    dummpy_set, scale_factor=cfg.DATASET.SCALE_FACTOR, color_factor=cfg.DATASET.COLOR_FACTOR,
    occlusion=cfg.DATASET.OCCLUSION, input_size=cfg.MODEL.IMAGE_SIZE, output_size=cfg.MODEL.HEATMAP_SIZE,
    depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM, bbox_3d_shape=bbox_3d_shape, rot=cfg.DATASET.ROT_FACTOR,
    sigma=cfg.MODEL.EXTRA.SIGMA, train=False, add_dpg=False, loss_type=cfg.LOSS['TYPE'])

try: det_model = fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT'); print("Loaded FasterRCNN (Default weights)")
except TypeError: det_model = fasterrcnn_resnet50_fpn(pretrained=True); print("Loaded FasterRCNN (pretrained)")
hybrik_model = builder.build_sppe(cfg.MODEL); print("Built HybrIK model")
flow_model = FlowIK_camnet(v_cfg); print("Built FlowIK model")

print(f'Loading HybrIK model from {CKPT}...'); save_dict = torch.load(CKPT, map_location='cpu', weights_only=False)
if type(save_dict) == dict: hybrik_model.load_state_dict(save_dict.get('model', save_dict.get('state_dict', save_dict)))
else: hybrik_model.load_state_dict(save_dict)
print("Loaded HybrIK weights.")

print(f'Loading LGD model from {V_CKPT}'); save_dict = torch.load(V_CKPT, map_location='cpu', weights_only=False)
flow_model.load_state_dict(save_dict, strict=False); print("Loaded FlowIK weights (non-strict).")

tmp_dict = torch.load(camnet_dict_path, map_location='cpu', weights_only=False)
new_tmp_dict = {};
for k, v in tmp_dict.items():
    if 'regressor.camnet' in k: new_tmp_dict[k[len('regressor.camnet.'):]] = v
if not new_tmp_dict: print(f"Warning: No keys starting with 'regressor.camnet.' found in {camnet_dict_path}")
else: flow_model.regressor.camnet.load_state_dict(new_tmp_dict); print("Loaded CamNet weights.")

if opt.gpu >= 0 and torch.cuda.is_available(): device = torch.device(f"cuda:{opt.gpu}")
else: device = torch.device("cpu")
print(f"Using device: {device}")
det_model.to(device); hybrik_model.to(device); flow_model.to(device)
det_model.eval(); hybrik_model.eval(); flow_model.eval()

# --- Video Processing Setup (Keep as is) ---
print('### Preparing Input/Output...')
video_basename = os.path.basename(opt.video_name).split('.')[0]; os.makedirs(opt.out_dir, exist_ok=True)
raw_image_folder = os.path.join(opt.out_dir, 'raw_images'); res_image_folder = os.path.join(opt.out_dir, 'res_images')
write_stream = None
if not opt.not_vis: # Only setup writers if visualizing
    os.makedirs(raw_image_folder, exist_ok=True); os.makedirs(res_image_folder, exist_ok=True)
    _, info, _ = get_video_info(opt.video_name); savepath = f'./{opt.out_dir}/res_niki_{video_basename}.mp4'; info['savepath'] = savepath # Added niki_ to filename
    try: write_stream = cv2.VideoWriter(*[info[k] for k in ['savepath', 'fourcc', 'fps', 'frameSize']]); assert write_stream.isOpened()
    except Exception as e:
        print(f"Video writer failed ({e}), trying fallback."); ext = info['savepath'].split('.')[-1]; fourcc, _ext = recognize_video_ext(ext)
        info['fourcc'] = fourcc; info['savepath'] = info['savepath'][:-4] + _ext
        try: write_stream = cv2.VideoWriter(*[info[k] for k in ['savepath', 'fourcc', 'fps', 'frameSize']]); assert write_stream.isOpened()
        except Exception as e2: print(f"Fallback video writer failed: {e2}"); opt.not_vis = True
    if not opt.not_vis and not write_stream.isOpened(): print("Error: Cannot open video writer."); opt.not_vis = True

# --- Image Path Reading (Keep as is) ---
print("### Reading image frames...")
if not os.path.exists(raw_image_folder) or not os.listdir(raw_image_folder): # Check if empty too
     print(f"Extracting frames using ffmpeg (output to {raw_image_folder})..."); os.makedirs(raw_image_folder, exist_ok=True)
     ffmpeg_cmd = f'ffmpeg -i "{opt.video_name}" "{os.path.join(raw_image_folder, f"{video_basename}-%06d.png")}" -y -loglevel error'
     print(f"Running: {ffmpeg_cmd}"); status = os.system(ffmpeg_cmd) # Capture status
     if status != 0: print(f"ERROR: ffmpeg command failed with status {status}"); sys.exit(1)
else: print(f"Using existing frames from {raw_image_folder}")
files = os.listdir(raw_image_folder); files.sort()
img_path_list = []
for file in files:
    if not os.path.isdir(file) and file.lower().endswith(('.jpg', '.png')): img_path_list.append(os.path.join(raw_image_folder, file))
if not img_path_list: print("ERROR: No image frames found or extracted."); sys.exit(1)
print(f"Found {len(img_path_list)} image frames.")

# --- Prepare for Model Execution (Keep as is) ---
prev_box = None; renderer = None
try: smpl_faces = torch.from_numpy(hybrik_model.smpl.faces.astype(np.int32)); print("Using SMPL faces.")
except AttributeError:
    print("Warning: Could not get SMPL faces from hybrik_model. Trying flow_model...")
    try: smpl_faces = torch.from_numpy(flow_model.smpl.faces.astype(np.int32)); print("Using SMPL faces from flow_model.")
    except AttributeError: print("Error: Could not get SMPL faces from either model."); smpl_faces = None; opt.not_vis = True
except Exception as e:
    print(f"Error getting SMPL faces: {e}"); smpl_faces = None; opt.not_vis = True


# --- STAGE 1: HybrIK Inference Loop (Keep as is) ---
print('### Running Stage 1: HybrIK Initial Pose Estimation...')
processed_frame_count = 0
for frame_idx, img_path in enumerate(tqdm(img_path_list, dynamic_ncols=True)): # Use enumerate for index
    basename = os.path.basename(img_path);
    with torch.no_grad():
        try: # Wrap frame processing in try-except
            input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            if input_image is None: raise IOError(f"Could not read image: {img_path}")
            det_input = det_transform(input_image).to(device); det_output = det_model([det_input])[0]
            current_det_box = get_one_box(det_output)
            if current_det_box is None:
                if prev_box is None:
                    if opt.verbose: print(f"W: No detection frame {basename} & no prev box. Skip.");
                    continue
                else: tight_bbox = prev_box;
            else: tight_bbox = current_det_box
            if prev_box is not None: tight_bbox = [0.6*p + 0.4*c for p, c in zip(prev_box, tight_bbox)]
            prev_box = tight_bbox
            pose_input, bbox_hybrik, img_center = transformation.test_transform(input_image, tight_bbox)
            pose_input = pose_input.to(device)[None, :, :, :]
            pose_output = hybrik_model(pose_input, flip_test=opt.flip_test,
                                    bboxes=torch.from_numpy(np.array(bbox_hybrik)).to(device).unsqueeze(0).float(),
                                    img_center=torch.from_numpy(img_center).to(device).unsqueeze(0).float(),
                                    do_hybrik=False)
            pred_uvd_jts = pose_output.pred_uvd_jts.reshape(-1, 3).cpu().data.numpy()
            pred_xyz_jts_29 = pose_output.pred_xyz_jts_29.reshape(-1, 3).cpu().data.numpy()
            pred_scores = pose_output.maxvals.cpu().data[:, :29].reshape(29).numpy()
            pred_betas = pose_output.pred_shape.squeeze(dim=0).cpu().data.numpy()
            pred_phi = pose_output.pred_phi.squeeze(dim=0).cpu().data.numpy()
            pred_cam_root = pose_output.cam_root.squeeze(dim=0).cpu().numpy()
            pred_sigma = pose_output.sigma.cpu().data.numpy()
            img_size = np.array((input_image.shape[1], input_image.shape[0]))
            res_db['pred_uvd'].append(pred_uvd_jts); res_db['pred_xyz_29'].append(pred_xyz_jts_29)
            res_db['pred_scores'].append(pred_scores); res_db['pred_sigma'].append(pred_sigma)
            res_db['f'].append(1000.0); res_db['pred_betas'].append(pred_betas)
            res_db['pred_phi'].append(pred_phi); res_db['pred_cam_root'].append(pred_cam_root)
            res_db['bbox'].append(np.array(tight_bbox))
            res_db['height'].append(img_size[1]); res_db['width'].append(img_size[0])
            res_db['img_path'].append(img_path); res_db['img_sizes'].append(img_size)
            processed_frame_count += 1
        except Exception as e:
            print(f"E: Stage 1 processing failed for frame {frame_idx} ({basename}): {e}")
            continue

print(f"Stage 1 finished. Processed {processed_frame_count} frames successfully.")
total_img = processed_frame_count
if total_img == 0: print("Error: No images processed successfully in Stage 1."); sys.exit(1)

# --- Stack Intermediate Data (Keep as is) ---
print("Stacking intermediate data...")
for k in res_keys:
    if len(res_db[k]) != total_img:
        print(f"E: Mismatch in length for key '{k}' (expected {total_img}, got {len(res_db[k])}). Cannot proceed."); sys.exit(1)
    try:
        if isinstance(res_db[k][0], np.ndarray): v = np.stack(res_db[k], axis=0)
        else: v = np.array(res_db[k])
    except ValueError as e:
        print(f"W: Could not stack intermediate key '{k}'. Storing as object array. Err: {e}")
        try: v = np.array(res_db[k], dtype=object)
        except Exception as e_obj: print(f"E: Failed to convert key '{k}' to object array: {e_obj}"); sys.exit(1)
    except Exception as e: print(f"E: Failed to stack intermediate key '{k}': {e}"); sys.exit(1)
    res_db[k] = v
    if opt.verbose: print(f"  - Intermediate key '{k}': Stacked shape {v.shape}")


# --- STAGE 2: FlowIK Refinement Loop ---
print('### Running Stage 2: FlowIK Temporal Refinement...')
seq_len = 16
if total_img < seq_len: print(f"Warning: Total processed frames ({total_img}) is less than sequence length ({seq_len}). Using all frames as one sequence."); seq_len = total_img
total_img_seq = (total_img // seq_len) * seq_len
if total_img % seq_len != 0: print(f"Warning: {total_img % seq_len} trailing frames will be ignored as they dont form a full sequence.")
if total_img_seq == 0: print("Error: Not enough frames for even one sequence."); sys.exit(1)

video_res_db = {}
video_res_db['transl'] = torch.zeros((total_img_seq, 3)); video_res_db['vertices'] = torch.zeros((total_img_seq, 6890, 3))
video_res_db['img_path'] = res_db['img_path'][:total_img_seq]; video_res_db['bbox_cs'] = torch.zeros((total_img_seq, 4))
video_res_db['final_thetas'] = torch.zeros((total_img_seq, 24, 3, 3)); video_res_db['final_betas'] = torch.zeros((total_img_seq, 10))
mean_beta = res_db['pred_betas'][:total_img_seq].mean(axis=0)
if mean_beta.shape[0] > 10: mean_beta = mean_beta[:10]
if opt.verbose: print(f"Using mean beta (top 10): {mean_beta}")
update_bbox = v_cfg.get('update_bbox', False); USE_HYBRIK_CAM = opt.hybrik_cam; printed_flow_debug = False

for i in tqdm(range(0, total_img_seq, seq_len), dynamic_ncols=True):
    slc = slice(i, i + seq_len)
    pred_xyz_29 = res_db['pred_xyz_29'][slc] * 2.2; pred_uv = res_db['pred_uvd'][slc, :, :2]
    pred_sigma_slice = res_db['pred_sigma'][slc]; pred_sigma = pred_sigma_slice.squeeze(1) if pred_sigma_slice.ndim == 4 and pred_sigma_slice.shape[1] == 1 else pred_sigma_slice
    if pred_sigma.shape != (seq_len, 29, 29): print(f"Warning: Unexpected pred_sigma shape after squeeze: {pred_sigma.shape}"); continue
    pred_beta = np.tile(mean_beta, (seq_len, 1)); pred_phi = res_db['pred_phi'][slc]; pred_cam_root = res_db['pred_cam_root'][slc]
    pred_cam = np.concatenate((1000.0 / (256 * pred_cam_root[:, [2]] + 1e-9), pred_cam_root[:, :2]), axis=1)
    bbox_xyxy = res_db['bbox'][slc]; bbox_cs = xyxy_to_center_scale_batch(bbox_xyxy)
    inp = {'pred_xyz_29': pred_xyz_29, 'pred_uv': pred_uv, 'pred_sigma': pred_sigma, 'pred_beta': pred_beta, 'pred_phi': pred_phi, 'pred_cam': pred_cam, 'bbox': bbox_cs, 'img_sizes': res_db['img_sizes'][slc]}
    for k in inp.keys():
        try: inp[k] = torch.from_numpy(inp[k]).float().cuda(device).unsqueeze(0)
        except Exception as e: print(f"Error converting input key '{k}' to tensor for sequence {i}: {e}"); continue
    if update_bbox:
        try: inp = reproject_uv(inp)
        except Exception as e: print(f"Error in reproject_uv for sequence {i}: {e}"); continue
    else: img_center = (inp['img_sizes']*0.5 - inp['bbox'][:,:,:2]) / inp['bbox'][:,:,[2]] * 256.0; inp['img_center'] = img_center
    with torch.no_grad():
        try: output = flow_model.forward_getcam(inp=inp)
        except Exception as e: print(f"E: FlowIK forward failed for sequence starting at {i}: {e}"); continue

        if not printed_flow_debug and opt.verbose: # Debug Block (Keep as is)
            print(f"\n--- Debug Info for flow_model output (Sequence starting {i}) ---"); print(f"Type: {type(output)}")
            flow_attrs = ['verts', 'transl', 'pred_theta_mat', 'pred_theta_aa', 'pred_beta', 'inv_pred2uv']
            for attr_name in flow_attrs:
                if hasattr(output, attr_name): val = getattr(output, attr_name); print(f"  - {attr_name}: Present, Type={type(val)}{', Shape='+str(val.shape) if isinstance(val, torch.Tensor) else ''}")
                else: print(f"  - {attr_name}: *** MISSING ***")
            print("--- End Flow Debug ---"); printed_flow_debug = True

        # Store FINAL results (Keep as is)
        video_res_db['vertices'][slc] = output.verts.cpu()[0]; video_res_db['bbox_cs'][slc] = inp['bbox'][0].cpu()
        final_transl = output.transl.cpu()[0]
        if USE_HYBRIK_CAM: final_transl = torch.from_numpy(pred_cam_root)
        video_res_db['transl'][slc] = final_transl
        video_res_db['final_betas'][slc] = torch.from_numpy(pred_beta)
        if opt.verbose and i==0: print("INFO: Using averaged input betas as final betas.")

        final_rotations_found = False; num_expected_joints = 24
        if hasattr(output, 'pred_theta_mat') and output.pred_theta_mat is not None: # Try matrix first
            mats = output.pred_theta_mat.cpu()[0]; num_joints_rot = mats.shape[1] // 9
            try:
                reshaped_mats = mats.reshape(seq_len, num_joints_rot, 3, 3)
                if reshaped_mats.shape[1] == num_expected_joints: video_res_db['final_thetas'][slc] = reshaped_mats; final_rotations_found = True;
                else: print(f"W: flow_model pred_theta_mat has {reshaped_mats.shape[1]} joints, expected {num_expected_joints}.")
            except Exception as e: print(f"W: Failed to reshape flow_model pred_theta_mat: {e}.")

        # --- MODIFIED: Use Scipy for axis-angle fallback ---
        if not final_rotations_found and hasattr(output, 'pred_theta_aa') and output.pred_theta_aa is not None:
             aas = output.pred_theta_aa.cpu()[0] # Shape (Seq, NumJoints*3)
             num_joints_aa = aas.shape[1] // 3
             try:
                 reshaped_aas = aas.reshape(seq_len, num_joints_aa, 3)
                 if reshaped_aas.shape[1] == num_expected_joints:
                     # --- Scipy Conversion ---
                     aas_numpy = reshaped_aas.reshape(-1, 3).numpy() # Reshape to (N, 3) and convert to numpy
                     rotmats_numpy = SciPyRotation.from_rotvec(aas_numpy).as_matrix() # Convert aa to matrices (N, 3, 3)
                     mats_from_aa_torch = torch.from_numpy(rotmats_numpy).reshape(seq_len, num_expected_joints, 3, 3) # Reshape and convert back to tensor
                     # --- End Scipy Conversion ---
                     video_res_db['final_thetas'][slc] = mats_from_aa_torch
                     final_rotations_found = True
                     if opt.verbose and i==0: print(f"INFO: Using final rotations converted from flow_model 'pred_theta_aa' ({num_expected_joints} joints) via Scipy.")
                 else: print(f"W: flow_model pred_theta_aa has {reshaped_aas.shape[1]} joints, expected {num_expected_joints}.")
             except ImportError: print("Error: Scipy is required for axis-angle conversion fallback. Please install it (`pip install scipy`).") # Specific error message
             except Exception as e: print(f"W: Failed to process/convert pred_theta_aa using Scipy: {e}.")
        # --- End Modification ---

        if not final_rotations_found: # Fallback to identity
             if opt.verbose and i==0: print(f"WARNING: Could not find final {num_expected_joints}-joint rotations in flow_model output. Final PK rotations will be identity.")
             video_res_db['final_thetas'][slc] = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(seq_len, num_expected_joints, 1, 1)

# --- STAGE 3: Rendering Loop (Keep as is) ---
if not opt.not_vis and smpl_faces is not None:
    print('### Running Stage 3: Rendering Final Pose...')
    smpl_faces = smpl_faces.to(device) # Move faces to device once
    for i in tqdm(range(total_img_seq), dynamic_ncols=True):
        img_path = video_res_db['img_path'][i]
        try:
            input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            if input_image is None: raise IOError("Could not read image for rendering")
            bbox_cs_frame = video_res_db['bbox_cs'][i]; vertices_frame = video_res_db['vertices'][[i]]; transl_frame = video_res_db['transl'][[i]]
            image = input_image.copy(); focal = 1000.0
            focal = focal / 256.0 * bbox_cs_frame[2] * max(image.shape[:2])/256.0
            verts_batch = vertices_frame.to(device); transl_batch = transl_frame.to(device)
            color_batch = render_mesh(vertices=verts_batch, faces=smpl_faces, translation=transl_batch, focal_length=focal, height=image.shape[0], width=image.shape[1])
            valid_mask_batch = (color_batch[:, :, :, [-1]] > 0); image_vis_batch = color_batch[:, :, :, :3] * valid_mask_batch
            image_vis_batch = (image_vis_batch * 255).cpu().numpy()
            color = image_vis_batch[0]; valid_mask = valid_mask_batch[0].cpu().numpy(); input_img = image; alpha = 0.9
            image_vis = alpha*color[:,:,:3]*valid_mask + (1-alpha)*input_img*valid_mask + (1-valid_mask)*input_img
            image_vis = image_vis.astype(np.uint8); image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)
            x1, y1, x2, y2 = center_scale_to_box(bbox_cs_frame[:2].numpy(), bbox_cs_frame[2].item(), bbox_cs_frame[3].item())
            image_vis = cv2.rectangle(image_vis, (int(x1), int(y1)), (int(x2), int(y2)), (154, 201, 219), 5)
            res_path = os.path.join(opt.out_dir, 'res_images', f'image-{i+1:06d}.jpg')
            cv2.imwrite(res_path, image_vis)
            if write_stream is not None and write_stream.isOpened(): write_stream.write(image_vis)
        except Exception as e:
            print(f"Error during rendering frame {i} ({os.path.basename(img_path)}): {e}")
            if write_stream is not None and write_stream.isOpened():
                 try: original_frame_bgr = cv2.imread(img_path); write_stream.write(original_frame_bgr)
                 except: pass
elif opt.not_vis: print("Skipping Stage 3: Rendering (disabled by --not-vis).")
else: print("Skipping Stage 3: Rendering (SMPL faces not available).")


# --- Release Video Resources (Keep as is) ---
if write_stream is not None and write_stream.isOpened(): print(f"Releasing video writer..."); write_stream.release()

# --- Finalize and Save FINAL BVH Pickle File (Keep as is) ---
if opt.save_pk:
    print(f"\nPreparing final BVH data for {total_img_seq} frames...")
    bvh_save_db['pred_thetas'] = video_res_db['final_thetas'].numpy()
    bvh_save_db['pred_betas'] = video_res_db['final_betas'].numpy()
    bvh_save_db['transl'] = video_res_db['transl'].numpy()
    transl_camsys_list = []
    img_heights = res_db['height'][:total_img_seq]; img_widths = res_db['width'][:total_img_seq]
    valid_bboxes_xyxy = res_db['bbox'][:total_img_seq]
    final_data_ok = True
    if len(valid_bboxes_xyxy) != total_img_seq: print(f"E: Bbox/frame count mismatch for transl_camsys ({len(valid_bboxes_xyxy)} vs {total_img_seq})"); final_data_ok = False
    else:
        for i in range(total_img_seq):
            orig_bbox_xyxy = valid_bboxes_xyxy[i]; box_w_orig = orig_bbox_xyxy[2] - orig_bbox_xyxy[0]; box_h_orig = orig_bbox_xyxy[3] - orig_bbox_xyxy[1]
            if box_w_orig <= 0 or box_h_orig <= 0: scale_factor = max(cfg.MODEL.IMAGE_SIZE) / 256.0;
            else: scale_factor = max(cfg.MODEL.IMAGE_SIZE) / max(box_w_orig, box_h_orig)
            transl_camsys = video_res_db['transl'][i].numpy() * scale_factor; transl_camsys_list.append(transl_camsys)
        bvh_save_db['transl_camsys'] = np.stack(transl_camsys_list)
        bvh_save_db['frame_idx'] = np.arange(total_img_seq); bvh_save_db['bbox'] = valid_bboxes_xyxy
        bvh_save_db['height'] = img_heights; bvh_save_db['width'] = img_widths; bvh_save_db['f'] = res_db['f'][:total_img_seq]
    if final_data_ok:
        for k in bvh_keys:
            if k not in bvh_save_db or bvh_save_db[k] is None: final_data_ok = False; print(f"E: Final BVH data missing key '{k}'."); break
            current_len = 0
            if isinstance(bvh_save_db[k], np.ndarray): current_len = bvh_save_db[k].shape[0]
            elif isinstance(bvh_save_db[k], list): current_len = len(bvh_save_db[k])
            else: final_data_ok = False; print(f"E: Final BVH data '{k}' is not NumPy array or list."); break
            if current_len != total_img_seq: final_data_ok = False; print(f"E: Frame count mismatch for final key '{k}' (expected {total_img_seq}, got {current_len})."); break
        if final_data_ok:
             print("Final BVH data structure check passed.")
             for k in bvh_keys:
                 if isinstance(bvh_save_db[k], np.ndarray): print(f"  - Final Key '{k}': NumPy Array, Shape={bvh_save_db[k].shape}")
                 elif isinstance(bvh_save_db[k], list): print(f"  - Final Key '{k}': List, Length={len(bvh_save_db[k])}")
             pk_output_path = os.path.join(opt.out_dir, f'niki_hybrik_output_{video_basename}.pk')
             try:
                 with open(pk_output_path, 'wb') as fid: pk.dump(bvh_save_db, fid, protocol=pk.HIGHEST_PROTOCOL)
                 print(f"\nSuccessfully saved FINAL BVH results ({total_img_seq} frames) to: {pk_output_path}")
             except Exception as e: print(f"\nError saving final BVH pickle file: {e}")
        else: print("\nSkipping final BVH pickle save due to missing or invalid data structure.")
    else: print("\nSkipping final BVH pickle save due to data preparation error (e.g., transl_camsys).")

print("\nProcessing finished.")