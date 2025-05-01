#!/usr/bin/env python3
# run with python hybrik_to_bvh_x.py --input res_sword_x/hybrik_output_sword.pk --output res_sword_x/sword_smplx_artroot_v3.bvh --smplx-path model_files/smplx/SMPLX_NEUTRAL.npz --fps 30
import argparse
import numpy as np
import pickle as pk
import os
import torch
import smplx
from scipy.spatial.transform import Rotation as R
import traceback
from tqdm import tqdm

# --- (Keep SMPLX_DATA_CACHE, LOG_FILE, log_message, get_smplx_data) ---
# (Assume get_smplx_data is the fixed version that returns the correct structure)
SMPLX_DATA_CACHE = {}
LOG_FILE = "smplx_model_inspection.log"
if os.path.exists(LOG_FILE): os.remove(LOG_FILE)
def log_message(message): print(message); open(LOG_FILE, "a").write(message + "\n")

BVH_ARTIFICIAL_ROOT_NAME = "root" # Using "root" as requested

def get_smplx_data(model_path, model_type='smplx', gender='neutral', ext='npz', use_pca=False, num_pca_comps=6):
    """Loads SMPL-X model and extracts hierarchy, offsets, names, and original root T-pose offset."""
    global SMPLX_DATA_CACHE
    cache_key = (model_path, model_type, gender, ext)
    if cache_key in SMPLX_DATA_CACHE: return SMPLX_DATA_CACHE[cache_key]
    log_message(f"Loading SMPL-X model definition from: {model_path}")
    bm = None
    try:
        with torch.no_grad():
            bm = smplx.create(model_path, model_type=model_type, gender=gender, use_pca=use_pca, num_pca_comps=num_pca_comps, ext=ext, batch_size=1, create_transl=False)
            log_message("smplx.create() successful.")
            parents = bm.parents.cpu().numpy()
            num_joints_from_parents = len(parents)
            log_message(f"Found {num_joints_from_parents} joints based on parents array length.")
            joint_names = None
            try:
                if hasattr(bm, 'JOINT_NAMES'): joint_names = bm.JOINT_NAMES
            except AttributeError: pass
            if joint_names is None:
                log_message("WARNING: Could not find JOINT_NAMES attribute. Using standard SMPL-X 55 names.")
                standard_joint_names = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'jaw', 'left_eye_smplhf', 'right_eye_smplhf', 'left_index1', 'left_index2', 'left_index3', 'left_middle1', 'left_middle2', 'left_middle3', 'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_ring1', 'left_ring2', 'left_ring3', 'left_thumb1', 'left_thumb2', 'left_thumb3', 'right_index1', 'right_index2', 'right_index3', 'right_middle1', 'right_middle2', 'right_middle3', 'right_pinky1', 'right_pinky2', 'right_pinky3', 'right_ring1', 'right_ring2', 'right_ring3', 'right_thumb1', 'right_thumb2', 'right_thumb3']
                if len(standard_joint_names) != num_joints_from_parents: raise ValueError(f"Standard joint name count ({len(standard_joint_names)}) != model parent count ({num_joints_from_parents}).")
                joint_names = standard_joint_names
            elif len(joint_names) != num_joints_from_parents: raise ValueError(f"Found bm.JOINT_NAMES count ({len(joint_names)}) != model parent count ({num_joints_from_parents}).")
            log_message("Calculating T-pose joint positions...")
            zero_betas = torch.zeros(1, bm.num_betas)
            zero_global_orient = torch.zeros(1, 3); zero_body_pose = torch.zeros(1, 21 * 3)
            zero_lhand_pose = torch.zeros(1, 15 * 3); zero_rhand_pose = torch.zeros(1, 15 * 3)
            zero_jaw_pose = torch.zeros(1, 3) if hasattr(bm, 'NUM_JAW_JOINTS') and bm.NUM_JAW_JOINTS > 0 else None
            zero_leye_pose = torch.zeros(1, 3) if hasattr(bm, 'NUM_EYE_JOINTS') and bm.NUM_EYE_JOINTS > 0 else None
            zero_reye_pose = torch.zeros(1, 3) if hasattr(bm, 'NUM_EYE_JOINTS') and bm.NUM_EYE_JOINTS > 0 else None
            forward_kwargs = {'betas': zero_betas, 'global_orient': zero_global_orient, 'body_pose': zero_body_pose, 'left_hand_pose': zero_lhand_pose, 'right_hand_pose': zero_rhand_pose, 'return_verts': False}
            if zero_jaw_pose is not None: forward_kwargs['jaw_pose'] = zero_jaw_pose
            if zero_leye_pose is not None: forward_kwargs['leye_pose'] = zero_leye_pose
            if zero_reye_pose is not None: forward_kwargs['reye_pose'] = zero_reye_pose
            t_pose_output = bm(**forward_kwargs)
            t_pose_joints = t_pose_output.joints.squeeze(0).cpu().numpy()
            log_message("T-pose calculation successful.")
            log_message("Calculating joint offsets...")
            offsets = {}
            joint_map = {name: i for i, name in enumerate(joint_names)}
            original_root_node = None
            for i, name in enumerate(joint_names):
                if parents[i] == -1:
                    original_root_node = name
                    # Offset for the 'pelvis' in BVH is its T-pose position relative to origin
                    offsets[name] = t_pose_joints[i] * 100.0
                else:
                    parent_name = joint_names[parents[i]]
                    parent_idx = joint_map[parent_name]
                    offset_vec = t_pose_joints[i] - t_pose_joints[parent_idx]
                    offsets[name] = offset_vec * 100.0
            log_message("Offset calculation successful.")
            if original_root_node is None: raise ValueError("Could not determine original SMPL-X root node.")
            hierarchy = {name: [] for name in joint_names}
            for i, name in enumerate(joint_names):
                if parents[i] != -1: hierarchy[joint_names[parents[i]]].append(name)
            smplx_data = {'joint_names': joint_names, 'parents': parents, 'hierarchy': hierarchy, 'original_root_node': original_root_node, 'offsets': offsets, 'joint_map': joint_map}
            SMPLX_DATA_CACHE[cache_key] = smplx_data
            log_message("SMPL-X data extraction complete.")
            return smplx_data
    except AttributeError as e: log_message(f"\n!!! AttributeError: {e} !!!"); log_message(traceback.format_exc()); raise e
    except Exception as e: log_message(f"\n!!! Error loading SMPL-X: {e} !!!"); log_message(traceback.format_exc()); raise e


# --- BVH Hierarchy Building (Artificial Root) ---
def build_bvh_hierarchy(smplx_data):
    """Builds the HIERARCHY section string with an artificial root named 'root'."""
    original_root_node = smplx_data['original_root_node']
    hierarchy_dict = smplx_data['hierarchy']
    offsets = smplx_data['offsets']
    num_smplx_joints = len(smplx_data['joint_names'])
    total_channels = 0 # Keep track of channels for validation

    # Inner function to build the SMPL-X part recursively
    def create_smplx_joint_string(joint_name, depth=0):
        nonlocal total_channels
        indent = "  " * depth
        offset = offsets[joint_name]
        children = hierarchy_dict.get(joint_name, [])
        lines = []
        lines.append(f"{indent}JOINT {joint_name}")
        lines.append(f"{indent}{{")
        lines.append(f"{indent}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}")
        lines.append(f"{indent}  CHANNELS 3 Zrotation Xrotation Yrotation")
        total_channels += 3 # Add channels for this joint
        for child in children: lines.extend(create_smplx_joint_string(child, depth + 1))
        if not children:
            lines.append(f"{indent}  End Site"); lines.append(f"{indent}  {{")
            lines.append(f"{indent}    OFFSET 0.000000 0.000000 0.000000"); lines.append(f"{indent}  }}")
        lines.append(f"{indent}}}")
        return lines

    # Start building the BVH string
    lines = ["HIERARCHY"]
    # Add the artificial root
    lines.append(f"ROOT {BVH_ARTIFICIAL_ROOT_NAME}")
    lines.append(f"{{")
    lines.append(f"  OFFSET 0.000000 0.000000 0.000000")
    lines.append(f"  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
    total_channels += 6 # Add channels for the root
    # Add the original SMPL-X hierarchy starting at depth 1
    lines.extend(create_smplx_joint_string(original_root_node, depth=1))
    lines.append(f"}}") # Close the artificial root

    log_message(f"Built BVH hierarchy with artificial root '{BVH_ARTIFICIAL_ROOT_NAME}'. Total expected channels per frame: {total_channels}")
    return "\n".join(lines), total_channels # Return total channels for validation

# --- BVH Joint Order (Artificial Root) ---
def get_bvh_joint_order(smplx_data):
    """Generates the ordered list of joints including the artificial root."""
    ordered_joints = [BVH_ARTIFICIAL_ROOT_NAME]
    original_root_node = smplx_data['original_root_node']
    hierarchy_dict = smplx_data['hierarchy']
    def traverse_smplx(joint_name):
        ordered_joints.append(joint_name)
        for child in hierarchy_dict.get(joint_name, []): traverse_smplx(child)
    traverse_smplx(original_root_node); return ordered_joints


# --- Coordinate System Conversion ---
def convert_translation(transl, scale=100.0):
    """Converts translation, applies scaling, and coordinate flip."""
    # Applying the original script's coordinate system flip
    xyz_convert = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    return transl.dot(xyz_convert) * scale

BVH_ROOT_TRANSFORM_MATRIX = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
def apply_pelvis_transform(rotation_matrix, apply_transform=True):
    """Applies the 180deg X rot coordinate system rotation to the pelvis orientation matrix."""
    if apply_transform: return np.dot(BVH_ROOT_TRANSFORM_MATRIX.T, rotation_matrix)
    else: return rotation_matrix

def matrix_to_euler_degrees(matrix):
    """Converts a 3x3 rotation matrix to ZXY Euler angles in degrees."""
    # Add check for invalid rotation matrices which can cause issues
    if not np.all(np.isfinite(matrix)):
        log_message(f"Warning: Invalid values (NaN/Inf) found in rotation matrix:\n{matrix}\nReplacing with identity.")
        matrix = np.identity(3)
    try:
        return R.from_matrix(matrix).as_euler('zxy', degrees=True)
    except ValueError as e:
         log_message(f"Warning: Error converting matrix to Euler: {e}. Matrix:\n{matrix}\nReplacing with zero rotation.")
         return np.array([0.0, 0.0, 0.0]) # Return zero Euler angles


# --- Main Conversion Function ---
def pickle_to_bvh(input_file, output_file, smplx_model_path, fps=30, gender='neutral', model_type='smplx', transform_pelvis=True):
    """Converts HybrIK SMPL-X pickle file to BVH motion file with artificial root."""
    log_message(f"--- Starting Conversion (Artificial Root v3) ---") # Version ID
    log_message(f"Input PK: {input_file}")
    log_message(f"Output BVH: {output_file}")
    log_message(f"SMPL-X Model: {smplx_model_path} (Type: {model_type}, Gender: {gender})")
    log_message(f"Target FPS: {fps}")
    log_message(f"Apply coord transform to Pelvis rotation: {transform_pelvis}")

    # --- 1. Load SMPL-X Model Definition ---
    try:
        smplx_data = get_smplx_data(smplx_model_path, gender=gender, model_type=model_type)
        joint_map = smplx_data['joint_map']
        original_root_node_name = smplx_data['original_root_node']
        smplx_joint_count = len(smplx_data['joint_names'])
    except Exception as e: log_message(f"FATAL ERROR: Failed to load SMPL-X data."); return

    # --- 2. Load Motion Data ---
    try:
        with open(input_file, 'rb') as f: motion_data = pk.load(f)
        log_message(f"Successfully loaded motion data from {input_file}")
    except Exception as e: log_message(f"FATAL ERROR: Error loading pickle {input_file}: {e}"); return

    # --- 3. Extract Required Data ---
    if 'transl' in motion_data and motion_data['transl'] is not None: translations = motion_data['transl']; log_message("Using 'transl' key.")
    elif 'transl_camsys' in motion_data and motion_data['transl_camsys'] is not None: translations = motion_data['transl_camsys']; log_message("Warning: Using 'transl_camsys'.")
    else: log_message("FATAL ERROR: No translation data found."); return
    if 'pred_thetas' in motion_data and motion_data['pred_thetas'] is not None: rotation_matrices = motion_data['pred_thetas']; log_message("Using 'pred_thetas' key.")
    else: log_message("FATAL ERROR: Rotation matrices ('pred_thetas') not found."); return

    # --- 4. Data Validation ---
    n_frames = translations.shape[0]
    if rotation_matrices.shape[0] != n_frames: log_message(f"FATAL ERROR: Frame count mismatch."); return
    if rotation_matrices.shape[1] != smplx_joint_count: log_message(f"FATAL ERROR: Joint count mismatch."); return
    if rotation_matrices.shape[2:] != (3, 3): log_message(f"FATAL ERROR: Rotation data shape invalid."); return
    log_message(f"Data validated: {n_frames} frames, {smplx_joint_count} SMPL-X joints.")

    # --- 5. Prepare BVH Structure ---
    frame_time = 1.0 / fps
    hierarchy_section, expected_channels = build_bvh_hierarchy(smplx_data) # Get expected channels
    bvh_joint_order = get_bvh_joint_order(smplx_data) # Includes artificial root "root"

    # --- 6. Write BVH File ---
    try:
        with open(output_file, 'w') as f:
            f.write(hierarchy_section); f.write("\n")
            f.write("MOTION\n"); f.write(f"Frames: {n_frames}\n"); f.write(f"Frame Time: {frame_time:.8f}\n")
            log_message("Writing motion frames...")
            first_y_pos = None

            # --- 7. Process and Write Each Frame (Artificial Root Logic) ---
            for frame_idx in tqdm(range(n_frames)):
                frame_motion_data = []
                num_values_written = 0 # Counter for validation

                # --- Get Translation (and apply coord fix) ---
                root_pos = convert_translation(translations[frame_idx])
                if frame_idx == 0: first_y_pos = root_pos[1]
                if first_y_pos is None: first_y_pos = 0 # Safety fallback
                root_pos[1] -= first_y_pos # Grounding

                # --- Get Global Orientation ---
                global_orient_matrix = rotation_matrices[frame_idx, 0]

                # --- Iterate through BVH Joint Order ---
                for joint_name in bvh_joint_order:
                    if joint_name == BVH_ARTIFICIAL_ROOT_NAME: # The artificial "root"
                        # Write Position channels (3 values)
                        frame_motion_data.extend([f"{p:.6f}" for p in root_pos])
                        # Write ZERO Rotation channels (3 values)
                        frame_motion_data.extend(["0.0", "0.0", "0.0"])
                        num_values_written += 6
                    else:
                        # This is an SMPL-X joint
                        smplx_idx = joint_map[joint_name]

                        if joint_name == original_root_node_name: # Is this the Pelvis?
                             final_matrix = apply_pelvis_transform(global_orient_matrix, apply_transform=transform_pelvis)
                        else:
                             final_matrix = rotation_matrices[frame_idx, smplx_idx]

                        # Convert final matrix to Euler angles
                        euler_angles = matrix_to_euler_degrees(final_matrix)
                        # Write only Rotation channels (3 values)
                        frame_motion_data.extend([f"{a:.6f}" for a in euler_angles])
                        num_values_written += 3

                # --- Validation Check ---
                if num_values_written != expected_channels:
                    log_message(f"FATAL ERROR: Frame {frame_idx}: Wrote {num_values_written} values, but expected {expected_channels} channels based on hierarchy.")
                    raise ValueError(f"Channel count mismatch on frame {frame_idx}") # Stop execution

                # Write the data for this frame
                f.write(" ".join(frame_motion_data) + "\n")

        log_message(f"BVH file created successfully: {output_file}")
        log_message("--- Conversion Finished ---")

    except Exception as e:
        log_message(f"\nFATAL ERROR: Error writing BVH file: {e}")
        log_message("\nDetailed traceback:"); log_message(traceback.format_exc())

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description='Convert HybrIK SMPL-X pickle files to BVH format with artificial root.')
    parser.add_argument('--input', '-i', required=True, help='Input pickle file')
    parser.add_argument('--output', '-o', required=True, help='Output BVH file path')
    parser.add_argument('--smplx-path', '-s', required=True, help='Path to the SMPL-X model file (.npz)')
    parser.add_argument('--fps', type=float, default=30.0, help='Frames per second (default: 30)')
    parser.add_argument('--model-type', default='smplx', choices=['smplx', 'smplh', 'smpl'], help='SMPL model type (default: smplx)')
    parser.add_argument('--gender', default='neutral', choices=['neutral', 'male', 'female'], help='SMPL model gender (default: neutral)')
    parser.add_argument('--transform-pelvis', default=True, dest='transform_pelvis', action=argparse.BooleanOptionalAction,
                        help='Apply coordinate transform (orig script fix) to Pelvis orientation (default: True)')

    args = parser.parse_args()
    if not os.path.exists(args.input): log_message(f"Error: Input file not found: {args.input}"); return
    if not os.path.exists(args.smplx_path): log_message(f"Error: SMPL-X model file not found: {args.smplx_path}"); return

    pickle_to_bvh(
        input_file=args.input,
        output_file=args.output,
        smplx_model_path=args.smplx_path,
        fps=args.fps,
        gender=args.gender,
        model_type=args.model_type,
        transform_pelvis=args.transform_pelvis
    )

if __name__ == '__main__':
    main()