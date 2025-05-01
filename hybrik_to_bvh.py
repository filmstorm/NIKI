#!/usr/bin/env python3
import argparse
import numpy as np
import pickle as pk
import os
import math
from scipy.spatial.transform import Rotation as R

# Define the bone mapping from the original script
part_match = {'root': 'root', 'bone_00': 'Pelvis', 'bone_01': 'L_Hip', 'bone_02': 'R_Hip',
              'bone_03': 'Spine1', 'bone_04': 'L_Knee', 'bone_05': 'R_Knee', 'bone_06': 'Spine2',
              'bone_07': 'L_Ankle', 'bone_08': 'R_Ankle', 'bone_09': 'Spine3', 'bone_10': 'L_Foot',
              'bone_11': 'R_Foot', 'bone_12': 'Neck', 'bone_13': 'L_Collar', 'bone_14': 'R_Collar',
              'bone_15': 'Head', 'bone_16': 'L_Shoulder', 'bone_17': 'R_Shoulder', 'bone_18': 'L_Elbow',
              'bone_19': 'R_Elbow', 'bone_20': 'L_Wrist', 'bone_21': 'R_Wrist', 'bone_22': 'L_Hand', 
              'bone_23': 'R_Hand'}

# FIX 1: Update the joint offsets to match the provided values with corrected left/right sides
JOINT_OFFSETS = {
    'root': [0.0, 0.0, 0.0],
    'Pelvis': [0.0, 95.0, 0.0],
    'L_Hip': [9.0, 0.0, 0.0],    # Changed from -9.0 to 9.0
    'R_Hip': [-9.0, 0.0, 0.0],   # Changed from 9.0 to -9.0
    'Spine1': [0.0, 7.0, 0.0],
    'L_Knee': [0.0, -45.0, 0.0],
    'R_Knee': [0.0, -45.0, 0.0],
    'Spine2': [0.0, 10.0, 0.0],
    'L_Ankle': [0.0, -45.0, 0.0],
    'R_Ankle': [0.0, -45.0, 0.0],
    'Spine3': [0.0, 10.0, 0.0],
    'L_Foot': [0.0, -10.0, 15.0],
    'R_Foot': [0.0, -10.0, 15.0],
    'Neck': [0.0, 10.0, 0.0],
    'L_Collar': [7.0, 3.0, 0.0],  # Changed from -7.0 to 7.0
    'R_Collar': [-7.0, 3.0, 0.0], # Changed from 7.0 to -7.0
    'Head': [0.0, 10.0, 0.0],
    'L_Shoulder': [10.0, 0.0, 0.0],  # Changed from -10.0 to 10.0
    'R_Shoulder': [-10.0, 0.0, 0.0], # Changed from 10.0 to -10.0
    'L_Elbow': [30.0, 0.0, 0.0],     # Changed from -30.0 to 30.0
    'R_Elbow': [-30.0, 0.0, 0.0],    # Changed from 30.0 to -30.0
    'L_Wrist': [25.0, 0.0, 0.0],     # Changed from -25.0 to 25.0
    'R_Wrist': [-25.0, 0.0, 0.0],    # Changed from 25.0 to -25.0
    'L_Hand': [8.0, 0.0, 0.0],       # Changed from -8.0 to 8.0
    'R_Hand': [-8.0, 0.0, 0.0],      # Changed from 8.0 to -8.0
}

def Rodrigues(rotvec):
    """Convert rotation vector to rotation matrix using Rodrigues formula"""
    theta = np.linalg.norm(rotvec)
    r = (rotvec / theta).reshape(3, 1) if theta > 0. else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]],
                      [r[2], 0, -r[0]],
                      [-r[1], r[0], 0]])
    return (cost * np.eye(3) + (1 - cost) * r.dot(r.T) + np.sin(theta) * mat)

def rodrigues2bshapes(pose):
    """Convert pose parameters to rotation matrices"""
    if pose.size == 24 * 9:
        rod_rots = np.asarray(pose).reshape(24, 3, 3)
        mat_rots = [rod_rot for rod_rot in rod_rots]
    else:
        rod_rots = np.asarray(pose).reshape(24, 3)
        mat_rots = [Rodrigues(rod_rot) for rod_rot in rod_rots]
    return mat_rots

def rotate180(rot):
    """Apply 180 degree rotation transformation - this is directly from the original Blender script"""
    xyz_convert = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ], dtype=np.float32)
    return np.dot(xyz_convert.T, rot)

def convert_transl(transl):
    """Convert translation vector to BVH coordinate system"""
    # This is directly from the original Blender script
    xyz_convert = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ], dtype=np.float32)
    # Scale up by 100x for visibility
    return transl.dot(xyz_convert) * 100.0

def matrix_to_euler(mat):
    """Convert rotation matrix to Euler angles (in degrees) in ZXY order for BVH"""
    r = R.from_matrix(mat)
    return r.as_euler('ZXY', degrees=True)

def create_hierarchy():
    """Create the BVH hierarchy section with proper bone offsets"""
    joint_hierarchy = {
        'root': ['Pelvis'],
        'Pelvis': ['L_Hip', 'R_Hip', 'Spine1'],
        'L_Hip': ['L_Knee'],
        'R_Hip': ['R_Knee'],
        'Spine1': ['Spine2'],
        'L_Knee': ['L_Ankle'],
        'R_Knee': ['R_Ankle'],
        'Spine2': ['Spine3'],
        'L_Ankle': ['L_Foot'],
        'R_Ankle': ['R_Foot'],
        'Spine3': ['Neck', 'L_Collar', 'R_Collar'],
        'Neck': ['Head'],
        'L_Collar': ['L_Shoulder'],
        'R_Collar': ['R_Shoulder'],
        'L_Shoulder': ['L_Elbow'],
        'R_Shoulder': ['R_Elbow'],
        'L_Elbow': ['L_Wrist'],
        'R_Elbow': ['R_Wrist'],
        'L_Wrist': ['L_Hand'],
        'R_Wrist': ['R_Hand'],
        'Head': [],
        'L_Foot': [],
        'R_Foot': [],
        'L_Hand': [],
        'R_Hand': []
    }
    
    def create_joint(joint_name, depth=0):
        """Recursively create joint definitions with proper indentation"""
        indent = "  " * depth
        offset = JOINT_OFFSETS[joint_name]
        
        if joint_name == 'root':
            result = f"{indent}ROOT {joint_name}\n"
            result += f"{indent}{{\n"
            result += f"{indent}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n"
            # Keep 6 channels for root to match version 4, but we'll only use the position
            result += f"{indent}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n"
        else:
            result = f"{indent}JOINT {joint_name}\n"
            result += f"{indent}{{\n"
            result += f"{indent}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n"
            result += f"{indent}  CHANNELS 3 Zrotation Xrotation Yrotation\n"
            
        children = joint_hierarchy[joint_name]
        for child in children:
            result += create_joint(child, depth + 1)
            
        if not children:  # End Site
            result += f"{indent}  End Site\n"
            result += f"{indent}  {{\n"
            if joint_name in ['L_Hand', 'R_Hand']:
                result += f"{indent}    OFFSET {5.0:.6f} {0.0:.6f} {0.0:.6f}\n"
            elif joint_name in ['L_Foot', 'R_Foot']:
                result += f"{indent}    OFFSET {0.0:.6f} {0.0:.6f} {10.0:.6f}\n"
            elif joint_name == 'Head':
                result += f"{indent}    OFFSET {0.0:.6f} {10.0:.6f} {0.0:.6f}\n"
            else:
                result += f"{indent}    OFFSET {0.0:.6f} {5.0:.6f} {0.0:.6f}\n"
            result += f"{indent}  }}\n"
            
        result += f"{indent}}}\n"
        return result
    
    hierarchy = "HIERARCHY\n"
    hierarchy += create_joint('root')
    return hierarchy

def create_joint_order():
    """Create a flat list of joints in the order they appear in the hierarchy"""
    joint_order = ['root']
    
    def traverse_hierarchy(joint_name):
        joint_hierarchy = {
            'root': ['Pelvis'],
            'Pelvis': ['L_Hip', 'R_Hip', 'Spine1'],
            'L_Hip': ['L_Knee'],
            'R_Hip': ['R_Knee'],
            'Spine1': ['Spine2'],
            'L_Knee': ['L_Ankle'],
            'R_Knee': ['R_Ankle'],
            'Spine2': ['Spine3'],
            'L_Ankle': ['L_Foot'],
            'R_Ankle': ['R_Foot'],
            'Spine3': ['Neck', 'L_Collar', 'R_Collar'],
            'Neck': ['Head'],
            'L_Collar': ['L_Shoulder'],
            'R_Collar': ['R_Shoulder'],
            'L_Shoulder': ['L_Elbow'],
            'R_Shoulder': ['R_Elbow'],
            'L_Elbow': ['L_Wrist'],
            'R_Elbow': ['R_Wrist'],
            'L_Wrist': ['L_Hand'],
            'R_Wrist': ['R_Hand'],
            'Head': [],
            'L_Foot': [],
            'R_Foot': [],
            'L_Hand': [],
            'R_Hand': []
        }
        
        children = joint_hierarchy[joint_name]
        for child in children:
            joint_order.append(child)
            traverse_hierarchy(child)
    
    traverse_hierarchy('root')
    return joint_order

def create_joint_to_bone_mapping():
    """Create a mapping from joint names to bone indices in the SMPL model"""
    joint_to_bone = {}
    for bone_id, joint_name in part_match.items():
        if bone_id != 'root':  # Skip root as it's handled separately
            bone_index = int(bone_id.split('_')[1])
            joint_to_bone[joint_name] = bone_index
    return joint_to_bone

def pickle_to_bvh(input_file, output_file, fps=30):
    """Convert HybrIK pickle file to BVH motion file"""
    print(f"Converting {input_file} to {output_file}")
    print("Fixes applied:")
    print("1. Fixed joint offset directions (swapped left/right sides)")
    print("2. Root rotation now set to 0 (rotation applied only to pelvis)")
    
    # Load motion data from pickle file
    with open(input_file, 'rb') as f:
        motion_data = pk.load(f)
    
    # Extract data
    transl_camsys = motion_data['transl_camsys']
    pred_thetas = motion_data['pred_thetas']
    nFrames = len(pred_thetas)
    
    # Get the frame time
    frame_time = 1.0 / fps
    
    # Create the BVH HIERARCHY structure
    hierarchy = create_hierarchy()
    
    # Create a list of joints in order of appearance in the hierarchy
    joint_order = create_joint_order()
    
    # Create a mapping from joint names to SMPL bone indices
    joint_to_bone = create_joint_to_bone_mapping()
    
    # Open the file for writing
    with open(output_file, 'w') as f:
        # Write the HIERARCHY section
        f.write(hierarchy)
        
        # Write the MOTION section
        f.write("MOTION\n")
        f.write(f"Frames: {nFrames}\n")
        f.write(f"Frame Time: {frame_time:.6f}\n")
        
        # Reference Y position from first frame for grounding
        first_y = None
        
        # Process each frame
        for frame in range(nFrames):
            # Get translation and pose data for this frame
            trans = convert_transl(transl_camsys[frame])
            pose = pred_thetas[frame]
            
            # Track the first frame's Y for grounding
            if frame == 0:
                first_y = trans[1]
            
            # Adjust Y value to keep character grounded
            trans[1] = trans[1] - first_y
            
            # Convert pose to rotation matrices
            mrots = rodrigues2bshapes(pose)
            
            # Process root rotation (to apply to pelvis later)
            root_rotation = rotate180(mrots[0])
            
            # Apply slight damping to root rotation
            if frame > 0:
                root_r = R.from_matrix(root_rotation)
                euler = root_r.as_euler('ZXY', degrees=True)
                
                # Light dampening on X and Z rotations to reduce wobble
                damping = 0.7  # Higher value = more natural movement
                euler[0] *= damping  # Z rotation
                euler[1] *= damping  # X rotation
                # Keep Y rotation for turning
                
                # Convert back to matrix
                root_rotation = R.from_euler('ZXY', euler, degrees=True).as_matrix()
            
            # Convert rotation matrices to Euler angles and write motion data
            frame_data = []
            
            # Process each joint in order
            for joint in joint_order:
                if joint == 'root':
                    # Root position plus zero rotation
                    frame_data.extend(trans)
                    # Add empty rotation values (no actual rotation)
                    frame_data.extend([0.0, 0.0, 0.0])
                elif joint == 'Pelvis':
                    # Apply the root rotation to pelvis
                    euler = matrix_to_euler(root_rotation)
                    frame_data.extend(euler)
                else:
                    # Find the bone index for this joint
                    bone_index = joint_to_bone.get(joint)
                    
                    if bone_index is not None:
                        # Add joint rotation
                        euler = matrix_to_euler(mrots[bone_index])
                        frame_data.extend(euler)
                    else:
                        # Joint not found in SMPL model, add default rotation
                        frame_data.extend([0.0, 0.0, 0.0])
            
            # Write frame data as a space-separated line
            f.write(' '.join([f"{x:.6f}" for x in frame_data]) + '\n')
    
    print(f"BVH file created successfully: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert HybrIK pickle files to BVH format with fixed joint directions')
    parser.add_argument('--input', '-i', required=True, help='Input pickle file with motion data')
    parser.add_argument('--output', '-o', required=True, help='Output BVH file')
    parser.add_argument('--fps', '-f', type=float, default=30.0, help='Frames per second (default: 30)')
    args = parser.parse_args()
    
    pickle_to_bvh(args.input, args.output, args.fps)

if __name__ == '__main__':
    main()