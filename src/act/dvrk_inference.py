###
# Note: for single arm inference, use the following parameters
# camera length: 54.90 (not identical to training, approximate)
# tool: large needle driver
# port location: the closest left one from the camera port
###

import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
import sys

# path_to_yay_robot = os.getenv('PATH_TO_YAY_ROBOT')
path_to_yay_robot = "/home/grapes/catkin_ws/src/yay_robot_jhu"

if path_to_yay_robot:
    sys.path.append(os.path.join(path_to_yay_robot, 'src'))

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data_dvrk # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions

from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos
from sklearn.preprocessing import normalize
from scipy.spatial.transform import Rotation as R
from pytransform3d.trajectories import plot_trajectory

from sim_env import BOX_POSE

from rostopics import ros_topics
import rospy
import cv2
import crtk
# from delete_me_2 import example_application
from mpl_toolkits import mplot3d
# from sksurgerycore.algorithms.averagequaternions import weighted_average_quaternions
from pytransform3d import rotations, batch_rotations, transformations, trajectories
from dvrk_scripts.dvrk_control import example_application
# from constants_inference import TASK_CONFIGS
from dvrk_scripts.constants_inference import TASK_CONFIGS


from aloha_pro.aloha_scripts.utils import initialize_model_and_tokenizer, encode_text

import time

import IPython
e = IPython.embed
set_seed(0)


parser = argparse.ArgumentParser()
# parser.add_argument('--action_mode', action='store', type=str, help='SE3, hybrid', required=True)
# parser.add_argument('--norm_scheme', action='store', type=str, help='std, min_max', required=True)
action_mode = 'hybrid'
parser.add_argument('--ckpt_dir', action='store', type=str, 
                    help='specify ckpt file path', 
                    required=True)
# ckpt_dir = "/home/grapes/catkin_ws/policy_epoch_20000_seed_0.ckpt"
# needed to avoid error for detr
parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
parser.add_argument('--use_language', action='store_true')
parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
args = parser.parse_args()


task_config = TASK_CONFIGS[args.task_name]
# print(task_config['action_mode'])
mean = task_config['action_mode'][1]['mean']
std = task_config['action_mode'][1]['std']
max_ = task_config['action_mode'][1]['max_']
min_ = task_config['action_mode'][1]['min_']
# max_ = None
# min_ = None

# specify hyperparameters here
image_size = [224, 224]

rt = ros_topics()
ral = crtk.ral('dvrk_arm_test')
psm1_app = example_application(ral, "PSM1", 1)
psm2_app = example_application(ral, "PSM2", 1)

# hyperparams
state_dim = 20
temporal_agg = False
max_timesteps = 25
num_inferences = 80
action_execution_horizon = 20


# hyperparams that never needs to change
chunk_size = 100
kl_weight = 10
hidden_dim = 512
dim_feedforward = 3200
lr_backbone = 1e-5
backbone = 'efficientnet_b3film'    # used to be 'resnet18'
enc_layers = 4
dec_layers = 7
nheads = 8
multi_gpu = None
use_language = args.use_language
language_encoder = "distilbert"
# command = "clipping first clip left tube"  #"grabbing gallbladder"  
command = "grabbing gallbladder"  
num_epochs = args.num_epochs
camera_names = ['left', 'right', 'left_wrist', 'right_wrist']
policy_config = {'lr': 1e-5,
                'num_queries': chunk_size,
                'kl_weight': kl_weight,
                'hidden_dim': hidden_dim,
                'dim_feedforward': dim_feedforward,
                'lr_backbone': lr_backbone,
                'backbone': backbone,
                'enc_layers': enc_layers,
                'num_epochs': num_epochs,
                'dec_layers': dec_layers,
                'nheads': nheads,
                'camera_names': camera_names,
                "multi_gpu": multi_gpu,
                }
# print(policy_config)
# load policy and stats
policy = ACTPolicy(policy_config)
checkpoint = torch.load(args.ckpt_dir)
# loading_status = policy.load_state_dict(checkpoint['model_state_dict'])
loading_status = policy.deserialize(checkpoint['model_state_dict'])
print(loading_status)
# print(loading_status)
policy.cuda()
policy.eval()
print(f'Loaded: {args.ckpt_dir}')

def get_image_dvrk():
    left_img = np.fromstring(rt.usb_image_left.data, np.uint8)
    left_img = cv2.imdecode(left_img, cv2.IMREAD_COLOR) # OpenCV >= 3.0:
    left_img = cv2.resize(left_img, (image_size[1], image_size[0])) # TODO: change this to 640 x 480, likewise during training
    left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    left_img = rearrange(left_img, 'h w c -> c h w')

    right_img = np.fromstring(rt.usb_image_right.data, np.uint8)
    right_img = cv2.imdecode(right_img, cv2.IMREAD_COLOR) # OpenCV >= 3.0:
    right_img = cv2.resize(right_img, (image_size[1], image_size[0])) # TODO: change this to 640 x 480, likewise during training
    right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
    right_img = rearrange(right_img, 'h w c -> c h w')

    lw_img = rt.endo_cam_psm2
    lw_img = cv2.resize(lw_img, (image_size[1], image_size[0])) # TODO: change this to 640 x 480, likewise during training
    lw_img = cv2.cvtColor(lw_img, cv2.COLOR_BGR2RGB)
    lw_img = rearrange(lw_img, 'h w c -> c h w')

    rw_img = rt.endo_cam_psm1
    rw_img = cv2.resize(rw_img, (image_size[1], image_size[0])) # TODO: change this to 640 x 480, likewise during training
    rw_img = cv2.cvtColor(rw_img, cv2.COLOR_BGR2RGB)
    rw_img = rearrange(rw_img, 'h w c -> c h w')

    curr_image = np.stack([left_img, right_img, lw_img, rw_img], axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    
    return curr_image

def convert_6d_rot_to_quat(rots):
    '''
    rots: convert 6D rotation into a quat [n x 6]
    quats: [n x 4] xyzw
    '''
    # Gram-schmidt procedure to make rot vectors orthonormal
    c1 = rots[:, 0:3] # t x 3
    c2 = rots[:, 3:6] # t x 3 
    c1 = normalize(c1, axis = 1) # t x 3
    dot_product = np.sum(c1 * c2, axis = 1).reshape(-1, 1)
    c2 = normalize(c2 - dot_product*c1, axis = 1)
    c3 = np.cross(c1, c2)
    r_mat = np.dstack((c1, c2, c3)) # t x 3 x 3
    rots = R.from_matrix(r_mat)
    return rots.as_quat()

def convert_actions_to_SE3_then_final_actions(dts, dquats, qpos_psm, jaw_angles):
    """
    dts: [n x 3] delta positions
    dquats: [n x 4] delta quats in xyzw convention
    qpos_psm: [n x 8] xyz quat (xyzw convention) jaw angle
    jaw_angles: [n x 1] jaw angles

    output: the trajectories in measured_cp [n x 8]  xyz xyzw jaw
    """
    # convert quats to wxyz convention
    dquats = batch_rotations.batch_quaternion_wxyz_from_xyzw(dquats) # wxyz convention
    qpos_psm[3:7] = rotations.quaternion_wxyz_from_xyzw(qpos_psm[3:7])

    # get positions
    dts_dquats = np.concatenate((dts, dquats), axis = 1)
    g_qpos = transformations.transform_from_pq(qpos_psm[0:7])
    g_actions = trajectories.transforms_from_pqs(dts_dquats)
    g_poses = trajectories.concat_one_to_many(g_qpos, g_actions) # n x 4 x 4

    # convert SE3 into original input form
    output = np.zeros((dquats.shape[0], 8)) # TODO: action hardcoded to be 8-dim (position, quat, jaw angle)
    output[:, 0:3] = g_poses[:, 0:3, 3]
    tmp = batch_rotations.quaternions_from_matrices(g_poses[:, 0:3, 0:3]) # [n x wxyz]
    output[:, 3:7] = batch_rotations.batch_quaternion_xyzw_from_wxyz(tmp) # convert wxyz to xyzw
    # print("pred jaw angle: ", output[:, 7])
    
    output[:, 7] = np.clip(jaw_angles, -0.35, 1.4)  # copy over gripper angles
    return output

def unnormalize_action(naction, norm_scheme):
    """
    only unnormalize the position and jaw angles, orientations remain the same
    """
    
    action = None

    if norm_scheme == "min_max":
        action = (naction + 1) / 2 * (max_ - min_) + min_
        action[:, 3:9] = naction[:, 3:9]
        action[:, 13:19] = naction[:, 13:19]
    
    elif norm_scheme == "std":
        action = unnormalize_positions_only_std(naction)
    else:
        raise NotImplementedError

    return action

def convert_delta_6d_to_taskspace_quat(all_actions, all_actions_converted, qpos):
    '''
    convert delta rot into task-space quaternion rot
    '''
    # Gram-schmidt
    c1 = all_actions[:, 3:6] # t x 3
    c2 = all_actions[:, 6:9] # t x 3 
    c1 = normalize(c1, axis = 1) # t x 3
    dot_product = np.sum(c1 * c2, axis = 1).reshape(-1, 1)
    c2 = normalize(c2 - dot_product*c1, axis = 1)
    c3 = np.cross(c1, c2)
    r_mat = np.dstack((c1, c2, c3)) # t x 3 x 3
    # transform delta rot into task space
    rots = R.from_matrix(r_mat)
    rot_init = R.from_quat(qpos[3:7])
    rots = (rot_init * rots).as_quat()
    all_actions_converted[:, 3:7] = rots
    return all_actions_converted

    
def unnormalize_positions_only_std(diffs):
    """
    diffs: n_actions x 20
    return: normalized n_actions x 20 (zero mean unit variance)
    Note: BOTH POSITIONS AND JAW ANGLES ARE NORMALIZED (orientations remain original)
    """
    unnormalized = diffs*std + mean

    # replace w/ originals for 6D rot
    unnormalized[:, 3:9] = diffs[:, 3:9]
    unnormalized[:, 13:19] = diffs[:, 13:19]

    return unnormalized


def generate_command_embedding(
    command, t, language_encoder, tokenizer, model, instructor=None
):
    print(f"Command at {t=}: {command}")

    command_embedding = encode_text(command, language_encoder, tokenizer, model)
    command_embedding = torch.tensor(command_embedding).cuda()
    if instructor is not None:
        command_embedding = instructor.get_nearest_embedding(command_embedding)[0]
    return command_embedding

def generate_command_embeddings(unique_phase_folder_names, encoder, tokenizer, model):
    # Returns a dictionary containing the phase command as key and a tuple of the phase command and phase embedding as value
    phase_command_embeddings_dict = {}
    for phase_folder_name in tqdm(unique_phase_folder_names, desc="Embedding phase commands"):
        # Extract the phase command from the folder name (removing the phase idx and the "_" in between the words)
        _, phase_command = phase_folder_name.split("_")[0], " ".join(phase_folder_name.split("_")[1:])
        embedding = encode_text(phase_command, encoder, tokenizer, model)
        phase_command_embeddings_dict[phase_folder_name]= (phase_command, embedding)

    return phase_command_embeddings_dict


if use_language:
    tokenizer, model = initialize_model_and_tokenizer(language_encoder)
    assert tokenizer is not None and model is not None

time.sleep(5)

with torch.inference_mode():
    for t in range(num_inferences):
        if t % chunk_size == 0:

            if use_language:

                command_embedding = generate_command_embedding(
                    command, t, language_encoder, tokenizer, model
                )

                assert command_embedding is not None

        qpos_zero = torch.zeros(1, state_dim).float().cuda()
        curr_image = get_image_dvrk()

        action = policy(qpos_zero, curr_image, command_embedding=command_embedding).cpu().numpy().squeeze() # 1 x 100 x state_dim (20) -> 100 x 20
        action = unnormalize_action(action, "std")

        actions_psm1 = None
        actions_psm2 = None

        qpos_psm1 = np.array((rt.psm1_pose.position.x, rt.psm1_pose.position.y, rt.psm1_pose.position.z,
                    rt.psm1_pose.orientation.x, rt.psm1_pose.orientation.y, rt.psm1_pose.orientation.z, rt.psm1_pose.orientation.w,
                    rt.psm1_jaw))   
        
        qpos_psm2 = np.array((rt.psm2_pose.position.x, rt.psm2_pose.position.y, rt.psm2_pose.position.z,
                    rt.psm2_pose.orientation.x, rt.psm2_pose.orientation.y, rt.psm2_pose.orientation.z, rt.psm2_pose.orientation.w,
                    rt.psm2_jaw))

        if action_mode == 'hybrid':

            actions_psm1 = np.zeros((chunk_size, 8)) # pos, quat, jaw
            actions_psm1[:, 0:3] = qpos_psm1[0:3] + action[:, 0:3] # convert to current translation
            actions_psm1 = convert_delta_6d_to_taskspace_quat(action[:, 0:10], actions_psm1, qpos_psm1)
            actions_psm1[:, 7] = np.clip(action[:, 9], -0.698, 0.698)  # copy over gripper angles
            
            actions_psm2 = np.zeros((chunk_size, 8)) # pos, quat, jaw
            actions_psm2[:, 0:3] = qpos_psm2[0:3] + action[:, 10:13] # convert to current translation
            actions_psm2 = convert_delta_6d_to_taskspace_quat(action[:, 10:], actions_psm2, qpos_psm2)
            # print(action[:, 19])
            actions_psm2[:, 7] = np.clip(action[:, 19], -0.698, 0.698)  # copy over gripper angles  
            
        if action_mode == 'ego':
            # compute actions for PSM1
            actions_psm1 = np.zeros((chunk_size, 8)) # pos (3), quat (4), jaw (1) 
            dts_psm1 = action[:, 0:3]
            dquats_psm1 = convert_6d_rot_to_quat(action[:, 3:9]) # [n x 4] xyzw convention
            actions_psm1 = convert_actions_to_SE3_then_final_actions(dts_psm1, dquats_psm1, qpos_psm1, action[:, 9]) # translation and quaternion
            
            # compute actions for PSM2
            actions_psm2 = np.zeros((chunk_size, 8)) # pos (3), quat (4), jaw (1) 
            dts_psm2 = action[:, 10:13]
            dquats_psm2 = convert_6d_rot_to_quat(action[:, 13:19]) # [n x 4] xyzw convention
            actions_psm2 = convert_actions_to_SE3_then_final_actions(dts_psm2, dquats_psm2, qpos_psm2, action[:, 19]) # translation and quaternion   
        
        # factor = 1000
        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.scatter(actions_psm1[:, 0], actions_psm1[:, 1], actions_psm1[:, 2], c ='r')
        # ax.scatter(actions_psm2[:, 0]*factor, actions_psm2[:, 1]*factor, actions_psm2[:, 2]*factor, c ='g', label = 'Generated trajectory')
        # ax.scatter(qpos_psm1[0], qpos_psm1[1], qpos_psm1[2], c = 'b')
        # ax.scatter(qpos_psm2[0]*factor, qpos_psm2[1]*factor, qpos_psm2[2]*factor, c = 'b', label = 'Current PSM2')
        # ax.set_xlabel('X (mm)')
        # ax.set_ylabel('Y (mm)')
        # ax.set_zlabel('Z (mm)')
        # n_bins = 7
        # ax.legend()
        # ax.xaxis.set_major_locator(plt.MaxNLocator(n_bins))
        # ax.yaxis.set_major_locator(plt.MaxNLocator(n_bins))
        # ax.zaxis.set_major_locator(plt.MaxNLocator(n_bins))
        # plt.show()
        # input("Press Enter to continue...")
        # assert(False)

        for jj in range(action_execution_horizon):
            ral.spin_and_execute(psm1_app.run_full_pose_goal, actions_psm1[jj])
            ral.spin_and_execute(psm2_app.run_full_pose_goal_tmp, actions_psm2[jj])