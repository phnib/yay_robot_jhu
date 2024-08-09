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
from std_msgs.msg import String, Float32MultiArray

path_to_yay_robot = "/home/grapes/catkin_ws/src/yay_robot_jhu"

if path_to_yay_robot:
    sys.path.append(os.path.join(path_to_yay_robot, 'src'))

import rospy
from rostopics import ros_topics
from policy import ACTPolicy, CNNMLPPolicy
from utils import set_seed # helper functions
from sklearn.preprocessing import normalize

import cv2
import crtk
from mpl_toolkits import mplot3d
from scipy.spatial.transform import Rotation as R
from pytransform3d import rotations, batch_rotations, transformations, trajectories
from dvrk_scripts.dvrk_control import example_application
from dvrk_scripts.constants_inference import TASK_CONFIGS
from aloha_pro.aloha_scripts.utils import initialize_model_and_tokenizer, encode_text
import time
import IPython
e = IPython.embed
set_seed(0)

class LowLevelPolicy:

    ## ----------------- initializations ----------------
    def __init__(self, args):
        self.args = args
        self.initialize_ros()
        self.setup_policy()
        self.setup_language_model()
        self.initialize_parameters()
        self.language_embedding = None
        self.language_instruction = None
        self.avail_commands = ["grabbing gallbladder", "clipping first clip left tube", "going back first clip left tube", 
            "clipping second clip left tube", "going back second clip left tube",
            "clipping third clip left tube", "going back third clip left tube",
            "go to the cutting position left tube", "go back from the cut left tube",
            "clipping first clip right tube", "going back first clip right tube",
            "clipping second clip right tube", "going back second clip right tube",
            "clipping third clip right tube", "going back third clip right tube",
            "go to the cutting position right tube", "go back from the cut right tube",
            ]
        command_idx = 1
        self.command = self.avail_commands[command_idx] ## can change
        self.temporal_agg = True
        self.debugging = True
        
    def initialize_ros(self):
        self.rt = ros_topics()
        self.ral = crtk.ral('dvrk_arm_test')
        self.psm1_app = example_application(self.ral, "PSM1", 1)
        self.psm2_app = example_application(self.ral, "PSM2", 1)
        self.instruction_sub = rospy.Subscriber("/instructor_prediction", String, self.language_instruction_callback, queue_size=10)
        self.embedding_sub = rospy.Subscriber("/instructor_embedding", Float32MultiArray, self.embeddings_callback, queue_size=10)
    
    def setup_policy(self):
        task_config = TASK_CONFIGS[self.args.task_name]
        self.mean = task_config['action_mode'][1]['mean']
        self.std = task_config['action_mode'][1]['std']
        self.max_ = task_config['action_mode'][1]['max_']
        self.min_ = task_config['action_mode'][1]['min_']
        self.action_mode = task_config['action_mode'][0]
        
        policy_config = {
            'lr': 1e-5,
            'num_queries': 100,
            'kl_weight': 10,
            'hidden_dim': 512,
            'dim_feedforward': 3200,
            'lr_backbone': 1e-5,
            'backbone': 'efficientnet_b3film',
            'enc_layers': 4,
            'num_epochs': self.args.num_epochs,
            'dec_layers': 7,
            'nheads': 8,
            'camera_names': ['left', 'left_wrist', 'right_wrist'],
            "multi_gpu": None,
        }
        
        self.policy = ACTPolicy(policy_config)
        checkpoint = torch.load(self.args.ckpt_dir)
        self.policy.deserialize(checkpoint['model_state_dict'])
        self.policy.cuda()
        self.policy.eval()
        print(f"Loaded: {self.args.ckpt_dir}")
    
    def setup_language_model(self):
        if self.args.use_language:
            self.tokenizer, self.model = initialize_model_and_tokenizer("distilbert")
            assert self.tokenizer is not None and self.model is not None
            print("language model and tokenizer set up completed")
    
    def initialize_parameters(self):
        self.num_inferences = 80
        self.action_execution_horizon = 35
        self.chunk_size = 100
        self.sleep_rate = 0.2
        self.language_encoder = "distilbert"
        self.max_timesteps = 400 
        self.num_queries = self.chunk_size
        self.state_dim = 16

    ## ------------ helper functions for action -------------
    def convert_6d_rot_to_quat(self, rots):
        c1 = rots[:, 0:3]
        c2 = rots[:, 3:6]
        c1 = normalize(c1, axis=1)
        dot_product = np.sum(c1 * c2, axis=1).reshape(-1, 1)
        c2 = normalize(c2 - dot_product * c1, axis=1)
        c3 = np.cross(c1, c2)
        r_mat = np.dstack((c1, c2, c3))
        rots = R.from_matrix(r_mat)
        return rots.as_quat()

    def convert_actions_to_SE3_then_final_actions(self, dts, dquats, qpos_psm, jaw_angles):
        dquats = batch_rotations.batch_quaternion_wxyz_from_xyzw(dquats)
        qpos_psm[3:7] = rotations.quaternion_wxyz_from_xyzw(qpos_psm[3:7])
        dts_dquats = np.concatenate((dts, dquats), axis=1)
        g_qpos = transformations.transform_from_pq(qpos_psm[0:7])
        g_actions = trajectories.transforms_from_pqs(dts_dquats)
        g_poses = trajectories.concat_one_to_many(g_qpos, g_actions)
        output = np.zeros((dquats.shape[0], 8))
        output[:, 0:3] = g_poses[:, 0:3, 3]
        tmp = batch_rotations.quaternions_from_matrices(g_poses[:, 0:3, 0:3])
        output[:, 3:7] = batch_rotations.batch_quaternion_xyzw_from_wxyz(tmp)
        output[:, 7] = np.clip(jaw_angles, -0.35, 1.4)
        return output

    def unnormalize_action(self, naction, norm_scheme):
        action = None
        if norm_scheme == "min_max":
            action = (naction + 1) / 2 * (self.max_ - self.min_) + self.min_
            action[:, 3:9] = naction[:, 3:9]
            action[:, 13:19] = naction[:, 13:19]
        elif norm_scheme == "std":
            action = self.unnormalize_positions_only_std(naction)
        else:
            raise NotImplementedError
        return action

    def unnormalize_positions_only_std(self, diffs):
        unnormalized = diffs * self.std + self.mean
        unnormalized[:, 3:9] = diffs[:, 3:9]
        unnormalized[:, 13:19] = diffs[:, 13:19]
        return unnormalized

    def convert_delta_6d_to_taskspace_quat(self, all_actions, all_actions_converted, qpos):
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
    
    def temporal_ensemble(self, all_time_actions, t, actions_psm1, actions_psm2):
        # Apply temporal assembling to both actions_psm1 and actions_psm2
        # for actions in [actions_psm1, actions_psm2]:
        # Temporal assembling logic for positions
        actions_for_curr_step = all_time_actions[:, t] # Obtain actions for current time step
        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
        actions_for_curr_step = actions_for_curr_step[actions_populated]

        # Exponential weights calculation
        k = 0.01
        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))

        exp_weights = exp_weights / exp_weights.sum()

        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
        # Apply temporal assembling to position
        psm1_position = (actions_for_curr_step[:, :3] * exp_weights).sum(dim=0, keepdim=True)
        psm2_position = (actions_for_curr_step[:, 8:11] * exp_weights).sum(dim=0, keepdim=True)
        
        # Handle quaternion averaging
        psm1_quaternions = actions_for_curr_step[:, 3:7]
        psm2_quaternions = actions_for_curr_step[:, 11:15]
        psm1_quaternion_avg = self.average_quaternions(psm1_quaternions, exp_weights)
        psm2_quaternion_avg = self.average_quaternions(psm2_quaternions, exp_weights)

        # Assign assembled actions back to actions_psm
        actions_psm1[:, :3] = psm1_position.cpu().numpy()
        actions_psm1[:, 3:7] = psm1_quaternion_avg
        actions_psm2[:, :3] = psm2_position.cpu().numpy()
        actions_psm2[:, 3:7] = psm2_quaternion_avg
        return actions_psm1, actions_psm2

    def average_quaternions(self, quaternions, weights):
        """
        Average a set of quaternions using weighted averaging.
        """
        if isinstance(weights, torch.Tensor):
            weights = weights.cpu().numpy()  # Move to CPU and convert to NumPy
            quaternions = quaternions.cpu().numpy()
        # Normalize weights to sum to 1
        weights = np.array(weights, dtype=np.float64)
        weights /= weights.sum()

        # Quaternion averaging: weighted mean in quaternion space
        avg_quat = np.zeros((4,))
        for i, quat in enumerate(quaternions):
            avg_quat += weights[i] * quat

        # Normalize the resulting quaternion
        avg_quat /= np.linalg.norm(avg_quat)
        return avg_quat
    
    
    ## --------------------- callbacks -----------------------
    def language_instruction_callback(self, msg):
        self.language_instruction = msg.data

    def embeddings_callback(self, msg):
        self.language_embedding = np.array(msg.data)

    ## ---------------------- helpers ------------------------

    def get_image_dvrk(self):
        # rt = ros_topics()
        left_img = np.fromstring(self.rt.usb_image_left.data, np.uint8)
        left_img = cv2.imdecode(left_img, cv2.IMREAD_COLOR)
        left_img = cv2.resize(left_img, (480, 360))
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        left_img = rearrange(left_img, 'h w c -> c h w')

        lw_img = self.rt.endo_cam_psm2
        # plt.imshow(lw_img)
        # plt.show()
        # assert(False)
        lw_img = cv2.resize(lw_img, (480, 360))
        lw_img = cv2.cvtColor(lw_img, cv2.COLOR_BGR2RGB)
        lw_img = rearrange(lw_img, 'h w c -> c h w')

        rw_img = self.rt.endo_cam_psm1
        rw_img = cv2.resize(rw_img, (480, 360))
        rw_img = cv2.cvtColor(rw_img, cv2.COLOR_BGR2RGB)
        rw_img = rearrange(rw_img, 'h w c -> c h w')

        curr_image = np.stack([left_img, lw_img, rw_img], axis=0)
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

        return curr_image
    
    def update_img(self):
        ui_image_size = (480, 640)
        left_img = np.fromstring(self.rt.usb_image_left.data, np.uint8)

        left_img = cv2.imdecode(left_img, cv2.IMREAD_COLOR)
        left_img = cv2.resize(left_img, (ui_image_size[1], ui_image_size[0]))
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)

        lw_img = self.rt.endo_cam_psm2

        lw_img = cv2.resize(lw_img, (ui_image_size[1], ui_image_size[0]))
        lw_img = cv2.cvtColor(lw_img, cv2.COLOR_BGR2RGB)

        rw_img = self.rt.endo_cam_psm1
        rw_img = cv2.resize(rw_img, (ui_image_size[1], ui_image_size[0]))
        rw_img = cv2.cvtColor(rw_img, cv2.COLOR_BGR2RGB)

        return left_img, lw_img, rw_img
    
    def generate_command_embedding(self, command):
        ## use language embeddings from high level policy rostopic
        # if self.args.use_language and self.language_embedding is not None:
        #     command_embedding = torch.tensor(self.language_embedding).float().cuda()
        #     print("using high level policy embeddings")
        #     return command_embedding
        
        ## use language instructions from high level policy rostopic
        if self.args.use_language and self.language_instruction is not None:
            command_embedding = encode_text(self.language_instruction, self.language_encoder, self.tokenizer, self.model)
            command_embedding = torch.tensor(command_embedding).cuda()
            print(f"\nusing high level policy command:{self.language_instruction}")

            return command_embedding
        
        ## use language command set in the low level policy
        else:
            command_embedding = encode_text(command, self.language_encoder, self.tokenizer, self.model)
            command_embedding = torch.tensor(command_embedding).cuda()
            print(f"use command:{command}")
            return command_embedding
        

    def execute_actions(self, actions_psm1, actions_psm2):
        for jj in range(self.action_execution_horizon):
            # print("Executing action: ", jj)
            self.ral.spin_and_execute(self.psm1_app.run_full_pose_goal, actions_psm1[jj])
            self.ral.spin_and_execute(self.psm2_app.run_full_pose_goal, actions_psm2[jj])
            time.sleep(self.sleep_rate)

    ## --------------------- main loop -----------------------

    def run(self):
        print("starting low level policy...")
        time.sleep(1)
        if self.temporal_agg:
            all_time_actions = torch.zeros(
                [self.max_timesteps, self.max_timesteps + self.num_queries, self.state_dim]
            ).cuda()

        with torch.inference_mode():
            for t in range(self.num_inferences):
                # if t % self.chunk_size == 0:
                command_embedding = self.generate_command_embedding(self.command)
                # assert command_embedding is not None

                qpos_zero = torch.zeros(1, 20).float().cuda()
                curr_image = self.get_image_dvrk()

                action = self.policy(qpos_zero, curr_image, command_embedding=command_embedding).cpu().numpy().squeeze()
                action = self.unnormalize_action(action, "std")

                qpos_psm1 = np.array((self.rt.psm1_pose.position.x, self.rt.psm1_pose.position.y, self.rt.psm1_pose.position.z,
                                      self.rt.psm1_pose.orientation.x, self.rt.psm1_pose.orientation.y, self.rt.psm1_pose.orientation.z, self.rt.psm1_pose.orientation.w,
                                      self.rt.psm1_jaw))

                qpos_psm2 = np.array((self.rt.psm2_pose.position.x, self.rt.psm2_pose.position.y, self.rt.psm2_pose.position.z,
                                      self.rt.psm2_pose.orientation.x, self.rt.psm2_pose.orientation.y, self.rt.psm2_pose.orientation.z, self.rt.psm2_pose.orientation.w,
                                      self.rt.psm2_jaw))

                if self.action_mode == 'hybrid':
                    actions_psm1 = np.zeros((self.chunk_size, 8)) # pos, quat, jaw
                    actions_psm1[:, 0:3] = qpos_psm1[0:3] + action[:, 0:3] # convert to current translation
                    actions_psm1 = self.convert_delta_6d_to_taskspace_quat(action[:, 0:10], actions_psm1, qpos_psm1)
                    actions_psm1[:, 7] = np.clip(action[:, 9], -0.698, 0.698)  # copy over gripper angles
                    
                    actions_psm2 = np.zeros((self.chunk_size, 8)) # pos, quat, jaw
                    actions_psm2[:, 0:3] = qpos_psm2[0:3] + action[:, 10:13] # convert to current translation
                    actions_psm2 = self.convert_delta_6d_to_taskspace_quat(action[:, 10:], actions_psm2, qpos_psm2)
                    actions_psm2[:, 7] = np.clip(action[:, 19], -0.698, 0.698)  # copy over gripper angles  


                    if self.temporal_agg:
                        # Convert lists to tensors
                        actions_psm1_tensor = torch.tensor(actions_psm1, dtype=torch.float32, device='cuda')
                        actions_psm2_tensor = torch.tensor(actions_psm2, dtype=torch.float32, device='cuda')
                        # Concatenate actions along the feature dimension (dim=1)
                        combined_actions = torch.cat([actions_psm1_tensor, actions_psm2_tensor], dim=1)  # Shape: [100, 16]

                        # Add a new dimension to match the shape for assignment
                        combined_actions = combined_actions.unsqueeze(0)  # Shape: [1, 100, 16]                        print(combined_actions.shape)
                        all_time_actions[[t], t : t + self.num_queries] = combined_actions
                        actions_psm1, actions_psm2 = self.temporal_ensemble(all_time_actions, t, actions_psm1, actions_psm2)

                # Send actions to the robot (assume methods are implemented)
                self.execute_actions(actions_psm1, actions_psm2)
                if self.debugging:
                    key = input("press enter to continue...")
                    if key == "q":
                        exit


## --------------------- main function -----------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', action='store', type=str, 
                        help='specify ckpt file path', 
                        required=True)
    # needed to avoid error for detr
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--use_language', action='store_true')
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)    
    # parser.add_argument('--command_idx', action='store', type=int, help='command_idx')    
    args = parser.parse_args()
        
    # rospy.init_node('dvrk_low_level_policy')
    system = LowLevelPolicy(args)
    system.run()
