import numpy as np
import torch
import os
import random

import h5py
import sys
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pytransform3d import rotations, batch_rotations, transformations, trajectories
from torchvision import transforms, utils
import albumentations as A
from albumentations.pytorch import ToTensorV2

import seaborn as sns
from tqdm import tqdm
import json

# path_to_yay_robot = "/home/jchen396/scr4_akriege1/chole/yay_robot_jhu"
path_to_yay_robot = os.getenv('PATH_TO_YAY_ROBOT')

if path_to_yay_robot:
    sys.path.append(os.path.join(path_to_yay_robot, 'src'))
from aloha_pro.aloha_scripts.utils import initialize_model_and_tokenizer, encode_text

import IPython
e = IPython.embed

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_command_embeddings(unique_phase_folder_names, encoder, tokenizer, model):
    # Returns a dictionary containing the phase command as key and a tuple of the phase command and phase embedding as value
    phase_command_embeddings_dict = {}
    for phase_folder_name in tqdm(unique_phase_folder_names, desc="Embedding phase commands"):
        if phase_folder_name.endswith("_recovery"):
            phase_folder_name = phase_folder_name[:-9]
        elif phase_folder_name.startswith("ACTUAL_CUTTING"):
            if phase_folder_name.endswith("_left"):
                phase_folder_name = "8_go_to_the_cutting_position_left_tube"
            elif phase_folder_name.endswith("_right"):
                phase_folder_name = "16_go_to_the_cutting_position_right_tube"
        # Extract the phase command from the folder name (removing the phase idx and the "_" in between the words)
        _, phase_command = phase_folder_name.split("_")[0], " ".join(phase_folder_name.split("_")[1:])
        embedding = encode_text(phase_command, encoder, tokenizer, model)
        phase_command_embeddings_dict[phase_folder_name]= (phase_command, embedding)

    return phase_command_embeddings_dict

def get_command_embeddings_from_json(unique_phase_folder_names, json_file_name):
    phase_command_embeddings_dict = {}

    with open(json_file_name, "r") as f:
        episode_data = json.load(f)

    for phase_folder_name in tqdm(unique_phase_folder_names, desc="Embedding phase commands"):

        # Extract the phase command from the folder name (removing the phase idx and the "_" in between the words)
        _, phase_command = phase_folder_name.split("_")[0], " ".join(phase_folder_name.split("_")[1:])

        # Search for the command in the JSON data
        found_embedding = None
        for item in episode_data:
            # print(item)
            if isinstance(item, dict) and item.get('command') == phase_command:

                found_embedding = item.get('embedding')
                break
        
        # Store the found embedding (if any)
        if found_embedding is not None:
            phase_command_embeddings_dict[phase_folder_name] = (phase_command, found_embedding)
        else:
            print(f"Embedding not found for command: {phase_command}")
    
    return phase_command_embeddings_dict


def split_tissue_samples(dataset_dir, num_tissue_samples, train_ratio, val_ratio, test_ratio):
    # Calculate the number of samples for each set
    num_train = int(train_ratio * num_tissue_samples)
    num_val = int(val_ratio * num_tissue_samples)
    num_test = num_tissue_samples - num_train - num_val

    # Generate a list of indices and shuffle them
    all_indices = list(range(1, num_tissue_samples+1))
    np.random.shuffle(all_indices)

    # Split the indices based on the calculated numbers
    # TODO: Check if the indices are the same for every training (by using the seed) even when training on a different machine - e.g., otherwise introducing bias when resuming training from last checkpoint
    # TODO: Alternative would be fixed indices for each tissue sample (but randomized assuming that the execution of the surgerymight evolve over newer tissue samples)
    train_indices = [idx for idx in all_indices[:num_train]]
    val_indices = [idx for idx in all_indices[num_train:num_train + num_val]]
    test_indices = [idx for idx in all_indices[num_train + num_val:]]

    return train_indices, val_indices, test_indices

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transforms.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class DataAug(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, img_hw):
        self.ratio = 0.95
        self.img_hw = img_hw # heigt width
        self.random_crop = transforms.RandomCrop(size=[int(self.img_hw[0] * self.ratio), 
                                                       int(self.img_hw[1] * self.ratio)])
        self.resize = transforms.Resize(self.img_hw, antialias=True)
        self.random_rot = transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False)
        self.composed = transforms.Compose([self.random_crop, 
                                            self.resize, 
                                            self.random_rot])
        self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08)

        ## albumentations (pixel dropout, blur, brightness, contrast, gaussian blur)
        self.albumentations_transforms = A.Compose([
            A.CoarseDropout(max_holes=8, max_height=img_hw[0] // 8, max_width=img_hw[1] // 8, min_holes=1, min_height=img_hw[0] // 32, min_width=img_hw[1] // 32, fill_value=0, p=0.5),
            A.OneOf([
                A.Blur(blur_limit=3, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            ], p=1.0),
            ToTensorV2()
        ])

    def __call__(self, sample):

        img_l, img_r, img_lw, img_rw = sample
        cat_img = torch.cat((img_l, img_r), axis = 0)
        cat_img_tf = self.composed(cat_img)
        img_lw = self.composed(img_lw)
        img_rw = self.composed(img_rw)

        img_l_tf, img_r_tf = cat_img_tf[0:3], cat_img_tf[3:]
        img_l_tf = self.color_jitter(img_l_tf)
        img_r_tf = self.color_jitter(img_r_tf)
        img_lw = self.color_jitter(img_lw)
        img_rw = self.color_jitter(img_rw)

        img_l_tf = self.albumentations_transforms(image=np.array(img_l_tf))['image']
        img_r_tf = self.albumentations_transforms(image=np.array(img_r_tf))['image']
        img_lw = self.albumentations_transforms(image=np.array(img_lw))['image']
        img_rw = self.albumentations_transforms(image=np.array(img_rw))['image']

        return {'img_l': img_l_tf,
                'img_r': img_r_tf,
                'img_lw': img_lw,
                'img_rw': img_rw}


class EpisodicDatasetDvrkGeneric(torch.utils.data.Dataset):
    def __init__(
        self,
        episode_ids,
        tissue_sample_ids, 
        dataset_dir, 
        camera_names, 
        camera_file_suffixes, 
        # num_episodes,
        task_config,
        norm_stats=None,
        max_len=None,
        command_list=None,
        use_language=False,
        language_encoder="distilbert",
        ):

        super(EpisodicDatasetDvrkGeneric).__init__()

        if len(tissue_sample_ids) == 0:
            raise ValueError("No tissue samples found in the dataset directory.")
        
        # self.episode_ids = episode_ids
        self.episode_ids = episode_ids if len(episode_ids) > 0 else [0]
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.camera_file_suffixes = camera_file_suffixes
        self.norm_stats = norm_stats
        self.max_len = max_len
        if command_list is not None:
            self.command_list = [cmd.strip("'\"") for cmd in command_list]
        self.total_items = 0
        self.use_language = use_language
        # self.num_episodes = num_episodes
        self.task_config = task_config
        self.action_mode = task_config['action_mode'][0]
        self.norm_scheme = task_config['norm_scheme']
        self.phantom = task_config['phantom']
        self.recovery_ratio = task_config['recovery_ratio']
        self.is_sim = None
        self.cutting_action_pad_size = task_config['cutting_action_pad_size']
        self.img_height, self.img_width = [480, 640]

        # Load the tissue samples and their phases and demos (for later stitching of the episodes)        
        self.tissue_phase_demo_dict = {}
        for tissue_sample_id in tissue_sample_ids:
            if self.phantom:
                tissue_sample_name = f"phantom_{tissue_sample_id}"
            else:
                tissue_sample_name = f"tissue_{tissue_sample_id}"
            tissue_sample_dir_path = os.path.join(dataset_dir, tissue_sample_name)
            phases = os.listdir(tissue_sample_dir_path)
            self.tissue_phase_demo_dict[tissue_sample_name] = {}

            for phase_sample in phases:
                demo_samples_path = os.path.join(tissue_sample_dir_path, phase_sample)

                if os.path.isfile(demo_samples_path):
                    continue  # Skip if the tissue sample path is not a directory

                demo_samples = os.listdir(demo_samples_path)

                ## remove corrections folder
                for demo_sample in demo_samples:
                    if demo_sample == "Corrections":
                        demo_samples.remove(demo_sample)

                ## initialize the dictionary for the tissue sample
                if tissue_sample_name not in self.tissue_phase_demo_dict:
                    self.tissue_phase_demo_dict[tissue_sample_name] = {}

                ## adjust the number of demos for the recovery phase
                if phase_sample.endswith("_recovery"):
                    num_of_perfect_demos = len(os.listdir(os.path.join(tissue_sample_dir_path, phase_sample[:-9])))
                    num_of_recovery_demos = int(num_of_perfect_demos * self.recovery_ratio)
                    print(f"Recovery phase: {phase_sample}, num of perfect demos: {num_of_perfect_demos}, num of recovery demos: {num_of_recovery_demos}")
                    demo_samples = demo_samples[:num_of_recovery_demos]

                # Add or update the demo samples in the dictionary
                self.tissue_phase_demo_dict[tissue_sample_name].setdefault(phase_sample, []).extend(demo_samples)

            ## create language embeddings
            if self.use_language:
                self.command_embeddings_dict = {}

                self.language_encoder = language_encoder
                tokenizer, model = initialize_model_and_tokenizer(self.language_encoder)
                unique_phase_folder_names = np.unique([phase_folder_name for tissue_sample in self.tissue_phase_demo_dict.values() for phase_folder_name in tissue_sample.keys()])

                # print("phase:", unique_phase_folder_names)
                print("\ngenerating command embeddings...\n")
                self.command_embeddings_dict[tissue_sample_name] = generate_command_embeddings(unique_phase_folder_names, self.language_encoder, tokenizer, model)
                # json_name = f"candidate_embeddings_{self.language_encoder}.json"
                # json_path = os.path.join(tissue_sample_dir_path, json_name)

                # self.command_embeddings_dict[tissue_sample_name] = generate_command_embeddings(unique_phase_folder_names, json_path)
                # print("embeddings:", self.command_embeddings_dict)

                del tokenizer, model
                # print(f"   {phase_sample}, {demo_samples}\n")
        print("num of tissues:", len(self.tissue_phase_demo_dict.keys()))

        print("num of samples:", sum(len(samples) for samples in self.tissue_phase_demo_dict.values()))
        total_count = 0
        for phase_dict in self.tissue_phase_demo_dict.values():
            for demo_samples in phase_dict.values():
                total_count += len(demo_samples)

        print("total count:", total_count)

        self.all_samples = [(tissue_sample, phase, sample) 
                            for tissue_sample in self.tissue_phase_demo_dict
                            for phase in self.tissue_phase_demo_dict[tissue_sample]
                            for sample in self.tissue_phase_demo_dict[tissue_sample][phase]]
        

        self.header_name_qpos_psm1 = ["psm1_pose.position.x", "psm1_pose.position.y", "psm1_pose.position.z",
                                "psm1_pose.orientation.x", "psm1_pose.orientation.y", "psm1_pose.orientation.z", "psm1_pose.orientation.w",
                                "psm1_jaw"]
        
        self.header_name_qpos_psm2 = ["psm2_pose.position.x", "psm2_pose.position.y", "psm2_pose.position.z",
                                "psm2_pose.orientation.x", "psm2_pose.orientation.y", "psm2_pose.orientation.z", "psm2_pose.orientation.w",
                                "psm2_jaw"]

        self.header_name_actions_psm1 = ["psm1_sp.position.x", "psm1_sp.position.y", "psm1_sp.position.z",
                                    "psm1_sp.orientation.x", "psm1_sp.orientation.y", "psm1_sp.orientation.z", "psm1_sp.orientation.w",
                                    "psm1_jaw_sp"]

        self.header_name_actions_psm2 = ["psm2_sp.position.x", "psm2_sp.position.y", "psm2_sp.position.z",
                                    "psm2_sp.orientation.x", "psm2_sp.orientation.y", "psm2_sp.orientation.z", "psm2_sp.orientation.w",
                                    "psm2_jaw_sp"]
        
        self.header_ecm = ["ecm_pose.position.x", "ecm_pose.position.y", "ecm_pose.position.z",
                            "ecm_pose.orientation.x", "ecm_pose.orientation.y", 
                            "ecm_pose.orientation.z", "ecm_pose.orientation.w"]
        
        self.quat_cp_psm1 = ["psm1_pose.orientation.x", "psm1_pose.orientation.y", "psm1_pose.orientation.z", "psm1_pose.orientation.w"]
        self.quat_cp_psm2 = ["psm2_pose.orientation.x", "psm2_pose.orientation.y", "psm2_pose.orientation.z", "psm2_pose.orientation.w"]

        # self.transforms = DataAug([224, 224])
        self.transforms = DataAug([self.img_height, self.img_width])

        # self.__getitem__(0) # initialize self.is_sim

    def compute_diff_actions(self, qpos, action):
        """
        qpos: current position [9]
        action: actions commanded by the user [n_actions x 9]
        returns: relative actions w.r.t qpos
        """
        # find diff first and then fill-in the quaternion differences properly
        diff = action - qpos

        quat_init = qpos[3:7]
        quat_actions = action[:, 3:7]

        # convert quaternions to rotation matrices
        r_init = R.from_quat(quat_init)
        r_actions = R.from_quat(quat_actions)
        # find their diff
        diff_rs = r_init.inv()*r_actions 
        # extract their first two columns
        diff_6d = diff_rs.as_matrix()[:,:,:2]
        diff_6d = diff_6d.transpose(0,2,1).reshape(-1, 6) # first column then second column
        
        diff_expand = np.zeros((diff.shape[0], 10)) # TODO: hard-coded dim (10) for a single arm
        diff_expand[:diff.shape[0], 0:diff.shape[1]] = diff 
        diff = diff_expand

        diff[:, 3:9] = diff_6d
        diff[:, 9] = action[:, -1] # fill in the jaw angle (note: jaw angle is not relative)
        return diff
    
    def compute_relative_actions_in_SE3(self, qpos, action):
        """
        Note: this is the proper implementation
        qpos: current position (measured_cp), xyz, xyzw, jaw angle (8-dim vector)
        action: set point on the dvrk (action_horizon x 8)
        
        returns: relative position and rotation w.r.t qpos
        """
        
        diff = np.zeros((action.shape[0], 10)) # TODO: hard-coded dim (10) for a single arm

        # convert current pose to SE(3)
        qpos_wxyz = rotations.quaternion_wxyz_from_xyzw(qpos[3:7])
        qpos_py3d = np.concatenate((qpos[0:3], qpos_wxyz))
        g_qpos = transformations.transform_from_pq(qpos_py3d) # no jaw angle!

        # convert actions to SE(3)
        action_wxyz = batch_rotations.batch_quaternion_wxyz_from_xyzw(action[:, 3:7]) 
        action_py3d = np.concatenate((action[:, 0:3], action_wxyz), axis = 1)
        g_action = trajectories.transforms_from_pqs(action_py3d)

        # invert current pose
        g_qpos_inv = transformations.invert_transform(g_qpos)
        diff_SE3 = trajectories.concat_one_to_many(g_qpos_inv, g_action)

        # construct 6d rot
        diff_6d = diff_SE3[:,0:3,:2]
        diff_6d = diff_6d.transpose(0,2,1).reshape(-1, 6) # first column then second column
        
        # fill in translation elements
        diff[:, 0:3] = diff_SE3[:, 0:3, 3] # replace the translations with the last column first three rows of SE3
        # fill in 6d rot
        diff[:, 3:9] = diff_6d
        # fill in jaw angle (note: jaw angle is absolute, not relative)
        diff[:, 9] = action[:, 7]
        return diff

    # misnomer: jaw angles are also being normalized
    def min_max_scale_positions_only(self, diffs):
        """
        diffs: n_actions x 20
        return: normalized n_actions x 20
        Note: BOTH POSITIONS AND JAW ANGLES ARE NORMALIZED (orientations remain original)
        """
        max_ = self.task_config['action_mode'][1]['max_']
        min_ = self.task_config['action_mode'][1]['min_']
        normalized = (diffs - min_) / (max_ - min_) * 2 - 1

        # replace w/ originals for 6D rot
        normalized[:, 3:9] = diffs[:, 3:9]
        normalized[:, 13:19] = diffs[:, 13:19]

        return normalized
    
    def standardize_positions_only(self, diffs):
        """
        diffs: n_actions x 20
        return: normalized n_actions x 20 (zero mean unit variance)
        Note: BOTH POSITIONS AND JAW ANGLES ARE NORMALIZED (orientations remain original)
        """
        mean = self.task_config['action_mode'][1]['mean']
        std = self.task_config['action_mode'][1]['std']
        # print("mean shape", mean.shape)
        # print("std shape", std.shape)
        normalized = (diffs - mean) / std

        # replace w/ originals for 6D rot
        normalized[:, 3:9] = diffs[:, 3:9]
        normalized[:, 13:19] = diffs[:, 13:19]

        return normalized

    def __len__(self):
        # if self.total_items == 0:
        #     for tissue_sample in self.tissue_phase_demo_dict.values():
        #         for phase_sample in tissue_sample.values():
        #             self.total_items += len(phase_sample)
        # return self.total_items        
        return len(self.episode_ids)


    def __getitem__(self, index):
        # sample_full_episode = False # hardcode
        max_len = self.max_len

        # Get the tissue sample, phase, and sample based on the index

        episode_id = self.episode_ids[index]
        tissue_sample, phase, sample = self.all_samples[episode_id]
        dataset_path = os.path.join(self.dataset_dir, f"{tissue_sample}/{phase}/{sample}")
        # print(dataset_path)
        csv_path = os.path.join(dataset_path, "ee_csv.csv")
        csv = pd.read_csv(csv_path)
        episode_len = len(csv)



        start_ts = np.random.choice(episode_len)
        if (phase.startswith("8_go_to_the_cutting_position_left_tube") or phase.startswith("16_go_to_the_cutting_position_right_tube")) and start_ts >= episode_len - self.cutting_action_pad_size:
            img_idx = episode_len - self.cutting_action_pad_size - 1
            image_path_l = os.path.join(dataset_path, "left_img_dir",
                                    "frame{:06d}".format(img_idx) + "_left.jpg")
            image_path_r = os.path.join(dataset_path, "right_img_dir",
                                    "frame{:06d}".format(img_idx) + "_right.jpg")
            image_path_lw = os.path.join(dataset_path, "endo_psm2",
                                    "frame{:06d}".format(img_idx) + "_psm2.jpg")
            image_path_rw = os.path.join(dataset_path, "endo_psm1",
                                    "frame{:06d}".format(img_idx) + "_psm1.jpg")
        else:
            image_path_l = os.path.join(dataset_path, "left_img_dir",
                                    "frame{:06d}".format(start_ts) + "_left.jpg")
            image_path_r = os.path.join(dataset_path, "right_img_dir",
                                    "frame{:06d}".format(start_ts) + "_right.jpg")
            image_path_lw = os.path.join(dataset_path, "endo_psm2",
                                    "frame{:06d}".format(start_ts) + "_psm2.jpg")
            image_path_rw = os.path.join(dataset_path, "endo_psm1",
                                    "frame{:06d}".format(start_ts) + "_psm1.jpg")
        
        img_l = cv2.imread(image_path_l)
        img_r = cv2.imread(image_path_r)
        img_lw = cv2.imread(image_path_lw)
        img_rw = cv2.imread(image_path_rw)
        # img_l = cv2.resize(img_l, [224, 224])
        # img_r = cv2.resize(img_r, [224, 224])
        # img_lw = cv2.resize(img_lw, [224, 224])
        # img_rw = cv2.resize(img_rw, [224, 224])
        img_l = cv2.resize(img_l, [self.img_height, self.img_width])
        img_r = cv2.resize(img_r, [self.img_height, self.img_width])
        img_lw = cv2.resize(img_lw, [self.img_height, self.img_width])
        img_rw = cv2.resize(img_rw, [self.img_height, self.img_width])
        
        img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
        img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
        img_lw = cv2.cvtColor(img_lw, cv2.COLOR_BGR2RGB)
        img_rw = cv2.cvtColor(img_rw, cv2.COLOR_BGR2RGB)

        # construct observations
        img_l = torch.from_numpy(img_l).float() # channel last
        img_r = torch.from_numpy(img_r).float()
        img_lw = torch.from_numpy(img_lw).float() # channel last
        img_rw = torch.from_numpy(img_rw).float() # channel last

        # bring channel to to the third
        img_l = torch.einsum('h w c -> c h w', img_l)
        img_r = torch.einsum('h w c -> c h w', img_r)
        img_lw = torch.einsum('h w c -> c h w', img_lw)
        img_rw = torch.einsum('h w c -> c h w', img_rw)

        # normalize image and change dtype to float
        img_l = img_l / 255.0
        img_r = img_r / 255.0
        img_lw = img_lw / 255.0
        img_rw = img_rw / 255.0

        # data aug
        tfmed = self.transforms([img_l, img_r, img_lw, img_rw])
        img_l = tfmed['img_l']
        img_r = tfmed['img_r']
        img_lw = tfmed['img_lw']
        img_rw = tfmed['img_rw']

        image_data = np.stack([img_l, img_r, img_lw, img_rw], axis = 0) 
        
        # get current position and actions
        qpos_psm1 = csv[self.header_name_qpos_psm1].iloc[start_ts, :].to_numpy()
        action_psm1 = csv[self.header_name_actions_psm1].iloc[start_ts:start_ts+400].to_numpy() # note 400 added here
        qpos_psm2 = csv[self.header_name_qpos_psm2].iloc[start_ts, :].to_numpy()
        action_psm2 = csv[self.header_name_actions_psm2].iloc[start_ts:start_ts+400].to_numpy() # note 400 added here
    
        diff_psm1 = None
        diff_psm2 = None

        # compute relative actions TODO: make it work for SE3 scenarios
        if self.action_mode == 'hybrid':
            diff_psm1 = self.compute_diff_actions(qpos_psm1, action_psm1)
            diff_psm2 = self.compute_diff_actions(qpos_psm2, action_psm2)
        elif self.action_mode == 'ego':
            diff_psm1 = self.compute_relative_actions_in_SE3(qpos_psm1, action_psm1)
            diff_psm2 = self.compute_relative_actions_in_SE3(qpos_psm2, action_psm2)
        else:
            raise(NotImplementedError) 

        # stack the actions along column dim
        action = np.column_stack((diff_psm1, diff_psm2))

        # normalize data
        if self.norm_scheme == 'min_max': 
          action = self.min_max_scale_positions_only(action)
        elif self.norm_scheme == 'std':
           action = self.standardize_positions_only(action)
        else:
            raise NotImplementedError

        action_len = min(episode_len - start_ts, 400) # TODO: a bit messy code
        padded_action = np.zeros((400, 20), dtype=np.float32) # TODO: this is hardcoded to be 400 
        # timesteps by default, but you will be taking a subset anyway later on i.e. chunk size, so it's probably ok and also
        # you will never be making predictions beyond 400 timestep horizon
        # also hardcoded to 10 dim per arm
        padded_action[:action_len] = action
        is_pad = np.zeros(400)
        is_pad[action_len:] = 1

        # set current poses to zeros (dvrk kinematics unreliable)
        qpos = np.zeros(20)

        # construct observations
        # image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        # image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        # image_data = image_data / 255.0
        # action_data = (action_data - mean) / std
      
        if self.use_language:
            if phase.endswith("_recovery"):
                phase = phase[:-9]
            elif phase.startswith("ACTUAL_CUTTING"):
                if phase.endswith("_left"):
                    phase = "8_go_to_the_cutting_position_left_tube"
                elif phase.endswith("_right"):
                    phase = "16_go_to_the_cutting_position_right_tube"
            phase_command, embedding = self.command_embeddings_dict[tissue_sample][phase]
            command_embedding = torch.tensor(embedding).squeeze()
            return image_data, qpos_data, action_data, is_pad, command_embedding
        else:
            return image_data, qpos_data, action_data, is_pad
"""
Test the EpisodicDatasetDvrkGeneric class.
"""
# if __name__ == "__main__":
#     seed = 42
#     set_seed(seed)
#     # Parameters for the test
#     path_to_dataset = os.getenv("PATH_TO_DATASET")
#     # path_to_dataset = "/home/imerse/chole_ws/data"

#     dataset_dir = os.path.join(path_to_dataset, "base_chole_clipping_cutting")
#     tissue_samples_ids = [1]
#     camera_names = ["left_img_dir", "right_img_dir", "endo_psm1", "endo_psm2"]
#     camera_file_suffixes = ["_left.jpg", "_right.jpg", "_psm1.jpg", "_psm2.jpg"]
#     num_episodes = 200 # Total number of episodes
#     use_language_flag = True
#     from dvrk_scripts.constants_dvrk import TASK_CONFIGS
#     task_config = TASK_CONFIGS['base_chole_clipping_cutting']
#     episode_ids = [i for i in range(num_episodes)]
#     dataset = EpisodicDatasetDvrkGeneric(
#                 episode_ids,
#                 tissue_samples_ids,
#                 dataset_dir,
#                 camera_names,
#                 camera_file_suffixes,
#                 # num_episodes,
#                 task_config,
#                 use_language=use_language_flag
#                 )

#     # Sample a random item from the dataset
#     rdm_idx = np.random.randint(0, len(dataset))
#     print("idx:", rdm_idx)
#     if use_language_flag:
#         image_data, qpos_data, action_data, is_pad, command_embedding = dataset[rdm_idx]
#         print(f"Image sequence shape: {image_data.shape}")
#         # print(f"Command: {command}")
#     else:
#         image_data, qpos_data, action_data, is_pad = dataset[rdm_idx]   


#         # Create a figure with subplots: one row per timestamp, one column per camera
#     fig, axes = plt.subplots(1, len(camera_names), figsize=(15, 10))
#     for cam_idx, cam_name in enumerate(camera_names):
#         img = image_data[cam_idx]  # Assuming image_data is a numpy array or compatible type

#         # Check and possibly transpose the shape if needed
#         if img.shape[0] == 3 and len(img.shape) == 3:
#             img = np.transpose(img, (1, 2, 0))  # Transpose to (height, width, channels)

#         axes[cam_idx].imshow(img)
#         axes[cam_idx].set_title(f"{cam_name}")
#         axes[cam_idx].axis('off')  # Optionally turn off the axis
#     plt.show()
