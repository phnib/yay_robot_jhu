import os
# import h5py
import json
import sys
from collections import defaultdict, deque
import math
import copy

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader, ConcatDataset

# import src code
path_to_yay_robot = os.getenv('PATH_TO_YAY_ROBOT')
if path_to_yay_robot:
    sys.path.append(os.path.join(path_to_yay_robot, 'src'))
else:
    raise EnvironmentError("Environment variable PATH_TO_YAY_ROBOT is not set")
from aloha_pro.aloha_scripts.utils import initialize_model_and_tokenizer, encode_text
from instructor.utils import center_crop_resize
from instructor.constants_daVinci import DATASET_CONFIGS # get task parameters
from instructor.utils import DAggerSampler, randintgaussian
    
def generate_command_embeddings(unique_phase_folder_names, encoder, tokenizer, model, reduced_base_class_set_flag):
    # Returns a dictionary containing the phase command as key and a tuple of the phase command and phase embedding as value
    phase_command_embeddings_dict = {}
    try: 
        unique_phase_folder_names_sorted = sorted(unique_phase_folder_names, key=lambda x: int(x.split('_')[0]))
    except:
        unique_phase_folder_names_sorted = unique_phase_folder_names
    
    if reduced_base_class_set_flag:
        instruction_to_phase_idx_mapping = {
            "Apply first clip left tube": [1,2,3],
            "Apply second clip left tube": [4,5],
            "Apply third clip left tube": [6,7],
            "Cut left tube": [8,9],
            "Apply first clip right tube": [10,11],
            "Apply second clip right tube": [12,13],
            "Apply third clip right tube": [14,15],
            "Cut right tube": [16,17],
        }
        phase_idx_to_instruction_mapping = {str(phase_idx): instruction for instruction, phase_idx_list in instruction_to_phase_idx_mapping.items() for phase_idx in phase_idx_list}
    for phase_folder_name in unique_phase_folder_names_sorted:
        # Extract the phase command from the folder name (removing the phase idx and the "_" in between the words) - No extra command for recovery phases
        phase_idx, phase_command = phase_folder_name.split("_")[0], " ".join(phase_folder_name.replace("_recovery", "").split("_")[1:])

        if reduced_base_class_set_flag:
            # Reduce base instruction set (keep finetuining instructions)
            if phase_idx in phase_idx_to_instruction_mapping:
                phase_command = phase_idx_to_instruction_mapping[phase_idx]
                
        embedding = encode_text(phase_command, encoder, tokenizer, model)
        phase_command_embeddings_dict[phase_folder_name]= (phase_command, embedding)

    return phase_command_embeddings_dict

def split_tissue_samples(dataset_dir, tissue_names, train_ratio, val_ratio, test_only_flag):
    # Calculate the number of samples for each set
    if dataset_dir == "base_chole_clipping_cutting":
        tissue_names.remove("tissue_1") # Remove tissue_1 from the dataset as not complete
            
    num_tissue_samples = len(tissue_names)
    num_train = int(train_ratio * num_tissue_samples) 
    num_val = math.ceil(val_ratio * num_tissue_samples) 

    # Generate a list of indices and shuffle them
    all_indices = list(range(0, num_tissue_samples))
    np.random.shuffle(all_indices)

    # Split the indices based on the calculated numbers
    train_tissue_names = [tissue_names[idx] for idx in all_indices[:num_train]]
    val_tissue_names = [tissue_names[idx] for idx in all_indices[num_train:num_train + num_val]]
    test_tissue_names = [tissue_names[idx] for idx in all_indices[num_train + num_val:]]

    if len(test_tissue_names) == 0 and test_only_flag:
        test_tissue_names = val_tissue_names

    return train_tissue_names, val_tissue_names, test_tissue_names

def extract_candidate_embeddings_and_commands(command_embeddings_dict):
    # Extract the candidate embeddings and commands
    candidate_embeddings = []
    candidate_texts = []
    for _, (phase_command, phase_embedding) in command_embeddings_dict.items():
        if phase_command not in candidate_texts: # Only add unique commands
            candidate_texts.append(phase_command)
            candidate_embeddings.append(torch.tensor(phase_embedding).squeeze())
        
    return torch.stack(candidate_embeddings), candidate_texts

def extract_phase_idx_to_instruction_mapping(command_embeddings_dict):
    # Extract the instruction to phase index mapping
    instruction_to_phase_idx_mapping = defaultdict(list)
    for phase_idx, (phase_command, _) in command_embeddings_dict.items():
        instruction_to_phase_idx_mapping[phase_command].append(phase_idx)
    # Get the phase index to instruction mapping
    phase_to_instruction_mapping = {str(phase_idx): instruction for instruction, phase_idx_list in instruction_to_phase_idx_mapping.items() for phase_idx in phase_idx_list}
    
    return phase_to_instruction_mapping

def get_valid_demo_start_end_indices(demo_folder_path, camera_names, before_phase_offset, after_phase_offset):
    # Load the start and end indices for the current demo as the valid range of the demo
    demo_num_frames_total = len(os.listdir(os.path.join(demo_folder_path, camera_names[0])))
    start, end = 0, demo_num_frames_total - 1
    indices_curated_file_path = os.path.join(demo_folder_path, "indices_curated.json")
    if os.path.exists(indices_curated_file_path):
        with open(indices_curated_file_path, 'r') as indices_curated_file:
            try:
                indices_curated_dict = json.load(indices_curated_file)
            except json.JSONDecodeError:
                print(f"Error reading indices_curated.json for {demo_folder_path}. Continue with max recording range.")
            if "start" in indices_curated_dict:
                start = max(indices_curated_dict['start'] - before_phase_offset, start)
            if "end" in indices_curated_dict:
                end = min(indices_curated_dict['end'] + after_phase_offset, end)
    demo_num_frames_valid = end - start + 1
    
    return start, end, demo_num_frames_valid

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split_name,
        tissue_sample_names,
        dataset_dir,
        camera_names,
        camera_file_suffixes,
        history_len=4,
        prediction_offset=15,
        history_step_size=30,
        num_episodes=200,
        input_transforms=None,
        center_crop_flag=False,
        reduced_base_class_set_flag=False,
        use_phase_history_flag=False,
        use_jaw_values_flag=False,
        phase_history_len=6,
        prediction_step_size=30,
        recovery_probability=0.2,
        phase_history_only_phase_switches_flag=True,
        verbose=False,
        
    ):
        super().__init__()
        
        if len(tissue_sample_names) == 0:
            raise ValueError("No tissue samples found in the dataset directory.")
        
        self.split_name = split_name
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.camera_file_suffixes = camera_file_suffixes
        self.history_len = history_len
        self.prediction_offset = prediction_offset
        self.history_step_size = history_step_size
        self.num_episodes = num_episodes
        self.input_transforms = input_transforms
        self.center_crop_flag = center_crop_flag
        self.reduced_base_class_set_flag = reduced_base_class_set_flag
        self.use_history_flag = use_phase_history_flag
        self.use_jaw_values_flag = use_jaw_values_flag
        self.phase_history_len = phase_history_len
        self.prediction_step_size = prediction_step_size
        self.recovery_probability = recovery_probability
        self.phase_history_only_phase_switches_flag = phase_history_only_phase_switches_flag
        self.verbose = verbose
        
        # Set the before_phase_offset and after_phase_offset
        dataset_name = os.path.basename(dataset_dir)
        dataset_config = DATASET_CONFIGS[dataset_name]
        self.before_phase_offset = dataset_config["before_phase_offset"]
        self.after_phase_offset = dataset_config["after_phase_offset"]
 
        # Initialize the phase_len_dict with defaultdict
        phase_len_dict = defaultdict(list)

        # Initialize tissue_phase_demo_dict
        self.tissue_phase_demo_dict = {}

        for tissue_sample_name in tissue_sample_names:
            tissue_sample_dir_path = os.path.join(dataset_dir, tissue_sample_name)
            phases = [file_name for file_name in os.listdir(tissue_sample_dir_path) if os.path.isdir(os.path.join(tissue_sample_dir_path, file_name)) and file_name.split('_')[0].isdigit()]
            phases_ordered = sorted(phases, key=lambda x: int(x.split('_')[0]))
            self.tissue_phase_demo_dict[tissue_sample_name] = {}
            for phase_sample in phases_ordered:
                files_in_phase_folder = os.listdir(os.path.join(tissue_sample_dir_path, phase_sample))
                demo_samples = [demo_sample for demo_sample in files_in_phase_folder if demo_sample[8] == "-"]
                self.tissue_phase_demo_dict[tissue_sample_name][phase_sample] = demo_samples
                # Add the length of the phase for current demo to phase_len_dict
                for demo_sample in demo_samples:
                    demo_num_frames_valid = get_valid_demo_start_end_indices(os.path.join(tissue_sample_dir_path, phase_sample, demo_sample), camera_names, self.before_phase_offset, self.after_phase_offset)[2]
                    phase_len_dict[phase_sample].append(demo_num_frames_valid)
                
        # Generate the embeddings for all phase commands
        encoder_name = "distilbert"
        tokenizer, model = initialize_model_and_tokenizer(encoder_name)
        unique_phase_folder_names = np.unique([phase_folder_name for tissue_sample in self.tissue_phase_demo_dict.values() for phase_folder_name in tissue_sample.keys()])
        self.command_embeddings_dict = generate_command_embeddings(unique_phase_folder_names, encoder_name, tokenizer, model, reduced_base_class_set_flag)
        del tokenizer, model
        
        # Compute the dataset statistics
        self.ds_statistics_dict = self.compute_dataset_statistics(phase_len_dict)
        
    def __len__(self):
        # Here this means the number of randomly generated stitched episodes
        return self.num_episodes

    def compute_dataset_statistics(self, phase_len_dict):
        # Compute the statistics of the dataset
        ds_statistics_dict = {}
        for phase_name, phase_len_list in phase_len_dict.items():
            ds_statistics_dict[phase_name] = {
                "min": min(phase_len_list),
                "max": max(phase_len_list),
                "mean": sum(phase_len_list) / len(phase_len_list),
                "std": np.std(phase_len_list),
                "num_demos": len(phase_len_list),
            }
            
        # Compute statistics for merged phases
        if self.reduced_base_class_set_flag:
            # Merge the phases from the self.command_embeddings_dict
            for phase_name, (phase_command, _) in self.command_embeddings_dict.items():
                if "recovery" in phase_name:
                    continue
                if phase_command in ds_statistics_dict:
                    ds_statistics_dict[phase_command]["min"] = ds_statistics_dict[phase_command]["min"] + ds_statistics_dict[phase_name]["min"]
                    ds_statistics_dict[phase_command]["max"] = ds_statistics_dict[phase_command]["max"] + ds_statistics_dict[phase_name]["max"]
                    ds_statistics_dict[phase_command]["mean"] = ds_statistics_dict[phase_command]["mean"] + ds_statistics_dict[phase_name]["mean"]
                    ds_statistics_dict[phase_command]["std"].append(ds_statistics_dict[phase_name]["std"])
                    ds_statistics_dict[phase_command]["num_demos"] *= ds_statistics_dict[phase_name]["num_demos"] 
                else:
                    ds_statistics_dict[phase_command] = ds_statistics_dict[phase_name].copy()
                    ds_statistics_dict[phase_command]["std"] = [ds_statistics_dict[phase_name]["std"]]
            
            phase_commands_list = [phase_command for phase_command, _ in self.command_embeddings_dict.values()]
            for phase_command in phase_commands_list:
                ds_statistics_dict[phase_command]["std"] = np.sqrt(np.sum(np.square(np.array(ds_statistics_dict[phase_command]["std"])))) # Std of Gaussian distribution is sqrt of sum of squares of stds
            
        return ds_statistics_dict

    def get_embedding_command_phase_for_ts(self, selected_phase_demo_dict, target_ts):
        # Returns the command embedding and the command for the target timestep
        for phase_segment in selected_phase_demo_dict.values():
            if phase_segment["phases_demo_start_idx"] <= target_ts <= phase_segment["phases_demo_end_idx"]:
                return torch.tensor(phase_segment["embedding"]).squeeze(), phase_segment["command"], phase_segment["phase_folder_name"]
        else:
            return None, None, None

    def get_current_phase_demo_folder_and_demo_frame_idx(self, selected_phase_demo_dict, target_ts):
        # Returns the phase and the demo frame index for the target timestep
        for phase_segment in selected_phase_demo_dict.values():
            if phase_segment["phases_demo_start_idx"] <= target_ts <= phase_segment["phases_demo_end_idx"]:
                demo_frame_idx = target_ts - phase_segment["phases_demo_start_idx"] + phase_segment["demo_rel_start_idx"]
                return phase_segment["phase_folder_name"], phase_segment["demo_folder_name"], demo_frame_idx
        else:
            raise ValueError(f"Could not find phase and demo frame index for target_ts {target_ts}.")

    def create_phase_demo_metadata_dict(self, selected_tissue_sample, selected_phases, selected_phase_demos):
        # Returns a dictionary containing the selected phase demos and the start and end timestep, and further metadata   
        selected_phase_demo_dict = {}
        episode_num_frames = 0
        next_phase_start_idx_counter = 0
        for phase, selected_phase_demo in zip(selected_phases, selected_phase_demos):
            # Add the selected phase demo to the selected_phase_demo_dict
            selected_phase_demo_dict[phase] = {}
            selected_phase_demo_dict[phase]["phase_folder_name"] = phase
            selected_phase_demo_dict[phase]["demo_folder_name"] = selected_phase_demo
            selected_phase_demo_dict[phase]["phases_demo_start_idx"] = next_phase_start_idx_counter
            selected_phase_demo_dict[phase]["command"], selected_phase_demo_dict[phase]["embedding"] = self.command_embeddings_dict[phase]
            
            # Load the start and end indices for the current demo as the valid range of the demo
            selected_demo_folder_path = os.path.join(self.dataset_dir, selected_tissue_sample, phase, selected_phase_demo)
            start, _, demo_num_frames_valid = get_valid_demo_start_end_indices(selected_demo_folder_path, self.camera_names, self.before_phase_offset, self.after_phase_offset)
            selected_phase_demo_dict[phase]["demo_rel_start_idx"] = start
            
            # Count the number of valid frames for the current demo
            episode_num_frames += demo_num_frames_valid
            next_phase_start_idx_counter += demo_num_frames_valid
            selected_phase_demo_dict[phase]["phases_demo_end_idx"] = next_phase_start_idx_counter - 1 # -1 because the phases_demo_end_idx is inclusive

        return selected_phase_demo_dict

    def get_jaw_psm2_psm1_data_sequence(self, selected_tissue_sample, selected_phase_demo_dict, start_ts, curr_ts):
        # Returns the jaw data sequence for psm2 and psm1 for time steps from start_ts to curr_ts
        
        # Get a mapping which value needs to be loaded from which csv file
        ts_kinematics_file_dame_frame_idx_dict = defaultdict(list)
        for ts in range(start_ts, curr_ts + 1, self.history_step_size):
            ts_corrected = max(0, ts) # If the episode is not long enough replicate the last frame/yaw value for the first recorded demo
            ts_phase_folder, ts_demo_folder, ts_demo_frame_idx = self.get_current_phase_demo_folder_and_demo_frame_idx(selected_phase_demo_dict, ts_corrected)
            kinematics_file_path = os.path.join(self.dataset_dir, selected_tissue_sample, ts_phase_folder, ts_demo_folder, "ee_csv.csv")
            ts_kinematics_file_dame_frame_idx_dict[kinematics_file_path].append(ts_demo_frame_idx)
        
        # Load the jaw data for the desired timesteps (from the corresponding csv files)
        jaw_psm2_data_sequence_list, jaw_psm1_data_sequence_list = [], []
        for kinematics_file_path, frame_indices in ts_kinematics_file_dame_frame_idx_dict.items():
            kinematics_data = pd.read_csv(kinematics_file_path)
            jaw_psm2_data_sequence_list.append(torch.tensor(kinematics_data.loc[frame_indices, "psm2_jaw"].values))
            jaw_psm1_data_sequence_list.append(torch.tensor(kinematics_data.loc[frame_indices, "psm1_jaw"].values))
        
        jaw_psm2_data_sequence = torch.concatenate(jaw_psm2_data_sequence_list)
        jaw_psm1_data_sequence = torch.concatenate(jaw_psm1_data_sequence_list)
        jaw_psm2_psm1_data_sequence = torch.stack((jaw_psm2_data_sequence, jaw_psm1_data_sequence), dim=1).to(dtype=torch.float32)
        
        return jaw_psm2_psm1_data_sequence 
            
    def get_phase_history(self, selected_phase_demo_dict, sorted_phases, curr_ts):
        # Returns the phase history for the last six performed phases or phase predictions (with padding if needed)
        
        # Compute the last prediction timestep (until which the history will be considered)
        last_pred_ts = curr_ts - self.prediction_step_size
        
        # Get the last n performed phases from the last prediction timestep on
        if self.phase_history_only_phase_switches_flag:
            # Get the last n performed phases from the current phase on
            last_pred_ts_phase = self.get_embedding_command_phase_for_ts(selected_phase_demo_dict, last_pred_ts)[2]
            last_pred_ts_phase_idx = sorted_phases.index(last_pred_ts_phase)
            phase_history_phase_folder_names = sorted_phases[max(0, last_pred_ts_phase_idx - self.phase_history_len + 1):last_pred_ts_phase_idx+1]
            phase_history = [self.command_embeddings_dict[phase_command][0] for phase_command in phase_history_phase_folder_names] # Transform to the commands corresponding to the phase folder names
            # Remove replicant merged phases
            if self.reduced_base_class_set_flag:
                phase_history = [phase for idx, phase in enumerate(phase_history) if idx == 0 or phase != phase_history[idx-1]]
            
            # Add padding (if needed)
            if len(phase_history) < self.phase_history_len:
                phase_history = ["padding"] * (self.phase_history_len - len(phase_history)) + phase_history
        else:            
            # Check first if we need to add further sample number information of precessor phases to phase demo dict (if not everything covered)
            first_phase_partially_stitched_episode = list(selected_phase_demo_dict.keys())[0].replace("_recovery", "")
            min_episode_idx = 0 # Note: Will be < 0 for the new precessor phases
            remaining_precessor_phases_reversed = sorted_phases[:sorted_phases.index(first_phase_partially_stitched_episode)][::-1]
            selected_phase_demo_dict_with_pre_phases = selected_phase_demo_dict.copy()
            for precessor_phase in remaining_precessor_phases_reversed:
                num_samples_til_pred_ts = last_pred_ts - min_episode_idx
                if num_samples_til_pred_ts >= self.prediction_step_size*self.phase_history_len:
                    break
                # Sample the length of the precessor phase
                precessor_phase_len_min = self.ds_statistics_dict[precessor_phase]["min"]
                precessor_phase_len_max = self.ds_statistics_dict[precessor_phase]["max"]
                precessor_phase_len_mean = self.ds_statistics_dict[precessor_phase]["mean"]
                precessor_phase_len_std = self.ds_statistics_dict[precessor_phase]["std"]
                precessor_phase_len = randintgaussian(precessor_phase_len_min, precessor_phase_len_max, mean=precessor_phase_len_mean, std_dev=precessor_phase_len_std) 
                min_episode_idx -= precessor_phase_len
                
                # Add the precessor phase to the phase demo dict
                selected_phase_demo_dict_with_pre_phases[precessor_phase] = {}
                selected_phase_demo_dict_with_pre_phases[precessor_phase]["phase_folder_name"] = precessor_phase
                selected_phase_demo_dict_with_pre_phases[precessor_phase]["phases_demo_start_idx"] = min_episode_idx
                selected_phase_demo_dict_with_pre_phases[precessor_phase]["phases_demo_end_idx"] = min_episode_idx + precessor_phase_len - 1
                selected_phase_demo_dict_with_pre_phases[precessor_phase]["command"], selected_phase_demo_dict_with_pre_phases[precessor_phase]["embedding"] = self.command_embeddings_dict[precessor_phase]
            
            # Get the last n performed phases from the current phase on
            phase_history = []
            pred_ts_limit = last_pred_ts - self.prediction_step_size*self.phase_history_len
            for pred_ts in range(last_pred_ts, pred_ts_limit, -self.prediction_step_size):
                pred_ts_command = self.get_embedding_command_phase_for_ts(selected_phase_demo_dict_with_pre_phases, pred_ts)[1]
                if pred_ts_command:
                    phase_history.append(pred_ts_command)
                else: 
                    phase_history.append("padding")
            # Reverse the phase history order at the end (as newest predictions should be last)
            phase_history = phase_history[::-1]
                            
        return phase_history


    def __getitem__(self, index):
        # Put together the stitched episode based on randomly getting a tissue id and a random index for a demo for each phase. Then sample a random timestep and get the corresponding image sequence and command embedding
       
        # Select a random tissue sample to generate the episode from
        selected_tissue_sample = np.random.choice(list(self.tissue_phase_demo_dict.keys()))
        
        # Go through the phases in fixed order of execution (ignore recovery phases here - included in next paragraph)
        phases = list(self.tissue_phase_demo_dict[selected_tissue_sample].keys())
        phases_wo_recovery = [phase for phase in phases if "recovery" not in phase]
        sorted_phases = sorted(phases_wo_recovery, key=lambda x: int(x.split('_')[0]))
        
        # Choose first phase as either the recovery or normal phase and the second phase as the next phase
        phase_1_wo_suffix = np.random.choice(sorted_phases[:-1]) # Exclude the last phase
        recovery_phase_flag = np.random.rand() <= self.recovery_probability
        recovery_available_flag = phase_1_wo_suffix + "_recovery" in phases
        phase_1_suffix = "_recovery" if recovery_phase_flag and recovery_available_flag else ""
        phase_1 = phase_1_wo_suffix + phase_1_suffix
        
        phase_1_idx = sorted_phases.index(phase_1_wo_suffix)
        phase_2 = sorted_phases[phase_1_idx + 1]
        
        # Sample the demos for the first and second phase
        demo_phase_1 = np.random.choice(self.tissue_phase_demo_dict[selected_tissue_sample][phase_1])
        demo_phase_2 = np.random.choice(self.tissue_phase_demo_dict[selected_tissue_sample][phase_2])
        
        # Store the selected phase demo and the start and end timestep
        phase_1_demo_folder_path = os.path.join(self.dataset_dir, selected_tissue_sample, phase_1, demo_phase_1)
        len_phase_1 = get_valid_demo_start_end_indices(phase_1_demo_folder_path, self.camera_names, self.before_phase_offset, self.after_phase_offset)[2]
        total_len_wo_p2 = len_phase_1 # Just use phase 1 as possible input (or potentially precessor phases)
        precessor_phases, precessor_phase_demos = deque(), deque() 
        required_num_samples = self.history_len * self.history_step_size + self.prediction_offset
        if total_len_wo_p2 < required_num_samples: # Check that the stitched partial episode is long enough (for all input data)
            if self.verbose:
                print(f"Total length {len_phase_1} of phase 1 ({phase_1}) is too short.")
                print(f"Adding precessor phases in front of phase 1 (and if not already the case, switch for phase 1 to normal instead of recovery, for clean cuts between precessor phases and phase 1).")
            phase_1 = phase_1.replace("_recovery", "")
            demo_phase_1 = np.random.choice(self.tissue_phase_demo_dict[selected_tissue_sample][phase_1])
            phase_1_demo_folder_path = os.path.join(self.dataset_dir, selected_tissue_sample, phase_1, demo_phase_1)
            total_len_wo_p2 = len_phase_1 = get_valid_demo_start_end_indices(phase_1_demo_folder_path, self.camera_names, self.before_phase_offset, self.after_phase_offset)[2]
        
            for phase_idx in range(phase_1_idx - 1, -1, -1): 
                if total_len_wo_p2 >= required_num_samples:
                    break
                precessor_phase = sorted_phases[phase_idx]
                precessor_phase_demo = np.random.choice(self.tissue_phase_demo_dict[selected_tissue_sample][precessor_phase])
                precessor_phase_folder_path = os.path.join(self.dataset_dir, selected_tissue_sample, precessor_phase, precessor_phase_demo)
                precessor_phase_demo_len = get_valid_demo_start_end_indices(precessor_phase_folder_path, self.camera_names, self.before_phase_offset, self.after_phase_offset)[2]
                total_len_wo_p2 += precessor_phase_demo_len
                precessor_phases.appendleft(precessor_phase)
                precessor_phase_demos.appendleft(precessor_phase_demo)
        precessor_phases, precessor_phase_demos = list(precessor_phases), list(precessor_phase_demos)
        
        # Create a dictionary to store the selected phase demos and the start and end timestep, and further metadata
        selected_phases = precessor_phases + [phase_1, phase_2]
        selected_phase_demos = precessor_phase_demos + [demo_phase_1, demo_phase_2]
        selected_phase_demo_dict = self.create_phase_demo_metadata_dict(selected_tissue_sample, selected_phases, selected_phase_demos)
        
        if self.verbose:
            print(f"Selected tissue sample: {selected_tissue_sample}")
            print(f"Selected phases: {phase_1}, {phase_2}")
            print(f"Selected demos: {demo_phase_1}, {demo_phase_2}")
        
        # Sample a random curr_ts and compute the start_ts and target_ts from concatinated phase 1+2 (via Gaussian distribution)
        if selected_phase_demo_dict[phase_1]["phases_demo_end_idx"] >= self.history_len * self.history_step_size:
            min_phase_1_phase_2_idx = self.history_len * self.history_step_size # If enough frames from previous demos 
        else: 
            min_phase_1_phase_2_idx = selected_phase_demo_dict[phase_1]["phases_demo_end_idx"] # If frames need to be padded (and therefore the num actual frames would be greater than phase 1)
        max_phase_1_phase_2_idx = selected_phase_demo_dict[phase_2]["phases_demo_end_idx"] - self.prediction_offset
        phase_1_phase_2_sampling_mean = selected_phase_demo_dict[phase_2]["phases_demo_start_idx"]
        phase_1_phase_2_sampling_std = (max_phase_1_phase_2_idx - min_phase_1_phase_2_idx) / (2*3)
        curr_ts = randintgaussian(min_phase_1_phase_2_idx, max_phase_1_phase_2_idx, mean=phase_1_phase_2_sampling_mean, std_dev=phase_1_phase_2_sampling_std)
        start_ts = curr_ts - self.history_len * self.history_step_size
        target_ts = curr_ts + self.prediction_offset
        
        # -----
        
        # Retrieve the language embedding for the target_ts
        command_embedding, command_gt, _ = self.get_embedding_command_phase_for_ts(selected_phase_demo_dict, target_ts)        
        if command_embedding is None:
            raise ValueError(f"Could not find embedding for target_ts {target_ts}.")
        
        if self.use_jaw_values_flag:
            # Read out the jaw values from start to end (when a certain flag is set)
            jaw_psm2_psm1_data_sequence = self.get_jaw_psm2_psm1_data_sequence(selected_tissue_sample, selected_phase_demo_dict, start_ts, curr_ts)
        
        # History information of the last six phases (with padding if needed)
        if self.use_history_flag:
            phase_history = self.get_phase_history(selected_phase_demo_dict, sorted_phases, curr_ts)
        else:
            self.phase_history_len = 1 # Required for eval of phase transitions
            phase_history = self.get_phase_history(selected_phase_demo_dict, sorted_phases, curr_ts)
        
        # Construct the image sequences for the desired timesteps
        image_sequence = []
        for ts in range(start_ts, curr_ts + 1, self.history_step_size):
            ts_corrected = max(0, ts) # If the episode is not long enough replicate the last frame/yaw value for the first recorded demo
            image_dict = {}
            ts_phase_folder, ts_demo_folder, ts_demo_frame_idx = self.get_current_phase_demo_folder_and_demo_frame_idx(selected_phase_demo_dict, ts_corrected)
            for cam_name, cam_file_suffix in zip(self.camera_names, self.camera_file_suffixes):
                cam_folder = os.path.join(self.dataset_dir, selected_tissue_sample, ts_phase_folder, ts_demo_folder, cam_name)
                frame_path = os.path.join(cam_folder, f"frame{str(ts_demo_frame_idx).zfill(6)}{cam_file_suffix}")
                img = torch.tensor(cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
                # Resize the image to 224x224 
                if self.center_crop_flag:
                    img_resized_224 = center_crop_resize(img, 224)
                else:
                    img_resized_224 = transforms.Resize((224, 224))(img)
                
                image_dict[cam_name] = img_resized_224
                
            all_cam_images = [
                image_dict[cam_name] for cam_name in self.camera_names
            ]
            all_cam_images = torch.stack(all_cam_images, dim=0)
            image_sequence.append(all_cam_images)

        # Apply the same transform for all camera images 
        image_sequence = torch.stack(image_sequence, dim=0) # Shape: ts, cam, c, h, w
        if self.split_name == "train" and self.input_transforms is not None:
            image_sequence = image_sequence.reshape(-1, image_sequence.size(2), image_sequence.size(3), image_sequence.size(4)) # Reshape to (ts*cam, c, h, w) for applying the same transform to all camera images
            image_sequence = self.input_transforms(image_sequence) 
            image_sequence = image_sequence.reshape(-1, len(self.camera_names), image_sequence.size(1), image_sequence.size(2), image_sequence.size(3)) # Reshape back to (ts, cam, c, h, w)
        image_sequence = image_sequence / 255.0 

        if self.use_jaw_values_flag:
            return image_sequence, command_embedding, command_gt, jaw_psm2_psm1_data_sequence, phase_history 
        else:
            return image_sequence, command_embedding, command_gt, phase_history 
    

def load_merged_data(
    dataset_dirs,
    num_episodes_list,
    camera_names,
    camera_file_suffixes,
    batch_size_train=32,
    batch_size_val=32,
    history_len=1,
    prediction_offset=10,
    history_step_size=1,
    test_only=False,
    input_transforms=None,
    dagger_ratio=None,
    center_crop_flag=False,
    reduced_base_class_set_flag=False,
    use_phase_history_flag=False,
    use_jaw_values_flag=False,
    phase_history_len=6,
    prediction_step_size=30,
    recovery_probability = 0.25,
    phase_history_only_phase_switches_flag = False
):
    
    print(f"{history_len=}, {history_step_size=}, {prediction_offset=}")

    if dagger_ratio is not None:
        assert 0 <= dagger_ratio <= 1, "dagger_ratio must be between 0 and 1."
    
    ds_metadata_dict = {}
    if dagger_ratio is None:
        # Obtain train/val/test split
        train_ratio = 0.9
        val_ratio = 0.1
        test_ratio = 1 - train_ratio - val_ratio
    else:
        train_ratio = 1
        val_ratio = test_ratio = 0

    # Save the metadata
    ds_metadata_dict["train_ratio"] = train_ratio
    ds_metadata_dict["val_ratio"] = val_ratio
    ds_metadata_dict["test_ratio"] = test_ratio
    ds_metadata_dict["train_tissues"] = {}
    ds_metadata_dict["val_tissues"] = {}
    ds_metadata_dict["test_tissues"] = {}
    ds_metadata_dict["train_ds_statistics"] = {}
    ds_metadata_dict["val_ds_statistics"] = {}
    ds_metadata_dict["test_ds_statistics"] = {}
    ds_metadata_dict["dagger_ratio"] = dagger_ratio
    ds_metadata_dict["history_len"] = history_len
    ds_metadata_dict["history_step_size"] = history_step_size
    ds_metadata_dict["prediction_offset"] = prediction_offset
    ds_metadata_dict["camera_names"] = camera_names
    ds_metadata_dict["test_only"] = test_only
    ds_metadata_dict["input_transforms"] = input_transforms
    ds_metadata_dict["dataset_dirs"] = dataset_dirs
    ds_metadata_dict["num_episodes_list"] = num_episodes_list    
    ds_metadata_dict["center_crop_flag"] = center_crop_flag
    ds_metadata_dict["reduced_base_class_set_flag"] = reduced_base_class_set_flag
    ds_metadata_dict["use_history_flag"] = use_phase_history_flag
    ds_metadata_dict["use_jaw_values_flag"] = use_jaw_values_flag
    ds_metadata_dict["phase_history_len"] = phase_history_len
    ds_metadata_dict["prediction_step_size"] = prediction_step_size
    ds_metadata_dict["recovery_probability"] = recovery_probability
    ds_metadata_dict["phase_history_only_phase_switches_flag"] = phase_history_only_phase_switches_flag

    # Construct the datasets and the dataset embeddings
    train_datasets, val_datasets, test_datasets = [], [], []
    command_embeddings_dict = {}
    val_command_embeddings_dict_add_datasets = {}
    class_occ_cnt_dict = defaultdict(lambda: 0)
    for dataset_dir, num_episodes in zip(dataset_dirs, num_episodes_list):
        # Load dataset dir and count number of tissue samples
        dataset_file_names = os.listdir(dataset_dir)
        dataset_name = os.path.basename(dataset_dir)
        dataset_config = DATASET_CONFIGS[dataset_name]
        incomplete_tissue_samples = dataset_config["incomplete_tissue_samples"] if "incomplete_tissue_samples" in dataset_config else []
        tissue_names = [tissue_name for tissue_name in dataset_file_names if tissue_name.startswith(("tissue", "phantom")) and tissue_name not in incomplete_tissue_samples]
        
        if dagger_ratio is None:
            # Split the tissue samples into train, val, test by randomly sampling until the ratios are fulfilled
            train_tissues, val_tissues, test_tissues = split_tissue_samples(
                dataset_dir, tissue_names, train_ratio, val_ratio, test_only
            )
            print(f"\nDataset: {dataset_dir}")
            print(f"Train tissues: {train_tissues}")
            print(f"Val tissues: {val_tissues}")
            print(f"Test tissues: {test_tissues}")
            
            ds_metadata_dict["train_tissues"][dataset_dir] = train_tissues
            ds_metadata_dict["val_tissues"][dataset_dir] = val_tissues
            ds_metadata_dict["test_tissues"][dataset_dir] = test_tissues
        else:
            train_tissues = tissue_names
            print(f"\nDataset: {dataset_dir}")
            print(f"Train tissues: {train_tissues}")
            ds_metadata_dict["train_tissues"][dataset_dir] = train_tissues 
        
        # ---------------------- Construct datasets -----------------------
        
        if dagger_ratio is not None and not test_only:
            raise NotImplementedError("DAgger not yet implemented.")
            train_datasets.append(SequenceDataset(
                        "train",
                        tissue_names,
                        dataset_dir,
                        camera_names,
                        camera_file_suffixes,
                        history_len,
                        prediction_offset,
                        history_step_size,
                        num_episodes,
                        input_transforms,
                        center_crop_flag,
                        reduced_base_class_set_flag,
                        use_phase_history_flag,
                        use_jaw_values_flag,
                        phase_history_len,
                        prediction_step_size,
                        recovery_probability,
                        phase_history_only_phase_switches_flag)
            )
            
            # Get dataset statistics
            train_ds_statistics_dict = train_datasets[-1].ds_statistics_dict
            ds_metadata_dict["train_ds_statistics"][dataset_dir] = train_ds_statistics_dict
            
            # Get the command embeddings for the train datasets and update the command embeddings dictionary
            train_command_embeddings_dict = train_datasets[-1].command_embeddings_dict
            command_embeddings_dict.update(train_command_embeddings_dict)
            
        elif not test_only:
            # Construct dataset and dataloader for each dataset dir and merge them
            train_datasets.append(SequenceDataset(
                        "train",
                        [tissue_name for tissue_name in train_tissues],
                        dataset_dir,
                        camera_names,
                        camera_file_suffixes,
                        history_len,
                        prediction_offset,
                        history_step_size,
                        num_episodes,
                        input_transforms,
                        center_crop_flag,
                        reduced_base_class_set_flag,
                        use_phase_history_flag,
                        use_jaw_values_flag,
                        phase_history_len,
                        prediction_step_size,
                        recovery_probability,
                        phase_history_only_phase_switches_flag)
            )
            val_datasets.append(SequenceDataset(
                        "val",
                        [tissue_name for tissue_name in val_tissues],
                        dataset_dir,
                        camera_names,
                        camera_file_suffixes,
                        history_len,
                        prediction_offset,
                        history_step_size,
                        num_episodes,
                        input_transforms,
                        center_crop_flag,
                        reduced_base_class_set_flag,
                        use_phase_history_flag,
                        use_jaw_values_flag,
                        phase_history_len,
                        prediction_step_size,
                        recovery_probability,
                        phase_history_only_phase_switches_flag)
            )
            
            # Get dataset statistics
            train_ds_statistics_dict = train_datasets[-1].ds_statistics_dict
            val_ds_statistics_dict = val_datasets[-1].ds_statistics_dict
            ds_metadata_dict["train_ds_statistics"][dataset_dir] = train_ds_statistics_dict
            ds_metadata_dict["val_ds_statistics"][dataset_dir] = val_ds_statistics_dict
            
            # Get the command embeddings for the train and val datasets
            train_command_embeddings_dict = train_datasets[-1].command_embeddings_dict
            val_command_embeddings_dict = val_datasets[-1].command_embeddings_dict
            
            # Update the command embeddings dictionary
            command_embeddings_dict.update(train_command_embeddings_dict)
            val_command_embeddings_dict_add_datasets.update(val_command_embeddings_dict)
            
            # Add the class occurence ratio to the class_occ_ratio_dict
            for command, _ in train_datasets[-1].command_embeddings_dict.values():
                class_occ_cnt_dict[command] += 1
                class_occ_cnt_dict["in_total"] += 1

        else: 
            test_datasets.append(SequenceDataset(
                        "test",
                        [tissue_name for tissue_name in test_tissues],
                        dataset_dir,
                        camera_names,
                        camera_file_suffixes,
                        history_len,
                        prediction_offset,
                        history_step_size,
                        num_episodes,
                        input_transforms,
                        center_crop_flag,
                        reduced_base_class_set_flag,
                        use_phase_history_flag,
                        use_jaw_values_flag,
                        phase_history_len,
                        prediction_step_size,
                        recovery_probability,
                        phase_history_only_phase_switches_flag)
            )
            
            # Get dataset statistics
            test_ds_statistics_dict = test_datasets[-1].ds_statistics_dict
            ds_metadata_dict["test_ds_statistics"][dataset_dir] = test_ds_statistics_dict
            
            # Get the command embeddings for the test datasets (should be the same as for train and val datasets)
            test_command_embeddings_dict = test_datasets[-1].command_embeddings_dict
            command_embeddings_dict.update(test_command_embeddings_dict)

    # ----------------------------- Construct the dataloaders -------------------------------
    
    if dagger_ratio is not None and not test_only:
        # Merge all datasets (e.g., base dataset + fine tuning (correction) datasets) into one big dataset
        merged_train_dataset = ConcatDataset(train_datasets)

        # TODO: Adjust DAgger code in utils.py depending on how we save the corrections data later (corrections data -> last_dataset_tissue_indices) 
            
        # dataset_sizes = {
        #     dataset_dir: num_episodes
        #     for dataset_dir, num_episodes in zip(dataset_dirs, num_episodes_list)
        # }

        # dagger_sampler = DAggerSampler(
        #     all_episode_indices,
        #     last_dataset_indices,
        #     batch_size_train,
        #     dagger_ratio,
        #     dataset_sizes,
        # )
        
        # train_dataloader = DataLoader(
        #     merged_train_dataset,
        #     batch_size=batch_size_train,
        #     shuffle=True,
        #     pin_memory=True,
        #     num_workers=8,
        #     prefetch_factor=16,
        #     persistent_workers=True,
        # )
        
        # Extract the candidate embeddings and commands
        candidate_embeddings, candidate_texts = extract_candidate_embeddings_and_commands(command_embeddings_dict)
        phase_to_instruction_mapping = extract_phase_idx_to_instruction_mapping(command_embeddings_dict)
        ds_metadata_dict["phase_to_instruction_mapping"] = phase_to_instruction_mapping
        ds_metadata_dict["candidate_texts"] = candidate_texts
        ds_metadata_dict["candidate_embeddings"] = candidate_embeddings
        return train_dataloader, ds_metadata_dict
    
    elif not test_only:
        # Check for if all val commands are in the training commands
        train_commands = set([command for command, _ in command_embeddings_dict.values()])
        val_commands = set([command for command, _ in val_command_embeddings_dict_add_datasets.values()])
        if not val_commands.issubset(train_commands):
            raise ValueError("Val commands are not subset of train commands.")
        
        
        # Merge all datasets (e.g., base dataset + fine tuning (correction) datasets) into one big dataset
        merged_train_dataset = ConcatDataset(train_datasets)
        merged_val_dataset = ConcatDataset(val_datasets)
        
        train_dataloader = DataLoader(
            merged_train_dataset,
            batch_size=batch_size_train,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
            prefetch_factor=16,
            persistent_workers=True,
        )
        val_dataloader = DataLoader(
            merged_val_dataset,
            batch_size=batch_size_val,
            shuffle=False,
            pin_memory=True,
            num_workers=8,
            prefetch_factor=16,
            persistent_workers=True,
        )
        
        # Extract the candidate embeddings and commands
        candidate_embeddings, candidate_texts = extract_candidate_embeddings_and_commands(command_embeddings_dict)
        phase_to_instruction_mapping = extract_phase_idx_to_instruction_mapping(command_embeddings_dict)
        ds_metadata_dict["phase_to_instruction_mapping"] = phase_to_instruction_mapping
        ds_metadata_dict["candidate_texts"] = candidate_texts
        ds_metadata_dict["candidate_embeddings"] = candidate_embeddings
    
        # Add class weights to the metadata (balanced class weights)
        total_samples = class_occ_cnt_dict["in_total"]
        del class_occ_cnt_dict["in_total"]
        num_classes = len(class_occ_cnt_dict)
        class_weights = {cls: total_samples / (num_classes * cnt) for cls, cnt in class_occ_cnt_dict.items()}
        class_weights_tensor = torch.tensor([class_weights[cls] for cls in candidate_texts], dtype=torch.float) # sort the class weights according to the candidate_texts (order for the model labels)
        ds_metadata_dict["class_weights"] = class_weights_tensor
        
        return train_dataloader, val_dataloader, ds_metadata_dict
    
    else:
        # Merge all datasets (e.g., base dataset + fine tuning (correction) datasets) into one big dataset
        merged_test_dataset = ConcatDataset(test_datasets) 

        test_dataloader = DataLoader(
            merged_test_dataset,
            batch_size=batch_size_val,
            shuffle=False,
            pin_memory=True,
            num_workers=16,
            prefetch_factor=1,
        )

        # Extract the candidate embeddings and commands
        candidate_embeddings, candidate_texts = extract_candidate_embeddings_and_commands(command_embeddings_dict)
        phase_to_instruction_mapping = extract_phase_idx_to_instruction_mapping(command_embeddings_dict)
        ds_metadata_dict["phase_to_instruction_mapping"] = phase_to_instruction_mapping
        ds_metadata_dict["candidate_texts"] = candidate_texts
        ds_metadata_dict["candidate_embeddings"] = candidate_embeddings

        return test_dataloader, ds_metadata_dict


"""
Test the SequenceDataset class.
"""
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from instructor.utils import set_seed

    # Parameters for the test
    dataset_name = "base_chole_clipping_cutting" # "base_chole_clipping_cutting" "phantom_chole" "debugging"
    dataset_dir = os.path.join(os.getenv("PATH_TO_DATASET"), dataset_name) 
    tissue_samples_ids = ["tissue_18"] # "phantom_1" "tissue_12"
    camera_names = ["endo_psm2", "left_img_dir", "right_img_dir", "endo_psm1"]
    camera_file_suffixes = ["_psm2.jpg", "_left.jpg", "_right.jpg", "_psm1.jpg"]
    history_len = 3
    prediction_offset = 12 # Get command for the current timestep
    history_step_size = 15
    num_episodes = 200 # Number of randlomy generated stitched episodes
    reduced_base_class_set_flag = True
    phase_history_len = 100
    prediction_step_size = 30
    recovery_probability = 0.4
    phase_history_only_phase_switches_flag = False
    verbose_flag = True

    # Define transforms/augmentations (resize transformation already applied in __getitem__ method)
    input_transforms = []
    
    # Note: Automatic augmentations
    # input_transforms.append(transforms.RandAugment())
    # input_transforms.append(transforms.TrivialAugmentWide())
    # input_transforms.append(transforms.AugMix())
    
    # Note: Manual augmetnations
    # input_transforms.append(transforms.RandomRotation(15))
    # input_transforms.append(transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)))
    # input_transforms.append(transforms.RandomResizedCrop(224, scale=(0.8, 1.0)))
    
    # input_transforms.append(v2.RandomPerspective(p=0.5))
    # input_transforms.append(v2.RandomPosterize(bits=7, p=0.25))
    # input_transforms.append(v2.RandomAdjustSharpness(2, p=0.25))
    # input_transforms.append(transforms.RandomApply([v2.GaussianBlur(kernel_size=5)], p=0.75))
    # input_transforms.append(v2.RandomPhotometricDistort(p=0.8))
    # input_transforms.append(transforms.RandomGrayscale(p=0.2))
    input_transforms = transforms.Compose(input_transforms)

    # Create a SequenceDataset instance
    dataset = SequenceDataset(
        "train",
        tissue_samples_ids,
        dataset_dir,
        camera_names,
        camera_file_suffixes,
        history_len,
        prediction_offset,
        history_step_size,
        num_episodes,
        input_transforms,
        center_crop_flag=False,
        reduced_base_class_set_flag=reduced_base_class_set_flag,
        use_phase_history_flag=True,
        use_jaw_values_flag=True,
        phase_history_len=phase_history_len,
        prediction_step_size=prediction_step_size,
        recovery_probability=recovery_probability,
        phase_history_only_phase_switches_flag=phase_history_only_phase_switches_flag,
        verbose=verbose_flag
    )

    # Sample a random item from the dataset
    rdm_idx = np.random.randint(0, len(dataset))
    image_sequence, command_embedding, command, jaw_values, phase_history = dataset[rdm_idx]

    print(f"Image sequence shape: {image_sequence.shape}")
    print(f"Language embedding shape: {command_embedding.shape}")
    print(f"Command: {command}")
    print(f"Phase history ({phase_history_len=}): {phase_history}")
    print(f"Jaw values ({history_len=}):\n{jaw_values}")

    # Create a figure with subplots: one row per timestamp, one column per camera
    fig, axes = plt.subplots(history_len + 1, len(camera_names), figsize=(15, 10))
    if history_len == 0:
        axes = axes[np.newaxis, :]

    # Loop over each timestamp and camera to plot the images
    for t in range(history_len + 1):
        for cam_idx, cam_name in enumerate(camera_names):
            ax = axes[t, cam_idx]  # Get the specific subplot axis
            img = image_sequence[t, cam_idx].permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.set_title(f"{cam_name} at timestep {t}")
            ax.axis('off')  # Optionally turn off the axis

    # Set title to command
    fig.suptitle(f"Command: {command} - Jaw values: {jaw_values}", fontsize=16)
    plt.tight_layout()
    example_dataset_plots_folder_path = os.path.join(path_to_yay_robot, "examples_plots", "dataset")
    if not os.path.exists(example_dataset_plots_folder_path):
        os.makedirs(example_dataset_plots_folder_path)
    file_name = os.path.join(example_dataset_plots_folder_path, f"dataset_img_{history_len=}_{history_step_size=}.png")
    file_path = os.path.join(example_dataset_plots_folder_path, file_name)
    plt.savefig(file_path)
    print(f"Saved {file_name}.")
    plt.close(fig)  # Close the figure to free memory
