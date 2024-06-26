import numpy as np
import torch
import os
# import h5py
import cv2
import json
import sys
sys.path.append("$PATH_TO_YAY_ROBOT/src")  # to import aloha 
from torch.utils.data import DataLoader, ConcatDataset

from aloha_pro.aloha_scripts.utils import crop_resize, random_crop, initialize_model_and_tokenizer, encode_text
# from act.utils import DAggerSampler
    
def generate_command_embeddings(tissue_phase_demo_dict, encoder, tokenizer, model):
    # Returns a dictionary containing the phase command as key and a tuple of the phase command and phase embedding as value
    phase_command_embeddings_dict = {}
    for phase_folder_name in tissue_phase_demo_dict.keys():
        # Extract the phase command from the folder name (removing the phase idx and the "_" in between the words)
        _, phase_command = phase_folder_name.split("_")[0], " ".join(phase_folder_name.split("_")[1:])
        embedding = encode_text(phase_command, encoder, tokenizer, model)
        phase_command_embeddings_dict[phase_folder_name]= (phase_command, embedding)

    return phase_command_embeddings_dict

def split_tissue_samples(dataset_dir, num_tissue_samples, train_ratio, val_ratio, test_ratio):
    # Calculate the number of samples for each set
    num_train = int(train_ratio * num_tissue_samples)
    num_val = int(val_ratio * num_tissue_samples)
    num_test = num_tissue_samples - num_train - num_val

    # Generate a list of indices and shuffle them
    all_indices = list(range(num_tissue_samples))
    np.random.shuffle(all_indices)

    # Split the indices based on the calculated numbers
    train_indices = [(dataset_dir, idx) for idx in all_indices[:num_train]]
    val_indices = [(dataset_dir, idx) for idx in all_indices[num_train:num_train + num_val]]
    test_indices = [(dataset_dir, idx) for idx in all_indices[num_train + num_val:]]

    return train_indices, val_indices, test_indices

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        episode_ids,
        dataset_dir,
        camera_names,
        camera_file_suffixes,
        history_len=5,
        prediction_offset=10,
        history_skip_frame=1,
        num_episodes=200,
        random_crop=False, # TODO: Add here more augmentations options
    ):
        super().__init__()
        self.episode_ids = episode_ids if len(episode_ids) > 0 else [0]
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.camera_file_suffixes = camera_file_suffixes
        self.history_len = history_len
        self.prediction_offset = prediction_offset
        self.history_skip_frame = history_skip_frame
        self.num_episodes = num_episodes
        self.random_crop = random_crop # TODO: Add here the other augmentations
        
        # Load the tissue samples and their phases and demos (for later stitching of the episodes)
        tissue_samples = os.listdir(dataset_dir)
        self.tissue_phase_demo_dict = {}
        for tissue_sample in tissue_samples:
            phases = os.listdir(os.path.join(dataset_dir, tissue_sample))
            self.tissue_phase_demo_dict[tissue_sample] = {}
            for phase_sample in phases:
                demo_samples = os.listdir(os.path.join(dataset_dir, tissue_sample, phase_sample))
                self.tissue_phase_demo_dict[tissue_sample][phase_sample] = demo_samples
                
        # Generate the embeddings for all phase commands
        encoder_name = "distilbert"
        tokenizer, model = initialize_model_and_tokenizer(encoder_name)
        self.command_embeddings_dict = generate_command_embeddings(self.tissue_phase_demo_dict, encoder_name, tokenizer, model)
        del tokenizer, model
        
    def __len__(self):
        # Here this means the number of randomly generated stitched episodes
        return len(self.num_episodes)

    def get_command_for_ts(self, selected_phase_demo_dict, target_ts):
        # Returns the command embedding and the command for the target timestep
        for phase_segment in selected_phase_demo_dict:
            if phase_segment["start_timestep"] <= target_ts <= phase_segment["end_timestep"]:
                return torch.tensor(phase_segment["embedding"]).squeeze(), phase_segment["command"] # TODO: Check if the squeeze is necessary
        else:
            raise ValueError(f"Could not find command for target_ts {target_ts}.")

    def get_current_phase_demo_folder_and_demo_frame_idx(self, selected_phase_demo_dict, target_ts):
        # Returns the phase and the demo frame index for the target timestep
        for phase_segment in selected_phase_demo_dict:
            if phase_segment["start_timestep"] <= target_ts <= phase_segment["end_timestep"]:
                return phase_segment["phase_folder_name"], phase_segment["demo_folder_name"], target_ts - phase_segment["start_timestep"]
        else:
            raise ValueError(f"Could not find phase and demo frame index for target_ts {target_ts}.")

    def __getitem__(self, index):
        # Put together the stitched episode based on randomly getting a tissue id and a random index for a demo for each phase. Then sample a random timestep and get the corresponding image sequence and command embedding
       # TODO: Downsize all to desired size (e.g. 224x224) - already done within the model - but rather put here?
       
        selected_tissue_sample = np.random.choice(list(self.tissue_phase_demo_dict.keys()))
        selected_phase_demo_dict = {}
        episode_num_frames = curr_phase_idx_counter = 0
        for phase in self.tissue_phase_demo_dict[selected_tissue_sample].keys():
            # Select a random demo for each phase
            selected_phase_demo = np.random.choice(self.tissue_phase_demo_dict[selected_tissue_sample][phase])
            
            # Store the selected phase demo and the start and end timestep
            selected_phase_demo_dict[phase] = {}
            selected_phase_demo_dict[phase]["phase_folder_name"] = phase
            selected_phase_demo_dict[phase]["demo_folder_name"] = selected_phase_demo
            selected_phase_demo_dict[phase]["start_timestep"] = curr_phase_idx_counter
            selected_phase_demo_dict[phase]["command"], selected_phase_demo_dict[phase]["embedding"] = self.command_embeddings_dict[phase]
            
            # Count the number of frames for the current demo
            demo_num_frames = len(os.listdir(os.path.join(self.dataset_dir, selected_tissue_sample, phase, selected_phase_demo)))
            episode_num_frames += demo_num_frames
            
            next_phase_idx_counter = curr_phase_idx_counter + demo_num_frames
            selected_phase_demo_dict[phase]["end_timestep"] = next_phase_idx_counter - 1 # -1 because the end_timestep is inclusive
            curr_phase_idx_counter = next_phase_idx_counter
        
        # Sample a random curr_ts and compute the start_ts and target_ts
        prediction_offset = self.prediction_offset
        curr_ts = np.random.randint(
            self.history_len * self.history_skip_frame,
            episode_num_frames - prediction_offset,
        )
        start_ts = curr_ts - self.history_len * self.history_skip_frame
        target_ts = curr_ts + prediction_offset
        
        # Retrieve the language embedding for the target_ts
        command_embedding, command_gt = self.get_command_for_ts(
            selected_phase_demo_dict, target_ts
        )
        
        if command_embedding is None:
            raise ValueError(f"Could not find embedding for target_ts {target_ts}.")
        
        # Construct the image sequences for the desired timesteps
        image_sequence = []
        for ts in range(start_ts, curr_ts + 1, self.history_skip_frame):
            image_dict = {}
            ts_phase_folder, ts_demo_folder, ts_demo_frame_idx = self.get_current_phase_demo_folder_and_demo_frame_idx(selected_phase_demo_dict, ts)
            for cam_name, cam_file_suffix in zip(self.camera_names, self.camera_file_suffixes):
                cam_folder = os.path.join(self.dataset_dir, selected_tissue_sample, ts_phase_folder, ts_demo_folder, cam_name)
                frame_path = os.path.join(cam_folder, f"frame{str(ts_demo_frame_idx).zfill(6)}{cam_file_suffix}")
                image_dict[cam_name] = cv2.imread(frame_path)
                image_dict[cam_name] = cv2.cvtColor(
                    image_dict[cam_name], cv2.COLOR_BGR2RGB
                )
            all_cam_images = [
                image_dict[cam_name] for cam_name in self.camera_names
            ]
            all_cam_images = np.stack(all_cam_images, axis=0)
            image_sequence.append(all_cam_images)

        # TODO: What about choosing half presision?
        image_sequence = np.array(image_sequence)
        image_sequence = torch.tensor(image_sequence, dtype=torch.float32)
        image_sequence = torch.einsum("t k h w c -> t k c h w", image_sequence)
        image_sequence = image_sequence / 255.0

        return image_sequence, command_embedding, command_gt


def load_merged_data(
    dataset_dirs,
    num_episodes_list,
    camera_names,
    camera_file_suffixes,
    batch_size_train,
    batch_size_val,
    history_len=1,
    prediction_offset=10,
    history_skip_frame=1,
    test_only=False,
    random_crop=False, # TODO: Integrate later here the other augmentations
    dagger_ratio=None, # TODO: Do we need it?
):
    print(f"{history_len=}, {history_skip_frame=}, {prediction_offset=}")
    if random_crop:
        print(f"Random crop enabled")
    # if dagger_ratio is not None: # TODO: do we need it?
    #     assert 0 <= dagger_ratio <= 1, "dagger_ratio must be between 0 and 1."
    
    # TODO: Adjust maybe later for debugging with a smaller number of tissues
    # Obtain train test split
    train_ratio = 0.90
    val_ratio = 0.05
    test_ratio = 1 - train_ratio - val_ratio

    train_datasets, val_datasets, test_datasets = [], [], []
    for dataset_dir, num_episodes in zip(dataset_dirs, num_episodes_list):
        # Load dataset dir and count number of tissue samples
        dataset_file_names = os.listdir(dataset_dir)
        tissue_folders = [f for f in dataset_file_names if f.startswith("tissue")]
        num_tissue_samples = len(tissue_folders)
        
        # Split the tissue samples into train, val, test by randomly sampling until the ratios are fulfilled
        train_indices, val_indices, test_indices = split_tissue_samples(
            dataset_dir, num_tissue_samples, train_ratio, val_ratio, test_ratio
        )
        
        if not test_only:
            # Construct dataset and dataloader for each dataset dir and merge them
            train_datasets.append(SequenceDataset(
                        [idx for d, idx in train_indices if d == dataset_dir],
                        dataset_dir,
                        camera_names,
                        camera_file_suffixes,
                        history_len,
                        prediction_offset,
                        history_skip_frame,
                        num_episodes,
                        random_crop)
            )
            val_datasets.append(SequenceDataset(
                        [idx for d, idx in val_indices if d == dataset_dir],
                        dataset_dir,
                        camera_names,
                        camera_file_suffixes,
                        history_len,
                        prediction_offset,
                        history_skip_frame,
                        num_episodes,
                        random_crop)
            )
            
            # Merge all datasets (from different tasks) into one big dataset
            merged_train_dataset = ConcatDataset(train_datasets)
            merged_val_dataset = ConcatDataset(val_datasets)
            
            # TODO: For testing purpose maybe use a small batc size and prefetcch factor
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
            
            return train_dataloader, val_dataloader
            
        else: 
            test_datasets.append(SequenceDataset(
                        [idx for d, idx in test_indices if d == dataset_dir],
                        dataset_dir,
                        camera_names,
                        camera_file_suffixes,
                        history_len,
                        prediction_offset,
                        history_skip_frame,
                        num_episodes,
                        random_crop)
            )
            
            # Merge all datasets (from different tasks) into one big dataset
            merged_test_dataset = ConcatDataset(test_datasets) 
    
            test_dataloader = DataLoader(
                merged_test_dataset,
                batch_size=1,
                shuffle=False,
                pin_memory=True,
                num_workers=16,
                prefetch_factor=1,
            )

        return test_dataloader


# TODO: Add here later the first test for the dataset
"""
Test the SequenceDataset class.

Example usage:
$ python src/instructor/dataset.py --dataset_dir /scr/lucyshi/dataset/aloha_bag_3_objects
"""
if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="Path to the dataset directory"
    )
    args = parser.parse_args()

    # Parameters for the test
    camera_names = ["cam_high", "cam_low"]
    history_len = 5
    prediction_offset = 10
    num_episodes = 10  # Just to sample from the first 10 episodes for testing

    # Create a SequenceDataset instance
    dataset = SequenceDataset(
        list(range(num_episodes)),
        args.dataset_dir,
        camera_names,
        history_len,
        prediction_offset,
    )

    # Sample a random item from the dataset
    idx = np.random.randint(0, len(dataset))
    image_sequence, command_embedding, _ = dataset[idx]

    print(f"Sampled episode index: {idx}")
    print(f"Image sequence shape: {image_sequence.shape}")
    print(f"Language embedding shape: {command_embedding.shape}")

    # Save the images in the sequence
    for t in range(history_len):
        plt.figure(figsize=(10, 5))
        for cam_idx, cam_name in enumerate(camera_names):
            plt.subplot(1, len(camera_names), cam_idx + 1)
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(
                image_sequence[t, cam_idx].permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB
            )
            plt.imshow(img_rgb)
            plt.title(f"{cam_name} at timestep {t}")
        plt.tight_layout()
        plt.savefig(f"plot/image_sequence_timestep_{t}.png")
        print(f"Saved image_sequence_timestep_{t}.png")
