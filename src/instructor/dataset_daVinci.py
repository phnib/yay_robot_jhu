import os
# import h5py
import json
import sys

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader, ConcatDataset

# import aloha
path_to_yay_robot = os.getenv('PATH_TO_YAY_ROBOT')
if path_to_yay_robot:
    sys.path.append(os.path.join(path_to_yay_robot, 'src'))
else:
    raise EnvironmentError("Environment variable PATH_TO_YAY_ROBOT is not set")
from aloha_pro.aloha_scripts.utils import crop_resize, random_crop, initialize_model_and_tokenizer, encode_text
from instructor.utils import center_crop_resize
# from act.utils import DAggerSampler
    
def generate_command_embeddings(unique_phase_folder_names, encoder, tokenizer, model):
    # Returns a dictionary containing the phase command as key and a tuple of the phase command and phase embedding as value
    phase_command_embeddings_dict = {}
    for phase_folder_name in unique_phase_folder_names:
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
    # TODO: Check if the indices are the same for every training (by using the seed) even when training on a different machine - e.g., otherwise introducing bias when resuming training from last checkpoint
    # TODO: Alternative would be fixed indices for each tissue sample (but randomized assuming that the execution of the surgerymight evolve over newer tissue samples)
    train_indices = [(dataset_dir, idx) for idx in all_indices[:num_train]]
    val_indices = [(dataset_dir, idx) for idx in all_indices[num_train:num_train + num_val]]
    test_indices = [(dataset_dir, idx) for idx in all_indices[num_train + num_val:]]

    return train_indices, val_indices, test_indices

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tissue_sample_ids,
        dataset_dir,
        camera_names,
        camera_file_suffixes,
        history_len=4,
        prediction_offset=15,
        history_skip_frame=30,
        num_episodes=200,
        framewise_transforms=None,
    ):
        super().__init__()
        
        if len(tissue_sample_ids) == 0:
            raise ValueError("No tissue samples found in the dataset directory.")
        
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.camera_file_suffixes = camera_file_suffixes
        self.history_len = history_len
        self.prediction_offset = prediction_offset
        self.history_skip_frame = history_skip_frame
        self.num_episodes = num_episodes
        self.framewise_transforms = framewise_transforms
        
        # Load the tissue samples and their phases and demos (for later stitching of the episodes)        
        self.tissue_phase_demo_dict = {}
        for tissue_sample_id in tissue_sample_ids:
            tissue_sample_name = f"tissue_{tissue_sample_id}"
            tissue_sample_dir_path = os.path.join(dataset_dir, tissue_sample_name)
            phases = os.listdir(tissue_sample_dir_path)
            self.tissue_phase_demo_dict[tissue_sample_name] = {}
            for phase_sample in phases:
                demo_samples = os.listdir(os.path.join(tissue_sample_dir_path, phase_sample))
                self.tissue_phase_demo_dict[tissue_sample_name][phase_sample] = demo_samples
                
        # Generate the embeddings for all phase commands
        encoder_name = "distilbert"
        tokenizer, model = initialize_model_and_tokenizer(encoder_name)
        unique_phase_folder_names = np.unique([phase_folder_name for tissue_sample in self.tissue_phase_demo_dict.values() for phase_folder_name in tissue_sample.keys()])
        self.command_embeddings_dict = generate_command_embeddings(unique_phase_folder_names, encoder_name, tokenizer, model)
        del tokenizer, model
        
    def __len__(self):
        # Here this means the number of randomly generated stitched episodes
        return self.num_episodes

    def get_command_for_ts(self, selected_phase_demo_dict, target_ts):
        # Returns the command embedding and the command for the target timestep
        for phase_segment in selected_phase_demo_dict.values():
            if phase_segment["start_timestep"] <= target_ts <= phase_segment["end_timestep"]:
                return torch.tensor(phase_segment["embedding"]).squeeze(), phase_segment["command"]
        else:
            raise ValueError(f"Could not find command for target_ts {target_ts}.")

    def get_current_phase_demo_folder_and_demo_frame_idx(self, selected_phase_demo_dict, target_ts):
        # Returns the phase and the demo frame index for the target timestep
        for phase_segment in selected_phase_demo_dict.values():
            if phase_segment["start_timestep"] <= target_ts <= phase_segment["end_timestep"]:
                return phase_segment["phase_folder_name"], phase_segment["demo_folder_name"], target_ts - phase_segment["start_timestep"]
        else:
            raise ValueError(f"Could not find phase and demo frame index for target_ts {target_ts}.")

    def __getitem__(self, index):
        # Put together the stitched episode based on randomly getting a tissue id and a random index for a demo for each phase. Then sample a random timestep and get the corresponding image sequence and command embedding
       
        selected_tissue_sample = np.random.choice(list(self.tissue_phase_demo_dict.keys()))
        selected_phase_demo_dict = {}
        episode_num_frames = curr_phase_idx_counter = 0
        
        # Go through the phases in fixed order of execution
        phases = list(self.tissue_phase_demo_dict[selected_tissue_sample].keys())
        sorted_phases = sorted(phases, key=lambda x: int(x.split('_')[0]))
        for phase in sorted_phases:
            # Select a random demo for each phase
            selected_phase_demo = np.random.choice(self.tissue_phase_demo_dict[selected_tissue_sample][phase])
            
            # Store the selected phase demo and the start and end timestep
            selected_phase_demo_dict[phase] = {}
            selected_phase_demo_dict[phase]["phase_folder_name"] = phase
            selected_phase_demo_dict[phase]["demo_folder_name"] = selected_phase_demo
            selected_phase_demo_dict[phase]["start_timestep"] = curr_phase_idx_counter
            selected_phase_demo_dict[phase]["command"], selected_phase_demo_dict[phase]["embedding"] = self.command_embeddings_dict[phase]
            
            # Count the number of frames for the current demo
            demo_num_frames = len(os.listdir(os.path.join(self.dataset_dir, selected_tissue_sample, phase, selected_phase_demo, self.camera_names[0])))
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
                img = torch.tensor(cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
                # TODO: Decide for either normal resize or to do a center crop and resize?
                # img_resized_224 = transforms.Resize((224, 224))(img)
                img_resized_224 = center_crop_resize(img, 224)
                image_dict[cam_name] = img_resized_224
                
            all_cam_images = [
                image_dict[cam_name] for cam_name in self.camera_names
            ]
            all_cam_images = torch.stack(all_cam_images, dim=0)
            all_cam_images_transformed = self.framewise_transforms(all_cam_images) # Apply same transform for all camera images
            image_sequence.append(all_cam_images_transformed)

        # TODO: What about choosing half presision?
        image_sequence = torch.stack(image_sequence, dim=0).to(dtype=torch.float32) # Shape: ts, cam, c, h, w
        image_sequence = image_sequence / 255.0

        return image_sequence, command_embedding, command_gt


# TODO: Check also if this works correctly
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
    framewise_transforms=None,
    dagger_ratio=None, # TODO: Do we need it?
):
    print(f"{history_len=}, {history_skip_frame=}, {prediction_offset=}")
    if random_crop:
        print(f"Random crop enabled")
    # if dagger_ratio is not None: # TODO: do we need it?
    #     assert 0 <= dagger_ratio <= 1, "dagger_ratio must be between 0 and 1."
    
    # TODO: Adjust maybe later for debugging with a smaller number of tissues
    # Obtain train/val/test split
    train_ratio = 0.90
    val_ratio = 0.05
    test_ratio = 1 - train_ratio - val_ratio

    # Construct the datasets and the dataset embeddings
    train_datasets, val_datasets, test_datasets = [], [], []
    command_embeddings_dict = {}
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
                        [tissue_id for tissue_id in train_indices],
                        dataset_dir,
                        camera_names,
                        camera_file_suffixes,
                        history_len,
                        prediction_offset,
                        history_skip_frame,
                        num_episodes,
                        framewise_transforms)
            )
            val_datasets.append(SequenceDataset(
                        [tissue_id for tissue_id in val_indices],
                        dataset_dir,
                        camera_names,
                        camera_file_suffixes,
                        history_len,
                        prediction_offset,
                        history_skip_frame,
                        num_episodes)
            )
            
            # Get the command embeddings for the train and val datasets
            train_command_embeddings_dict = train_datasets[-1].command_embeddings_dict
            val_command_embeddings_dict = val_datasets[-1].command_embeddings_dict

            # Check for same commands in train and val datasets
            train_commands = set(train_command_embeddings_dict.keys())
            val_commands = set(val_command_embeddings_dict.keys())
            if train_commands != val_commands:
                raise ValueError(f"Commands for validation does not match training commands.")
            
            # TODO: Check whether the embeddings for the same command are the same in train and val datasets
            
            # Update the command embeddings dictionary
            command_embeddings_dict.update(train_command_embeddings_dict)
            
        else: 
            test_datasets.append(SequenceDataset(
                        [tissue_id for tissue_id in test_indices],
                        dataset_dir,
                        camera_names,
                        camera_file_suffixes,
                        history_len,
                        prediction_offset,
                        history_skip_frame,
                        num_episodes,
                        framewise_transforms)
            )
            
            # Get the command embeddings for the test datasets (should be the same as for train and val datasets)
            test_command_embeddings_dict = test_datasets[-1].command_embeddings_dict
            command_embeddings_dict.update(test_command_embeddings_dict)
            
    # Construct the dataloaders
    if not test_only:
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
        
        return train_dataloader, val_dataloader, command_embeddings_dict
    
    else:
        # Merge all datasets (e.g., base dataset + fine tuning (correction) datasets) into one big dataset
        merged_test_dataset = ConcatDataset(test_datasets) 

        test_dataloader = DataLoader(
            merged_test_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=16,
            prefetch_factor=1,
        )
        
        return test_dataloader, command_embeddings_dict


"""
Test the SequenceDataset class.
"""
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from instructor.utils import set_seed

    seed = 42
    set_seed(seed)

    # Parameters for the test
    dataset_dir = os.getenv("PATH_TO_DATASET")
    tissue_samples_ids = [1]
    camera_names = ["left_img_dir", "right_img_dir", "endo_psm1", "endo_psm2"]
    camera_file_suffixes = ["_left.jpg", "_right.jpg", "_psm1.jpg", "_psm2.jpg"]
    history_len = 3
    prediction_offset = 0 # Get command for the current timestep
    history_skip_frame = 30
    num_episodes = 200 # Number of randlomy generated stitched episodes

    # Define transforms/augmentations (resize transformation already applied in __getitem__ method)
    # TODO: Decide for the best augmentations
    framewise_transforms = []
    framewise_transforms.append(transforms.RandomRotation(30))
    framewise_transforms.append(transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)))
    # framewise_transforms.append(v2.RandomPerspective(p=0.5))
    # framewise_transforms.append(v2.RandomPosterize(bits=7, p=0.25))
    # framewise_transforms.append(v2.RandomAdjustSharpness(2, p=0.25))
    # framewise_transforms.append(transforms.RandomApply([v2.GaussianBlur(kernel_size=5)], p=0.75))
    # framewise_transforms.append(transforms.RandomApply([transforms.RandomResizedCrop(224, scale=(0.5, 1.0))]))
    # framewise_transforms.append(v2.RandomPhotometricDistort(p=0.8))
    # framewise_transforms.append(transforms.RandomGrayscale(p=0.2))
    framewise_transforms = transforms.Compose(framewise_transforms)

    # Create a SequenceDataset instance
    dataset = SequenceDataset(
        tissue_samples_ids,
        dataset_dir,
        camera_names,
        camera_file_suffixes,
        history_len,
        prediction_offset,
        history_skip_frame,
        num_episodes,
        framewise_transforms
    )

    # Sample a random item from the dataset
    rdm_idx = np.random.randint(0, len(dataset))
    image_sequence, command_embedding, command = dataset[rdm_idx]

    print(f"Image sequence shape: {image_sequence.shape}")
    print(f"Language embedding shape: {command_embedding.shape}")
    print(f"Command: {command}")

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
    fig.suptitle(f"Command: {command}")
    plt.tight_layout()
    example_dataset_plots_folder_path = os.path.join(path_to_yay_robot, "examples_plots", "dataset")
    if not os.path.exists(example_dataset_plots_folder_path):
        os.makedirs(example_dataset_plots_folder_path)
    file_name = os.path.join(example_dataset_plots_folder_path, f"dataset_img_{history_len=}_{history_skip_frame=}.png")
    file_path = os.path.join(example_dataset_plots_folder_path, file_name)
    plt.savefig(file_path)
    print(f"Saved {file_name}.")
    plt.close(fig)  # Close the figure to free memory
