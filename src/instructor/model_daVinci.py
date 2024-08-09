import sys
import os

import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import random

# import aloha
path_to_yay_robot = os.getenv('PATH_TO_YAY_ROBOT')
if path_to_yay_robot:
    sys.path.append(os.path.join(path_to_yay_robot, 'src'))
else:
    raise EnvironmentError("Environment variable PATH_TO_YAY_ROBOT is not set")
from instructor.backbone_models_daVinci import extract_features, init_feature_extractor_model, preprocess_inputs, init_processor
from instructor.dataset_daVinci import SequenceDataset

class Instructor(nn.Module):
    def __init__(
        self,
        device,
        history_len,
        history_step_size,
        prediction_offset,
        camera_names,
        output_dim=768, # DestillBert embedding space size
        hidden_dim=256, # For the MLP
        num_heads=8, # For the Transformer
        num_layers=6, # For the Transformer
        candidate_embeddings=None,
        candidate_texts=None,
        command_to_index=None,
        one_hot_flag=False,
        backbone_model_name="clip",
        model_init_weights=None,
        freeze_backbone_until="all",
        use_jaw_values_flag=False,
        jaw_values_output_dim=256,
        use_phase_history_flag=False,
        phase_history_len=6,
        phase_emb_dim=4,
        history_output_dim=256,
        use_image_emb_transformer_flag=False,
        phase_to_instruction_mapping=None,
        phase_history_only_phase_switches_flag=False,
        camera_dropout_prob=0.1,
        jaw_values_dropout_prob=0.1,
        phase_history_dropout_prob=0.1,
        image_dim=224,
        llava_anyres_flag=False,
        no_llava_anyres_global_image_flag=False,
        wrist_images_rel_width=3/4,
        llava_anyres_rel_width=1/2
    ):
        super().__init__()

        # Load pretrained backbone model
        self.backbone_model, self.backbone_output_dim = init_feature_extractor_model(backbone_model_name, model_init_weights, device, freeze_backbone_until, image_dim)
        self.processor = init_processor(backbone_model_name, model_init_weights)
        
        num_camera_patches = SequenceDataset.get_num_patches(camera_names, llava_anyres_flag, no_llava_anyres_global_image_flag)
        if use_image_emb_transformer_flag:
            # Transformer for processing sequences of image embeddings
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.backbone_output_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim,
                    batch_first=True
                ),
                num_layers=num_layers,
            )
            
            # Positional Encoding
            self.positional_encoding = self.create_sinusoidal_embeddings(
                self.backbone_output_dim, (history_len + 1) * num_camera_patches
            )
            image_output_dim = self.backbone_output_dim
        else:
            image_output_dim = self.backbone_output_dim * num_camera_patches * (history_len + 1) # As image features are concatenated then
    
        if use_jaw_values_flag:
            # MLP for processing jaw values
            num_jaw_values = 2 * (history_len + 1) # 2 values per timestep
            self.jaw_values_mlp = nn.Linear(num_jaw_values, jaw_values_output_dim)
            image_jaw_mlp_input_dim = image_output_dim + jaw_values_output_dim
            self.image_jaw_mlp = nn.Sequential(
                nn.Linear(image_jaw_mlp_input_dim, hidden_dim),
                nn.ReLU()
            )
            
        if use_phase_history_flag:
            # Embedding for phase history
            self.phase_embedding = nn.Embedding(len(candidate_texts)+1, phase_emb_dim)
            history_mlp_input_dim = phase_emb_dim * phase_history_len
            self.history_mlp = nn.Sequential(
                nn.Linear(history_mlp_input_dim, history_output_dim),
                nn.ReLU()
            )
            
            # Add mapping from commands to indices (with padding)
            self.history_phase_to_index = command_to_index
            self.history_phase_to_index = {k: v + 1 for k, v in self.history_phase_to_index.items()}
            self.history_phase_to_index["padding"] = 0            

        # MLP for processing the final output
        if use_jaw_values_flag and use_phase_history_flag:
            mlp_input_dim = hidden_dim + history_output_dim
        elif use_jaw_values_flag:
            mlp_input_dim = hidden_dim
        elif use_phase_history_flag:
            mlp_input_dim = image_output_dim + history_output_dim
        else:
            mlp_input_dim = image_output_dim

        if one_hot_flag:
            output_dim = len(candidate_texts)

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Learnable temperature
        self.temperature = nn.Parameter(torch.ones(1))

        # Store the rest of the parameters (for logging)
        self.one_hot_flag = one_hot_flag
        self.history_len = history_len
        self.history_step_size = history_step_size
        self.prediction_offset = prediction_offset
        self.camera_names = camera_names
        self.candidate_embeddings = candidate_embeddings 
        self.candidate_texts = candidate_texts
        self.command_to_index = command_to_index 
        self.backbone_model_name = backbone_model_name
        self.model_init_weights = model_init_weights
        self.freeze_backbone_until = freeze_backbone_until
        self.device = device
        self.use_jaw_values_flag = use_jaw_values_flag
        self.use_phase_history_flag = use_phase_history_flag
        self.use_image_emb_transformer_flag = use_image_emb_transformer_flag
        if use_jaw_values_flag:
            self.jaw_values_output_dim = jaw_values_output_dim
        if use_phase_history_flag:
            self.phase_emb_dim = phase_emb_dim
            self.history_output_dim = history_output_dim
            self.phase_history_len = phase_history_len
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        if use_image_emb_transformer_flag:
            self.num_heads = num_heads
            self.num_layers = num_layers
        self.phase_to_instruction_mapping = phase_to_instruction_mapping
        self.phase_history_only_phase_switches_flag = phase_history_only_phase_switches_flag
        self.image_dim = image_dim
        self.llava_anyres_flag = llava_anyres_flag
        self.no_llava_anyres_global_image_flag = no_llava_anyres_global_image_flag
        self.wrist_images_rel_width = wrist_images_rel_width
        self.llava_anyres_rel_width = llava_anyres_rel_width

        # Dropout probabilities
        self.camera_dropout_prob = camera_dropout_prob
        self.jaw_values_dropout_prob = jaw_values_dropout_prob
        self.phase_history_dropout_prob = phase_history_dropout_prob

        total, trainable = count_parameters(self)
        print(f"Total parameters: {total / 1e6:.2f}M")
        print(f"Trainable parameters: {trainable / 1e6:.2f}M")

    def forward(self, images, psm2_psm1_jaw_values=None, phase_history=None):
        
        if self.use_phase_history_flag:
            assert len(phase_history[0]) == self.phase_history_len, f"Phase history should have length {self.phase_history_len}"
        
        # Given images of shape (b, t, k, c, h, w)
        batch_size, timesteps, num_camera_patches, c, h, w = images.shape

        # Apply camera dropout
        if self.training and self.camera_dropout_prob > 0:
            camera_dropout_mask = torch.bernoulli(torch.ones(batch_size, num_camera_patches, device=self.device) * (1 - self.camera_dropout_prob))
            camera_dropout_mask = camera_dropout_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            images = images * camera_dropout_mask

        # Check if padding is required
        if timesteps < self.history_len + 1:
            padding_needed = self.history_len + 1 - timesteps
            padding = torch.zeros(
                (batch_size, padding_needed, num_camera_patches, c, h, w), device=images.device
            )
            images = torch.cat([padding, images], dim=1)
            timesteps = self.history_len + 1  # Update timesteps to reflect the new length

        # Reshape images to (b*t*k, c, h, w) for processing through backbone model
        images_reshaped = images.reshape(batch_size * timesteps * num_camera_patches, c, h, w)

        # Apply transformations for backbone model --> backbone model expects images to be normalized and resized e.g. to 224*224
        images_transformed = preprocess_inputs(images_reshaped, self.backbone_model_name, self.model_init_weights, self.device, self.processor) 

        # Get image features from backbone model
        image_features = extract_features(self.backbone_model, self.backbone_model_name, self.model_init_weights, self.image_dim, images_transformed)

        # Reshape the image features to [batch_size, timesteps*cameras, feature_dim]
        image_features_reshaped = image_features.reshape(
            batch_size, timesteps * num_camera_patches, -1
        ).to(torch.float32)

        # Use the transformer to process the image features or concatenate them
        if self.use_image_emb_transformer_flag:
            # Add positional encoding
            image_features_reshaped += self.positional_encoding[
                : timesteps * num_camera_patches, :
            ].to(image_features_reshaped.device)

            # Pass the concatenated features through the Transformer
            transformer_out = self.transformer(
                image_features_reshaped.transpose(0, 1)
            ).transpose(0, 1)

            # Extract the final output of the Transformer for each sequence in the batch
            final_image_output = transformer_out[:, -1, :]
        else:
            # Concatenate the image features
            final_image_output = image_features_reshaped.reshape(batch_size, -1)

        # Process the jaw values (if available)
        if self.use_jaw_values_flag:
            # Apply jaw values dropout
            if self.training and self.jaw_values_dropout_prob > 0:
                jaw_input_dropout_mask = torch.bernoulli(torch.ones((batch_size, 1, 2), device=self.device) * (1 - self.jaw_values_dropout_prob))
                psm2_psm1_jaw_values = psm2_psm1_jaw_values * jaw_input_dropout_mask
            psm2_psm1_jaw_values_flattened = psm2_psm1_jaw_values.reshape(batch_size, -1) # Flatten the jaw values: (batch_size, timesteps, 2) -> (batch_size, timesteps*2)
            
            psm2_psm1_jaw_values_features = self.jaw_values_mlp(psm2_psm1_jaw_values_flattened)
            psm2_psm1_jaw_values_image_features = torch.cat((final_image_output, psm2_psm1_jaw_values_features), dim=1)
            final_output = self.image_jaw_mlp(psm2_psm1_jaw_values_image_features)          
        else:
            final_output = final_image_output

        # Process the phase history (if available)
        if self.use_phase_history_flag:
            # Apply phase history dropout
            if self.training and self.phase_history_dropout_prob > 0:
                phase_history_dropout_mask = torch.bernoulli(torch.ones((batch_size, 1), device=self.device) * (1 - self.phase_history_dropout_prob))
                phase_history = (phase_history * phase_history_dropout_mask).to(torch.int64)
            
            phase_history_embeddings = self.phase_embedding(phase_history)
            phase_history_embeddings_reshaped = phase_history_embeddings.reshape(batch_size, -1)
            phase_history_output = self.history_mlp(phase_history_embeddings_reshaped)
            
            # Concatenate the final output with the phase history output
            final_output = torch.cat((final_output, phase_history_output), dim=1)

        if self.one_hot_flag:
            # Directly predict the logits for each command
            logits = self.mlp(final_output)
            command_emb_pred = final_output # From transformer model
        else:
            # Predict the command embedding
            command_emb_pred = self.mlp(final_output)
            # Compute the similarity scores as logits
            logits = self.compute_similarities(command_emb_pred) / self.temperature.clamp(
                min=1e-8
            )

        return logits, self.temperature, command_emb_pred

    def compute_similarities(self, embeddings):
        # Compute the cosine similarities
        cosine_similarities = (embeddings @ self.candidate_embeddings.T) / (
            embeddings.norm(dim=-1, keepdim=True)
            * self.candidate_embeddings.norm(dim=-1, keepdim=True).T
        )

        return cosine_similarities

    @staticmethod
    def create_sinusoidal_embeddings(d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def decode_logits(self, logits, temperature):
        # Returns the command with the highest logit 
                
        # Compute the probabilities
        probs = (
            logits
            if self.one_hot_flag
            else torch.nn.functional.softmax(logits / temperature, dim=-1)
        )

        # Find the indices of the max logit for each example in the batch
        _, max_indices = torch.max(probs, dim=-1)

        return [self.candidate_texts[index] for index in max_indices.cpu().numpy()]

    def get_nearest_embedding(self, embeddings):
        # Compute cosine similarities
        similarities = self.compute_similarities(embeddings)

        # Get the index of the maximum similarity for each prediction
        indices = similarities.argmax(dim=-1)

        # Print the top 5 candidates
        probs = torch.nn.functional.softmax(similarities, dim=-1)
        top_probs, top_indices = torch.topk(probs[0], 5)
        normalized_top_probs = top_probs / top_probs.sum()
        for i, (index, prob) in enumerate(zip(top_indices, normalized_top_probs)):
            print(
                f"Candidate {i}: {self.candidate_texts[index]}, Normalized Prob: {prob:.4f}"
            )

        # Map the indices back to the embeddings
        return [self.candidate_embeddings[i] for i in indices.cpu().numpy()]

    def get_random_from_top_k(self, embeddings, k=3):
        similarities = (embeddings @ self.candidate_embeddings.T) / (
            embeddings.norm(dim=-1, keepdim=True)
            * self.candidate_embeddings.norm(dim=-1, keepdim=True).T
        )
        top_k_indices = similarities.topk(k, dim=-1)[1]

        # Randomly select one from the top-k for each row
        selected_indices = [
            random.choice(indices_row) for indices_row in top_k_indices.cpu().numpy()
        ]

        return [self.candidate_texts[i] for i in selected_indices]

    def sample_with_temperature(self, embeddings, temperature=1.0):
        similarities = (embeddings @ self.candidate_embeddings.T) / (
            embeddings.norm(dim=-1, keepdim=True)
            * self.candidate_embeddings.norm(dim=-1, keepdim=True).T
        )
        probs = torch.nn.functional.softmax(similarities / temperature, dim=-1)
        sampled_indices = torch.multinomial(
            probs, 1
        ).squeeze()  # Squeezing to potentially remove singleton dimensions
        # Check if sampled_indices is a scalar (0-dim) or an array
        if sampled_indices.ndim == 0:
            # If it's a scalar, we make it a one-element array
            sampled_indices = [sampled_indices.item()]
        else:
            # Otherwise, we convert it to a list
            sampled_indices = sampled_indices.tolist()

        return [self.candidate_texts[i] for i in sampled_indices]


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


# Example usage:
if __name__ == "__main__":      
    import os
    import matplotlib.pyplot as plt
    from torchvision.transforms import v2
    import albumentations as A
    
    from instructor.dataset_daVinci import load_merged_data, SequenceDataset
    from instructor.utils import set_seed

    seed = 42
    set_seed(seed)    

    # Parameters for the test
    gpu = 1
    tissue_samples_ids = [1]
    camera_names = ["endo_psm2", "left_img_dir", "endo_psm1"] # "right_img_dir"
    camera_file_suffixes = ["_psm2.jpg", "_left.jpg", "_psm1.jpg"] # "_right.jpg"
    dataset_names = ["base_chole_clipping_cutting", "phantom_chole"] # "base_chole_clipping_cutting" "phantom_chole" "debugging"
    datasets_dir = [os.path.join(os.getenv("PATH_TO_DATASET"), dataset_name) for dataset_name in dataset_names]
    num_episodes_list = [200]*len(datasets_dir)
    batch_size_train = batch_size_val=  16
    history_len = 3
    prediction_offset = 0 # Get command for the current timestep
    history_step_size = 30
    num_episodes = 200 # Number of randlomy generated stitched episodes
    use_phase_history_flag = True
    phase_history_len = 6
    use_jaw_values_flag = True
    use_transformer_flag = False 
    reduced_base_class_set_flag = False
    one_hot_flag = False
    backbone_model_name = "clip"
    model_init_weights = "imagenet"
    prediction_step_size = 30
    recovery_probability = 0.4
    phase_history_only_phase_switches_flag = False
    camera_dropout_prob = jaw_values_dropout_prob = phase_history_dropout_prob=0.4
    image_dim = 336
    llava_anyres_flag = True
    no_llava_anyres_global_image_flag = False
    wrist_images_rel_width = 3/4
    llava_anyres_rel_width = 2/3
    

    # Define transforms/augmentations (resize transformation already applied in __getitem__ method)
    torch_input_transforms = []
    
    # Note: Automatic augmentations
    torch_input_transforms.append(transforms.RandAugment())
    # input_transforms.append(transforms.TrivialAugmentWide())
    # input_transforms.append(transforms.AugMix())
    
    # Note: Manual augmentations
    # Torch augmentations
    torch_input_transforms.append(transforms.RandomResizedCrop([image_dim,image_dim], scale=(0.8, 1.0)))
    # input_transforms.append(transforms.RandomRotation(15))
    # input_transforms.append(transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)))
    
    # input_transforms.append(v2.RandomPerspective(p=0.5))
    # input_transforms.append(v2.RandomPosterize(bits=7, p=0.25))
    # input_transforms.append(v2.RandomAdjustSharpness(2, p=0.25))
    # input_transforms.append(transforms.RandomApply([v2.GaussianBlur(kernel_size=5)], p=0.75))
    # input_transforms.append(v2.RandomPhotometricDistort(p=0.8))
    # input_transforms.append(transforms.RandomGrayscale(p=0.2))
    
    # Albumentations augmentations
    albumentation_input_transforms = []

    min_height, min_width = max(1, image_dim // 40), max(1, image_dim // 40)
    max_height, max_width = min(image_dim // 30, image_dim), min(image_dim // 30, image_dim) 
    albumentation_input_transforms.append(A.CoarseDropout(max_holes=128, max_height=max_height, max_width=max_width, min_holes=1, min_height=min_height, 
                    min_width=min_width, fill_value=0, p=0.5))
    
    # Store the transforms in a dictionary
    torch_input_transforms = transforms.Compose(torch_input_transforms)
    num_patches = SequenceDataset.get_num_patches(camera_names, llava_anyres_flag, no_llava_anyres_global_image_flag)
    num_input_images = num_patches * (history_len + 1)
    albumentations_additional_targets = dict(zip([f"image{i}" for i in range(num_input_images)], ["image"] * num_input_images))    
    albumentation_input_transforms = A.Compose(albumentation_input_transforms, additional_targets=albumentations_additional_targets)
    input_transforms = {"torch": torch_input_transforms, "albumentations": albumentation_input_transforms}

    # Load the dataloader
    train_dataloader, val_dataloader, ds_metadata_dict = load_merged_data(
        dataset_dirs=datasets_dir,
        num_episodes_list=num_episodes_list,
        camera_names=camera_names,
        camera_file_suffixes=camera_file_suffixes,
        history_len=history_len,
        prediction_offset=prediction_offset,
        history_step_size=history_step_size,
        batch_size_train=batch_size_train,
        batch_size_val=batch_size_val,
        input_transforms=input_transforms,
        use_phase_history_flag=use_phase_history_flag,
        use_jaw_values_flag=use_jaw_values_flag,
        reduced_base_class_set_flag=reduced_base_class_set_flag,
        phase_history_len=phase_history_len,
        prediction_step_size=prediction_step_size,
        recovery_probability=recovery_probability,
        phase_history_only_phase_switches_flag=phase_history_only_phase_switches_flag,
        image_dim=image_dim,
        llava_anyres_flag=llava_anyres_flag,
        no_llava_anyres_global_image_flag=no_llava_anyres_global_image_flag,
        wrist_images_rel_width=wrist_images_rel_width,
        llava_anyres_rel_width=llava_anyres_rel_width
    )    
    candidate_embeddings = ds_metadata_dict["candidate_embeddings"]
    candidate_texts = ds_metadata_dict["candidate_texts"]
    
    # Load the model
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    candidate_embeddings = candidate_embeddings.to(device)
    command_to_index = {command: i for i, command in enumerate(candidate_texts)}
    model = Instructor(
        device=device,
        history_len=history_len,
        history_step_size=history_step_size,
        prediction_offset=prediction_offset,
        candidate_embeddings=candidate_embeddings,
        candidate_texts=candidate_texts,
        command_to_index=command_to_index,
        camera_names=camera_names,
        use_jaw_values_flag=use_jaw_values_flag,
        use_phase_history_flag=use_phase_history_flag,
        use_image_emb_transformer_flag=use_transformer_flag,
        one_hot_flag=one_hot_flag,
        backbone_model_name=backbone_model_name,
        model_init_weights=model_init_weights,
        phase_history_len=phase_history_len,
        phase_history_only_phase_switches_flag=phase_history_only_phase_switches_flag,
        camera_dropout_prob=camera_dropout_prob,
        jaw_values_dropout_prob=jaw_values_dropout_prob,
        phase_history_dropout_prob=phase_history_dropout_prob,
        image_dim=image_dim,
        llava_anyres_flag=llava_anyres_flag,
        no_llava_anyres_global_image_flag=no_llava_anyres_global_image_flag,
        wrist_images_rel_width=wrist_images_rel_width,
        llava_anyres_rel_width=llava_anyres_rel_width
    )
    model.to(device)

    for split_name, dataloader in [("train", train_dataloader), ("val", val_dataloader)]:
        # Fetch a batch of data and pass it through the model
        idx_in_batch = 0
        for image_sequence, command_embedding, gt_command, jaw_values, phase_history in dataloader:
            image_sequence = image_sequence.to(device)
            if use_jaw_values_flag:
                jaw_values = jaw_values.to(device)
            if use_phase_history_flag:
                phase_history_indexed = [[model.history_phase_to_index[phase_command_list[batch_idx]] for batch_idx in range(len(phase_command_list))] for phase_command_list in phase_history]
                phase_history_indexed = torch.tensor(phase_history_indexed).transpose(0, 1).to(device)
            predictions_logits, temperature, _ = model(image_sequence, jaw_values, phase_history_indexed)
            pred_command = model.decode_logits(predictions_logits, temperature)
            
            print(f"\nSplit: {split_name}")
            print(f"Image sequence shape: {image_sequence.shape}")
            print(f"Language data shape: {command_embedding.shape}")
            print(f"Predictions shape: {predictions_logits.shape}")
            print(f"Ground truth command ({prediction_offset=}): {gt_command[idx_in_batch]}")
            print(f"Predicted command [untrained] ({prediction_offset=}): {pred_command[idx_in_batch]}\n")
            print(f"Phase history (idx=0): {[phase[0] for phase in phase_history]}")
            print(f"Jaw values (idx=0): {jaw_values[0]}")
            
            break

        # Create a figure with subplots: one row per timestamp, one column per camera
        fig_height = 4 * (history_len + 1)
        camera_patch_names = SequenceDataset.get_camera_patch_names(camera_names, llava_anyres_flag, no_llava_anyres_global_image_flag)
        fig_width = len(camera_patch_names) * 3
        fig, axes = plt.subplots(history_len + 1, len(camera_patch_names), figsize=(fig_width, fig_height))
        if history_len == 0:
            axes = axes[np.newaxis, :]

        # Loop over each timestamp and camera to plot the images
        for t in range(history_len + 1):
            for cam_idx, cam_patch_name in enumerate(camera_patch_names):
                ax = axes[t, cam_idx]  # Get the specific subplot axis
                img = image_sequence[0, t, cam_idx].permute(1, 2, 0).detach().cpu().numpy()
                ax.imshow(img)
                ax.set_title(f"{cam_patch_name} at timestep {t}")
                ax.axis('off')  # Optionally turn off the axis

        # Set title to command
        fig.suptitle(f"Gt Command: {gt_command[idx_in_batch]} - Prediction [untrained]: {pred_command[idx_in_batch]}")
        plt.tight_layout()
        example_dataset_plots_folder_path = os.path.join(path_to_yay_robot, "examples_plots", "untrained_model_pred")
        if not os.path.exists(example_dataset_plots_folder_path):
            os.makedirs(example_dataset_plots_folder_path)
        file_name = os.path.join(example_dataset_plots_folder_path, f"untrained_model_pred_img_{split_name=}{history_len=}_{history_step_size=}_{image_dim=}_{wrist_images_rel_width=}_{llava_anyres_rel_width=}_{llava_anyres_flag=}_{no_llava_anyres_global_image_flag}.png")
        file_path = os.path.join(example_dataset_plots_folder_path, file_name)
        plt.savefig(file_path)
        print(f"Saved {file_name}\n\n---------")
        plt.close(fig)  # Close the figure to free memory