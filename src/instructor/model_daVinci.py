import sys
import os

import torch
import torch.nn as nn
import numpy as np
from clip import load
import torchvision.transforms as transforms
import random

# import aloha
path_to_yay_robot = os.getenv('PATH_TO_YAY_ROBOT')
if path_to_yay_robot:
    sys.path.append(os.path.join(path_to_yay_robot, 'src'))
else:
    raise EnvironmentError("Environment variable PATH_TO_YAY_ROBOT is not set")


clip_transform = transforms.Compose(
    [
        # transforms.Resize((224, 224)), already done in dataset (more performant)
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        ),
    ]
)

class Instructor(nn.Module):
    def __init__(
        self,
        device,
        history_len,
        output_size=768,
        hidden_size=512,
        num_heads=8,
        num_layers=6,
        candidate_embeddings=None,
        candidate_texts=None,
        command_to_index=None,
        num_cameras=4,
        one_hot_flag=False,
    ):
        super().__init__()
        self.one_hot_flag = one_hot_flag

        # Load the pretrained CLIP model
        self.clip_model, self.clip_text_model = load("ViT-B/32", device=device)
        for param in self.clip_model.parameters():
            param.requires_grad = False  # Freeze the CLIP model parameters

        # Transformer for processing sequences of image embeddings
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.clip_model.visual.output_dim,
                nhead=num_heads,
                dim_feedforward=hidden_size,
                batch_first=True
            ),
            num_layers=num_layers,
        )

        if one_hot_flag:
            output_size = len(candidate_texts)

        self.mlp = nn.Sequential(
            nn.Linear(self.clip_model.visual.output_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

        # Learnable temperature
        self.temperature = nn.Parameter(torch.ones(1))

        # Positional Encoding
        self.positional_encoding = self.create_sinusoidal_embeddings(
            self.clip_model.visual.output_dim, (history_len + 1) * num_cameras
        )

        self.history_len = history_len
        self.candidate_embeddings = candidate_embeddings 
        self.candidate_texts = candidate_texts
        self.command_to_index = command_to_index 

        total, trainable = count_parameters(self)
        print(f"Total parameters: {total / 1e6:.2f}M")
        print(f"Trainable parameters: {trainable / 1e6:.2f}M")

    def forward(self, images):
        # Given images of shape (b, t, k, c, h, w)
        batch_size, timesteps, num_cameras, c, h, w = images.shape

        # Check if padding is required
        if timesteps < self.history_len + 1:
            padding_needed = self.history_len + 1 - timesteps
            padding = torch.zeros(
                (batch_size, padding_needed, num_cameras, c, h, w), device=images.device
            )
            images = torch.cat([padding, images], dim=1)
            timesteps = self.history_len + 1  # Update timesteps to reflect the new length

        # Reshape images to (b*t*k, c, h, w) for processing through CLIP
        images_reshaped = images.reshape(batch_size * timesteps * num_cameras, c, h, w)

        # Apply transformations for CLIP
        images_transformed = clip_transform(
            images_reshaped
        )  # CLIP model expects images to be normalized and resized to 224*224

        # Get image features from CLIP
        image_features = self.clip_model.encode_image(images_transformed)

        # Reshape the image features to [batch_size, timesteps*cameras, feature_dim]
        image_features_reshaped = image_features.reshape(
            batch_size, timesteps * num_cameras, -1
        ).to(torch.float32)

        # Add positional encoding
        image_features_reshaped += self.positional_encoding[
            : timesteps * num_cameras, :
        ].to(image_features_reshaped.device)

        # Pass the concatenated features through the Transformer
        transformer_out = self.transformer(
            image_features_reshaped.transpose(0, 1)
        ).transpose(0, 1)

        # Extract the final output of the Transformer for each sequence in the batch
        final_output = transformer_out[:, -1, :]

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
    
    from instructor.dataset_daVinci import load_merged_data
    from instructor.utils import set_seed

    seed = 42
    set_seed(seed)    

    # Parameters for the test
    datasets_dir = os.getenv("PATH_TO_DATASET")
    tissue_samples_ids = [1]
    camera_names = ["left_img_dir", "right_img_dir", "endo_psm1", "endo_psm2"]
    camera_file_suffixes = ["_left.jpg", "_right.jpg", "_psm1.jpg", "_psm2.jpg"]
    history_len = 2
    prediction_offset = 0 # Get command for the current timestep
    history_skip_frame = 30
    num_episodes = 200 # Number of randlomy generated stitched episodes

    # Define transforms/augmentations (resize transformation already applied in __getitem__ method)
    framewise_transforms = []
    framewise_transforms.append(transforms.RandomRotation(30))
    framewise_transforms.append(transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)))
    framewise_transforms.append(v2.RandomPerspective(p=0.5))
    framewise_transforms.append(v2.RandomPosterize(bits=7, p=0.25))
    framewise_transforms.append(v2.RandomAdjustSharpness(2, p=0.25))
    framewise_transforms.append(transforms.RandomApply([v2.GaussianBlur(kernel_size=5)], p=0.75))
    framewise_transforms.append(transforms.RandomApply([transforms.RandomResizedCrop(224, scale=(0.5, 1.0))]))
    framewise_transforms.append(v2.RandomPhotometricDistort(p=0.8))
    framewise_transforms.append(transforms.RandomGrayscale(p=0.2))
    framewise_transforms = transforms.Compose(framewise_transforms)

    # Dataset and Dataloader parameters
    datasets_dir = [os.path.join(os.getenv("PATH_TO_DATASET"), "debugging")]
    num_episodes_list = [200]*len(datasets_dir)
    camera_names = ["left_img_dir", "right_img_dir", "endo_psm1", "endo_psm2"]
    camera_file_suffixes = ["_left.jpg", "_right.jpg", "_psm1.jpg", "_psm2.jpg"]
    batch_size_train = 2
    batch_size_val = 2

    # Load the dataloader
    train_dataloader, val_dataloader, (candidate_embeddings, candidate_texts) = load_merged_data(
        dataset_dirs=datasets_dir,
        num_episodes_list=num_episodes_list,
        camera_names=camera_names,
        camera_file_suffixes=camera_file_suffixes,
        history_len=history_len,
        prediction_offset=prediction_offset,
        history_skip_frame=history_skip_frame,
        batch_size_train=batch_size_train,
        batch_size_val=batch_size_val,
        framewise_transforms=framewise_transforms,
    )    
    
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate_embeddings = candidate_embeddings.to(device)
    model = Instructor(
        device=device,
        history_len=history_len,
        candidate_embeddings=candidate_embeddings,
        candidate_texts=candidate_texts,
    )
    model.to(device)

    for split_name, dataloader in [("train", train_dataloader), ("val", val_dataloader)]:
        # Fetch a batch of data and pass it through the model
        idx_in_batch = 0
        for image_sequence, command_embedding, gt_command in dataloader:
            image_sequence = image_sequence.to(device)
            predictions_logits, temperature, _ = model(image_sequence)
            pred_command = model.decode_logits(predictions_logits, temperature)
            
            print(f"\nSplit: {split_name}")
            print(f"Image sequence shape: {image_sequence.shape}")
            print(f"Language data shape: {command_embedding.shape}")
            print(f"Predictions shape: {predictions_logits.shape}")
            print(f"Ground truth command ({prediction_offset=}): {gt_command[idx_in_batch]}")
            print(f"Predicted command [untrained] ({prediction_offset=}): {pred_command[idx_in_batch]}\n")
            
            break

        # Create a figure with subplots: one row per timestamp, one column per camera
        fig, axes = plt.subplots(history_len + 1, len(camera_names), figsize=(15, 10))
        if history_len == 0:
            axes = axes[np.newaxis, :]

        # Loop over each timestamp and camera to plot the images
        for t in range(history_len + 1):
            for cam_idx, cam_name in enumerate(camera_names):
                ax = axes[t, cam_idx]  # Get the specific subplot axis
                img = image_sequence[0, t, cam_idx].permute(1, 2, 0).detach().cpu().numpy()
                ax.imshow(img)
                ax.set_title(f"{cam_name} at timestep {t}")
                ax.axis('off')  # Optionally turn off the axis

        # Set title to command
        fig.suptitle(f"Gt Command: {gt_command[idx_in_batch]} - Prediction [untrained]: {pred_command[idx_in_batch]}")
        plt.tight_layout()
        example_dataset_plots_folder_path = os.path.join(path_to_yay_robot, "examples_plots", "untrained_model_pred")
        if not os.path.exists(example_dataset_plots_folder_path):
            os.makedirs(example_dataset_plots_folder_path)
        file_name = os.path.join(example_dataset_plots_folder_path, f"untrained_model_pred_img_{split_name=}{history_len=}_{history_skip_frame=}.png")
        file_path = os.path.join(example_dataset_plots_folder_path, file_name)
        plt.savefig(file_path)
        print(f"Saved {file_name}\n---------")
        plt.close(fig)  # Close the figure to free memory