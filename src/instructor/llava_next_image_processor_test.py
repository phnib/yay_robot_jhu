import os
import time

import numpy as np
import torch
from torchvision import transforms
from transformers import CLIPImageProcessor, CLIPVisionModel
import matplotlib.pyplot as plt


# Define the parameters
model_input_dim = 336 # 224
num_patches = 3 # 2 # Must be 2 or 3
patch_dim = [model_input_dim,2*model_input_dim]
tissue_folder_path = "/data/corsair/chole/data/base_chole_clipping_cutting/tissue_8/2_clipping_first_clip_left_tube/20240712-122648-748167"
num_ts = 2
start_idx = 0
camera_name_file_suffix_dict = {"endo_psm2": "_psm2.jpg", "left_img_dir": "_left.jpg", "endo_psm1": "_psm1.jpg"}
gpu = 0

device = "cpu" # torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------------

# Dataloader
start_time = time.time()
resize_fct = transforms.Resize(patch_dim)
input_images = []
for camera_folder_name, cam_file_suffix in camera_name_file_suffix_dict.items():
    camera_images = []
    for ts_demo_frame_idx in range(start_idx, start_idx+num_ts):    
        image_path = os.path.join(tissue_folder_path, camera_folder_name, f"frame{str(ts_demo_frame_idx).zfill(6)}{cam_file_suffix}")
        image = plt.imread(image_path)
        plt.imshow(image)
        plt.axis('off')
        plt.savefig(f"original_camera_{camera_folder_name}_ts_{ts_demo_frame_idx}.png", bbox_inches='tight', pad_inches=0)
        print(f"Saved original_camera_{camera_folder_name}_ts_{ts_demo_frame_idx}.png")
        image = torch.tensor(image).permute(2, 0, 1)  # Convert to tensor and move to GPU
        image_resized = resize_fct(image) 
        camera_images.append(image_resized)    
    print(f"\nCamera: {camera_folder_name} - Original image shape:", image.shape, "Resized image shape:", image_resized.shape)
    camera_images_tensor = torch.stack(camera_images)
    input_images.append(camera_images_tensor)
input_images = torch.stack(input_images, dim=1).to(device) # Shape: [num_ts, num_cameras, C, H, W]
_, num_cams, C, H, W = input_images.shape

# Dividing into patches
input_images_patched = input_images.reshape(num_ts*num_cams, C, H, W) # Shape: [num_ts*num_cameras, C, H, W]
start_time = time.time()
# Extract two/three patches from the image (left, center, right)
if num_patches == 3:
    resize_fct_336 = transforms.Resize([336, 336])
    input_images_resized = resize_fct_336(input_images_patched)
input_images_left_half = input_images_patched[:, :, :, :model_input_dim]
input_images_right_half = input_images_patched[:, :, :, -model_input_dim:]
input_images_patched = torch.stack([input_images_resized, input_images_left_half, input_images_right_half], dim=1).to(torch.float32) 
input_images_patched = input_images_patched.reshape(num_ts*num_cams*num_patches, C, model_input_dim, model_input_dim) # Shape: [num_ts*num_cameras*num_patches, C, H, W]

# Apply mean and std normalization (apply to last 3 dimensions)
normalized_patches = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))(input_images_patched) # CLIP imagenet normalization

end_time = time.time()
print(f"\nResizing time: {1000*(end_time-start_time):.2f} ms")
print(f"Reshaped input images shape: {normalized_patches.shape}\n")  


# # TODO: Load CLIP image processor and compare the output with the normalized_patches
# # Load CLIP image processor
# image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336", do_convert_rgb=False, do_center_crop=False) 
# start_time = time.time()
# clip_input_images_patched = input_images_patched_reshaped.reshape(num_ts*num_cams*num_patches, C, model_input_dim, model_input_dim)
# clip_normalized_patches = image_processor(clip_input_images_patched)["pixel_values"]
# clip_normalized_patches_tensor = torch.tensor(np.array(clip_normalized_patches)).to(device)
# end_time = time.time()
# print(f"CLIP Image Processor time: {1000*(end_time-start_time):.2f} ms")


# Load CLIP model (choose correct model depending on the model input size)
model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336").to(device)
config = model.config

# Set the number of features
num_features = config.hidden_size 
print(f"Number of features: {num_features}")

# Apply CLIP model
normalized_patches = normalized_patches.to(device)
num_patches_to_process = normalized_patches.shape[0] # Full batch
outputs = model(normalized_patches[:num_patches_to_process]) # Desired input shape: [batch_size (-> num_ts*num_cams*num_patches), C, H, W]
pooled_output = outputs.pooler_output  # pooled CLS states
print(f"Shape of pooled_output: {pooled_output.shape}")

# Save all the images (for cameras, patches, and time steps)
input_images_patched_reshaped = input_images_patched.reshape(num_ts, num_cams, num_patches, C, model_input_dim, model_input_dim)
for ts_idx in range(num_ts):
    for camera_idx, camera_name in enumerate(camera_name_file_suffix_dict.keys()):
        for patch_idx in range(num_patches):
            patch_image = input_images_patched_reshaped[ts_idx, camera_idx, patch_idx].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
            # Save the image
            plt.imshow(patch_image)
            plt.axis('off')
            cam_ts_patch_file_name = f"camera_{camera_name}_ts_{ts_idx}_patch_{patch_idx}.png"
            plt.savefig(cam_ts_patch_file_name, bbox_inches='tight', pad_inches=0)
            print(f"Saved {cam_ts_patch_file_name}")
            plt.close()
