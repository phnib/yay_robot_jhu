import os
import time

import numpy as np
import torch
from torchvision import transforms
from transformers import CLIPImageProcessor, CLIPVisionModel, CLIPFeatureExtractor
import matplotlib.pyplot as plt


# Define the parameters
model_input_dim = 336 # 224
patch_dim = [model_input_dim,2*model_input_dim]
tissue_folder_path = "/data/corsair/chole/data/base_chole_clipping_cutting/tissue_8/2_clipping_first_clip_left_tube/20240712-122648-748167"
num_ts = 2
start_idx = 0
camera_name_file_suffix_dict = {"endo_psm2": "_psm2.jpg", "left_img_dir": "_left.jpg", "endo_psm1": "_psm1.jpg"}
gpu = 0

device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
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
resize_fct_336 = transforms.Resize([336, 336])
input_images_patched = input_images.reshape(num_ts*num_cams, C, H, W) # Shape: [num_ts*num_cameras, C, H, W]
start_time = time.time()
input_images_resized = resize_fct_336(input_images_patched)
input_images_left_half = input_images_patched[:, :, :, :model_input_dim]
input_images_right_half = input_images_patched[:, :, :, -model_input_dim:]
input_images_patched = torch.stack([input_images_resized, input_images_left_half, input_images_right_half], dim=1).to(torch.float32) # Shape: [num_ts*num_cameras, num_patches, C, H, W]
input_images_patched_reshaped = input_images_patched.reshape(num_ts, num_cams, 3, C, model_input_dim, model_input_dim)

# Apply mean and std normalization (apply to last 3 dimensions)
normalized_patches = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))(input_images_patched_reshaped) # CLIP imagenet normalization

end_time = time.time()
print(f"\nResizing time: {1000*(end_time-start_time):.2f} ms")
print(f"Reshaped input images shape: input_images_patched_reshaped.shape\n")  

# # TODO: Load CLIP image processor and compare the output with the normalized_patches
# # Load CLIP image processor
# image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336", do_convert_rgb=False, do_center_crop=False) 
# start_time = time.time()
# clip_input_images_patched = input_images_patched_reshaped.reshape(num_ts*num_cams*3, C, model_input_dim, model_input_dim)
# clip_normalized_patches = image_processor(clip_input_images_patched)["pixel_values"]
# clip_normalized_patches_tensor = torch.tensor(np.array(clip_normalized_patches)).to(device)
# clip_normalized_patches_tensor_reshaped = clip_normalized_patches_tensor.reshape(num_ts, num_cams, 3, C, model_input_dim, model_input_dim)
# end_time = time.time()
# print(f"CLIP Image Processor time: {1000*(end_time-start_time):.2f} ms")


# Load CLIP model (choose correct model depending on the model input size)
# TODO: Or rather using CLIP Feature Extractor?
model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336") # TODO: Put on device # TODO: Load ckpt correctly

# Set the number of features
num_features = None

# Replace the head of the model with new classification heads for the specific tasks
model.classifier = torch.nn.Identity()  # Remove the pre-trained classification head if present

# Apply CLIP model
normalized_patches = normalized_patches.to(device)
output = model(normalized_patches)

# Print the output
print("Output shape:", output.logits.shape)
print("Output:", output.logits) # TODO: Does it have logits or is it just the embeddings?

# TODO: Checkout how the images of left and right wrist look like
# Save all the images (for cameras, patches, and time steps)
for ts_idx in range(num_ts):
    for camera_idx, camera_name in enumerate(camera_name_file_suffix_dict.keys()):
        for patch_idx in range(3):
            patch_image = input_images_patched_reshaped[ts_idx, camera_idx, patch_idx].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
            # Save the image
            plt.imshow(patch_image)
            plt.axis('off')
            cam_ts_patch_file_name = f"camera_{camera_name}_ts_{ts_idx}_patch_{patch_idx}.png"
            plt.savefig(cam_ts_patch_file_name, bbox_inches='tight', pad_inches=0)
            print(f"Saved {cam_ts_patch_file_name}")
            plt.close()
