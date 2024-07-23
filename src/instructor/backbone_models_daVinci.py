from pathlib import Path
import os
import sys

from clip import load
import torch
import torchvision.transforms as T
import torch.nn as nn
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from torchvision import models
from functools import partial
from timm.models.vision_transformer import VisionTransformer
from huggingface_hub import snapshot_download
from transformers import ViTForImageClassification, ViTImageProcessor

path_to_yay_robot = os.getenv('PATH_TO_YAY_ROBOT')
if path_to_yay_robot:
    sys.path.append(os.path.join(path_to_yay_robot, 'src'))
else:
    raise EnvironmentError("Environment variable PATH_TO_YAY_ROBOT is not set")
from instructor.submodules.gsvit_submodule.gsvit_ae_model import EfficientViTAutoEncoder

# --------------------------- Model init functions ----------------------------
  
def load_gsvit_fe(model_init_weights, device):
    
    # Load the EfficientViT model as feature extractor
    model = EfficientViTAutoEncoder()

    # Load all weights from the pretrained model (from its encoder)
    models_folder_path = Path(__file__).resolve().parent / "submodules" / "gsvit_submodule" / "models"
    if model_init_weights == "general":
        model_path = models_folder_path / "GSViT.pkl"
    elif model_init_weights == "cholecystectomy":
        model_path = models_folder_path / "saved_network_cholesystectomy_0_41.pkl" # Use either "GSViT.pkl" or "saved_network_cholesystectomy_0_41.pkl"
    elif model_init_weights == "imagenet":
        pass # As imagenet weights are loaded by default
    else:
        raise ValueError(f"Model weights {model_init_weights} not supported yet!")
    
    if model_init_weights in ["general", "cholecystectomy"]:
        # Load the weights from the model
        # model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(pretrained_dict)
    
    # # Check if model_dict and pretrained_dict keys are the same
    # common_keys = set(model_dict.keys()) & set(pretrained_dict.keys())
    # print("\nKeys present in both the model's encoder architecture and pretrained SSL weights:", common_keys)
    
    num_features = 384 
    
    return model, num_features
  
def load_endovit_fe(model_init_weights, img_size, patch_size = 16, embed_dim = 768, depth = 12, num_heads = 12, mlp_ratio = 4, qkv_bias = True, norm_layer = partial(nn.LayerNorm, eps=1e-6)):
       
    if model_init_weights == "endo700k":
        # Define the huggingface repository and model filename
        repo_id = "egeozsoy/EndoViT"
        model_filename = "pytorch_model.bin"
        
        # Download model files
        model_path = snapshot_download(repo_id=repo_id, revision="main")
        model_init_weights_path = Path(model_path) / model_filename

        # Load model weights
        model_init_weights = torch.load(model_init_weights_path)['model']

        # Define the model (ensure this matches your model's architecture)
        model = VisionTransformer(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer)

        # Load the weights into the model
        model.load_state_dict(model_init_weights, strict=False)             
        
        # Assuming the model is a vision transformer, we need to get the dimension of the last layer
        num_features = model.head.in_features # If not using GlobalAveragePooling * (img_size[0]//patch_size * img_size[1]//patch_size + 1)
        
        # Replace the head of the model with new classification heads for the specific tasks
        model.head = nn.Identity()  # Remove the pre-trained classification head if present
    elif model_init_weights == "imagenet":
        # Using huggingface imagenet pretrained weights (https://huggingface.co/google/vit-base-patch16-224)
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        
        # Set the number of features
        num_features = model.classifier.in_features
        
        # Replace the head of the model with new classification heads for the specific tasks
        model.classifier = nn.Identity()  # Remove the pre-trained classification head if present
    else:
        raise ValueError(f"Model weights {model_init_weights} not supported yet!")
    
    return model, num_features
    
def load_resnet_fe(model_init_weights):
    if model_init_weights == "imagenet":
        model = models.resnet50(weights='IMAGENET1K_V2')
        num_features = model.fc.in_features
    # Load pretrained  SelfSupSurg weights
    elif model_init_weights in ["mocov2", "simclr", "swav", "dino"]:
        self_sup_surg_models_folder = Path(__file__).resolve().parent / "submodules" / "selfsupsurg" / "models"
        if model_init_weights == "mocov2":
            # Init the ResNet model
            model = models.resnet50()
            num_features = model.fc.in_features
            model.fc = torch.nn.Identity() # Remove the original classification head
            
            # Load the weights from the MoCoV2 model
            mocov2_model_path = self_sup_surg_models_folder / "model_final_checkpoint_moco_v2_surg.torch"
            base_model_checkpoint = torch.load(mocov2_model_path)["classy_state_dict"]["base_model"]["model"]["trunk"]
            pretrained_dict = {param_name.replace('_feature_blocks.', ''): param_weight for param_name, param_weight in base_model_checkpoint.items()}
            model.load_state_dict(pretrained_dict, strict=False)
        elif model_init_weights == "simclr":
            # Init the ResNet model
            model = models.resnet50()
            num_features = model.fc.in_features
            model.fc = torch.nn.Identity() # Remove the original classification head
            
            # Load the weights from the SimCLR model
            simclr_model_path = self_sup_surg_models_folder / "model_final_checkpoint_simclr_surg.torch"
            base_model_checkpoint = torch.load(simclr_model_path)["classy_state_dict"]["base_model"]["model"]["trunk"]
            pretrained_dict = {param_name.replace('_feature_blocks.', ''): param_weight for param_name, param_weight in base_model_checkpoint.items()}
            model.load_state_dict(pretrained_dict, strict=False)            
        elif model_init_weights == "swav":
            # Init the ResNet model
            model = models.resnet50()
            num_features = model.fc.in_features
            model.fc = torch.nn.Identity() # Remove the original classification head
            
            # Load the weights from the SwAV model
            swav_model_path = self_sup_surg_models_folder / "model_final_checkpoint_swav_surg.torch"
            base_model_checkpoint = torch.load(swav_model_path)["classy_state_dict"]["base_model"]["model"]["trunk"]
            pretrained_dict = {param_name.replace('_feature_blocks.', ''): param_weight for param_name, param_weight in base_model_checkpoint.items()}
            model.load_state_dict(pretrained_dict, strict=False)
        elif model_init_weights == "dino":
            # Init the ResNet model
            model = models.resnet50()
            num_features = model.fc.in_features
            model.fc = torch.nn.Identity() # Remove the original classification head
            
            # Load the weights from the DINO model
            dino_model_path = self_sup_surg_models_folder / "model_final_checkpoint_dino_surg.torch"
            base_model_checkpoint = torch.load(dino_model_path)["classy_state_dict"]["base_model"]["model"]["trunk"]
            pretrained_dict = {param_name.replace('_feature_blocks.', ''): param_weight for param_name, param_weight in base_model_checkpoint.items()}
            model.load_state_dict(pretrained_dict, strict=False)        
    else:
        raise ValueError(f"Model weights {model_init_weights} not supported yet!")
    
    # Remove the original fully connected layer
    model = nn.Sequential(*list(model.children())[:-1])  # Remove the original classification head
    
    return model, num_features
    
def load_clip_fe(model_init_weights, device):    
    if model_init_weights == "imagenet":
        # Load the CLIP model
        model = load("ViT-B/32", device=device)[0]
    elif model_init_weights == "sda":
        # Load the SDA-CLIP model
        model_weights_path = Path(__file__).resolve().parent / "submodules" / "clip" / "models" / "soft_task.pt"
        model = load("ViT-B/16", device=device)[0]
        sda_clip_state_dict = torch.load(model_weights_path, map_location=device)["model_state_dict"]
        model.load_state_dict(sda_clip_state_dict)
    else:
        raise ValueError(f"Model weights {model_init_weights} not supported yet!")
        
    # Set the number of features
    num_features = model.visual.output_dim
    
    return model, num_features
    
def init_feature_extractor_model(fe_model_name, model_init_weights, device, freeze_fe_until_layer, img_size=(224,224)):
    
    # Load the desired feature extractor model
    if fe_model_name == "gsvit":
        # Load a pre-trained GSViT model (either with SSL or ImageNet weights)
        fe, num_features = load_gsvit_fe(model_init_weights, device)
    elif fe_model_name == "endovit":
        # Load a pre-trained EndoViT model (either with SSL or ImageNet weights)
        fe, num_features = load_endovit_fe(model_init_weights, img_size=img_size)
    elif fe_model_name == "resnet":
        # Load a pre-trained ResNet50 model (either with SelfSupSurg or ImageNet weights)
        fe, num_features = load_resnet_fe(model_init_weights)
    elif fe_model_name == "clip":
        # Load a pre-trained CLIP model
        fe, num_features = load_clip_fe(model_init_weights, device)
        
    # Freeze the feature extractor
    for layer_idx, param in enumerate(fe.parameters()):
        if freeze_fe_until_layer != "all" and (freeze_fe_until_layer == "none" or layer_idx == freeze_fe_until_layer):
            break
        param.requires_grad = False
        
    return fe, num_features

# ------------------------- Model preprocessing functions -------------------------
    
def preprocess_inputs_gsvit(images):    
    """
    Flip color channels, e.g., from RGB to BGR
    
    Args:
        images (torch.Tensor): Input images
    
    Returns:
        images (torch.Tensor): Images with flipped color channels
    """
    
    tmp = images[:, 0, :, :].clone()
    images[:, 0, :, :] = images[:, 2, :, :]
    images[:, 2, :, :] = tmp
    return images

def preprocess_inputs_endovit(images, model_init_weights, processor, device):
    
    if model_init_weights == "endo700k":
        # Define the normalization transformation
        dataset_mean=[0.3464, 0.2280, 0.2228]
        dataset_std=[0.2520, 0.2128, 0.2093]
        normalize = T.Normalize(mean=dataset_mean, std=dataset_std)

        # Apply the normalization to the batch of images
        normalized_images = normalize(images)
    elif model_init_weights == "imagenet":
        normalized_images = processor(images=images, do_rescale=False, return_tensors="pt").pixel_values.to(device)

    return normalized_images

def preprocess_inputs_resnet(images, model_init_weights):
    
    if model_init_weights == "imagenet":
        # Preprocess the images with the same normalization and resizing as the ImageNet dataset
        normalized_images = models.ResNet50_Weights.IMAGENET1K_V2.transforms()(images)
    else:
        # Normalize the images with the dataset mean and std from the SelfSupSurg dataset
        dataset_mean=[0.485, 0.456, 0.406]
        dataset_std=[0.229, 0.224, 0.225]
        normalized_images = T.Normalize(mean=dataset_mean, std=dataset_std)(images)
    
    return normalized_images
   
def preprocess_inputs_clip(images):
    
    clip_transform = T.Compose(
        [
            # transforms.Resize((224, 224)), already done in dataset (more performant)
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # Note: same normalization in SDA-CLIP used
        ]
    )
    normalized_images = clip_transform(images)
    return normalized_images
    
def preprocess_inputs(images, fe_model_name, model_init_weights, device, processor=None):

    if fe_model_name == "gsvit":
        # Process the inputs
        images = preprocess_inputs_gsvit(images)            
    elif fe_model_name == "endovit":
        # Process the inputs
        images = preprocess_inputs_endovit(images, model_init_weights, processor, device)
    elif fe_model_name == "resnet":
        images = preprocess_inputs_resnet(images, model_init_weights)
    elif fe_model_name == "clip":
        images = preprocess_inputs_clip(images)
        
    return images

def init_processor(fe_model_name, model_init_weights):
    
    if fe_model_name == "endovit" and model_init_weights == "imagenet":
        return ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    else:
        return None

# ------------------------- FE + classifier functions ----------------------------------

def extract_features(fe, fe_model_name, model_init_weights, x):
    
    if fe_model_name == "gsvit":
        features = fe.evit(x)
        # Apply 2D Global Average Pooling
        batch_size, num_features = features.shape[:2]
        flattened_tensor = x.view(batch_size, num_features, -1)  # shape: (Batchsize, num_features, 4x4)
        # Compute average pooling across the last dimension (spatial dimensions)
        features = torch.mean(flattened_tensor, dim=-1)  # shape: (Batchsize, num_features)
    elif fe_model_name == "endovit" and model_init_weights == "endo700k":
        features = fe.forward_features(x)
        # Apply 1D Global Average Pooling
        features = torch.mean(features, dim=1)
    elif fe_model_name == "endovit" and model_init_weights == "imagenet":
        features = fe(x).logits
    elif fe_model_name == "clip":
        features = fe.encode_image(x)
    else:
        features = fe(x)
        
    return features

# TODO: Maybe also add later -> output_logits = self.classifier(features.reshape(features.size(0), -1)) in instructor model + advanced_classifier_flag = False in instructor training
def init_classifier(num_features, num_outputs, advanced_classifier_flag, complexity_level=4):
    # Complexity level: 0, 1, 2, 3, 4 (-> 4 is the original complexity level)
    
    if advanced_classifier_flag:
        # Define a more complex classifier
        complexity_factor = 2**complexity_level # -> 1, 2, 4, 8, 16
        classifier = nn.Sequential(
            nn.Linear(num_features, 128*complexity_factor), 
            nn.ELU(),  # Apply ELU activation
            nn.LayerNorm(128*complexity_factor),  # Apply Layer Normalization
            nn.Dropout(p=0.1),  # Add dropout with 10% dropout chance
            nn.Linear(128*complexity_factor, 32*complexity_factor),
            nn.ELU(),  # Apply ELU activation
            nn.LayerNorm(32*complexity_factor),  # Apply Layer Normalization
            nn.Dropout(p=0.1),  # Add dropout with 10% dropout chance
            nn.Linear(32*complexity_factor, num_outputs),
            # No activation here, outputting logits
        )  
    else:
        # Define the classification head
        classifier = nn.Linear(num_features, num_outputs)

    return classifier