import random
import numpy as np
import torch
from torchvision import transforms

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def center_crop_resize(img, size):
    # Center crop the image to biggest possible square and resize it to the desired size
    
    min_img_dim = min(img.shape[1:])
    transform = transforms.Compose([
        transforms.CenterCrop(min_img_dim),
        transforms.Resize(size)
    ])
    img_transformed = transform(img)
    return img_transformed