import requests
import sys
import os

import torch

from PIL import Image
from pathlib import Path

# import aloha 
path_to_yay_robot = os.getenv('PATH_TO_YAY_ROBOT')
if path_to_yay_robot:
    sys.path.append(os.path.join(path_to_yay_robot, 'src'))
else:
    raise EnvironmentError("Environment variable PATH_TO_YAY_ROBOT is not set")

from prismatic import load

# For gated LMs like Llama-2, make sure to request official access, and generate an access token
hf_token = Path(".hf_token").read_text().strip() # TODO: How does it work with HF token? Env var?
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load a pretrained VLM (either local path, or ID to auto-download from the HF Hub) 
# TODO: Load here Sam's model (with bfloat16 precision)
model_id = "prism-dinosiglip+7b"
vlm = load(model_id, hf_token=hf_token)
vlm.to(device, dtype=torch.bfloat16)

# Download an image and specify a prompt
# TODO: Add here an image URL - check for the data format
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png" 
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
user_prompt = "Which medical procedure is performed on this image?"

# TODO: Think of the actual user prompt
hl_policy_first_prompt = "You're an AI assistant guiding a bimanual robot that performs a cholecystectomy saying which task is it currently performing. Every 200 timesteps (4 seconds), you can issue one instruction from a provided list. You'll receive images at these intervals from four cameras: top, front, left wrist, and right wrist (left to right). You will select your instruction from the following list:"
hl_policy_prompt = ""

# Build prompt
prompt_builder = vlm.get_prompt_builder()
prompt_builder.add_turn(role="human", message=user_prompt)
prompt_text = prompt_builder.get_prompt()

# Generate!
# TODO: Measure the time it needs to generate the text
generated_text = vlm.generate(
    image,
    prompt_text,
    do_sample=True,
    temperature=0.4,
    max_new_tokens=512,
    min_length=1,
)