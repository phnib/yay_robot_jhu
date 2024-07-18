import requests
from pathlib import Path
from tqdm import tqdm

# List of model URLs
model_urls = [
    "https://s3.unistra.fr/camma_public/github/selfsupsurg/models/model_final_checkpoint_moco_v2_surg.torch",
    "https://s3.unistra.fr/camma_public/github/selfsupsurg/models/model_final_checkpoint_simclr_surg.torch",
    "https://s3.unistra.fr/camma_public/github/selfsupsurg/models/model_final_checkpoint_swav_surg.torch",
    "https://s3.unistra.fr/camma_public/github/selfsupsurg/models/model_final_checkpoint_dino_surg.torch"
]

# Create the 'models' directory if it does not exist
models_dir = Path(__file__).parent / Path('models')
models_dir.mkdir(parents=True, exist_ok=True)

# Function to download a file from a URL with progress bar
def download_file(url, folder):
    local_filename = folder / url.split('/')[-1]
    
    # Stream the request
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get the total file size
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kilobyte
    
    # Use TQDM to create a progress bar
    t = tqdm(total=total_size, unit='iB', unit_scale=True, desc=local_filename.name)
    
    with open(local_filename, 'wb') as f:
        for chunk in response.iter_content(block_size):
            t.update(len(chunk))
            f.write(chunk)
    t.close()
    
    if total_size != 0 and t.n != total_size:
        print("ERROR: Something went wrong")
    else:
        print(f"Downloaded {local_filename}")

# Download each model
for model_url in model_urls:
    download_file(model_url, models_dir)
