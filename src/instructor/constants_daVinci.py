import os

### Dataset parameters
DATA_DIR = os.getenv("PATH_TO_DATASET")

# Note: Choose the dataset name as the dataset dir folder name
DATASET_CONFIGS = {
    "base_chole_clipping_cutting": {
        "dataset_dir": os.path.join(DATA_DIR, "base_chole_clipping_cutting"),
        "num_episodes": 2500,
        "incomplete_tissue_samples": ["tissue_1", "phantom_1"], # Should not be used for HL policy training - good for LL and ML policy training 
        "camera_names": ["endo_psm2", "left_img_dir", "right_img_dir", "endo_psm1"], 
        "camera_file_suffixes": ["_psm2.jpg", "_left.jpg", "_right.jpg", "_psm1.jpg"],
        "after_phase_offset": 6,
        "before_phase_offset": 6, 
    },
    "phantom_chole": {
        "dataset_dir": os.path.join(DATA_DIR, "phantom_chole"),
        "num_episodes": 750, 
        "incomplete_tissue_samples": [], # Should not be used for HL policy training - good for LL and ML policy training 
        "camera_names": ["endo_psm2", "left_img_dir", "right_img_dir", "endo_psm1"], 
        "camera_file_suffixes": ["_psm2.jpg", "_left.jpg", "_right.jpg", "_psm1.jpg"],
        "after_phase_offset": 10,
        "before_phase_offset": 10,   
    },
}