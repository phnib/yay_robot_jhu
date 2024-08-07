import os

### Dataset parameters
DATA_DIR = os.getenv("PATH_TO_DATASET")

# Note: Choose the dataset name as the dataset dir folder name
DATASET_CONFIGS = {
    "base_chole_clipping_cutting": {
        "dataset_dir": os.path.join(DATA_DIR, "base_chole_clipping_cutting"),
        "num_episodes": 1000,
        "incomplete_tissue_samples": ["tissue_1", "tissue_4", "phantom_1"], # Should not be used for HL policy training - good for LL and ML policy training 
        "camera_names": ["endo_psm2", "left_img_dir", "right_img_dir", "endo_psm1"], 
        "camera_file_suffixes": ["_psm2.jpg", "_left.jpg", "_right.jpg", "_psm1.jpg"],
        "after_phase_offset": 10,
        "before_phase_offset": 0, 
        "exclude_grabbing_gallbladder_tissues": [f"tissue_{idx}" for idx in [1,4,5,6,8,12,13]],
        "correct_psm1_rotation_tissues": [f"tissue_{idx}" for idx in [5, 6, 8, 12, 13, 14, 18]]
    },
    "phantom_chole": {
        "dataset_dir": os.path.join(DATA_DIR, "phantom_chole"),
        "num_episodes": 500, 
        "incomplete_tissue_samples": [], # Should not be used for HL policy training - good for LL and ML policy training 
        "camera_names": ["endo_psm2", "left_img_dir", "right_img_dir", "endo_psm1"], 
        "camera_file_suffixes": ["_psm2.jpg", "_left.jpg", "_right.jpg", "_psm1.jpg"],
        "after_phase_offset": 10,
        "before_phase_offset": 10,   
    },
}