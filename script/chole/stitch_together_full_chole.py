import random
from pathlib import Path
import json

import cv2
import pandas as pd

def create_combined_video(base_path, output_path, num_videos=10, num_phases=17, tissue_idx=1, after_phase_offset = 5, before_phase_offset = 5, phase_text_flag = True, also_right_camera_flag=False):
   
   # Create the parent directory if it does not exist
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
        
    # Create metadata file (with the parameters used for the video generation)
    metadata_file_path = Path(output_path) / "metadata.json"
    metadata = {
        "num_phases": num_phases,
        "after_phase_offset": after_phase_offset,
        "before_phase_offset": before_phase_offset
    }
    with open(metadata_file_path, 'w') as metadata_file:
        json.dump(metadata, metadata_file)
    
    for vid_index in range(num_videos):
        # Define the final video path for each run
        final_video_path = Path(output_path) / f"randomly_stitched_episode_tissue_{tissue_idx}_{vid_index + 1}.avi"
        
        # Create the parent directory if it does not exist
        if not final_video_path.parent.exists():
            final_video_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Define video writer
        out = None

        # Get the defined tissue folder
        tissue_folder_path = Path(base_path) / f"tissue_{tissue_idx}"

        # Iterate through each of phase folders
        for phase_idx in range(1, num_phases + 1):
            # Get the current phase folder
            phase_folder_start = f"{phase_idx}_*"        
            try:   
                phase_folder_path = list(tissue_folder_path.glob(phase_folder_start))[0]
            except IndexError:
                raise ValueError(f"No folder found for phase index {phase_idx}") 
            
            # Get all demo folders for that specific tissue and phase
            date_folders = list(phase_folder_path.glob('*-*'))
            if not date_folders:
                continue
            selected_date_folder_path = random.choice(date_folders)
            
            # Get number of frames from the kinematics csv file
            kinematics_csv_path = selected_date_folder_path / 'ee_csv.csv'
            if not kinematics_csv_path.exists():
                print(f"No kinematics csv file found for {selected_date_folder_path}")
                continue # Skip if no kinematics csv file found
            df = pd.read_csv(kinematics_csv_path)
            dataset_length = len(df)
            start, end = 0, dataset_length - 1
            
            # Check for "indices_curated.json" file (for more accurate start and end indices)
            indices_curated_file_path = selected_date_folder_path / "indices_curated.json"
            if indices_curated_file_path.exists():
                with open(indices_curated_file_path, 'r') as indices_curated_file:
                    try:
                        indices_curated_dict = json.load(indices_curated_file)
                    except json.JSONDecodeError:
                        print(f"Error reading indices_curated.json for {selected_date_folder_path}. Continue with max recording range.")

                    # Get start and end indices from the json file
                    if "start" in indices_curated_dict:
                        start = max(indices_curated_dict['start'] - before_phase_offset, start)
                    if "end" in indices_curated_dict:
                        end = min(indices_curated_dict['end'] + after_phase_offset, end)
            
            # Process images for each frame index
            for frame_idx in range(start, end + 1):
                images = []
                widths = []
                # Get image for both wrist cameras and left (and maybe right) image from stereo camera 
                if not also_right_camera_flag:
                    subfolder_file_suffix_mapping = [('endo_psm2', '_psm2.jpg'), ('left_img_dir', '_left.jpg'), ('endo_psm1', '_psm1.jpg')]
                else:
                    subfolder_file_suffix_mapping = [('endo_psm2', '_psm2.jpg'), ('left_img_dir', '_left.jpg'), ('right_img_dir', '_right.jpg'), ('endo_psm1', '_psm1.jpg')]
                for sub_folder, suffix in subfolder_file_suffix_mapping:
                    img_path = selected_date_folder_path / sub_folder / f"frame{str(frame_idx).zfill(6)}{suffix}"
                    if img_path.exists():
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            if sub_folder == 'left_img_dir' or sub_folder == 'right_img_dir':
                                height = 480
                                width = int(img.shape[1] * (height / img.shape[0]))
                                img = cv2.resize(img, (width, height))
                            images.append(img)
                            widths.append(img.shape[1])
                        else:
                            raise ValueError(f"Image corrupt for {img_path}")

                # Concatenate the images
                final_image = cv2.hconcat(images)
                
                if phase_text_flag:
                    # Calculate text position to center over the 'left_img_dir' image
                    text = f"Folder: {phase_folder_path.stem}"
                    text_position_x = widths[0] + (widths[1] // 2) - 325  # Center text on the second image
                    cv2.putText(final_image, text, (text_position_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Write the final image to the video
                if out is None:
                    h, w, _ = final_image.shape
                    out = cv2.VideoWriter(final_video_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (w, h))
                out.write(final_image)

        # Release the video writer
        if out:
            out.release()
        print(f"Video {vid_index + 1} saved at {final_video_path}")

if __name__ == "__main__":
    from pathlib import Path
    from datetime import datetime
    import os

    # Define the parameters for the video generation
    num_videos = 1
    num_phases = 17
    tissue = 4  # tissue_1 has no continuous phases, so we are using >= tissue_4
    before_phase_offset, after_phase_offset = 10, 10
    phase_text_flag = False
    also_right_camera_flag = False
    
    # Set the base and output path
    chole_scripts_path = Path(__file__).parent
    base_path =  os.path.join(os.getenv("PATH_TO_DATASET"), "base_chole_clipping_cutting")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_path = Path(os.path.join(chole_scripts_path, "GeneratedStitchedEpisodes", f"{tissue=}_{timestamp}"))
    
    # Generate the combined video
    create_combined_video(base_path, output_path, num_videos, num_phases, tissue, after_phase_offset, before_phase_offset, phase_text_flag, also_right_camera_flag)

