import random
from pathlib import Path
import json

import pandas as pd
import cv2

def create_combined_video(base_path, output_path, num_videos=10, num_phases=17, tissue_idx=1, after_phase_offset = 5, before_phase_offset = 5, phase_text_flag = True, desired_camera_names=None, save_each_n_frame_as_image_wo_text=None):
   
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
        video_name = f"randomly_stitched_episode_tissue_{tissue_idx}_{vid_index + 1}"
        final_video_path = Path(output_path) / f"{video_name}.avi"
        
        # Create the parent directory if it does not exist
        if not final_video_path.parent.exists():
            final_video_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Define video writer
        out = None

        # Get the defined tissue folder
        tissue_folder_path = Path(base_path) / f"tissue_{tissue_idx}"

        # Create the full kinematics episode csv file
        full_kinematics_episode_output_path = Path(output_path) / "full_kinematics_episode.csv"
        full_kinematics_episode = pd.DataFrame()

        if save_each_n_frame_as_image_wo_text:
            # Create a directory for the current tissue and video index
            frames_folder_output_path = Path(output_path) / f"{video_name}_frames"
            if not frames_folder_output_path.exists():
                frames_folder_output_path.mkdir(parents=True, exist_ok=True)

            # Create log file to put in current time stamp, psm2 and psm1 jaw information every n frames (to input into VLM)
            yaw_info_log_file_path = Path(output_path) / "yaw_info_log.txt"

            episode_frame_idx = 0

        # Iterate through each of phase folders
        for phase_idx in range(1, num_phases + 1):
            # Get the current phase folder
            phase_folder_start = f"{phase_idx}_*[^recovery]" # Exclude recovery phases - for now (as preventing continous stitching)
            try:   
                phase_folder_path = list(tissue_folder_path.glob(phase_folder_start))[0]
            except IndexError:
                raise ValueError(f"No folder found for phase index {phase_idx}") 
            
            # Get all demo folders for that specific tissue and phase
            date_folders = list(phase_folder_path.glob('*-*'))
            if not date_folders:
                continue
            selected_date_folder_path = random.choice(date_folders)
            
            # Get number of frames from the left image directory
            left_img_dir_path = selected_date_folder_path / "left_img_dir"
            if not left_img_dir_path.exists():
                print(f"No left image directory found for {selected_date_folder_path}")
                continue
            dataset_length = len(list(left_img_dir_path.glob("*.jpg")))
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
            
            # Get number of frames from the kinematics csv file
            kinematics_csv_path = selected_date_folder_path / 'ee_csv.csv'
            if not kinematics_csv_path.exists():
                print(f"No kinematics csv file found for {selected_date_folder_path}")
                continue # Skip if no kinematics csv file found
            demo_kinematics = pd.read_csv(kinematics_csv_path)
            valid_demo_kinematics = demo_kinematics.iloc[start:end + 1]
            # Concatenate the kinematics data
            full_kinematics_episode = pd.concat([full_kinematics_episode, valid_demo_kinematics], ignore_index=True)            
            
            # Process images for each frame index
            for frame_idx in range(start, end + 1):
                images = []
                widths = []
                # Get image for both wrist cameras and left (and maybe right) image from stereo camera 
                subfolder_file_suffix_mapping = [('endo_psm2', '_psm2.jpg'), ('left_img_dir', '_left.jpg'), ('right_img_dir', '_right.jpg'), ('endo_psm1', '_psm1.jpg')]
                if desired_camera_names:
                    subfolder_file_suffix_mapping = [(camera_name, camera_name_suffix) for camera_name, camera_name_suffix in subfolder_file_suffix_mapping if camera_name in desired_camera_names]
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
                    else:
                        raise ValueError(f"Image not found for {img_path}")

                # Concatenate the images
                final_image = cv2.hconcat(images)
                
                if save_each_n_frame_as_image_wo_text and episode_frame_idx % save_each_n_frame_as_image_wo_text == 0:
                    # Save the image without text
                    frame_output_path = frames_folder_output_path / f"frame{str(episode_frame_idx).zfill(6)}.jpg"
                    cv2.imwrite(str(frame_output_path), final_image)
                    
                    # Save the jaw information for psm2 and psm1
                    psm2_yaw = valid_demo_kinematics.iloc[frame_idx]['psm2_jaw']
                    psm1_yaw = valid_demo_kinematics.iloc[frame_idx]['psm1_jaw']
                    with open(yaw_info_log_file_path, 'a') as yaw_info_log_file:
                        yaw_info_log_file.write(f"t={episode_frame_idx}, jaw-psm2: {psm2_yaw}, jaw-psm1: {psm1_yaw}\n")

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

                episode_frame_idx += 1

        # Release the video writer
        if out:
            out.release()
        print(f"Video {vid_index + 1} saved at {final_video_path}")
        
        # Save the full kinematics episode to a csv file
        full_kinematics_episode.to_csv(full_kinematics_episode_output_path, index=False)
        print(f"Full kinematics episode saved at {full_kinematics_episode_output_path}")

if __name__ == "__main__":
    from pathlib import Path
    from datetime import datetime
    import os

    # Define the parameters for the video generation
    num_videos = 1
    num_phases = 17
    tissue = 12  # tissue_1 has no continuous phases, so we are using >= tissue_4
    before_phase_offset, after_phase_offset = 0, 10
    phase_text_flag = False
    desired_camera_names = ["endo_psm2", "left_img_dir", "endo_psm1"] # ["endo_psm2", "left_img_dir", "endo_psm1"] # ["left_img_dir"] # None ["left_img_dir", "endo_psm1"] ["endo_psm2", "left_img_dir", "endo_psm1"]
    save_each_n_frame_as_image_wo_text = 30
    
    # Set the base and output path
    chole_scripts_path = Path(__file__).parent
    base_path =  os.path.join(os.getenv("PATH_TO_DATASET"), "base_chole_clipping_cutting")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_path = Path(os.path.join(chole_scripts_path, "GeneratedStitchedEpisodes", f"{tissue=}_{timestamp}"))
    
    # Generate the combined video
    create_combined_video(base_path, output_path, num_videos, num_phases, tissue, after_phase_offset, before_phase_offset, phase_text_flag, desired_camera_names, save_each_n_frame_as_image_wo_text)

