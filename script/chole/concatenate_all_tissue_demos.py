from pathlib import Path
import json

import cv2

def create_combined_video_all_demos(base_path, output_path, tissue_idx, timestamp, after_phase_offset = 5, before_phase_offset = 5,
                                    with_label_flag=True, desired_camera_names=None, tissue_prefix="tissue"):
    
    # Define the final video path for each run
    final_video_path = Path(output_path) / f"all_demos_combined_{tissue_prefix}_{tissue_idx}_{timestamp}.avi"
    
    # Create the parent directory if it does not exist
    if not final_video_path.parent.exists():
        final_video_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Define video writer
    out = None

    # Get the defined tissue folder
    tissue_folder_path = Path(base_path) / f"{tissue_prefix}_{tissue_idx}"

    # Get and sort phase folders based on the number before the first "_"
    phase_folders = [phase_folder for phase_folder in tissue_folder_path.glob('*_*') if phase_folder.name.split('_')[0].isdigit()]
    phase_folders_sorted = sorted(phase_folders, key=lambda p: int(p.name.split('_')[0]))

    # Iterate through each of the sorted phase folders
    for phase_folder_path in phase_folders_sorted:
        # Get all demo folders for that specific tissue and phase
        date_folders = list(phase_folder_path.glob('*-*'))
        if not date_folders:
            print(f"No demo folders found for {phase_folder_path}")
            continue

        date_folder_sorted = sorted(date_folders, key=lambda p: p.name)
        for selected_date_folder_path in date_folder_sorted:
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
            
            # Process images for each frame index
            for frame_idx in range(start, end + 1):
                images = []
                widths = []
                # Get image for both wrist cameras and left image from stereo camera 
                camera_name_suffix_mapping = [('endo_psm2', '_psm2.jpg'), 
                                            ('left_img_dir', '_left.jpg'), 
                                            ('endo_psm1', '_psm1.jpg')]
                if desired_camera_names:
                    camera_name_suffix_mapping = [(camera_name, camera_name_suffix) for camera_name, camera_name_suffix in camera_name_suffix_mapping if camera_name in desired_camera_names]
                for sub_folder, suffix in camera_name_suffix_mapping:
                    img_path = selected_date_folder_path / sub_folder / f"frame{str(frame_idx).zfill(6)}{suffix}"
                    if img_path.exists():
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            if sub_folder == 'left_img_dir' and desired_camera_names != ['left_img_dir']:
                                height = 480
                                width = int(img.shape[1] * (height / img.shape[0]))
                                img = cv2.resize(img, (width, height))
                            images.append(img)
                            widths.append(img.shape[1])
                        else:
                            raise ValueError(f"Image corrupt for {img_path}")
                    else:
                        print(f"Image not found for {img_path}")

                # Concatenate the images
                final_image = cv2.hconcat(images)
                
                if with_label_flag:
                    # Calculate text position to center over the 'left_img_dir' image
                    text_position_x = 0  # Center text on the second image
                    text = f"Phase: {phase_folder_path.stem}, Demo: {selected_date_folder_path.stem}"
                    cv2.putText(final_image, text, (text_position_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Write the final image to the video
                if out is None:
                    h, w, _ = final_image.shape
                    out = cv2.VideoWriter(final_video_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (w, h))
                out.write(final_image)

    # Release the video writer
    if out:
        out.release()
    print(f"Saved all concatenated demos video for tissue {tissue_idx} in {final_video_path}")

if __name__ == "__main__":
    from datetime import datetime
    import os

    tissue_indices = [50] # [1,4,5,6,8,12,13] # [14,18,19,22,23,30,32,35,39,40,41,47,49]
    tissue_prefix = "tissue" # "tissue" "phantom"
    dataset_name = "base_chole_clipping_cutting" # "base_chole_clipping_cutting" "phantom_chole" "debugging" "debugging2"

    # Set the base and output path
    chole_scripts_path = Path(__file__).parent
    base_path =  os.path.join(os.getenv("PATH_TO_DATASET"), dataset_name)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_path = Path(os.path.join(chole_scripts_path, "AllDemosVideos"))

    # Generate the combined video
    before_phase_offset, after_phase_offset = 0, 10
    with_label_flag = True
    desired_camera_names = None # ["left_img_dir"] # None 
    for tissue_idx in tissue_indices:
        create_combined_video_all_demos(base_path, output_path, tissue_idx, timestamp, after_phase_offset, before_phase_offset, with_label_flag, desired_camera_names, tissue_prefix)
