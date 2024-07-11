from pathlib import Path
import json

import cv2
import pandas as pd

def create_combined_video_all_demos(base_path, output_path, tissue_idx, timestamp, after_phase_offset = 5, before_phase_offset = 5):
    
    # Define the final video path for each run
    final_video_path = Path(output_path) / f"all_demos_combined_{tissue_idx}_{timestamp}.avi"
    
    # Create the parent directory if it does not exist
    if not final_video_path.parent.exists():
        final_video_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Define video writer
    out = None

    # Get the defined tissue folder
    tissue_folder_path = Path(base_path) / f"tissue_{tissue_idx}"

    # Get and sort phase folders based on the number before the first "_"
    phase_folders = sorted(tissue_folder_path.glob('*_*'), key=lambda p: int(p.name.split('_')[0]))

    # Iterate through each of the sorted phase folders
    for phase_folder_path in phase_folders:
        # Get all demo folders for that specific tissue and phase
        date_folders = list(phase_folder_path.glob('*-*'))
        if not date_folders:
            print(f"No demo folders found for {phase_folder_path}")
            continue

        for selected_date_folder_path in date_folders:
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
                # Get image for both wrist cameras and left image from stereo camera 
                for sub_folder, suffix in [('endo_psm2', '_psm2.jpg'), 
                                            ('left_img_dir', '_left.jpg'), 
                                            ('endo_psm1', '_psm1.jpg')]:
                    img_path = selected_date_folder_path / sub_folder / f"frame{str(frame_idx).zfill(6)}{suffix}"
                    if img_path.exists():
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            if sub_folder == 'left_img_dir':
                                height = 480
                                width = int(img.shape[1] * (height / img.shape[0]))
                                img = cv2.resize(img, (width, height))
                            images.append(img)
                            widths.append(img.shape[1])
                        else:
                            raise ValueError(f"Image corrupt for {img_path}")
                    else:
                        print(f"Image not found for {img_path}")
                        continue

                # Concatenate the images
                final_image = cv2.hconcat(images)
                
                # Calculate text position to center over the 'left_img_dir' image
                text_position_x = widths[0] + (widths[1] // 2) - 500  # Center text on the second image
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
    
    # Set the base and output path
    chole_scripts_path = Path(__file__).parent
    base_path =  os.path.join(os.getenv("PATH_TO_DATASET"), "base_chole_clipping_cutting")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_path = Path(os.path.join(chole_scripts_path, "AllDemosVideos"))

    # Generate the combined video
    tissue_idx = 1
    before_phase_offset, after_phase_offset = 0, 0 # 5, 5
    create_combined_video_all_demos(base_path, output_path, tissue_idx, timestamp, after_phase_offset, before_phase_offset)
