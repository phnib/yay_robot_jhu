import os
import shutil

def copy_files_with_suffix(input_folder, target_filename, suffix):
    # Iterate through the directory and its subdirectories
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file == target_filename:
                # Construct the full file path
                file_path = os.path.join(root, file)
                
                # Split the file name and extension
                file_name, file_ext = os.path.splitext(file)
                
                # Create the new file name with the suffix
                new_file_name = f"{file_name}{suffix}{file_ext}"
                
                # Construct the new file path
                new_file_path = os.path.join(root, new_file_name)
                
                # Copy the file with the new name
                shutil.copy2(file_path, new_file_path)
                print(f"Copied: {file_path} to {new_file_path}")

def copy_files_with_parent_structure(input_folder, target_filename, output_folder):
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate through the directory and its subdirectories
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file == target_filename:
                # Calculate relative path to maintain directory structure
                rel_dir = os.path.relpath(root, input_folder)
                dest_dir = os.path.join(output_folder, rel_dir)
                
                # Ensure the destination directory exists
                os.makedirs(dest_dir, exist_ok=True)
                
                # Copy the file
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dest_dir, file)
                shutil.copy2(src_file, dst_file)
                print(f"Copied: {src_file} to {dst_file}")

if __name__ == "__main__":
    # tissue_idx = 14
    # tissue_prefix = "tissue" # "tissue" "phantom"
    # tissue_name = f"{tissue_prefix}_{str(tissue_idx)}"
    # dataset_name = "base_chole_clipping_cutting" # "base_chole_clipping_cutting" "phantom_chole"

    # # Set the base and output path
    # tissue_folder_path =  os.path.join(os.getenv("PATH_TO_DATASET"), dataset_name, tissue_name)
    # target_filename = "indices_curated.json"
    # suffix = "_pull_clip"
    
    # # Make copy of all file version of specific file (e.g., indices_curated.json)
    # copy_files_with_suffix(tissue_folder_path, target_filename, suffix)

    # ----------

    dataset_name = "base_chole_clipping_cutting" # "base_chole_clipping_cutting" "phantom_chole" 
    dataset_folder = os.path.join(os.getenv("PATH_TO_DATASET"), dataset_name)
    target_filename = "indices_curated.json"
    output_folder = os.path.join(os.getenv("PATH_TO_DATASET"), "curated_indices_data")

    # Copy files with parent directory structure - e.g., for all indices_curated.json files after data curation to transfer to other computers
    copy_files_with_parent_structure(dataset_folder, target_filename, output_folder)
