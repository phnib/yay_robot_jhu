import os
import json

def get_duct_mapping(info_json_path):
    with open(info_json_path, 'r') as file:
        data = json.load(file)
    return data['anatomy']['duct']

def rename_folders(base_path, reverse=False):
    
    # --- Get duct mapping ---
    info_json_path = os.path.join(base_path, 'info.json')
    if os.path.exists(info_json_path):
        duct_mapping = get_duct_mapping(info_json_path)
    else:
        duct_mapping = "left" # Left by default
    
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path):
            # --- Rename folders ---            
            if reverse:
                new_folder_name = folder_name.replace("left_tube", "duct").replace("right_tube", "artery") if duct_mapping == "left" else folder_name.replace("right_tube", "duct").replace("left_tube", "artery")
            else:
                new_folder_name = folder_name.replace("duct", "left_tube").replace("artery", "right_tube") if duct_mapping == "left" else folder_name.replace("duct", "right_tube").replace("artery", "left_tube")
            
            new_folder_path = os.path.join(base_path, new_folder_name)
            
            if new_folder_name != folder_name:
                os.rename(folder_path, new_folder_path)
                print(f"Renamed '{folder_name}' to '{new_folder_name}'")

if __name__ == "__main__":
    dataset_name = "debugging3"
    reverse = True  # Set this to True if you want to reverse the renaming
    chole_dataset_path = os.getenv("PATH_TO_DATASET")
    dataset_path = os.path.join(chole_dataset_path, dataset_name)
    tissue_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    for tissue_name in tissue_folders:
        tissue_folder_path = os.path.join(chole_dataset_path, dataset_name, tissue_name)
        rename_folders(tissue_folder_path, reverse)