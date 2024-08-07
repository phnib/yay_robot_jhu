
import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def visualize_robot_trajectory(qpos):

    factor = 1000
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    cutting_start = 208  #145
    cutting_end = 213  #149
    # ax.scatter(actions_psm1[:, 0], actions_psm1[:, 1], actions_psm1[:, 2], c ='r')
    # ax.scatter(actions_psm2[:, 0]*factor, actions_psm2[:, 1]*factor, actions_psm2[:, 2]*factor, c ='g', label = 'Generated trajectory')
    # ax.scatter(qpos_psm1[0], qpos_psm1[1], qpos_psm1[2], c = 'b')
    ax.scatter(qpos[0:cutting_start, 0]*factor, qpos[0:cutting_start, 1]*factor, qpos[0:cutting_start, 2]*factor, c = 'b', label = 'PSM1 position')
    ax.scatter(qpos[cutting_start:cutting_end, 0]*factor, qpos[cutting_start:cutting_end, 1]*factor, qpos[cutting_start:cutting_end, 2]*factor, c = 'r', label = 'PSM1 cutting')
    ax.scatter(qpos[cutting_end:, 0]*factor, qpos[cutting_end:, 1]*factor, qpos[cutting_end:, 2]*factor, c = 'k', label = 'PSM1 after cutting')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    n_bins = 7
    ax.legend()
    ax.xaxis.set_major_locator(plt.MaxNLocator(n_bins))
    ax.yaxis.set_major_locator(plt.MaxNLocator(n_bins))
    ax.zaxis.set_major_locator(plt.MaxNLocator(n_bins))
    plt.show()

def generate_cutting_motion(csv_df, filename, repeating_num):
    # Define the constants
    closing_angle = -0.349096
    closing_rate = 0.5
    if csv_df.empty:
        print("The DataFrame is empty. Please check the CSV file.")
        return
    # Get the last row
    last_row = csv_df.iloc[-1].to_dict()
    
    # Create a list to hold new rows
    new_rows = []
    
    for i in range(repeating_num):
        jaw_angle = deepcopy(last_row["psm1_jaw"])

        if jaw_angle > 0:
            last_row["psm1_jaw"] = jaw_angle - closing_rate * i
            # print(jaw_angle - closing_rate * i)
        else:
            last_row["psm1_jaw"] = closing_angle
            # print(closing_angle)
        
        new_rows.append(last_row.copy())

    # Convert new rows to a DataFrame
    new_rows_df = pd.DataFrame(new_rows)
    
    # Concatenate the original data and new rows
    csv_df = pd.concat([csv_df, new_rows_df], ignore_index=True)
    csv_df.to_csv(filename, index=False)

    # input("enter to continue...")
def generate_cutting_motion_sp(csv_df, filename, repeating_num):
    # Define the constants
    closing_angle = -0.349096
    closing_rate = 0.5
    if csv_df.empty:
        print("The DataFrame is empty. Please check the CSV file.", filename)
        return
    # Update the last 10 rows of 'psm1_jaw_sp'
    for i in range(2, repeating_num+1):
        csv_df.loc[csv_df.index[-i], 'psm1_jaw_sp'] = max(csv_df.loc[csv_df.index[-i+1], 'psm1_jaw'] - 0.1, closing_angle)

    csv_df.loc[csv_df.index[-1], 'psm1_jaw_sp'] = max(csv_df.loc[csv_df.index[-1], 'psm1_jaw'] - 0.1, closing_angle)

    # Save the updated DataFrame to a new CSV file
    csv_df.to_csv(filename, index=False)

def remove_extra_row(csv_df, filename, sample_path, repeating_num):
    image_dir = os.path.join(sample_path, "left_img_dir")
    image_list = os.listdir(image_dir)
    
    # Count the number of images
    num_images = len(image_list)
    
    # Calculate the threshold for removing redundant rows
    threshold = num_images + repeating_num
    
    # If the number of rows in the DataFrame exceeds the threshold, remove the redundant rows
    if len(csv_df) > threshold:
        print(f"Removing redundant rows from {filename}, csv length: {len(csv_df)}, num of img: {num_images}")
        csv_df = csv_df.iloc[:threshold]

    csv_df.to_csv(filename, index=False)


if __name__ == "__main__":
    tissue_ids = [53]
    # dataset_path = "/home/imerse/chole_ws/data/phantom_chole/phantom_1/ACTUAL_CUTTING_right"
    phases = ["8_go_to_the_cutting_position_left_tube",
              "8_go_to_the_cutting_position_left_tube_recovery", 
              "16_go_to_the_cutting_position_right_tube",
              "16_go_to_the_cutting_position_right_tube_recovery"]
    
    for tissue_id in tissue_ids:
        for phase in phases:

            dataset_path = f"/home/imerse/chole_ws/data/base_chole_clipping_cutting/tissue_{tissue_id}/{phase}"
            if not os.path.exists(dataset_path):
                print(f"dataset path not found in {dataset_path}")
                continue
            samples = os.listdir(dataset_path)
            for sample in samples:
                sample_dir = os.path.join(dataset_path, sample)

                if not os.path.exists(os.path.join(sample_dir, "ee_csv.csv")):
                    print(f"ee state csv file not found in {sample_dir}")
                    continue

                csv_path = os.path.join(sample_dir, "ee_csv.csv")
                csv = pd.read_csv(csv_path)

                generate_cutting_motion(csv, csv_path, 10)
                # remove_extra_row(csv, csv_path, sample_dir, 10)

            print(f"Done generating cutting for tissue {tissue_id} in phase {phase}")

    for tissue_id in tissue_ids:
        for phase in phases:

            dataset_path = f"/home/imerse/chole_ws/data/base_chole_clipping_cutting/tissue_{tissue_id}/{phase}"
            if not os.path.exists(dataset_path):
                print(f"dataset path not found in {dataset_path}")
                continue
            samples = os.listdir(dataset_path)
            for sample in samples:
                sample_dir = os.path.join(dataset_path, sample)

                if not os.path.exists(os.path.join(sample_dir, "ee_csv.csv")):
                    print(f"ee state csv file not found in {sample_dir}")
                    continue

                csv_path = os.path.join(sample_dir, "ee_csv.csv")
                csv = pd.read_csv(csv_path)

                remove_extra_row(csv, csv_path, sample_dir, 10)

            print(f"Done removing extra rows for tissue {tissue_id} in phase {phase}")

    
    for tissue_id in tissue_ids:
        for phase in phases:

            dataset_path = f"/home/imerse/chole_ws/data/base_chole_clipping_cutting/tissue_{tissue_id}/{phase}"
            if not os.path.exists(dataset_path):
                print(f"dataset path not found in {dataset_path}")
                continue
            samples = os.listdir(dataset_path)
            for sample in samples:
                sample_dir = os.path.join(dataset_path, sample)

                if not os.path.exists(os.path.join(sample_dir, "ee_csv.csv")):
                    print(f"ee state csv file not found in {sample_dir}")
                    continue

                csv_path = os.path.join(sample_dir, "ee_csv.csv")
                csv = pd.read_csv(csv_path)

                generate_cutting_motion_sp(csv, csv_path, 10)

            print(f"Done generating sp for tissue {tissue_id} in phase {phase}")

    print("job finished")
