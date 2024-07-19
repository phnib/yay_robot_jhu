
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



if __name__ == "__main__":
    tissue_ids = [1, 2]
    # dataset_path = "/home/imerse/chole_ws/data/phantom_chole/phantom_1/ACTUAL_CUTTING_right"
    phase = "16_go_to_the_cutting_position_right_tube_recovery"
    for tissue_id in tissue_ids:
        dataset_path = f"/home/imerse/chole_ws/data/phantom_chole/phantom_{tissue_id}/{phase}"
        samples = os.listdir(dataset_path)
        for sample in samples:
            sample_dir = os.path.join(dataset_path, sample)

            if not os.path.exists(os.path.join(sample_dir, "ee_csv.csv")):
                print(f"ee state csv file not found in {sample_dir}")
                exit


            csv_path = os.path.join(sample_dir, "ee_csv.csv")
            csv = pd.read_csv(csv_path)

            header_name_qpos_psm1 = ["psm1_pose.position.x", "psm1_pose.position.y", "psm1_pose.position.z",
                                "psm1_pose.orientation.x", "psm1_pose.orientation.y", "psm1_pose.orientation.z", "psm1_pose.orientation.w",
                                "psm1_jaw"]
            
            episode_len = len(csv)
            # print("episode_len: ", episode_len)
            qpos_psm1 = csv[header_name_qpos_psm1].iloc[:, 0:episode_len].to_numpy()
            # print(qpos_psm1.shape)
            # visualize_robot_trajectory(qpos_psm1)
            generate_cutting_motion(csv, csv_path, 10)
        print(f"Done for tissue {tissue_id}")

    print("job finished")
