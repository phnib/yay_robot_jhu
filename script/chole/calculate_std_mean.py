import os
import numpy as np
import pandas as pd
from natsort import natsorted
from scipy.spatial.transform import Rotation as R
from pytransform3d import rotations, batch_rotations, transformations, trajectories

def compute_relative_actions_in_SE3(qpos, action):
    """
    Note: this is the proper implementation
    qpos: current position (measured_cp), xyz, xyzw, jaw angle (8-dim vector)
    action: set point on the dvrk (action_horizon x 8)
    
    returns: relative position and rotation w.r.t qpos
    """
    
    diff = np.zeros((action.shape[0], 10)) # TODO: hard-coded dim (10) for a single arm

    # convert current pose to SE(3)
    qpos_wxyz = rotations.quaternion_wxyz_from_xyzw(qpos[3:7])
    qpos_py3d = np.concatenate((qpos[0:3], qpos_wxyz))
    g_qpos = transformations.transform_from_pq(qpos_py3d) # no jaw angle!

    # convert actions to SE(3)
    action_wxyz = batch_rotations.batch_quaternion_wxyz_from_xyzw(action[:, 3:7]) 
    action_py3d = np.concatenate((action[:, 0:3], action_wxyz), axis = 1)
    g_action = trajectories.transforms_from_pqs(action_py3d)

    # invert current pose
    g_qpos_inv = transformations.invert_transform(g_qpos)
    diff_SE3 = trajectories.concat_one_to_many(g_qpos_inv, g_action)

    # construct 6d rot
    diff_6d = diff_SE3[:,0:3,:2]
    diff_6d = diff_6d.transpose(0,2,1).reshape(-1, 6) # first column then second column
    
    # fill in translation elements
    diff[:, 0:3] = diff_SE3[:, 0:3, 3] # replace the translations with the last column first three rows of SE3
    # fill in 6d rot
    diff[:, 3:9] = diff_6d
    # fill in jaw angle (note: jaw angle is absolute, not relative)
    diff[:, 9] = action[:, 7]
    return diff

def compute_diffs(ids, data_dir, chunk_size=100, phantoms=False):
    cp_psm1 = [ "psm1_pose.position.x", "psm1_pose.position.y", "psm1_pose.position.z",
            "psm1_pose.orientation.x", "psm1_pose.orientation.y", "psm1_pose.orientation.z", "psm1_pose.orientation.w",
            "psm1_jaw"]

    sp_psm1 = ["psm1_sp.position.x", "psm1_sp.position.y", "psm1_sp.position.z",
            "psm1_sp.orientation.x", "psm1_sp.orientation.y", "psm1_sp.orientation.z", "psm1_sp.orientation.w",
            "psm1_jaw_sp"]

    cp_psm2 = [ "psm2_pose.position.x", "psm2_pose.position.y", "psm2_pose.position.z",
            "psm2_pose.orientation.x", "psm2_pose.orientation.y", "psm2_pose.orientation.z", "psm2_pose.orientation.w",
            "psm2_jaw"]

    sp_psm2 = ["psm2_sp.position.x", "psm2_sp.position.y", "psm2_sp.position.z",
            "psm2_sp.orientation.x", "psm2_sp.orientation.y", "psm2_sp.orientation.z", "psm2_sp.orientation.w",
            "psm2_jaw_sp"]

    t = 0
    samples = {}

    for id in ids:
        samples[id] = {}
        if phantoms:
            root = os.path.join(data_dir, f"phantom_{id}")
        else:
            root = os.path.join(data_dir, f"tissue_{id}")
        dirlist = [item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]
        dirlist = natsorted(dirlist)

        total_demo_num = 0
        for dir in dirlist:
            phase = os.path.join(root, dir)
            samples[id][dir] = []
            for item in os.listdir(phase):
                samples[id][dir].append(item)
            total_demo_num += len(samples[id][dir])
        t += total_demo_num
        print(id, ", total demo num =", total_demo_num)
    print("total demo num =", t)
    
    diffs = []

    for id in ids:
        print("id:", id)
        if phantoms:
            root = os.path.join(data_dir, f"phantom_{id}")
        else:
            root = os.path.join(data_dir, f"tissue_{id}")
        dirlist = [item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]
        dirlist = natsorted(dirlist)
        for phase in samples[id].keys():

            sample = samples[id][phase]
            for s in sample:
                if s == "Corrections":
                    sample_dir = os.path.join(root, phase, s)
                    new_sample = os.listdir(sample_dir)
                    for ss in new_sample:
                        sample_dir = os.path.join(sample_dir, ss)
                        break
                    pth = os.path.join(sample_dir, "ee_csv.csv")
                else:
                    pth = os.path.join(root, phase, s, "ee_csv.csv")
                csv = pd.read_csv(pth)

                for jj in range(len(csv)):
                    
                    first_el_psm1 = csv[cp_psm1].iloc[jj, :].to_numpy()
                    chunk_el_psm1 = csv[sp_psm1].iloc[jj:jj+chunk_size, :].to_numpy()
                    diff_psm1 = compute_relative_actions_in_SE3(first_el_psm1, chunk_el_psm1)
                    
                    first_el_psm2 = csv[cp_psm2].iloc[jj, :].to_numpy()
                    chunk_el_psm2 = csv[sp_psm2].iloc[jj:jj+chunk_size, :].to_numpy()
                    diff_psm2 = compute_relative_actions_in_SE3(first_el_psm2, chunk_el_psm2)

                    diff_stacked = np.column_stack((diff_psm1, diff_psm2))
                    diffs.append(diff_stacked)

        print(len(diffs))

    diffs_np = np.concatenate(diffs, axis=0)
    mean = diffs_np.mean(axis=0)
    std = diffs_np.std(axis=0).clip(1e-2, 10)
    min = diffs_np.min(axis = 0)
    max = diffs_np.max(axis = 0)

    return mean, std, min, max

# Define the main function to generate the task configuration file
def generate_task_config():
    phantoms = False
    if phantoms:
        ids = [1, 2, 3]
        data_dir = "/home/imerse/chole_ws/data/phantom_chole/"
    else:
        ids = [4, 5, 6, 8, 12, 13, 14, 18, 19, 22, 23, 30, 32, 35, 39, 40]
        data_dir = "/home/imerse/chole_ws/data/base_chole_clipping_cutting/"
    
    mean, std, min, max = compute_diffs(ids, data_dir)

    std_str = ', '.join(map(str, std))
    mean_str = ', '.join(map(str, mean))
    min_str = ', '.join(map(str, min))
    max_str = ', '.join(map(str, max))

    print("mean:", mean_str)
    print("std:", std_str)
    print("min:", min_str)
    print("max:", max_str)

    # write the results into a txt file
    with open("./std_mean.txt", "w") as f:
        f.write(f"tissue ids: {ids}\n")
        f.write(f"mean: {mean_str}\n")
        f.write(f"std: {std_str}\n")
        f.write(f"min: {min_str}\n")
        f.write(f"max: {max_str}\n")


# Run the main function to generate the task configuration
generate_task_config()
