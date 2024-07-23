import pathlib
import numpy as np
import os

"""
Directions: set task name, action mode (e.g. hybrid, ego, etc), and the correct normalization params (usually just mean/std).
Make sure action mode is exactly what you want!
"""


### Task parameters
# DATA_DIR = "/home/imerse/chole_ws/data"
DATA_DIR = os.getenv("PATH_TO_DATASET")
TASK_CONFIGS = {
    'phantom_chole':{
        'dataset_dir': DATA_DIR + "/phantom_chole/",
        'num_episodes': 1444,
        'phantom': True,
        'tissue_samples_ids': [1, 2, 3],
        'camera_file_suffixes':  ["_left.jpg", "_right.jpg", "_psm1.jpg", "_psm2.jpg"],
        'episode_len': 400, # not to be confused with number of demos
        'cutting_action_pad_size': 10,
        "recovery_ratio": 0.1,
        'action_mode': ['hybrid', 
                        {'max_': np.array([4.23075504e-04, 9.21509775e-04, 6.13676510e-04, 1.00000000e+00,
                                            4.95288465e-03, 2.07531063e-03, 3.96580366e-03, 1.00000000e+00,
                                            1.72866365e-02, 6.03863632e-01, 2.51159512e-03, 2.53343572e-02,
                                            1.48695231e-02, 1.00000000e+00, 2.44695645e-02, 4.82827872e-02,
                                            4.49850287e-01, 1.00000000e+00, 5.61511748e-01, 1.39638370e+00]), # jaw angle

                        'min_': np.array([-2.20925486e-04, -4.97738164e-04, -2.13495468e-04,  9.99985730e-01,
                                            -3.95869664e-03, -2.87697585e-03, -4.97288428e-03,  9.99844203e-01,
                                            -6.99162754e-03,  5.75808443e-01, -2.28192639e-02, -3.98780204e-03,
                                            -7.45012130e-03,  8.70544978e-01, -3.52216873e-01, -4.44619862e-01,
                                            -2.06905363e-02,  7.59440435e-01, -1.76271664e-02, -3.49066000e-01]), # jaw angle

                        'mean': np.array([-1.12857951e-03, -5.55928830e-04,  1.96000871e-03,  9.93965839e-01,
                                        -1.32512325e-02, -1.88838127e-02,  8.76573960e-03,  9.76772472e-01,
                                        -1.07177619e-02,  5.47486902e-01, -6.68116961e-04,  1.28479974e-04,
                                        2.38041489e-04,  9.98855082e-01, -4.33842069e-03, -4.91651007e-03,
                                        5.37581220e-03,  9.98161916e-01,  5.55527884e-03, -3.13479974e-01]),

                        'std': np.array([0.01,       0.01207889, 0.01,       0.01114384, 0.07472094, 0.07610724,
                                        0.07463249, 0.04631447, 0.19495843, 0.46001799, 0.01,       0.01,
                                        0.01,       0.01,       0.0333083,  0.0328184,  0.03904109, 0.01376217,
                                        0.04358239, 0.15126492]) }],
                        
        'norm_scheme': 'std',
        'save_frequency': 250,
        'camera_names': ['left', 'right', 'left_wrist', 'right_wrist'],
    },
    'base_chole_clipping_cutting':{
        'dataset_dir': DATA_DIR + "/base_chole_clipping_cutting/",
        'phantom': False,
        'num_episodes': 4581,
        'tissue_samples_ids': [4, 5, 8, 12, 13, 14, 18, 19],
        'camera_file_suffixes':  ["_left.jpg", "_right.jpg", "_psm1.jpg", "_psm2.jpg"],
        'episode_len': 400, # not to be confused with number of demos
        'cutting_action_pad_size': 10,
        "recovery_ratio": 0.1,
        'action_mode': ['hybrid', 
                        {'max_': np.array([4.23075504e-04, 9.21509775e-04, 6.13676510e-04, 1.00000000e+00,
                                            4.95288465e-03, 2.07531063e-03, 3.96580366e-03, 1.00000000e+00,
                                            1.72866365e-02, 6.03863632e-01, 2.51159512e-03, 2.53343572e-02,
                                            1.48695231e-02, 1.00000000e+00, 2.44695645e-02, 4.82827872e-02,
                                            4.49850287e-01, 1.00000000e+00, 5.61511748e-01, 1.39638370e+00]), # jaw angle

                        'min_': np.array([-2.20925486e-04, -4.97738164e-04, -2.13495468e-04,  9.99985730e-01,
                                            -3.95869664e-03, -2.87697585e-03, -4.97288428e-03,  9.99844203e-01,
                                            -6.99162754e-03,  5.75808443e-01, -2.28192639e-02, -3.98780204e-03,
                                            -7.45012130e-03,  8.70544978e-01, -3.52216873e-01, -4.44619862e-01,
                                            -2.06905363e-02,  7.59440435e-01, -1.76271664e-02, -3.49066000e-01]), # jaw angle

                        'mean': np.array([-1.12857951e-03, -5.55928830e-04,  1.96000871e-03,  9.93965839e-01,
                                          -1.32512325e-02, -1.88838127e-02,  8.76573960e-03,  9.76772472e-01,
                                          -1.07177619e-02,  5.47486902e-01, -6.68116961e-04,  1.28479974e-04,
                                           2.38041489e-04,  9.98855082e-01, -4.33842069e-03, -4.91651007e-03,
                                           5.37581220e-03,  9.98161916e-01,  5.55527884e-03, -3.13479974e-01]),

                        'std': np.array([0.01,      0.01207889, 0.01,       0.01114384, 0.07472094, 0.07610724,
                                        0.07463249, 0.04631447, 0.19495843, 0.46001799, 0.01,       0.01,
                                        0.01,       0.01,       0.0333083,  0.0328184,  0.03904109, 0.01376217,
                                        0.04358239, 0.15126492]) }],
                        
        'norm_scheme': 'std',
        'save_frequency': 250,
        'camera_names': ['left', 'right', 'left_wrist', 'right_wrist'],
    },
}
