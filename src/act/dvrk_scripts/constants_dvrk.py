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
    'base_chole_clipping_cutting':{
        'dataset_dir': DATA_DIR + "/base_chole_clipping_cutting/",
        'num_episodes': 224,
        'tissue_samples_ids': [1],
        'camera_file_suffixes':  ["_left.jpg", "_right.jpg", "_psm1.jpg", "_psm2.jpg"],
        'episode_len': 400, # not to be confused with number of demos
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

                        'mean': np.array([-1.07592371e-03,  6.39197726e-04,  1.33060653e-03,  9.87509484e-01,
                                            2.66880921e-02,  2.47990212e-02, -2.24158183e-02,  9.87458769e-01,
                                        -1.40400757e-02, -2.64201908e-01, -2.21153217e-04,  8.88786546e-04,
                                            1.59460341e-03,  9.77774663e-01, -2.50561688e-02, -5.64664921e-02,
                                            9.36258713e-03,  9.74861881e-01, -2.83120580e-02,  7.98996343e-02]),

                        'std': np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                                        0.01,       0.01,       0.01,       0.01,       0.01,       0.01,
                                        0.01,       0.0206547,  0.0746277,  0.08575524, 0.09589779, 0.03703402,
                                        0.11229542, 0.54967358]) }],
                        
        'norm_scheme': 'std',
        'save_frequency': 250,
        'camera_names': ['left', 'right', 'left_wrist', 'right_wrist'],
    },

}
