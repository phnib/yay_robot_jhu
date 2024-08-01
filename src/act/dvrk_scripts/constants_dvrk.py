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
        'camera_file_suffixes':  ["_left.jpg", "_psm1.jpg", "_psm2.jpg"],
        'episode_len': 400, # not to be confused with number of demos
        'cutting_action_pad_size': 10,
        "recovery_ratio": 0.2,
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
        'save_frequency': 150,
        'camera_names': ['left', 'left_wrist', 'right_wrist'],
    },
    'base_chole_clipping_cutting':{
        'dataset_dir': DATA_DIR + "/base_chole_clipping_cutting/",
        'phantom': False,
        'num_episodes': 5087,
        'tissue_samples_ids': [4, 5, 6, 8, 12, 13, 14, 18],
        'camera_file_suffixes':  ["_left.jpg", "_psm1.jpg", "_psm2.jpg"],
        'episode_len': 400, # not to be confused with number of demos
        'cutting_action_pad_size': 10,
        "recovery_ratio": 0.2,
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
        'save_frequency': 150,
        'camera_names': ['left', 'left_wrist', 'right_wrist'],

    },
    'exvivo_11':{
        'dataset_dir': DATA_DIR + "/base_chole_clipping_cutting/",
        'phantom': False,
        'num_episodes': 6392,
        'tissue_samples_ids': [4, 5, 6, 8, 12, 13, 14, 18, 19, 22],
        'camera_file_suffixes':  ["_left.jpg", "_psm1.jpg", "_psm2.jpg"],
        'episode_len': 400, # not to be confused with number of demos
        'cutting_action_pad_size': 10,
        "recovery_ratio": 1.0,
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

                        'mean': np.array([-1.04358107e-03, -1.44092798e-04,  2.12065951e-03,  9.94326056e-01,
                                          -1.27530813e-02, -1.88901459e-02,  8.54350753e-03,  9.76271403e-01,
                                          -3.02171681e-03,  5.49041745e-01, -6.44904918e-04,  1.31217346e-04,
                                           2.45878723e-04,  9.98964884e-01, -4.14443666e-03, -4.47166073e-03,
                                           5.01863478e-03,  9.98282330e-01,  5.00672862e-03, -3.17354488e-01,]),

                        'std': np.array([0.01,       0.01217403, 0.01,       0.01064037, 0.07222305, 0.07393801,
                                         0.07149193, 0.04634726, 0.19887901, 0.47374915, 0.01,       0.01,
                                         0.01,       0.01,       0.03242789, 0.03051706, 0.03738663, 0.01292145,
                                         0.04263112, 0.14474135,]) }],

        'norm_scheme': 'std',
        'save_frequency': 100,
        'camera_names': ['left', 'left_wrist', 'right_wrist'],
        'available_phase_commands': {
            (1, 3): "apply first clip on the left tube", 
            (4, 5):  "apply second clip on the left tube", 
            (6, 7): "apply third clip on the left tube",
            (8, 9): "cut the left tube",
            (10, 11): "apply first clip on the right tube",
            (12, 13): "apply second clip on the right tube",
            (14, 15): "apply third clip on the right tube",
            (16, 17): "cut the right tube"
        },
        'merging_subtasks': True,

    },
    'exvivo_13':{
        'dataset_dir': DATA_DIR + "/base_chole_clipping_cutting/",
        'phantom': False,
        'num_episodes': 8000,
        'tissue_samples_ids': [4, 5, 6, 8, 12, 13, 14, 18, 19, 22, 23, 30, 32],
        'camera_file_suffixes':  ["_left.jpg", "_psm1.jpg", "_psm2.jpg"],
        'episode_len': 400, # not to be confused with number of demos
        'cutting_action_pad_size': 10,
        "recovery_ratio": 1.0,
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

                        'mean': np.array([-1.28326786e-03,  8.63239905e-04,  2.46955407e-03,  9.93370251e-01,
                                          -8.48731692e-03, -2.77646603e-02,  8.33757196e-03,  9.74587386e-01,
                                           1.37758725e-02,  5.47324932e-01, -5.72959012e-04,  1.22170840e-04,
                                           1.89320416e-04,  9.99095827e-01, -4.12592430e-03, -4.04169945e-03,
                                           4.86193567e-03,  9.98518887e-01,  4.25297238e-03, -3.20388251e-01,]),

                        'std': np.array([0.01,       0.01266959, 0.01,       0.01280918, 0.07194699, 0.08385815,
                                         0.07089388, 0.04826008, 0.20631326, 0.481729,   0.01,       0.01,
                                         0.01,       0.01,       0.0305749,  0.02825818, 0.03499589, 0.01170378,
                                         0.03945398, 0.14041393,]) }],

        'norm_scheme': 'std',
        'save_frequency': 100,
        'camera_names': ['left', 'left_wrist', 'right_wrist'],
        'available_phase_commands': {
            (1, 3): "apply first clip on the left tube", 
            (4, 5):  "apply second clip on the left tube", 
            (6, 7): "apply third clip on the left tube",
            (8, 9): "cut the left tube",
            (10, 11): "apply first clip on the right tube",
            (12, 13): "apply second clip on the right tube",
            (14, 15): "apply third clip on the right tube",
            (16, 17): "cut the right tube"
        },
        'merging_subtasks': True,
    },
    
    'exvivo_16':{
        'dataset_dir': DATA_DIR + "/base_chole_clipping_cutting/",
        'phantom': False,
        'num_episodes': 9789,
        'tissue_samples_ids': [4, 5, 6, 8, 12, 13, 14, 18, 19, 22, 23, 30, 32, 35, 39, 40],
        'camera_file_suffixes':  ["_left.jpg", "_psm1.jpg", "_psm2.jpg"],
        'episode_len': 400, # not to be confused with number of demos
        'cutting_action_pad_size': 10,
        "recovery_ratio": 1.0,
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

                        'mean': np.array([-0.001442905859290911, 0.0013345987724556144, 0.0026566287460412933, 0.9924688882801028, 
                                          -0.0041728440068513565, -0.03455507975556743, 0.0071885281460733865, 0.9735069466285956, 
                                           0.02364083321057046, 0.5430185038927395, -0.0005160879512865046, 0.00011223049528998244, 
                                           0.0001346816543208265, 0.9991787398568377, -0.004061638146868762, -0.003636024959821714, 
                                           0.004703551915752909, 0.9986432066907925, 0.003516683276670445, -0.3227010764087532]),

                        'std': np.array([0.01, 0.012598459054031445, 0.01, 0.01416700448111247, 0.07343185351055137, 
                                         0.09055991670154613, 0.07079933375292484, 0.048625085874357946, 0.21046785100166998, 
                                         0.4857690755881234, 0.01, 0.01, 0.01, 0.01, 0.02922561270263285, 0.02688321081885941,
                                         0.03313355361902432, 0.01081921695173057, 0.038240913417313366, 0.1385132136507323]) }],
\
        'norm_scheme': 'std',
        'save_frequency': 100,
        'camera_names': ['left', 'left_wrist', 'right_wrist'],
        'merging_subtasks': False,
    },

}
