import argparse
import time
from datetime import datetime
import os
from collections import defaultdict, deque
import contextlib
import sys
import signal

import cv2
import pandas as pd
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String, Float32MultiArray
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import rospy
from sklearn.metrics import f1_score, accuracy_score

# Import the necessary modules from this package
PATH_TO_YAY_ROBOT = os.getenv('PATH_TO_YAY_ROBOT')
if PATH_TO_YAY_ROBOT:
    sys.path.append(os.path.join(PATH_TO_YAY_ROBOT, 'src'))
else:
    raise EnvironmentError("Environment variable PATH_TO_YAY_ROBOT is not set")

from instructor.train_daVinci import build_instructor, log_confusion_matrix, log_combined_image
from aloha_pro.aloha_scripts.constants_daVinci import DATASET_CONFIGS
from instructor.dataset_daVinci import get_valid_demo_start_end_indices
from instructor.utils import center_crop_resize

# Context manager to measure the execution time of a code block
@contextlib.contextmanager
def measure_execution_time(label, execution_times_dict):
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        execution_time = end_time - start_time
        execution_times_dict[label].append(execution_time)

# Triggering smooth stopping of the instructor pipeline when pressing Ctrl+c
exit_flag = False
def signal_handler(sig, frame):
    global exit_flag
    print('\nStopping the instructor pipeline...\n')
    exit_flag = True
signal.signal(signal.SIGINT, signal_handler)

# -------------- ROS Subscriber Callbacks --------------

# Init the global variables for the camera images
image_left = image_right = image_psm1_wrist = image_psm2_wrist = psm1_jaw = psm2_jaw = None

ros_cv2_bridge = CvBridge() # Initialize the CvBridge

# Callback function for the left camera
def left_camera_callback(data):
    global image_left
    image_left = ros_cv2_bridge.imgmsg_to_cv2(data, desired_encoding = 'passthrough')
    
# Callback function for the right camera
def right_camera_callback(data):
    global image_right
    image_right = ros_cv2_bridge.imgmsg_to_cv2(data, desired_encoding = 'passthrough')
    
# Callback function for the PSM1 wrist camera
def psm1_wrist_camera_callback(data):
    global image_psm1_wrist
    image_psm1_wrist = ros_cv2_bridge.imgmsg_to_cv2(data, desired_encoding = 'passthrough')
    
def psm2_wrist_camera_callback(data):
    global image_psm2_wrist
    image_psm2_wrist = ros_cv2_bridge.imgmsg_to_cv2(data, desired_encoding = 'passthrough')
    
def psm1_jaw_callback(data):
    global psm1_jaw
    psm1_jaw = data.position[0]
    
def psm2_jaw_callback(data):
    global psm2_jaw
    psm2_jaw = data.position[0]
    
# --------------- Utils function ---------------

def create_random_chole_episode(dataset_dir, selected_tissue_sample, camera_names, camera_name_file_suffix_dict, downsampling_shape, center_crop_flag, use_jaw_values_flag=True):
    # Put together the stitched episode based on randomly getting a tissue id and a random index for a demo for each phase. Then sample a random timestep and get the corresponding image sequence and command embedding
       
    print("\nGenerating a random episode sequence...\n")
       
    # Init the episode sequence and the ground truth instruction sequence
    if use_jaw_values_flag:
        episode_jaw_values_sequence = []
    episode_frame_sequence = []
    episode_gt_instruction_sequence = []
       
    # Check within the dataset directory for tissue characteristics (e.g., before and after phase offset)
    dataset_name = os.path.basename(dataset_dir)
    before_phase_offset = DATASET_CONFIGS[dataset_name]["before_phase_offset"]
    after_phase_offset = DATASET_CONFIGS[dataset_name]["after_phase_offset"]
    
    # Go through the phases in fixed order of execution
    tissue_sample_dir_path = os.path.join(dataset_dir, selected_tissue_sample)
    phases_folder_names = [file_name for file_name in os.listdir(tissue_sample_dir_path) if os.path.isdir(os.path.join(tissue_sample_dir_path, file_name))]
    phases_folder_names = [phase_folder_name for phase_folder_name in phases_folder_names if "recovery" not in phase_folder_name]
    sorted_phases = sorted(phases_folder_names, key=lambda x: int(x.split('_')[0]))
    for phase_folder_name in sorted_phases:
        # Select a random demo for the current phase
        files_in_phase_folder = os.listdir(os.path.join(dataset_dir, selected_tissue_sample, phase_folder_name))
        demos_folder_names = [demo_sample for demo_sample in files_in_phase_folder if demo_sample[8] == "-"]
        selected_phase_demo_folder_name = np.random.choice(demos_folder_names)
        
        # Load the start and end indices for the current demo as the valid range of the demo
        selected_demo_folder_path = os.path.join(dataset_dir, selected_tissue_sample, phase_folder_name, selected_phase_demo_folder_name)
        start_idx, end_idx = get_valid_demo_start_end_indices(selected_demo_folder_path, camera_names, before_phase_offset, after_phase_offset)
        num_frames = end_idx - start_idx + 1
        
        # Extract the phase command from the folder name (removing the phase idx and the "_" in between the words) 
        _, phase_instruction = phase_folder_name.split("_")[0], " ".join(phase_folder_name.split("_")[1:]) 
        episode_gt_instruction_sequence += [phase_instruction]*num_frames # Add the instruction for the current demo
        
        # Get the jaw values for the current demo
        if use_jaw_values_flag:
            kinematics_csv_path = os.path.join(selected_demo_folder_path, 'ee_csv.csv')
            demo_kinematics = pd.read_csv(kinematics_csv_path)
            valid_jaw_values = torch.tensor(demo_kinematics.iloc[start_idx:end_idx + 1][["psm2_jaw", "psm1_jaw"]].values)
            episode_jaw_values_sequence.append(valid_jaw_values)
        
        # Append the frames of the selected demo
        for ts_demo_frame_idx in range(start_idx, end_idx + 1):
            camera_frame_dict = {}
            for camera_name in camera_names:
                camera_file_suffix = camera_name_file_suffix_dict[camera_name]
                camera_folder_name = os.path.join(dataset_dir, selected_tissue_sample, phase_folder_name, selected_phase_demo_folder_name, camera_name)
                frame_path = os.path.join(camera_folder_name, f"frame{str(ts_demo_frame_idx).zfill(6)}{camera_file_suffix}")
                frame = torch.tensor(cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
                # Resize the image
                if center_crop_flag:
                    frame_resized = center_crop_resize(frame, downsampling_shape[0])
                else:
                    frame_resized = transforms.Resize(downsampling_shape)(frame)
                camera_frame_dict[camera_name] = frame_resized
            
            # Stack the camera frames together
            all_cam_images = [camera_frame_dict[cam_name] for cam_name in camera_names]
            all_cam_images = torch.stack(all_cam_images, dim=0)
            episode_frame_sequence.append(all_cam_images)

    # Stack the episode frame sequence and the ground truth instruction sequence
    episode_frame_sequence_tensor = torch.stack(episode_frame_sequence, dim=0) / 255.0
    if use_jaw_values_flag:
        episode_jaw_value_sequence_tensor = torch.concatenate(episode_jaw_values_sequence, dim=0).to(dtype=torch.float32)
        return episode_frame_sequence_tensor, episode_jaw_value_sequence_tensor, episode_gt_instruction_sequence # Jaw values shape should be: num_frames, 2
    else:
        return episode_frame_sequence_tensor, episode_gt_instruction_sequence # Frame episode shape should be: num_frames, cam, c, h, w


def get_current_jaw_values(input_type, frame_idx=None, random_episode_jaw_value_sequence=None):
    # Based on the input type (live, random) get the current jaw values (+ if successful - indicating end of the episode for offline data)
    
    if input_type == "live":
        # Access the ROS subscribers and get the current jaw values
        if psm1_jaw is None or psm2_jaw is None:
            return False, None
        else:
            return True, torch.tensor([psm2_jaw, psm1_jaw])
    else:
        # Access the random generated episode jaw values
        if frame_idx >= len(random_episode_jaw_value_sequence):
            return False, None
        else:
            return True, random_episode_jaw_value_sequence[frame_idx]


def get_current_frames(input_type, downsampling_shape, center_crop_flag, frame_idx=None, random_episode_frame_sequence=None, camera_names=None):
    # Based on the input type (live, random) get the current frames stacked together
    
    if input_type == "live":
        # Access the ROS subscribers and get the current frames - do a copy of it + apply transformations - size + color
        camera_frames_dict = {}
        if "left_img_dir" in camera_names:
            camera_frames_dict["left_img_dir"] = image_left.copy()
        if "right_img_dir" in camera_names:
            camera_frames_dict["right_img_dir"] = image_right.copy()
        if "endo_psm1" in camera_names:
            camera_frames_dict["endo_psm1"] = image_psm1_wrist.copy()
        if "endo_psm2" in camera_names:
            camera_frames_dict["endo_psm2"] = image_psm2_wrist.copy()
        # Sort the camera frames based on the camera names
        camera_frames = [camera_frames_dict[camera_name] for camera_name in camera_names]
        
        # Apply transformations (resize, color, ..)
        current_frames_transformed = []
        for camera_frame, camera_name in zip(camera_frames, camera_names):
            if camera_frame is None:
                return False, None
            else:
                # Resize the image
                if center_crop_flag: 
                    camera_frame = center_crop_resize(camera_frame, downsampling_shape[0])
                else:
                    camera_frame = cv2.resize(camera_frame, downsampling_shape)
                
                # Convert the image to RGB
                if camera_name not in ["left_img_dir", "right_img_dir"]: 
                    camera_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                current_frames_transformed.append(torch.tensor(camera_frame.transpose(2, 0, 1)) / 255.0) # Shape: c, h, w
        
        return True, torch.stack(current_frames_transformed, dim=0) # Shape: cam, c, h, w 
        
        
    elif input_type == "random":
        # Access the random generated episode frames
        if frame_idx >= len(random_episode_frame_sequence):
            return False, None
        else:
            current_frames_transformed = random_episode_frame_sequence[frame_idx]  # Shape: cam, c, h, w
            return True, current_frames_transformed
    
    
def visualize_current_frames(current_frames, language_instruction_prediction, language_instruction_ground_truth=None, upscaling_factor=2, 
                             visualization_flag=True, current_jaw_values=None, phase_history=None):
    # Visualize the current frames (with the current language instruction prediction - if also gt is given then also visualize that)
    
    # Concatenate the frames together
    current_frames_concatenated = (torch.cat(list(current_frames), dim=2).detach().to("cpu").numpy().transpose(1, 2, 0)*255).astype(np.uint8)
    
    # Add the predicted language instruction on the image (and if gt then color the prediction green if correct, red if not + show also gt instruction)                
    text_position_x = 0
    prediction_text = f"Prediction: {language_instruction_prediction}"
    if language_instruction_ground_truth:
        prediction_text_color = (0, 255, 0) if language_instruction_prediction == language_instruction_ground_truth else (0, 0, 255)
        gt_text = f"GT: {language_instruction_ground_truth}"
    else:
        prediction_text_color = (255, 255, 255)
    current_frames_concatenated_bgr = cv2.cvtColor(current_frames_concatenated, cv2.COLOR_RGB2BGR) 
    current_frames_concatenated_bgr_upscaled = cv2.resize(current_frames_concatenated_bgr, (current_frames_concatenated_bgr.shape[1]*upscaling_factor, current_frames_concatenated_bgr.shape[0]*upscaling_factor))
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale, thickness = 0.75, 2
    cv2.putText(current_frames_concatenated_bgr_upscaled, prediction_text, (text_position_x, 25), font, font_scale, prediction_text_color, thickness, cv2.LINE_AA)
    if language_instruction_ground_truth:
        cv2.putText(current_frames_concatenated_bgr_upscaled, gt_text, (text_position_x, 60), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    # Add the jaw values to right side of the image
    if current_jaw_values is not None:
        jaw_text = f"Jaw values: PSM2: {current_jaw_values[1]:.2f}, PSM1: {current_jaw_values[0]:.2f}"
        jaw_text_position_x = current_frames_concatenated_bgr_upscaled.shape[1] - 500
        cv2.putText(current_frames_concatenated_bgr_upscaled, jaw_text, (jaw_text_position_x, 25), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    if phase_history is not None:
        phase_history_text = f"Phase history: {phase_history}"
        phase_history_text_position_x = current_frames_concatenated_bgr_upscaled.shape[1] - 500
        phase_history_text_position_y = 60 if current_jaw_values is not None else 25
        cv2.putText(current_frames_concatenated_bgr_upscaled, phase_history_text, (phase_history_text_position_x, phase_history_text_position_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    # Show the concatenated frames
    if visualization_flag:
        cv2.imshow("HL Policy Prediction", current_frames_concatenated_bgr_upscaled)
    
    return current_frames_concatenated_bgr_upscaled # Return the annotated frame for saving the video


def evaluate_instruction_prediction(predictions, ground_truths, all_phases, timestamp, save_folder_path):
    # Init metrics dict
    metrics_dict = {}
    
    # Compute the accuracy, f1 score, confusion matrix
    metrics_dict["accuracy"] = accuracy_score(ground_truths, predictions)
    metrics_dict["f1_score"] = f1_score(ground_truths, predictions, average="macro")
    
    # Save the confusion matrix function from the training script 
    save_path = os.path.join(save_folder_path, f"confusion_matrix_{timestamp}.png")
    log_confusion_matrix(ground_truths, predictions, all_phases, save_path=save_path, log_wandb_flag=False)

    return metrics_dict

# ------------------------------------- Main function --------------------------------------

def instructor_pipeline(args):
    # ------------- Access the command line parameters -------------

    # Output the command line parameters
    print("\n----------------------- Command line parameters ----------------------------\n")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("\n-----------------------------------------------------------------------------\n")

    # Define the output folder
    timestamp = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
    ckpt_name = os.path.basename(args.ckpt_path)
    output_folder_path = os.path.join(PATH_TO_YAY_ROBOT, "evaluation", "hl_policy_pipeline", args.input_type, ckpt_name, f"tissue_{args.tissue_name}_{timestamp=}")
    
    # Create the recordings folder if it does not exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    # -------------- Initialize the instructor model --------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init instructor model + load parameters and weigths from checkpoint
    checkpoint = torch.load(args.ckpt_path)
    history_len = checkpoint.history_len 
    candidate_texts = checkpoint.candidate_texts 
    candidate_embeddings = checkpoint.candidate_embeddings
    prediction_offset = checkpoint.prediction_offset 
    history_step_size = checkpoint.history_step_size 
    one_hot_flag = checkpoint.one_hot_flag
    model_camera_names = checkpoint.camera_names
    center_crop_flag = checkpoint.center_crop_flag
    backbone_model_name = checkpoint.backbone_model_name
    model_init_weights = checkpoint.model_init_weights
    freeze_backbone_until = checkpoint.freeze_backbone_until
    use_jaw_values_flag = checkpoint.use_jaw_values_flag
    use_phase_history_flag = checkpoint.use_phase_history_flag
    phase_history_len = checkpoint.phase_history_len
    use_transformer_flag = checkpoint.use_image_emb_transformer_flag
    instructor_model = build_instructor(history_len, history_step_size, prediction_offset, candidate_embeddings, candidate_texts, device, one_hot_flag, 
                                        model_camera_names, center_crop_flag, backbone_model_name, model_init_weights, freeze_backbone_until, use_jaw_values_flag, 
                                        use_phase_history_flag, phase_history_len, use_transformer_flag)  
    
    # Load the model weights
    instructor_model.load_state_dict(checkpoint.state_dict())    
    instructor_model.to(device)
    del checkpoint # Free up memory

    # Check if the model cameras matches the cameras selected as args
    if set(model_camera_names) != set(args.camera_names):
        raise ValueError(f"The model was trained on the cameras {model_camera_names} but the selected cameras are {args.camera_names}")

    # Set the model to evaluation mode
    instructor_model.eval()
    
    # ------------- Initialize ROS communication -------------
    
    if args.input_type == "live":        
        rospy.init_node('hl_policy_pipepline', anonymous=True)
    
        # Set the rate of execution
        rate = rospy.Rate(args.fps)
        
        # Instructor publisher for the language instruction prediction
        instruction_publisher = rospy.Publisher("/instructor_prediction", String, queue_size=args.publisher_queue_size) 
        instruction_embedding_publisher = rospy.Publisher("/instructor_embedding", Float32MultiArray, queue_size=args.publisher_queue_size)
        
        # Wrist camera subs
        if "endo_psm1" in args.camera_names:
            endo_psm1_sub = rospy.Subscriber("/PSM1/endoscope_img", Image, psm1_wrist_camera_callback, queue_size=args.subscriber_queue_size)
        if "endo_psm2" in args.camera_names:
            endo_psm2_sub = rospy.Subscriber("/PSM2/endoscope_img", Image, psm2_wrist_camera_callback, queue_size=args.subscriber_queue_size)
        
        # Endoscope imgs
        if "left_img_dir" in args.camera_names:
            left_img_dir_sub = rospy.Subscriber("/jhu_daVinci/left/image_raw", Image, left_camera_callback, queue_size=args.subscriber_queue_size)
        if "right_img_dir" in args.camera_names:
            right_img_dir_sub = rospy.Subscriber("/jhu_daVinci/right/image_raw", Image, right_camera_callback, queue_size=args.subscriber_queue_size)
            
        # Jaw values
        if use_jaw_values_flag:
            psm1_jaw_sub = rospy.Subscriber("PSM1/jaw/measured_js", JointState, psm1_jaw_callback, queue_size=args.subscriber_queue_size)
            psm2_jaw_sub = rospy.Subscriber("PSM2/jaw/measured_js", JointState, psm2_jaw_callback, queue_size=args.subscriber_queue_size)
        
        time.sleep(1) # Wait for the subscribers to be initialized
        
    # -------------- Init the recorded episode data (if random) --------------
    
    if args.input_type == "random":
        # Generate a random episode
        frame_episode_sequence, jaw_values_episode_sequence, episode_gt_instruction_sequence = create_random_chole_episode(args.dataset_dir, args.tissue_name, args.camera_names, args.camera_name_file_suffix_dict, args.downsampling_shape, center_crop_flag) 
    
        # Check if the gt instructions are within the learned instructions of the model
        gt_instructions_set = set(episode_gt_instruction_sequence)
        candidate_texts_set = set(candidate_texts)
        if not gt_instructions_set.issubset(candidate_texts_set):
            raise ValueError(f"The ground truth instructions are not within the learned instructions of the model. Missing instructions: {gt_instructions_set - candidate_texts_set}")
    
    # -------------- Initialize evaluation variables --------------
    
    # Initialize the language instruction prediction and ground truth lists
    instruction_pred_list, instruction_gt_list = [], []
    
    # Evaluation timing variables
    if args.save_video_flag:
        concat_frame_list = [] # Array to store the concatenated frames (for later saving as video)
    execution_times_dict = defaultdict(list)
    
    
    # ------------- Main loop -------------
    
    # Init phase history list (with the phase indices up to the start phase)
    if use_phase_history_flag:
        if not args.input_type == "live" or not args.starting_phase_idx:
            phase_history = [0]*phase_history_len 
        else:
            phase_history = [0]*(max(0, phase_history_len-args.starting_phase_idx-1)) + list(range(args.starting_phase_idx))[-phase_history_len:]
            phase_history_commands = [candidate_texts[phase_idx-1] for phase_idx in phase_history]
            print(f"Starting phase: {candidate_texts[args.starting_phase_idx-1]} - Phase history: {phase_history_commands}")
        model_input_phase_history = torch.tensor(phase_history).to(device).unsqueeze(0) # Shape: batch_size (=1), phase_history_len
    else:
        model_input_phase_history = None
    
    # Keeping the last frames (for the history length)
    model_input_frames = deque(maxlen=history_len+1)
    if use_jaw_values_flag:
        model_input_jaw_values = deque(maxlen=history_len+1)
    else:
        model_input_jaw_values_tensor = None
    
    frame_idx = 0
    while not exit_flag:
        with measure_execution_time("Total inference time", execution_times_dict):
            # -------------- Load current jaw values + camera frames --------------

            if args.visualization_flag or args.save_video_flag or frame_idx % history_step_size*args.ll_policy_slowness_factor == 0: # Only load the frames if they are needed (for visualization, saving video, or for the model input)
                if use_jaw_values_flag:
                    with measure_execution_time("Loading jaw values time", execution_times_dict):
                        if args.input_type == "random":
                            success, current_jaw_values = get_current_jaw_values(args.input_type, frame_idx=frame_idx, random_episode_jaw_value_sequence=jaw_values_episode_sequence)
                        elif args.input_type == "live":
                            success, current_jaw_values = get_current_jaw_values(args.input_type)
                        # Break out of the loop if the jaw values could not be loaded (e.g., end of the episode)
                        if not success:
                            break
                else:
                    current_jaw_values = None

                with measure_execution_time("Loading video frame time", execution_times_dict):
                    if args.input_type == "random":
                        success, current_frames = get_current_frames(args.input_type, args.downsampling_shape, center_crop_flag, frame_idx=frame_idx, random_episode_frame_sequence=frame_episode_sequence)
                    elif args.input_type == "live":
                        success, current_frames = get_current_frames(args.input_type, args.downsampling_shape, center_crop_flag, camera_names=args.camera_names)
                    # Break out of the loop if the image could not be loaded (e.g., frame could not be loaded)
                    if not success:
                        break   # Stop the loop and save the video up to here (if desired)
                    # Add the current frames to the model input frames + generate current input tensor
                    
                    # Add frames and jaw values to model input based on the history step size (and dependent of the slower speed of the low level policy)
                    if frame_idx % history_step_size*args.ll_policy_slowness_factor == 0:
                        model_input_frames.append(current_frames)
                        model_input_frames_tensor = torch.stack(list(model_input_frames), dim=0).to(device).unsqueeze(0) # Shape: batch_size (=1), history_len, cam, c, h, w 
                
                        if use_jaw_values_flag:
                            model_input_jaw_values.append(current_jaw_values)
                            model_input_jaw_values_tensor = torch.stack(list(model_input_jaw_values), dim=0).to(device).unsqueeze(0) # Shape: batch_size (=1), history_len, 2
            
            # -------------- Predict (+publish) the language instruction --------------
            
            # Start with predictions after the history length is reached and predict every x frames and not at the beginning (as starting with the first phase)
            # TODO: Maybe adjust later that it already is able to predict with less frames (after training with masking)
            if frame_idx % args.prediction_stride*args.ll_policy_slowness_factor == 0 and len(model_input_frames) == history_len+1 and frame_idx != 0:
                with measure_execution_time("Instructor_inference_time", execution_times_dict):
                    # Apply the model on the current frames
                    logits, temperature, predicted_embedding = instructor_model(model_input_frames_tensor, model_input_jaw_values_tensor, model_input_phase_history)
                    print(f"\nFrame Idx: Jaw values (PSM2, PSM1): {model_input_jaw_values_tensor}, phase history: {model_input_phase_history}")
                    if args.input_type == "live":
                        print(f"Frame Idx: {frame_idx} - Predicted instruction: {predicted_instruction}")
                    # Decode the model output to the language instruction
                    predicted_instruction = instructor_model.decode_logits(logits, temperature)[0]
                    instruction_pred_list.append(predicted_instruction)
                    
                    # Publish the predicted language instruction
                    if args.input_type == "live": 
                        instruction_publisher.publish(predicted_instruction)
                        instruction_embedding_publisher.publish(Float32MultiArray(data=predicted_embedding))  
                    
                    # # TODO: Remove later again - debugging
                    # save_path = os.path.join(output_folder_path, f"frame_{frame_idx}.png")
                    # log_combined_image(model_input_frames_tensor[0], predicted_instruction, predicted_instruction, save_path=save_path)
                    # print(f"Java values: {model_input_jaw_values_tensor}")
                    # print(f"Phase history: {model_input_phase_history}")
                    # exit()
                    
                    # Evaluate the prediction (if gt instruction is available --> in offline case)
                    if args.input_type == "random":
                        gt_instruction = episode_gt_instruction_sequence[frame_idx + prediction_offset] if frame_idx + prediction_offset < len(episode_gt_instruction_sequence) else gt_instruction # Get the gt instruction (if available) - repeat last gt instruction if end of episode (by the prediction offset)
                        instruction_gt_list.append(gt_instruction)
                        
                        if gt_instruction == predicted_instruction:
                            print(f"Frame Idx: {frame_idx} - Predicted instruction: {predicted_instruction} - GT instruction: {gt_instruction}")
                        else:
                            print(f"Frame Idx: {frame_idx} - Predicted instruction: {predicted_instruction} - GT instruction: {gt_instruction} --> Wrong prediction")
                        
                        # # TODO: Remove later again - debugging
                        # save_path = os.path.join(output_folder_path, f"frame_{frame_idx}.png")
                        # log_combined_image(model_input_frames_tensor[0], gt_instruction, predicted_instruction, save_path=save_path)
                        # print(f"Java values: {model_input_jaw_values_tensor}")
                        # print(f"Phase history: {model_input_phase_history}")
                        # exit()
        
            elif frame_idx == 0:
                # Wait with predictions until the history length is reached and begin with the first instruction (which is the same for all demos)
                starting_command_idx = 0 if not args.starting_phase_idx else args.starting_phase_idx - 1
                predicted_instruction = candidate_texts[starting_command_idx]
                predicted_embedding = candidate_embeddings[starting_command_idx]
                if args.input_type == "random":
                    gt_instruction = episode_gt_instruction_sequence[frame_idx + prediction_offset]
            
                # Publish the predicted language instruction
                if args.input_type == "live": 
                    instruction_publisher.publish(predicted_instruction)
                    instruction_embedding_publisher.publish(Float32MultiArray(data=predicted_embedding))      
            
            # Update the phase history list
            if use_phase_history_flag:
                predicted_instruction_idx = candidate_texts.index(predicted_instruction) + 1 # Because of padding index
                # If the predicted instruction is different from the last instruction, add it to the phase history
                if predicted_instruction_idx != phase_history[-1]:
                    phase_history.append(predicted_instruction_idx)
                    # Keep the phase history list at the desired length
                    phase_history = phase_history[1:]
                    model_input_phase_history = torch.tensor(phase_history).to(device).unsqueeze(0) # Shape: batch_size (=1), phase_history_len
            
            # -------------- Visualize the frame --------------
            
            if args.visualization_flag or args.save_video_flag:
                with measure_execution_time("Visualizing frame time", execution_times_dict):
                    # Visualization of the predicted language instruction
                    if args.input_type == "random":
                        annotated_frame = visualize_current_frames(current_frames, predicted_instruction, gt_instruction, upscaling_factor=args.upscaling_factor,
                                                                   visualization_flag=args.visualization_flag, current_jaw_values=current_jaw_values, phase_history=phase_history)
                    else:
                        annotated_frame = visualize_current_frames(current_frames, predicted_instruction, upscaling_factor=args.upscaling_factor,
                                                                   visualization_flag=args.visualization_flag, current_jaw_values=current_jaw_values, phase_history=phase_history)
            
            if args.save_video_flag:
                concat_frame_list.append(annotated_frame)

        # -------------- Update + log execution times --------------  
            
        if args.log_execution_times_during_execution_flag:
            # Log the average execution times of the last n samples
            print("") # Add empty line for better readability
            last_n_samples = 100
            num_samples_time_eval = min(len(execution_times_dict["Total inference time"]), last_n_samples)
            for key, value in execution_times_dict.items():
                if key != "Total inference time":
                    print(f"{key} (last {num_samples_time_eval} samples): {np.mean(value[-last_n_samples:])*1000:.2f} ms")
            # For better visibility (+ with infos on the number of frames - as higher at the beginning at start of the process)
            # Note: The average total inference time might be lower than the individual components as not every frame is processed (by the model)
            print(f"--> Total inference time (last {num_samples_time_eval} samples): {np.mean(execution_times_dict['Total inference time'][-last_n_samples:])*1000:.2f} ms")

            print("\n------------------------------------------------------")

        frame_idx += 1 # Increase the frame index

        # -------------- Stop the pipeline + clean up --------------

        # Break out of the loop if the user pressed q
        if cv2.waitKey(1) == ord('q'):
            break
        
        # Sleep to ensure given rate
        if args.input_type == "live":
            rate.sleep()
        else:
            time_to_sleep = max(1/args.fps - execution_times_dict["Total inference time"][-1], 0) # Sleep for the remaining time of the desired rate
            time.sleep(time_to_sleep)

    cv2.destroyAllWindows()
    plt.close('all')
    
    # -------------- Save the detected images as video --------------
    
    # Save the detected images as video (if desired)
    if args.save_video_flag:
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    
        # Create the video file name
        if args.input_type == "live":
            video_file_path = os.path.join(output_folder_path, f'{args.input_type}_{timestamp=}.mp4')
        else:
            additional_info = args.tissue_name
            video_file_path = os.path.join(output_folder_path, f'{args.input_type}_{additional_info}_{timestamp=}.mp4')
        video_dim = concat_frame_list[-1].shape[:2][::-1]
        out = cv2.VideoWriter(video_file_path, fourcc, args.fps, video_dim)
        # Write the detected images to the video
        for concat_frame in concat_frame_list:
            out.write(concat_frame)
        out.release()
        print(f"\nSaved the video to {video_file_path}")

    # ----------------------- Evaluation ----------------------------

    print("\n----------------------- Evaluation ----------------------------")

    # Evaluate the language instruction prediction
    if args.input_type == "random":
        metrics_dict = evaluate_instruction_prediction(instruction_pred_list, instruction_gt_list, candidate_texts, timestamp, output_folder_path)

        # Phase prediction evaluation
        print(f"\nAccuracy: {metrics_dict['accuracy']*100:.2f}")
        print(f"F1 score: {metrics_dict['f1_score']*100:.2f}")
    
    # Print average timings (for all different components, with keys in the dict + the total inference time)
    print("\nAverage timings:")
    for key, value in execution_times_dict.items():
        if key != "Total inference time":
            print(f"{key}: {np.mean(value)*1000:.2f} ms")
    print(f"Total inference time: {np.mean(execution_times_dict['Total inference time'])*1000:.2f} ms") 
    

# -------------------------------- Create parser --------------------------------

def parse_pipeline_args():
    """
    Define a parser that reads out the command line arguments and returns the parsed arguments.
    """
    
    parser = argparse.ArgumentParser(description='Surgical Tool Tracking Pipeline')

    # --------------------------- Instruction model parameters ---------------------------
    
    # Instructor model path
    default_ckpt_folder_name = "base_chole_clipping_cutting_clip_sda_full_phase_set_one_ts_jaw_history_w_transformer_h_len_3_cosine_scheduler"
    default_ckpt_folder_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "model_ckpts", "hl", default_ckpt_folder_name)
    default_ckpt_file_name = "best_val_loss_epoch=548"
    parser.add_argument('--ckpt_path', type=str, default=os.path.join(default_ckpt_folder_path, f"{default_ckpt_file_name}.ckpt"),
                        help="Path to the instructor model")

    # Prediction stride value to only predict every x frames
    parser.add_argument('--prediction_stride', type=int, default=30, help="Prediction stride value (e.g., predict every x frames)")


    # ---------------------------------- Data parameters -------------------------------------
    
    # Input type (testing it with live data, random generated episodes
    parser.add_argument('--input_type', type=str, default="random",
                        help="Can be either 'live' or 'random' (for random generated episode)")
    
    # Image size
    parser.add_argument('--downsampling_shape', type=tuple, default=(224, 224), help="Desired size of the input image")
    
    # Camera name file suffix dict
    default_camera_name_file_suffix_dict = {"endo_psm2": "_psm2.jpg", "left_img_dir": "_left.jpg", "right_img_dir": "_right.jpg", "endo_psm1": "_psm1.jpg"}
    parser.add_argument('--camera_name_file_suffix_dict', type=dict, default=default_camera_name_file_suffix_dict, help="Dictionary with the camera names and their corresponding file suffixes")
    
    # Starting phase 
    parser.add_argument('--starting_phase_idx', type=int, default=None, help="Starting phase index for the random generated episode (None or 0 when starting from the beginning)")
    
    # Low level policy speed ratio (as the low level policy is slower than the high level policy) - set when ll policy will be used
    parser.add_argument('--ll_policy_slowness_factor', type=int, default=1, help="Speed ratio of the low level policy compared to the high level policy")
    
    # Camera names - based on the order of the cameras, the images will be stacked together as model input
    parser.add_argument('--camera_names', type=str, nargs='+', default=["endo_psm2", "left_img_dir", "endo_psm1"], help="Names of the cameras") # Possible cameras: "endo_psm2", "left_img_dir", "right_img_dir", "endo_psm1"
    
    # Add dataset directory
    default_dataset_name = "base_chole_clipping_cutting"
    local_dataset_path = os.getenv("PATH_TO_DATASET")
    if local_dataset_path:
        default_dataset_dir = os.path.join(local_dataset_path, default_dataset_name)
    else:
        default_dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chole_data", default_dataset_name)
    parser.add_argument('--dataset_dir', type=str, default=default_dataset_dir, help="Path to the dataset directory")
    
    # Add tissue id
    default_tissue_sample = "tissue_14"
    parser.add_argument('--tissue_name', type=str, default=default_tissue_sample, help="Tissue id for the random generated episode")
    
    
    # --------------------------- Visualization & Evaluation parameters ---------------------------
    
    # Upscaling factor for the visualization
    parser.add_argument('--upscaling_factor', type=int, default=2, help="Upscaling factor for the visualization of the frames")
    
    # Add save video flag
    parser.add_argument('--save_video_flag', action='store_true', default=True,
                    help="Flag to save the phase prediction recording")
    
    # Add visualization flag
    parser.add_argument('--visualization_flag', action='store_true', default=False,
                    help="Flag to visualize phase prediction during execution.")
    
    # Add log execution times flag
    parser.add_argument('--log_execution_times_during_execution_flag', action='store_true', default=False,
                    help="Flag to log the execution times of the different components")


    # --------------------------- ROS communication parameters ---------------------------
    
    # Fps (for ros communication)
    parser.add_argument('--fps', type=int, default=30, help="FPS of the ROS node")
    
    # Subscriber queue size    
    parser.add_argument('--subscriber_queue_size', type=int, default=1, help="Queue size for the ROS subscriber")
    
    # Publisher queue size
    parser.add_argument('--publisher_queue_size', type=int, default=0, help="Queue size for the ROS publisher")

    # ----------------------------------------------------------------------------------------------

    # Parse the command line arguments
    args = parser.parse_args()
    
    return args  

if __name__ == '__main__':
    args = parse_pipeline_args()
    instructor_pipeline(args)  