import argparse
import time
from datetime import datetime
import os
from collections import defaultdict, deque
import contextlib
import sys

import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String,  Float32MultiArray
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import rospy
from sklearn.metrics import f1_score

# Import the necessary modules from this package
path_to_yay_robot = os.getenv('PATH_TO_YAY_ROBOT')
if path_to_yay_robot:
    sys.path.append(os.path.join(path_to_yay_robot, 'src'))
else:
    raise EnvironmentError("Environment variable PATH_TO_YAY_ROBOT is not set")

from instructor.train_daVinci import build_instructor, log_confusion_matrix
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

# -------------- ROS Subscriber Callbacks --------------

# Init the global variables for the camera images
image_left = image_right = image_psm1_wrist = image_psm2_wrist = None

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
    
# --------------- Utils function ---------------

def create_random_chole_episode(dataset_dir, camera_names, camera_file_suffixes_aspect_ratio, downsampling_shape, center_crop_flag):
    # Put together the stitched episode based on randomly getting a tissue id and a random index for a demo for each phase. Then sample a random timestep and get the corresponding image sequence and command embedding
       
    # Init the episode sequence and the ground truth instruction sequence
    episode_frame_sequence = []
    episode_gt_instruction_sequence = []
       
    # Check within the dataset directory for the possible tissue ids + for tissue characteristics (e.g., before and after phase offset)
    dataset_name = os.path.basename(dataset_dir)
    incomplete_tissues = DATASET_CONFIGS["incomplete_tissue_samples"] if "incomplete_tissue_samples" in DATASET_CONFIGS else []
    tissue_folder_names = [tissue_name for tissue_name in os.listdir(dataset_dir) if tissue_name not in incomplete_tissues]
    selected_tissue_sample = np.random.choice(tissue_folder_names)
    before_phase_offset = DATASET_CONFIGS[dataset_name]["before_phase_offset"]
    after_phase_offset = DATASET_CONFIGS[dataset_name]["after_phase_offset"]
    
    # Go through the phases in fixed order of execution
    tissue_sample_dir_path = os.path.join(dataset_dir, selected_tissue_sample)
    phases_folder_names = [file_name for file_name in os.listdir(tissue_sample_dir_path) if os.path.isdir(os.path.join(tissue_sample_dir_path, file_name))]
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
        
        # Append the frames of the selected demo
        for ts_demo_frame_idx in range(start_idx, end_idx + 1):
            camera_frame_dict = {}
            for camera_name in camera_names:
                camera_file_suffix = camera_file_suffixes_aspect_ratio[camera_name][0]
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
    return episode_frame_sequence_tensor, episode_gt_instruction_sequence # TODO: Check if the shape is: num_frames, cam, c, h, w


def get_current_frames(input_type, downsampling_shape, center_crop_flag, frame_idx=None, random_episode_sequence=None, video_capture=None, camera_names=None, camera_file_suffixes_aspect_ratio=None):
    # Based on the input type (video, live, random) get the current frames stacked together
    
    if input_type == "video":
        # Load the current video frame
        success, image = video_capture.read()

        # Raise error if the image could not be loaded
        if not success:
            return success, None
        
        # Separate the concatenated images into the individual camera frames
        frame_height = image.shape[0]
        camera_frames = []
        width_offset = 0
        # TODO: Check if everything works with the aspect ratio
        for camera_name in camera_names:
            _, aspect_ratio = camera_file_suffixes_aspect_ratio[camera_name]
            image_width = int(frame_height * aspect_ratio)
            camera_frame = image[:, width_offset:width_offset + image_width]
            if camera_frame.shape[0] != frame_height or camera_frame.shape[1] != image_width:
                # Resize the image to the desired shape
                if center_crop_flag:
                    camera_frame = center_crop_resize(camera_frame, downsampling_shape[0])
                else:
                    camera_frame = cv2.resize(camera_frame, downsampling_shape)
            camera_frame_rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
            camera_frames.append(camera_frame_rgb)
            
        camera_frames_tensor = torch.tensor(camera_frames).permute(0, 3, 1, 2) # TODO: Check if it has the shape: cam, c, h, w
        return success, camera_frames_tensor
    
    elif input_type == "live":
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
                if camera_name in ["left_img_dir", "right_img_dir"]:
                    camera_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                current_frames_transformed.append(camera_frame)
        
        return True, torch.stack(current_frames_transformed, dim=0) # TODO: Check if it has the shape: cam, c, h, w
        
        
    elif input_type == "random":
        # Access the random generated episode frames
        current_frames_transformed = random_episode_sequence[frame_idx]  # TODO: Check if it has the shape: cam, c, h, w
        return True, current_frames_transformed
    
    
def visualize_current_frames(current_frames, language_instruction_prediction, language_instruction_ground_truth=None):
    # Visualize the current frames (with the current language instruction prediction - if also gt is given then also visualize that)
    
    # Concatenate the frames together
    current_frames_concatenated = (torch.cat(list(current_frames), dim=2).detach().to("cpu").numpy().transpose(1, 2, 0)*255).astype(np.uint8)
    
    # Add the predicted language instruction on the image (and if gt then color the prediction green if correct, red if not + show also gt instruction)                
    text_position_x = 0
    prediction_text = f"Prediction: {language_instruction_prediction}"
    if language_instruction_ground_truth:
        prediction_text_color = (0, 255, 0) if language_instruction_prediction == language_instruction_ground_truth else (0, 0, 255)
        prediction_text += f"      GT: {language_instruction_ground_truth}"
    else:
        prediction_text_color = (255, 255, 255)
    # TODO: Color of the text correct?
    current_frames_concatenated_bgr = cv2.cvtColor(current_frames_concatenated, cv2.COLOR_RGB2BGR) 
    cv2.putText(current_frames_concatenated_bgr, prediction_text, (text_position_x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, prediction_text_color, 1, cv2.LINE_AA)
    
    # Show the concatenated frames
    cv2.imshow("HL Policy Prediction", current_frames_concatenated_bgr)


def evaluate_instruction_prediction(predictions, ground_truths, timestamp):
    # Init metrics dict
    metrics_dict = {}
    
    # TODO
    
    # Compute the accuracy, f1 score, confusion matrix
    metrics_dict["accuracy"] = (predictions == ground_truths).mean()
    metrics_dict["f1_score"] = f1_score(predictions, ground_truths)
    
    # Save the confusion matrix function from the training script 
    language_commands = list(np.unique(ground_truths))
    # TODO: Check if saving works and the path is correct?
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation", "hl_policy_pipeline", "confusion_matrix_{timestamp}.png")
    log_confusion_matrix(ground_truths, predictions, language_commands, save_path=save_path)

    return metrics_dict

# ------------------------------------- Main function --------------------------------------

def instructor_pipeline(args):
    # ------------- Access the command line parameters -------------

    # Output the command line parameters
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

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
    instructor_model = build_instructor(history_len, history_step_size, prediction_offset, candidate_embeddings, candidate_texts, device, one_hot_flag, model_camera_names, center_crop_flag)
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
        
        # Set the rate of the ROS node
        ros_fps = args.ros_fps
        rate = rospy.Rate(ros_fps)
        
        # Instructor publisher for the language instruction prediction
        instruction_publisher = rospy.Publisher("/instructor_prediction", String) 
        instruction_embedding_publisher = rospy.Publisher("/instructor_embedding", Float32MultiArray) # TODO: Check if this is the correct type?
        
        # Wrist camera subs
        if "endo_psm1" in args.camera_names:
            endo_psm1_sub = rospy.Subscriber("/PSM1/endoscope_img", Image, psm1_wrist_camera_callback)
        if "endo_psm2" in args.camera_names:
            endo_psm2_sub = rospy.Subscriber("/PSM2/endoscope_img", Image, psm2_wrist_camera_callback)
        
        # Endoscope imgs
        if "left_img_dir" in args.camera_names:
            left_img_dir_sub = rospy.Subscriber("/jhu_daVinci/left/image_raw", Image, left_camera_callback)
        if "right_img_dir" in args.camera_names:
            right_img_dir_sub = rospy.Subscriber("/jhu_daVinci/right/image_raw", Image, right_camera_callback)

    # -------------- Init the recorded episode data (if video or random) --------------

    if args.input_type == "video": 
        # Capture the video stream
        cap = cv2.VideoCapture(args.video_file_path)
    
    if args.input_type == "random":
        # Generate a random episode
        episode_sequence, episode_gt_instruction_sequence = create_random_chole_episode(args.dataset_dir, args.camera_names, args.camera_file_suffixes_aspect_ratio, args.downsampling_shape, center_crop_flag) 
    
        # Check if the gt instructions are within the learned instructions of the model
        gt_instructions_set = set(episode_gt_instruction_sequence)
        candidate_texts_set = set(candidate_texts)
        if not gt_instructions_set.issubset(candidate_texts_set):
            raise ValueError(f"The ground truth instructions are not within the learned instructions of the model. Missing instructions: {gt_instructions_set - candidate_texts_set}")
    
    # -------------- Initialize evaluation variables --------------
    
    # Initialize the language instruction prediction and ground truth lists
    predictions, ground_truths = [], []
    
    # Evaluation timing variables
    timestamp = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
    inference_time_last_100_samples = [] 
    fps_list = []
    if args.save_video_flag:
        concat_frame_list = [] # Array to store the concatenated frames (for later saving as video)
    execution_times_dict = defaultdict(list)
    
    
    # ------------- Main loop -------------
    
    # Keeping the last frames (for the history length)
    model_input_frames = deque(maxlen=history_len+1)
    
    frame_idx = 0
    while True:
        # Track the inference time
        with measure_execution_time("Total inference time", execution_times_dict):
            # -------------- Load current camera frames --------------

            # TODO: Check integrated history step size
            with measure_execution_time("Loading video frame time", execution_times_dict):
                if frame_idx % history_step_size == 0:
                    if args.input_type == "random":
                        success, current_frames = get_current_frames(args.input_type, args.downsampling_shape, center_crop_flag, frame_idx=frame_idx, random_episode_sequence=episode_sequence)
                    elif args.input_type == "video":
                        success, current_frames = get_current_frames(args.input_type, args.downsampling_shape, center_crop_flag, video_capture=cap, camera_names=args.camera_names, camera_file_suffixes_aspect_ratio=args.camera_file_suffixes_aspect_ratio)
                    elif args.input_type == "live":
                        success, current_frames = get_current_frames(args.input_type, args.downsampling_shape, center_crop_flag, camera_names=args.camera_names, camera_file_suffixes_aspect_ratio=args.camera_file_suffixes_aspect_ratio)
                    # Break out of the loop if the image could not be loaded (e.g., end of video)
                    if not success:
                        break   # Stop the loop if the video has ended - and save the video up to here (if desired)
                    # Add the current frames to the model input frames + generate current input tensor
                    model_input_frames.append(current_frames)
                    model_input_frames_tensor = torch.stack(list(model_input_frames), dim=0).to(device).unsqueeze(0) # Shape: batch_size (=1), history_len, cam, c, h, w # TODO: Check if this is the correct dim
            
            # -------------- Predict (+publish) the language instruction --------------
            
            if frame_idx % args.prediction_stride == 0 and len(model_input_frames) == history_len+1:
                with measure_execution_time("Instructor_inference_time", execution_times_dict):
                    # Apply the model on the current frames
                    logits, temperature, predicted_embedding = instructor_model(model_input_frames_tensor)
                
                    # Decode the model output to the language instruction
                    predicted_instruction = instructor_model.decode_logits(logits, temperature)[0]
            elif len(model_input_frames) < history_len+1:
                # Wait with predictions until the history length is reached and begin with the first instruction (which is the same for all demos)
                # TODO: Can I get the first command like that here not to do this differently? Or hardcode Grabbing the gallbladder (+ its embedding)?
                predicted_instruction = candidate_texts[0]
                predicted_embedding = candidate_embeddings[0]
            
            # Publish the predicted language instruction
            if args.input_type == "live":
                instruction_publisher.publish(predicted_instruction)
                instruction_embedding_publisher.publish(predicted_embedding) # TODO: Check if I need to convert to cpu and numpy array first?          
            
            # -------------- Visualize the frame --------------
            
            with measure_execution_time("Visualizing frame time", execution_times_dict):
                # Visualization of the predicted language instruction
                if args.input_type == "random":
                    gt_instruction = episode_gt_instruction_sequence[frame_idx + prediction_offset]
                    annotated_frame = visualize_current_frames(current_frames, predicted_instruction, gt_instruction)
                else:
                    annotated_frame = visualize_current_frames(current_frames, predicted_instruction)
            
            if args.save_video_flag:
                concat_frame_list.append(annotated_frame)

        # -------------- Update + log execution times --------------  
        
        inference_time = execution_times_dict["Total inference time"][-1]
        fps_list.append(1 / inference_time)
        inference_time_last_100_samples.append(inference_time)
        if len(inference_time_last_100_samples) > 100:
            inference_time_last_100_samples.pop(0)
            
        # Log the average execution times of the last 100 samples
        print("") # Add empty line for better readability
        for key, value in execution_times_dict.items():
            if key != "Total inference time":
                print(f"{key} (last {len(inference_time_last_100_samples)} samples): {np.mean(value[-100:])*1000:.2f} ms")
        print(f"--> Total inference time (last {len(inference_time_last_100_samples)} samples): {np.mean(inference_time_last_100_samples)*1000:.2f} ms")

        print("\n------------------------------------------------------")

        frame_idx += 1 # Increase the frame index

        # -------------- Stop the pipeline + clean up --------------

        # Break out of the loop if the user pressed q
        if cv2.waitKey(1) == ord('q'):
            break
        
        # --------------------- Ros rate ---------------------
        
        if args.input_type == "live":
            rate.sleep()

    cap.release()
    cv2.destroyAllWindows()
    plt.close('all')
    
    # -------------- Save the detected images as video --------------
    
    # Save the detected images as video (if desired)
    if args.save_video_flag:
        # Define the output folder
        output_folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "evaluation", "hl_policy_pipeline", args.input_type)

        # Create the recordings folder if it does not exist
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if args.input_type in ["video", "random"]:
            video_fps = 30 # FPS of the recording
        else:
            video_fps = np.mean(fps_list)
            
        # Create the video file name
        if args.input_type == "live":
            video_file_path = os.path.join(output_folder_path, f'{args.input_type}_{timestamp=}.mp4')
        else:
            additional_info = args.video_file_path.split("/")[-1].split(".")[0] if args.input_type == "video" else args.tissue_id
            video_file_path = os.path.join(output_folder_path, f'{args.input_type}_{additional_info}_{timestamp=}.mp4')
        video_dim = concat_frame_list[-1].shape[:2][::-1]
        out = cv2.VideoWriter(video_file_path, fourcc, video_fps, video_dim)
        # Write the detected images to the video
        for concat_frame in concat_frame_list:
            out.write(concat_frame)
        out.release()
        print(f"\nSaved the video to {video_file_path}")

    # ----------------------- Evaluation ----------------------------

    print("\n----------------------- Evaluation ----------------------------\n")

    # Evaluate the language instruction prediction
    metrics_dict = evaluate_instruction_prediction(predictions, ground_truths, timestamp)

    # Detection + triangulation evaluation
    print(f"Accuracy: {metrics_dict['accuracy']*100:.2f}")
    print(f"F1 score: {metrics_dict['f1_score']*100:.2f}")
    
    # Print average timings (for all different components, with keys in the dict + the total inference time + avg fps)
    print("\nAverage timings:")
    for key, value in execution_times_dict.items():
        if key != "Total inference time":
            print(f"{key}: {np.mean(value)*1000:.2f} ms")
    print(f"Total inference time: {np.mean(execution_times_dict['Total inference time'])*1000:.2f} ms")
    print(f"Median FPS: {np.median(fps_list):.2f}")   



# -------------------------------- Create parser --------------------------------

def parse_pipeline_args():
    """
    Define a parser that reads out the command line arguments and returns the parsed arguments.
    """
    
    parser = argparse.ArgumentParser(description='Surgical Tool Tracking Pipeline')

    # --------------------------- Instruction model parameters ---------------------------
    
    # Instructor model path
    default_ckpt_folder_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "model_ckpts", "hl", "debugging2")
    default_ckpt_file_name = "best_val_loss_epoch=71"
    parser.add_argument('--ckpt_path', type=str, default=os.path.join(default_ckpt_folder_path, f"{default_ckpt_file_name}.ckpt"),
                        help="Path to the YOLO model file")

    # Prediction stride value to only predict every x frames
    parser.add_argument('--prediction_stride', type=int, default=1, help="Prediction stride value (e.g., predict every x frames)")

    # ---------------------------------- Data parameters -------------------------------------
    
    # Offline testing flag (on own video)
    parser.add_argument('--input_type', type=str, default="random",
                        help="Can be either 'live', 'video' or 'random' (for random generated episode)")
    
    # Image size
    parser.add_argument('--downsampling_shape', type=tuple, default=(224, 224), help="Desired size of the input image")
    
    # Add video file path
    default_video_path = "/home/phansen/JHU-Projects/yay_robot_jhu/script/chole/GeneratedStitchedEpisodes/tissue=4_20240715-000355-095607/randomly_stitched_episode_tissue_4_1.avi"
    parser.add_argument('--video_file_path', type=str, default=default_video_path,  
                        help="Path to recording to test the system offline with the KUKA simulation)")
    
    # ROS fps
    parser.add_argument('--ros_fps', type=int, default=30, help="FPS of the ROS node")
    
    # Camera names - based on the order of the cameras, the images will be stacked together as model input
    parser.add_argument('--camera_names', type=str, nargs='+', default=["endo_psm2", "left_img_dir", "endo_psm1"], help="Names of the cameras") # "right_img_dir"
    
    # Camera file suffixes and camera dimension dict (width, height)
    default_camera_file_suffixes_aspect_ratio_dict = {
        "endo_psm2": ("_psm2.jpg", 3/2),
        "left_img_dir": ("_left.jpg", 16/9),
        "right_img_dir": ("_right.jpg", 16/9),
        "endo_psm1": ("_psm1.jpg", 3/2)
        }
    parser.add_argument('--camera_file_suffixes_aspect_ratio', type=dict, default=default_camera_file_suffixes_aspect_ratio_dict,
                        help="Dictionary with the camera file suffixes and dimensions")
    
    # Add dataset directory
    default_dataset_name = "debugging2" # TODO: Later use: base_chole_clipping_cutting
    default_dataset_dir = os.path.join(os.getenv("PATH_TO_DATASET"), default_dataset_name)
    parser.add_argument('--dataset_dir', type=str, default=default_dataset_dir, help="Path to the dataset directory")
    
    # Add tissue id
    parser.add_argument('--tissue_id', type=int, default=4, help="Tissue id for the random generated episode")
    
    # Add save video flag
    parser.add_argument('--save_video_flag', action='store_true', default=True,
                    help="Flag to save the instrument detection video")

    # ----------------------------------------------------------------------------------------------

    # Parse the command line arguments
    args = parser.parse_args()
    
    return args  

if __name__ == '__main__':
    args = parse_pipeline_args()
    instructor_pipeline(args)  