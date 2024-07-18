"""
Example usage:

python src/instructor/train_daVinci.py     --dataset_name base_chole_clipping_cutting      --ckpt_dir $YOUR_CKPT_PATH/hl/base_chole_clipping_cutting_clip_reduced_set     --batch_size 128     --num_epochs 15000     --lr 1e-4     --history_step_size 30    --prediction_offset 12     --history_len 3     --seed 3   --load_best_ckpt_flag --one_hot_flag --plot_val_images_flag --max_num_images 5 --cameras_to_use left_img_dir endo_psm1 --backbone_model clip --model_init_weights dino --freeze_backbone_until all --reduced_base_class_set_flag
"""

import os
import argparse
import threading
import sys

import torch
import torch.optim as optim
import numpy as np
import wandb
import torchvision.transforms as transforms
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score

# import aloha 
path_to_yay_robot = os.getenv('PATH_TO_YAY_ROBOT')
if path_to_yay_robot:
    sys.path.append(os.path.join(path_to_yay_robot, 'src'))
else:
    raise EnvironmentError("Environment variable PATH_TO_YAY_ROBOT is not set")
# from aloha_pro.aloha_scripts.utils import crop_resize, random_crop, initialize_model_and_tokenizer, encode_text
from aloha_pro.aloha_scripts.utils import memory_monitor
from instructor.dataset_daVinci import load_merged_data
from instructor.model_daVinci import Instructor



def train(model, dataloader, optimizer, criterion, device, ckpt_dir, current_epoch, max_num_images=10):
    model.train()
    total_loss = 0.0
    for batch_idx, batch in enumerate(dataloader):
        images, _, commands = batch
        images = images.to(device)

        optimizer.zero_grad()
        logits, temperature, _ = model(images)

        # Convert ground truth command strings to indices using the pre-computed dictionary
        commands_idx = [model.command_to_index[cmd] for cmd in commands] 
        commands_idx = torch.tensor(commands_idx, device=device)

        loss = criterion(logits, commands_idx)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if args.log_wandb:
            wandb.log({"Train Loss": loss.item(), "Temperature": temperature.item()})
            
        # Save images from the last batch (to see, e.g., the augmentation applied)
        if batch_idx == len(dataloader) - 1:
            saved_img_cnt = 0
            for img_idx in range(len(images)):
                if saved_img_cnt >= max_num_images:
                    break
                
                gt = commands[img_idx]
                pred = model.decode_logits(logits[img_idx].unsqueeze(0), temperature)[0]

                save_path = os.path.join(ckpt_dir, "training_images", f"epoch_{current_epoch}_{batch_idx}_{img_idx}.jpg")
                log_combined_image(images[img_idx], gt, pred, save_path)
                
                if args.log_wandb:
                    wandb.log({f"Training Image {saved_img_cnt}": wandb.Image(save_path, caption=f"Epoch {current_epoch}, Batch {batch_idx}, Image {img_idx}")})

                saved_img_cnt += 1
            
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader: 
            images, _, commands = batch
            images = images.to(device)

            logits, temperature, _ = model(images)

            # Convert ground truth command strings to indices using the pre-computed dictionary
            commands_idx = [model.command_to_index[cmd] for cmd in commands]
            commands_idx = torch.tensor(commands_idx, device=device)

            loss = criterion(logits, commands_idx)
            total_loss += loss.item()

            if args.log_wandb:
                wandb.log({"Eval Loss": loss.item(), "Temperature": temperature.item()})
        
    return total_loss / len(dataloader)


def test(model, dataloader, split_name, device, current_epoch, one_hot_flag, ckpt_dir, max_num_images = 10, plot_images_flag=True, log_wandb_flag=True, reduced_base_class_set_flag=False):

    model.eval()

    all_commands_gt = []
    all_decoded_texts = []
    all_decoded_texts_masked = []

    if not one_hot_flag:
        all_predicted_embeddings = []
        all_gt_embeddings = []

    if plot_images_flag:
        incorrect_img_cnt = correct_img_cnt = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images, command_embedding_gt, command_gt = batch
            images = images.to(device)

            logits, temperature, predicted_embedding = model(images)
            
            # TODO: Maybe integrate later again, but would need phase index of the episode
            if not reduced_base_class_set_flag:
                # Only consider current (gt) command and next command for the mask
                # TODO: Add later also the recovery commands (and spatial commands)
                # Get a list of the current commands
                current_gt_command_idx = [model.command_to_index[cmd] for cmd in command_gt]
                # Save here a mask that will only consider the gt command and the next command (later also recovery commands)
                current_command_mask = torch.zeros_like(logits)
                for idx, command_idx in enumerate(current_gt_command_idx):
                    current_command_mask[idx, command_idx] = 1
                    if command_idx < len(model.candidate_texts) - 1:
                        current_command_mask[idx, command_idx+1] = 1
                        
                # Apply softmax and then set everything to 0 using the mask
                logits_masked = torch.nn.functional.softmax(logits, dim=-1) * current_command_mask 
                
                # Get text for each prediction in the batch (masked by the current command)
                decoded_texts_masked = model.decode_logits(logits_masked, temperature)

            decoded_texts = model.decode_logits(logits, temperature) # Also get the text for the unmasked logits

            # Store the ground truth and predicted commands for the confusion matrix
            all_commands_gt.extend(command_gt)
            all_decoded_texts.extend(decoded_texts)
            if not reduced_base_class_set_flag:
                all_decoded_texts_masked.extend(decoded_texts_masked)

            if not one_hot_flag:
                all_predicted_embeddings.extend(predicted_embedding.cpu().numpy())
                all_gt_embeddings.extend(command_embedding_gt.cpu().numpy())

            for img_idx, (gt, pred) in enumerate(zip(command_gt, decoded_texts)):    
                if plot_images_flag:            
                    # Save incorrect prediction
                    if pred != gt and incorrect_img_cnt < max_num_images:
                        incorrect_img_cnt += 1
                        save_path = os.path.join(ckpt_dir, "predictions", f"{current_epoch=}_incorrect_{batch_idx=}_{img_idx}.jpg")
                        log_combined_image(images[img_idx], gt, pred, save_path)
                        if args.log_wandb:
                            wandb.log({f"Incorrect Prediction": wandb.Image(save_path, caption=f"Epoch {current_epoch}, Batch {batch_idx}, Image {img_idx}")})
                    # Save correct prediction
                    if pred == gt and correct_img_cnt < max_num_images:
                        correct_img_cnt += 1
                        save_path = os.path.join(ckpt_dir, "predictions", f"epoch_{current_epoch}_correct_{batch_idx}_{img_idx}.jpg")
                        log_combined_image(images[img_idx], gt, pred, save_path)
                        if args.log_wandb:
                            wandb.log({f"Correct Prediction": wandb.Image(save_path, caption=f"Epoch {current_epoch}, Batch {batch_idx}, Image {img_idx}")})
                
    # Visualize embeddings
    if not one_hot_flag:
        # Save the t-SNE visualization of the embeddings of the last batch
        tnse_plots_folder_path = os.path.join(ckpt_dir, "tsne_plots")
        if not os.path.exists(tnse_plots_folder_path):
            os.makedirs(tnse_plots_folder_path, exist_ok=True)
        save_path = os.path.join(tnse_plots_folder_path, f"embeddings_tsne_epoch_{epoch}.png")
        log_tsne_plot(candidate_embeddings, candidate_texts, all_predicted_embeddings, all_decoded_texts, all_gt_embeddings, current_epoch, save_path)

    # Create and save confusion matrix
    conf_matrix_folder_path = os.path.join(ckpt_dir, "confusion_matrices")
    if not os.path.exists(conf_matrix_folder_path):
        os.makedirs(conf_matrix_folder_path, exist_ok=True)
    save_path = os.path.join(conf_matrix_folder_path, f"{split_name}_confusion_matrix_epoch_unmasked_{current_epoch}.png")
    log_confusion_matrix(all_commands_gt, all_decoded_texts, candidate_texts, split_name, current_epoch, save_path, log_wandb_flag, add_info="Unmasked")
    if not reduced_base_class_set_flag:
        log_confusion_matrix(all_commands_gt, all_decoded_texts_masked, candidate_texts, split_name, current_epoch, save_path.replace("Unmasked", "Masked"), log_wandb_flag, add_info="Masked")

    # Compute metrics for unmasked and masked predictions
    print()
    for name, decoded_texts in zip(["Unmasked", "Masked"], [all_decoded_texts, all_decoded_texts_masked]):
        # Compute the success rate -> accurarcy
        accurarcy_curr_epoch = accuracy_score(all_commands_gt, decoded_texts)
        if args.log_wandb:
            wandb.log({f"{name} Accurarcy": accurarcy_curr_epoch})
            
        # Compute the (macro) F1 score
        f1_score_curr_epoch = f1_score(all_commands_gt, decoded_texts, average='macro')
        if args.log_wandb:
            wandb.log({f"{name} F1 Score": f1_score_curr_epoch})
        
        print(f"Epoch {current_epoch}: {name} Accuracy = {accurarcy_curr_epoch * 100:.2f}% - {name} F1 Score = {f1_score_curr_epoch * 100:.2f}%")
        
        if reduced_base_class_set_flag:
            break # Only compute the metrics for the unmasked predictions when using the reduced base class set

# ----------------------------

def log_combined_image(image, gt_text, pred_text, save_path=None):
    # image = image[:, :, [2, 1, 0]]

    num_ts = image.shape[0]
    
    if num_ts <= 5:
        # Extract frames for all timesteps and concatenate across width
        for t in range(num_ts):
            combined_image = torch.cat([image[t, cam_idx] for cam_idx in range(image.shape[1])], dim=-1)
            if t == 0:
                combined_image_all = combined_image
            else:
                combined_image_all = torch.cat([combined_image_all, combined_image], dim=-2)
        combined_image = combined_image_all
    else:
        # Extract last frame and concatenate across width
        combined_image = torch.cat([image[-1, cam_idx] for cam_idx in range(image.shape[1])], dim=-1)

    # Convert to PIL image
    combined_image_pil = transforms.ToPILImage()(combined_image)

    # Create a blank canvas to add text
    canvas = Image.new(
        "RGB", (combined_image_pil.width, combined_image_pil.height + 100), "black"
    )
    canvas.paste(combined_image_pil, (0, 100))

    # Add GT and predicted text
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=38)
    draw.text((10, 10), "GT: " + gt_text, font=font, fill="white")
    if gt_text != pred_text:
        draw.text((10, 50), "Pred: " + pred_text, font=font, fill="red")
    else:
        draw.text((10, 50), "Pred: " + pred_text, font=font, fill="green")

    canvas.save(save_path)


def log_confusion_matrix(y_true, y_pred, classes, split_name=None, epoch=None, save_path=None, log_wandb_flag=True, add_info="Unmasked"):
    """
    Compute the confusion matrix for each criteria.
    
    Args:
        y_true_all_criteria (torch.Tensor): True labels for all criteria
        y_pred_all_criteria (torch.Tensor): Predicted labels for all criteria
        split_name (str): Name of the split (e.g., "train", "val")
        epoch (int): Current epoch - If None, no epoch is logged (for final confusion matrix after training)
    """
    
    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        
        # Create a new fig
        fig = plt.figure(figsize=(8, 8))
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar(shrink=0.7)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        
        return fig
    
    # Log the confusion matrix with WandB
    if epoch is not None:
        fig = plot_confusion_matrix(confusion_matrix(y_true, y_pred, labels=classes), classes=classes, title=f"Confusion Matrix ({add_info}) (Epoch {epoch})")
    else:
        fig = plot_confusion_matrix(confusion_matrix(y_true, y_pred, labels=classes), classes=classes, title=f"Confusion Matrix ({add_info})")
    if log_wandb_flag:
        if split_name is None:
            wandb.log({"confusion_matrix": fig})
        else:
            wandb.log({f"{split_name=}_confusion_matrix": fig})
    
    # Save with the epoch in the filename
    plt.savefig(save_path)
    plt.close()


# TODO: check if it works corrrectly - close points in high-dimensional space should be close in the 2D space
def log_tsne_plot(candidate_embeddings, candidate_commands, predicted_embeddings, predicted_commands, gt_embeddings, epoch, save_path):
    
    # Convert lists to numpy arrays
    candidate_embeddings = np.array(candidate_embeddings)
    gt_embeddings = np.array(gt_embeddings)
    predicted_embeddings = np.array(predicted_embeddings)

    # Check that all predicted commands are within the candidate commands
    all_unique_commands_set = set(candidate_commands)
    all_unique_predicted_commands_set = set(predicted_commands)
    if not all_unique_predicted_commands_set.issubset(all_unique_commands_set):
        print(f"\nCommands that are not in the candidate commands: {all_unique_predicted_commands_set - all_unique_commands_set}")
        raise ValueError("All predicted commands should be within the candidate commands")

    # Generate a color palette
    base_colors = sns.color_palette("husl", len(candidate_commands))
    color_map = {command: color for command, color in zip(candidate_commands, base_colors)}

    # Stack embeddings and apply t-SNE
    all_embeddings = np.vstack([predicted_embeddings, gt_embeddings, candidate_embeddings])
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    # Split the 2D embeddings back (not interested in the gt_embeddings, only for stability of the t-SNE)
    predicted_2d = embeddings_2d[: len(predicted_embeddings)]
    candidate_2d = embeddings_2d[len(predicted_embeddings):]

    # Plot the results
    plt.figure(figsize=(12, 10))

    # Plot candidate embeddings
    for i, command in enumerate(candidate_commands):
        plt.scatter(candidate_2d[i, 0], candidate_2d[i, 1], color=color_map[command], alpha= 1, label=f"{command}" if command not in candidate_commands[:i] else "")

    # Plot predicted embeddings
    for i, command in enumerate(predicted_commands):
        # TODO: Show the predicted command via marker filling and the gt command via marker edge
        plt.scatter(predicted_2d[i, 0], predicted_2d[i, 1], color=color_map[command], alpha=0.5)

    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title(f"t-SNE Visualization of Embeddings (Epoch {epoch})")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # Save with the epoch in the filename
    plt.savefig(save_path)

    # Log the image to wandb if logging is enabled
    if args.log_wandb:
        wandb.log({"t-SNE Visualization": [wandb.Image(save_path, caption=f"Epoch {epoch}")]})
    plt.close()

# -----------------------------

def build_instructor(history_len, history_step_size, prediction_offset, candidate_embeddings, candidate_texts, device, one_hot_flag, camera_names, center_crop_flag, backbone_model_name, model_init_weights, freeze_backbone_until):
    # Map command texts to indices
    command_to_index = {command: index for index, command in enumerate(candidate_texts)}

    # Build model
    candidate_embeddings = candidate_embeddings.to(device)
    model = Instructor(
        device=device,
        history_len=history_len,
        history_step_size=history_step_size,
        prediction_offset=prediction_offset,
        candidate_embeddings=candidate_embeddings,
        candidate_texts=candidate_texts,
        command_to_index=command_to_index,
        one_hot_flag=one_hot_flag,
        camera_names=camera_names,
        center_crop_flag=center_crop_flag,
        backbone_model_name=backbone_model_name,
        model_init_weights=model_init_weights,
        freeze_backbone_until=freeze_backbone_until,
    ).to(device)
    return model


def best_checkpoint(ckpt_dir):
    """
    Returns the best checkpoint file from the given directory (if exists best).
    """

    # Starts with "best_val_loss_" and ends with ".ckpt" - could be multiple from different ckpt runs - take the last one
    best_val_ckpt_name_list = [
        file_name
        for file_name in os.listdir(ckpt_dir)
        if file_name.startswith("best_val_loss_") and file_name.endswith(".ckpt")
    ]
    
    epoch_numbers = [int(file_name.split("=")[1].split(".")[0]) for file_name in best_val_ckpt_name_list]

    # If no valid checkpoints are found, return None
    if not epoch_numbers:
        return None, None

    latest_best_idx = max(epoch_numbers)
    next_idx = latest_best_idx + 1
    return os.path.join(ckpt_dir, f"best_val_loss_epoch={latest_best_idx}.ckpt"), next_idx

def latest_checkpoint(ckpt_dir):
    """
    Returns the latest checkpoint file from the given directory.
    """
    all_ckpts = [
        f
        for f in os.listdir(ckpt_dir)
        if f.startswith("epoch_") and f.endswith(".ckpt")
    ]
    epoch_numbers = [int(file_name.split("_")[1].split(".")[0]) for file_name in all_ckpts]

    # If no valid checkpoints are found, return None
    if not epoch_numbers:
        return None, None

    latest_idx = max(epoch_numbers)
    return os.path.join(ckpt_dir, f"epoch_{latest_idx}.ckpt"), latest_idx


if __name__ == "__main__":
    from instructor.utils import set_seed
    from aloha_pro.aloha_scripts.constants_daVinci import DATASET_CONFIGS # get dataset parameters
    
    threading.Thread(target=memory_monitor, daemon=True).start()

    parser = argparse.ArgumentParser(description="Train and evaluate command prediction model using CLIP.")
    parser.add_argument('--dataset_names', nargs='+', type=str, help='List of dataset names', required=True)
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    parser.add_argument('--weight_decay', action='store', type=float, help='weight_decay', default=0.01)
    parser.add_argument('--log_wandb', action='store_true')
    parser.add_argument('--gpu', action='store', type=int, help='gpu', default=0)
    parser.add_argument('--history_len', action='store', type=int, help='history_len', default=3)
    parser.add_argument('--prediction_offset', action='store', type=int, help='prediction_offset', default=15)
    parser.add_argument('--history_step_size', action='store', type=int, help='history_step_size', default=30)
    parser.add_argument('--test_only_flag', action='store_true', help='Test the model using the latest checkpoint and exit')
    parser.add_argument('--one_hot_flag', action='store_true', help='Use one hot encoding for the commands')
    parser.add_argument('--dagger_ratio', action='store', type=float, help='dagger_ratio', default=None)
    parser.add_argument('--validation_interval', action='store', type=int, help='validation_interval', default=3)
    parser.add_argument('--save_ckpt_interval', action='store', type=int, help='save_ckpt_interval', default=100)
    parser.add_argument('--early_stopping_interval', action='store', type=int, help='early_stopping_interval', default=None)
    parser.add_argument('--load_best_ckpt_flag', action='store_true', help='Use the best checkpoint based on the validation loss if continue training on available checkpoint')
    parser.add_argument('--center_crop_flag', action='store_true', help='Center crop the images during preprocessing, preventing unnatural rescaling, but potentially cutting off important information')
    parser.add_argument('--plot_val_images_flag', action='store_true', help='Plot images for correct and incorrect predictions')
    parser.add_argument('--max_num_images', action='store', type=int, help='Maximum number of images to plot for correct and incorrect predictions', default=10)
    parser.add_argument('--cameras_to_use', nargs='+', type=str, help='List of camera names to use', default=["endo_psm2", "left_img_dir", "right_img_dir", "endo_psm1"])
    parser.add_argument('--reduced_base_class_set_flag', action='store_true', help='Use a reduced set of base classes')
    parser.add_argument('--backbone_model_name', action='store', type=str, help='backbone_model_name', default="clip")
    # gsvit - possible weights: general | cholecystectomy | imagenet
    # resnet - possible weights: imagenet | mocov2 | simclr | swav | dino 
    # endovit - possible weights: endo700k | imagenet
    parser.add_argument('--model_init_weights', action='store', type=str, help='model_init_weights', default="imagenet")
    parser.add_argument('--freeze_backbone_until', action='store', type=str, help='freeze_backbone_until', default="all") 
    
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Device setting
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}\n")

    dataset_dirs = []
    num_episodes_list = []

    for dataset in args.dataset_names:
        dataset_config = DATASET_CONFIGS[dataset]
        dataset_dirs.append(dataset_config["dataset_dir"])
        num_episodes_list.append(dataset_config["num_episodes"])
    camera_names = [camera_name for camera_name in dataset_config["camera_names"] if camera_name in args.cameras_to_use]
    camera_file_suffixes = [camera_file_suffix for camera_file_suffix, camera_name in zip(dataset_config["camera_file_suffixes"], dataset_config["camera_names"]) if camera_name in args.cameras_to_use]

    # ---------------------- Define dataloaders and model ----------------------

    # Define transforms/augmentations (resize transformation already applied in __getitem__ method)
    # TODO: Decide for the best augmentations - maybe load only these defined in the args?!
    framewise_transforms = []
    framewise_transforms.append(transforms.RandomRotation(15))
    framewise_transforms.append(transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)))
    framewise_transforms.append(transforms.RandomResizedCrop(224, scale=(0.8, 1.0)))
    
    # framewise_transforms.append(v2.RandomPerspective(p=0.5))
    # framewise_transforms.append(v2.RandomPosterize(bits=7, p=0.25))
    # framewise_transforms.append(v2.RandomAdjustSharpness(2, p=0.25))
    # framewise_transforms.append(transforms.RandomApply([v2.GaussianBlur(kernel_size=5)], p=0.75))
    # framewise_transforms.append(transforms.RandomApply([transforms.RandomResizedCrop(224, scale=(0.5, 1.0))]))
    # framewise_transforms.append(v2.RandomPhotometricDistort(p=0.8))
    # framewise_transforms.append(transforms.RandomGrayscale(p=0.2))
    framewise_transforms = transforms.Compose(framewise_transforms)

    # Data loading
    if args.dagger_ratio is not None and not args.test_only_flag:
        train_dataloader, ds_metadata_dict = load_merged_data(
            dataset_dirs=dataset_dirs,
            num_episodes_list=num_episodes_list,
            camera_names=camera_names,
            camera_file_suffixes=camera_file_suffixes,
            batch_size_train=args.batch_size,
            batch_size_val=args.batch_size,
            history_len=args.history_len,
            prediction_offset=args.prediction_offset,
            history_step_size=args.history_step_size,
            test_only=args.test_only_flag,
            framewise_transforms=framewise_transforms,
            dagger_ratio=args.dagger_ratio,
            center_crop_flag=args.center_crop_flag,
            reduced_base_class_set_flag=args.reduced_base_class_set_flag,
        )
    elif not args.test_only_flag:
        train_dataloader, val_dataloader, ds_metadata_dict = load_merged_data(
            dataset_dirs=dataset_dirs,
            num_episodes_list=num_episodes_list,
            camera_names=camera_names,
            camera_file_suffixes=camera_file_suffixes,
            batch_size_train=args.batch_size,
            batch_size_val=args.batch_size,
            history_len=args.history_len,
            prediction_offset=args.prediction_offset,
            history_step_size=args.history_step_size,
            test_only=args.test_only_flag,
            framewise_transforms=framewise_transforms,
            dagger_ratio=args.dagger_ratio,
            center_crop_flag=args.center_crop_flag,
            reduced_base_class_set_flag=args.reduced_base_class_set_flag,
        )
    else:
        test_dataloader, ds_metadata_dict = load_merged_data(
            dataset_dirs=dataset_dirs,
            num_episodes_list=num_episodes_list,
            camera_names=camera_names,
            camera_file_suffixes=camera_file_suffixes,
            batch_size_train=args.batch_size,
            batch_size_val=args.batch_size,
            history_len=args.history_len,
            prediction_offset=args.prediction_offset,
            history_step_size=args.history_step_size,
            test_only=args.test_only_flag,
            framewise_transforms=framewise_transforms,
            dagger_ratio=args.dagger_ratio,
            center_crop_flag=args.center_crop_flag,
            reduced_base_class_set_flag=args.reduced_base_class_set_flag,
        )

    # Merge ds_metadata_dict with args (use as wandb config)
    wandb_metadata = ds_metadata_dict.copy()  # Create a copy to avoid modifying the original dict
    wandb_metadata.update(vars(args))

    # WandB initialization
    if args.log_wandb:
        wandb_entity = os.getenv("WANDB_ENTITY")
        run_name = "instructor." + args.ckpt_dir.split("/")[-1] + f".{args.seed}"
        wandb_run_id_path = os.path.join(args.ckpt_dir, "wandb_run_id.txt")
        # check if it exists
        if os.path.exists(wandb_run_id_path): 
            with open(wandb_run_id_path, "r") as f:
                saved_run_id = f.read().strip()
            wandb.init(
                project="yay-surgical-robot", entity=wandb_entity, name=run_name, resume=saved_run_id, config=wandb_metadata
            )
        else:
            wandb.init(
                project="yay-surgical-robot",
                entity=wandb_entity,
                name=run_name,
                config=wandb_metadata,
                resume="allow",
            )
            # Ensure the directory exists before trying to open the file
            os.makedirs(os.path.dirname(wandb_run_id_path), exist_ok=True)
            with open(wandb_run_id_path, "w") as f:
                f.write(wandb.run.id)


    # Build the model
    candidate_embeddings = ds_metadata_dict["candidate_embeddings"]
    candidate_texts = ds_metadata_dict["candidate_texts"]    
    model = build_instructor(args.history_len, args.history_step_size, args.prediction_offset, candidate_embeddings, 
                             candidate_texts, device, args.one_hot_flag, camera_names, args.center_crop_flag,
                             args.backbone_model_name, args.model_init_weights, args.freeze_backbone_until)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    # Load the most recent checkpoint if available
    if not os.path.isdir(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
        next_idx = 0
    else:
        # Load the most recent checkpoint if available
        if args.load_best_ckpt_flag:
            latest_ckpt, next_idx = best_checkpoint(args.ckpt_dir)
        else:
            latest_ckpt, next_idx = latest_checkpoint(args.ckpt_dir)
        if latest_ckpt:
            print(f"\nLoading checkpoint: {latest_ckpt}")
            latest_ckpt_dict = torch.load(latest_ckpt, map_location=device).state_dict()
            model.load_state_dict(latest_ckpt_dict)
        else:
            print("\nNo checkpoint found.")
            next_idx = 0

    # ---------------------- Training loop ----------------------

    # Create a directory to save predictions for the current run
    predictions_dir = os.path.join(args.ckpt_dir, "predictions")
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)
        
    # Create a directory to save training images for the current run
    training_images_dir = os.path.join(args.ckpt_dir, "training_images")
    if not os.path.exists(training_images_dir):
        os.makedirs(training_images_dir)

    # Test the model using the latest checkpoint - don't train
    if args.test_only_flag:
        latest_idx = next_idx-1
        test(model, test_dataloader, "test", device, latest_idx, args.one_hot_flag, args.ckpt_dir, 
             log_wandb_flag=args.log_wandb, reduced_base_class_set_flag=args.reduced_base_class_set_flag)
        exit()

    # Training loop
    pbar_epochs = tqdm(range(next_idx, args.num_epochs), desc="Epochs")
    best_val_loss = float("inf")
    first_iteration_flag = True
    for epoch in pbar_epochs:
        if args.log_wandb:
            wandb.log({"Epoch": epoch})
        
        train_loss = train(model, train_dataloader, optimizer, criterion, device, args.ckpt_dir, epoch, args.max_num_images)
        if args.dagger_ratio is None: 
            val_loss = evaluate(model, val_dataloader, criterion, device)
            
            # Test the model and log success rate every 200 epochs
            if (epoch + 1) % args.validation_interval == 0:
                test(model, val_dataloader, "val", device, epoch, args.one_hot_flag, args.ckpt_dir, plot_images_flag=args.plot_val_images_flag, 
                     log_wandb_flag=args.log_wandb, reduced_base_class_set_flag=args.reduced_base_class_set_flag)

        pbar_epochs.set_postfix({"Train Loss": train_loss, "Val Loss": val_loss if args.dagger_ratio is None else None})

        if args.log_wandb:
            wandb.log({"Epoch Train Loss": train_loss})
            if args.dagger_ratio is None:
                wandb.log({"Epoch Eval Loss": val_loss})
        else:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # -------------------------- Checkpoints --------------------------

        # Save a checkpoint every 100 epochs
        if epoch % args.save_ckpt_interval == 0 and epoch > 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"epoch_{epoch}.ckpt")
            torch.save(model, ckpt_path)

            # Pruning: this removes the checkpoint save_ckpt_interval epochs behind the current one
            # except for the ones at multiples of prune_freq epochs
            prune_freq = args.save_ckpt_interval * 3
            prune_epoch = epoch - args.save_ckpt_interval
            if prune_epoch % prune_freq != 0:
                prune_path = os.path.join(args.ckpt_dir, f"epoch_{prune_epoch}.ckpt")
                if os.path.exists(prune_path):
                    os.remove(prune_path)
                    
        # Save always the best performing model based on the validation loss
        if args.dagger_ratio is None and val_loss < best_val_loss:
            best_val_loss = val_loss
            if not first_iteration_flag:
                prev_best_val_epoch = best_val_epoch
            best_val_epoch = epoch
            best_ckpt_path = os.path.join(args.ckpt_dir, f"best_val_loss_{epoch=}.ckpt")
            torch.save(model, best_ckpt_path)
            # Remove the previous best checkpoint if it exists
            if not first_iteration_flag:
                prev_best_ckpt_path = os.path.join(args.ckpt_dir, f"best_val_loss_epoch={prev_best_val_epoch}.ckpt")
                if os.path.exists(prev_best_ckpt_path):
                    os.remove(prev_best_ckpt_path)
            first_iteration_flag = False
            
        # Early stopping: Stop training if the validation loss has not improved for specific number of epochs
        if args.early_stopping_interval is not None:
            if epoch - best_val_epoch >= args.early_stopping_interval:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    if args.log_wandb:
        wandb.finish()
