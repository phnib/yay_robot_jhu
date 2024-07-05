"""
Example usage:

python src/instructor/train_daVinci.py \
    --task_name debugging \
    --ckpt_dir $YOUR_CKPT_PATH/debugging_ckpt \
    --batch_size 64 \
    --num_epochs 15000 \
    --lr 1e-4 \
    --history_skip_frame 30 \
    --prediction_offset 15 \
    --history_len 3 \
    --seed 0 \
    --log_wandb
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
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import f1_score

# import aloha # TODO: Rather do this via absolute imports
path_to_yay_robot = os.getenv('PATH_TO_YAY_ROBOT')
if path_to_yay_robot:
    sys.path.append(os.path.join(path_to_yay_robot, 'src'))
else:
    raise EnvironmentError("Environment variable PATH_TO_YAY_ROBOT is not set")
# from aloha_pro.aloha_scripts.utils import crop_resize, random_crop, initialize_model_and_tokenizer, encode_text
from aloha_pro.aloha_scripts.utils import memory_monitor
from instructor.dataset_daVinci import load_merged_data
from instructor.model_daVinci import Instructor



def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
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


def test(model, dataloader, split_name, device, current_epoch, one_hot_flag, max_num_images = 5):

    model.eval()

    # Initialize variables for the confusion matrix + tsne plots
    total_correct = 0
    total_predictions = 0

    all_predicted_embeddings = []
    all_gt_embeddings = []
    all_commands_gt = []
    all_decoded_texts = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images, command_embedding_gt, command_gt = batch
            images = images.to(device)

            logits, temperature, predicted_embedding = model(images)
            # Get nearest text for each prediction in the batch
            decoded_texts = model.decode_logits(logits, temperature)

            # Store the ground truth and predicted commands for the confusion matrix
            all_commands_gt.extend(command_gt)
            all_decoded_texts.extend(decoded_texts)

            all_predicted_embeddings.extend(predicted_embedding.cpu().numpy())
            all_gt_embeddings.extend(command_embedding_gt.cpu().numpy())

            incorrect_img_cnt = correct_img_cnt = 0
            for img_idx, (gt, pred) in enumerate(zip(command_gt, decoded_texts)):                
                # Save incorrect prediction
                if pred != gt and incorrect_img_cnt < max_num_images:
                    incorrect_img_cnt += 1
                    save_path = os.path.join(ckpt_dir, "predictions", f"{current_epoch=}_incorrect_{batch_idx=}_{img_idx}.jpg")
                    log_combined_image(images[img_idx], gt, pred, save_path)
                    if args.log_wandb:
                        wandb.log({f"Incorrect Prediction": wandb.Image(save_path, caption=f"Epoch {current_epoch}, Batch {batch_idx}, Image {img_idx}")})
                # Save correct prediction
                if pred == gt and correct_img_cnt < max_num_images:
                    save_path = os.path.join(ckpt_dir, "predictions", f"epoch_{current_epoch}_correct_{batch_idx}_{img_idx}.jpg")
                    log_combined_image(images[img_idx], gt, pred, save_path)
                    if args.log_wandb:
                        wandb.log({f"Correct Prediction": wandb.Image(save_path, caption=f"Epoch {current_epoch}, Batch {batch_idx}, Image {img_idx}")})

                total_correct += int(pred == gt)
                total_predictions += 1
                
    # Visualize embeddings
    if not one_hot_flag:
        # Plot the t-SNE visualization of the embeddings of the last batch
        log_tsne_plot(candidate_embeddings, candidate_texts, all_predicted_embeddings, all_decoded_texts, all_gt_embeddings, current_epoch)

    # Create confusion matrix
    log_confusion_matrix(all_commands_gt, all_decoded_texts, split_name, candidate_texts, current_epoch)

    # Compute the success rate -> accurarcy
    accurarcy_curr_epoch = total_correct / total_predictions
    if args.log_wandb:
        wandb.log({"Accurarcy": accurarcy_curr_epoch})
        
    # Compute the (macro) F1 score
    f1_score_curr_epoch = f1_score(all_commands_gt, all_decoded_texts, average='macro')
    if args.log_wandb:
        wandb.log({"F1 Score": f1_score_curr_epoch})
        
    print(f"\nEpoch {current_epoch}: Accuracy = {accurarcy_curr_epoch * 100:.2f}% - F1 Score = {f1_score_curr_epoch * 100:.2f}%")

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
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 30
    )
    draw.text((10, 10), "GT: " + gt_text, font=font, fill="white")
    draw.text((10, 50), "Pred: " + pred_text, font=font, fill="red")

    if save_path is not None:
        canvas.save(save_path)


def log_confusion_matrix(y_true, y_pred, split_name, classes, epoch=None):
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
    
    # TODO: Maybe need to shorten the names for the plots
    # Log the confusion matrix with WandB
    if epoch is not None:
        fig = plot_confusion_matrix(confusion_matrix(y_true, y_pred, labels=classes), classes=classes, title=f"Confusion Matrix (Epoch {epoch})")
        wandb.log({f"{split_name=}_confusion_matrix": fig})
        plt.close()
    else:
        fig = plot_confusion_matrix(confusion_matrix(y_true, y_pred, labels=classes), classes=classes, title=f"Confusion Matrix")
        wandb.log({f"{split_name=}_confusion_matrix": fig})
        plt.close()


# TODO: Double check if it works corrrectly - close points in high-dimensional space should be close in the 2D space
def log_tsne_plot(candidate_embeddings, candidate_commands, predicted_embeddings, predicted_commands, gt_embeddings, epoch):
    
    # Convert lists to numpy arrays
    candidate_embeddings = np.array(candidate_embeddings)
    gt_embeddings = np.array(gt_embeddings)
    predicted_embeddings = np.array(predicted_embeddings)

    # Check that all predicted commands are within the candidate commands
    all_unique_commands_set = set(candidate_commands)
    all_unique_predicted_commands_set = set(predicted_commands)
    if not all_unique_predicted_commands_set.issubset(all_unique_commands_set):
        print(f"Commands that are not in the candidate commands: {all_unique_predicted_commands_set - all_unique_commands_set}")
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
        plt.scatter(predicted_2d[i, 0], predicted_2d[i, 1], color=color_map[command], alpha=0.5)

    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title(f"t-SNE Visualization of Embeddings (Epoch {epoch})")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # Save with the epoch in the filename
    tnse_plots_folder_path = os.path.join(ckpt_dir, "tsne_plots")
    os.makedirs(tnse_plots_folder_path, exist_ok=True)
    image_save_path = os.path.join(tnse_plots_folder_path, f"embeddings_tsne_epoch_{epoch}.png")
    plt.savefig(image_save_path)

    # Log the image to wandb if logging is enabled
    if args.log_wandb:
        wandb.log(
            {
                "t-SNE Visualization": [
                    wandb.Image(image_save_path, caption=f"Epoch {epoch}")
                ]
            },
        )
        
    plt.close()

# -----------------------------

def build_instructor(history_len, candidate_embeddings, candidate_texts, device, one_hot_flag=False):
    # Map command texts to indices
    command_to_index = {command: index for index, command in enumerate(candidate_texts)}

    # Build model
    candidate_embeddings = candidate_embeddings.to(device)
    model = Instructor(
        device=device,
        history_len=history_len,
        candidate_embeddings=candidate_embeddings,
        candidate_texts=candidate_texts,
        command_to_index=command_to_index,
        one_hot_flag=one_hot_flag,
    ).to(device)
    return model


def latest_checkpoint(ckpt_dir):
    """
    Returns the latest checkpoint file from the given directory.
    """
    all_ckpts = [
        f
        for f in os.listdir(ckpt_dir)
        if f.startswith("epoch_") and f.endswith(".ckpt")
    ]
    epoch_numbers = [int(f.split("_")[1].split(".")[0]) for f in all_ckpts]

    # If no valid checkpoints are found, return None
    if not epoch_numbers:
        return None, None

    latest_idx = max(epoch_numbers)
    return os.path.join(ckpt_dir, f"epoch_{latest_idx}.ckpt"), latest_idx


if __name__ == "__main__":
    from instructor.utils import set_seed
    from aloha_pro.aloha_scripts.constants_daVinci import DATASET_CONFIGS # get task parameters
    
    threading.Thread(target=memory_monitor, daemon=True).start()

    parser = argparse.ArgumentParser(description="Train and evaluate command prediction model using CLIP.")
    parser.add_argument('--task_name', nargs='+', type=str, help='List of task names', required=True)
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    parser.add_argument('--log_wandb', action='store_true')
    parser.add_argument('--gpu', action='store', type=int, help='gpu', default=0)
    parser.add_argument('--history_len', action='store', type=int, help='history_len', default=3)
    parser.add_argument('--prediction_offset', action='store', type=int, help='prediction_offset', default=15)
    parser.add_argument('--history_skip_frame', action='store', type=int, help='history_skip_frame', default=30)
    parser.add_argument('--test_only', action='store_true', help='Test the model using the latest checkpoint and exit')
    parser.add_argument('--one_hot_flag', action='store_true', help='Use one hot encoding for the commands')
    parser.add_argument('--dagger_ratio', action='store', type=float, help='dagger_ratio', default=None)
    parser.add_argument('--validation_interval', action='store', type=int, help='validation_interval', default=3)
    # TODO: Maybe add later a list of transformations/augmentations that should be applied as args

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Device setting
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    dataset_dirs = []
    num_episodes_list = []

    for task in args.task_name:
        task_config = DATASET_CONFIGS[task]
        dataset_dirs.append(task_config["dataset_dir"])
        num_episodes_list.append(task_config["num_episodes"])
        camera_names = task_config["camera_names"]
        camera_file_suffixes = task_config["camera_file_suffixes"]
    ckpt_dir = args.ckpt_dir
    dagger_ratio = args.dagger_ratio # TODO: Integrate later

    # WandB initialization
    if args.log_wandb:
        wandb_entity = os.getenv("WANDB_ENTITY")
        run_name = "instructor." + ckpt_dir.split("/")[-1] + f".{args.seed}"
        wandb_run_id_path = os.path.join(ckpt_dir, "wandb_run_id.txt")
        # check if it exists
        if os.path.exists(wandb_run_id_path) and False: # TODO: Add later again - ignore for now bc of the warnings
            with open(wandb_run_id_path, "r") as f:
                saved_run_id = f.read().strip()
            wandb.init(
                project="yay-surgical-robot", entity=wandb_entity, name=run_name, resume=saved_run_id
            )
        else:
            wandb.init(
                project="yay-surgical-robot",
                entity=wandb_entity,
                name=run_name,
                config=args,
                resume="allow",
            )
            # Ensure the directory exists before trying to open the file
            os.makedirs(os.path.dirname(wandb_run_id_path), exist_ok=True)
            with open(wandb_run_id_path, "w") as f:
                f.write(wandb.run.id)

    # ---------------------- Define dataloaders and model ----------------------

    # Define transforms/augmentations (resize transformation already applied in __getitem__ method)
    # TODO: Decide for the best augmentations - maybe load only these defined in the args?!
    framewise_transforms = []
    # framewise_transforms.append(transforms.RandomRotation(30))
    # framewise_transforms.append(transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)))
    # framewise_transforms.append(transforms.RandomApply([transforms.RandomResizedCrop(224, scale=(0.5, 1.0))]))
    # framewise_transforms.append(v2.RandomPerspective(p=0.5))
    # framewise_transforms.append(v2.RandomPosterize(bits=7, p=0.25))
    # framewise_transforms.append(v2.RandomAdjustSharpness(2, p=0.25))
    # framewise_transforms.append(transforms.RandomApply([v2.GaussianBlur(kernel_size=5)], p=0.75))
    # framewise_transforms.append(v2.RandomPhotometricDistort(p=0.8))
    # framewise_transforms.append(transforms.RandomGrayscale(p=0.2))
    framewise_transforms = transforms.Compose(framewise_transforms)

    # Data loading
    if not args.test_only:
        train_dataloader, val_dataloader, (candidate_embeddings, candidate_texts) = load_merged_data(
            dataset_dirs=dataset_dirs,
            num_episodes_list=num_episodes_list,
            camera_names=camera_names,
            camera_file_suffixes=camera_file_suffixes,
            batch_size_train=args.batch_size,
            batch_size_val=args.batch_size,
            history_len=args.history_len,
            prediction_offset=args.prediction_offset,
            history_skip_frame=args.history_skip_frame,
            test_only=args.test_only,
            framewise_transforms=framewise_transforms,
            dagger_ratio=dagger_ratio,
        )
    else:
        test_dataloader, (candidate_embeddings, candidate_texts) = load_merged_data(
            dataset_dirs=dataset_dirs,
            num_episodes_list=num_episodes_list,
            camera_names=camera_names,
            camera_file_suffixes=camera_file_suffixes,
            batch_size=args.batch_size,
            history_len=args.history_len,
            prediction_offset=args.prediction_offset,
            history_skip_frame=args.history_skip_frame,
            test_only=args.test_only,
            framewise_transforms=framewise_transforms,
            dagger_ratio=dagger_ratio,
        )

    one_hot_flag = True # args.one_hot_flag # TODO: Add later via arg -> remove this and take directly from args
    model = build_instructor(args.history_len, candidate_embeddings, candidate_texts, device, one_hot_flag)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr) # TODO: Add here later also further parameters like weight decay, ..
    criterion = torch.nn.CrossEntropyLoss()

    # Load the most recent checkpoint if available
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
        latest_idx = 0
    else:
        # Load the most recent checkpoint if available # TODO: Later rather load the best model based on the validation loss?!
        latest_ckpt, latest_idx = latest_checkpoint(args.ckpt_dir)
        if latest_ckpt:
            print(f"Loading checkpoint: {latest_ckpt}")
            model.load_state_dict(torch.load(latest_ckpt, map_location=device))
        else:
            print("No checkpoint found.")
            latest_idx = 0

    # ---------------------- Training loop ----------------------

    # Create a directory to save predictions for the current run
    predictions_dir = os.path.join(ckpt_dir, "predictions")
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

    # Test the model using the latest checkpoint - don't train
    if args.test_only:
        test(model, test_dataloader, "test", device, latest_idx, one_hot_flag)
        exit()

    # Training loop
    pbar_epochs = tqdm(range(latest_idx, args.num_epochs), desc="Epochs")
    for epoch in pbar_epochs:
        wandb.log({"Epoch": epoch}) # TODO: Add later maybe also other hyperparameters
        
        train_loss = train(model, train_dataloader, optimizer, criterion, device)
        if dagger_ratio is None: # TODO: Integrate back later
            val_loss = evaluate(model, val_dataloader, criterion, device)
            
            # Test the model and log success rate every 200 epochs
            if epoch % args.validation_interval == 0 and (epoch > 0 or dagger_ratio is not None):
                test(model, val_dataloader, "val", device, epoch, one_hot_flag)

        pbar_epochs.set_postfix({"Train Loss": train_loss, "Val Loss": val_loss if dagger_ratio is None else None})

        if args.log_wandb:
            wandb.log({"Epoch Train Loss": train_loss})
            if dagger_ratio is None:
                wandb.log({"Epoch Eval Loss": val_loss})

        # Save a checkpoint every 100 epochs
        save_ckpt_every = 100
        if epoch % save_ckpt_every == 0 and epoch > 0 and False: # TODO: Change later back again
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}.ckpt")
            torch.save(model.state_dict(), ckpt_path)

            # TODO: Rather keeping the best model based on the validation loss?! + early stopping?!
            # Pruning: this removes the checkpoint save_ckpt_every epochs behind the current one
            # except for the ones at multiples of prune_freq epochs
            prune_freq = 300
            prune_epoch = epoch - save_ckpt_every
            if prune_epoch % prune_freq != 0:
                prune_path = os.path.join(ckpt_dir, f"epoch_{prune_epoch}.ckpt")
                if os.path.exists(prune_path):
                    os.remove(prune_path)

    if args.log_wandb:
        wandb.finish()
