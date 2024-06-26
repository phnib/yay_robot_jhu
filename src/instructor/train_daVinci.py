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
import torch
import torch.optim as optim
import argparse
import os
import numpy as np
import wandb
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import threading
import sys
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from sklearn.manifold import TSNE
from collections import OrderedDict

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


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        images, _, commands = batch
        images = images.to(device)

        optimizer.zero_grad()
        logits, temperature = model(images)

        
        # Convert ground truth command strings to indices using the pre-computed dictionary
        commands_idx = [
            model.command_to_index[
                cmd #.replace("the back", "the bag").replace("mmove", "move")
            ]
            for cmd in commands
        ]
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

            logits, temperature = model(images)

            # Convert ground truth command strings to indices using the pre-computed dictionary
            commands_idx = [
                model.command_to_index[
                    cmd#.replace("the back", "the bag")
                    ]
                for cmd in commands
            ]
            commands_idx = torch.tensor(commands_idx, device=device)

            loss = criterion(logits, commands_idx)
            total_loss += loss.item()

            if args.log_wandb:
                wandb.log({"Eval Loss": loss.item(), "Temperature": temperature.item()})
    return total_loss / len(dataloader)


def test(model, dataloader, device, current_epoch):
    model.eval()

    total_correct = 0
    total_predictions = 0

    # predicted_embeddings = []
    # gt_embeddings = []

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            images, command_embedding_gt, command_gt = batch
            images = images.to(device)

            logits, temperature = model(images)
            # Get nearest text for each prediction in the batch
            decoded_texts = model.decode_logits(logits, temperature)

            # predicted_embeddings.extend(predictions.cpu().numpy())
            # gt_embeddings.extend(command_embedding_gt.cpu().numpy())

            for i, (gt, pred) in enumerate(zip(command_gt, decoded_texts)):
                # Save incorrect prediction
                # if pred != gt:
                #     save_path = os.path.join(ckpt_dir, "predictions", f"epoch_{current_epoch}_incorrect_{idx}_{i}.jpg")
                #     save_combined_image(images[i].squeeze(0), gt, pred, save_path)
                #     if args.log_wandb:
                #         wandb.log({f"Incorrect Prediction": wandb.Image(save_path, caption=f"Epoch {current_epoch}, Batch {idx}, Image {i}")})
                # elif i < 5:
                #     save_path = os.path.join(ckpt_dir, "predictions", f"epoch_{current_epoch}_correct_{idx}_{i}.jpg")
                #     save_combined_image(images[i].squeeze(0), gt, pred, save_path)

                total_correct += int(pred == gt)
                total_predictions += 1
                print(f"Ground truth: {gt} \t Predicted text: {pred}")

    # TODO: Add the visualization of the embeddings again - or just add it too WandB
    # Visualize embeddings
    # tsne_visualize(predicted_embeddings, gt_embeddings, candidate_embeddings, current_epoch)

    success_rate = total_correct / total_predictions
    print(f"Epoch {current_epoch}: Success Rate = {success_rate * 100:.2f}%")

    if args.log_wandb:
        wandb.log({"Success Rate": success_rate})

    return success_rate


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


def save_combined_image(image, gt_text, pred_text, save_path=None):
    # image = image[:, :, [2, 1, 0]]

    # Extract first frame t=0 and concatenate across width
    combined_image = torch.cat([image[0, i] for i in range(image.shape[1])], dim=-1)

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
    else:
        return canvas


def tsne_visualize(predicted_embeddings, gt_embeddings, candidate_embeddings, epoch):
    # Convert lists to numpy arrays
    predicted_embeddings = np.array(predicted_embeddings)
    gt_embeddings = np.array(gt_embeddings)

    assert (
        predicted_embeddings.shape == gt_embeddings.shape
    ), "The number of predicted and ground truth embeddings do not match."

    # Stack embeddings and apply t-SNE
    all_embeddings = np.vstack(
        [predicted_embeddings, gt_embeddings, candidate_embeddings.cpu().numpy()]
    )
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    # Split the 2D embeddings back
    predicted_2d = embeddings_2d[: len(predicted_embeddings)]
    gt_2d = embeddings_2d[
        len(predicted_embeddings) : len(predicted_embeddings) + len(gt_embeddings)
    ]
    candidate_2d = embeddings_2d[len(predicted_embeddings) + len(gt_embeddings) :]

    # Plot the results
    plt.figure(figsize=(10, 10))
    plt.scatter(
        candidate_2d[:, 0], candidate_2d[:, 1], marker="o", color="g", label="Dataset"
    )
    plt.scatter(gt_2d[:, 0], gt_2d[:, 1], marker="o", color="b", label="Ground Truth")
    plt.scatter(
        predicted_2d[:, 0], predicted_2d[:, 1], marker="o", color="r", label="Predicted"
    )

    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title(f"t-SNE Visualization of Embeddings (Epoch {epoch})")
    plt.legend()

    # Save with the epoch in the filename
    image_save_path = os.path.join(ckpt_dir, f"embeddings_tsne_epoch_{epoch}.png")
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


# Note: Currently not needed as reading it from the datasets directly
def load_candidate_texts(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        # Extract the instruction (text before the colon), strip whitespace, and then strip quotation marks
        candidate_texts = [line.split(":")[0].strip().strip("'\"") for line in lines]
    return candidate_texts

# Note: Currently not needed as reading it from the datasets directly
def load_candidate_texts_and_embeddings(dataset_dirs, device=torch.device("cuda")):
    
    candidate_texts = []
    candidate_embeddings = []

    for dataset_dir in dataset_dirs:
        embeddings_path = os.path.join(
            dataset_dir, "candidate_embeddings_distilbert.npy"
        )
        # Load pre-computed embeddings
        candidate_embedding = (
            torch.tensor(np.load(embeddings_path).astype(np.float32))
            .to(device)
            .squeeze()
        )
        candidate_embeddings.append(candidate_embedding)
        candidate_texts_path = os.path.join(dataset_dir, "count.txt")
        current_candidate_texts = load_candidate_texts(candidate_texts_path)
        candidate_texts.extend(current_candidate_texts)
    candidate_embeddings = torch.cat(candidate_embeddings, dim=0).to(device)

    def remove_duplicates(candidate_texts, candidate_embeddings):
        unique_entries = OrderedDict()

        for text, embedding in zip(candidate_texts, candidate_embeddings):
            if text not in unique_entries:
                unique_entries[text] = embedding

        # Rebuild the lists without duplicates
        filtered_texts = list(unique_entries.keys())
        filtered_embeddings = torch.stack(list(unique_entries.values()))

        return filtered_texts, filtered_embeddings

    candidate_texts, candidate_embeddings = remove_duplicates(
        candidate_texts, candidate_embeddings
    )
    return candidate_texts, candidate_embeddings


def build_instructor(history_len, candidate_embeddings, candidate_texts, device):
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
    ).to(device)
    return model


if __name__ == "__main__":
    from instructor.utils import set_seed
    from aloha_pro.aloha_scripts.constants_daVinci import TASK_CONFIGS # get task parameters
    
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
    parser.add_argument('--dagger_ratio', action='store', type=float, help='dagger_ratio', default=None) # TODO: Still needed?
    # TODO: Maybe add later a list of transformations/augmentations that should be applied as arg

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Device setting
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    dataset_dirs = []
    num_episodes_list = []

    for task in args.task_name:
        task_config = TASK_CONFIGS[task]
        dataset_dirs.append(task_config["dataset_dir"])
        num_episodes_list.append(task_config["num_episodes"])
        camera_names = task_config["camera_names"]
        camera_file_suffixes = task_config["camera_file_suffixes"]
    ckpt_dir = args.ckpt_dir
    dagger_ratio = args.dagger_ratio # TODO: Still needed?

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

    model = build_instructor(args.history_len, candidate_embeddings, candidate_texts, device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr) # TODO: Add here later also further parameters like weight decay, ..
    criterion = torch.nn.CrossEntropyLoss()

    # WandB initialization
    if args.log_wandb:
        wandb_entity = os.getenv("WANDB_ENTITY")
        run_name = "instructor." + ckpt_dir.split("/")[-1] + f".{args.seed}"
        wandb_run_id_path = os.path.join(ckpt_dir, "wandb_run_id.txt")
        # check if it exists
        if os.path.exists(wandb_run_id_path):
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

    # Create a directory to save predictions for the current run
    predictions_dir = os.path.join(ckpt_dir, "predictions")
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

    # Test the model using the latest checkpoint - don't train
    if args.test_only:
        test(model, test_dataloader, device, latest_idx)
        exit()

    # Training loop
    pbar_epochs = tqdm(range(latest_idx, args.num_epochs), desc="Epochs")
    for epoch in pbar_epochs:
        train_loss = train(model, train_dataloader, optimizer, criterion, device)
        if dagger_ratio is None: # TODO: Do we still need the dagger_ratio?
            eval_loss = evaluate(model, val_dataloader, criterion, device)
            
            # Test the model and log success rate every 200 epochs
            if epoch % 200 == 0 and (epoch > 0 or dagger_ratio is not None):
                test(model, val_dataloader, device, epoch)

        pbar_epochs.set_postfix({"Train Loss": train_loss})

        if args.log_wandb:
            wandb.log({"Epoch Train Loss": train_loss})#, step=epoch)
            if dagger_ratio is None:
                wandb.log({"Epoch Eval Loss": eval_loss})#, step=epoch)

        # Save a checkpoint every 100 epochs
        save_ckpt_every = 100
        if epoch % save_ckpt_every == 0 and epoch > 0:
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
