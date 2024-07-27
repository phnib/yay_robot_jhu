import random
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Sampler

def center_crop_resize(img, size):
    # Center crop the image to biggest possible square and resize it to the desired size
    
    min_img_dim = min(img.shape[1:])
    transform = transforms.Compose([
        transforms.CenterCrop(min_img_dim),
        transforms.Resize(size)
    ])
    img_transformed = transform(img)
    return img_transformed

def randintgaussian(low, high, mean, std_dev):
    """
    Generate a random integer using a Gaussian distribution and clip it to the specified range.

    Parameters:
    low (int): The minimum value (inclusive).
    high (int): The maximum value (exclusive).
    mean (float): The mean of the Gaussian distribution.
    std_dev (float): The standard deviation of the Gaussian distribution.

    Returns:
    int: A random integer within the specified range.
    """
    # Generate a random number from a Gaussian distribution
    value = int(np.random.normal(mean, std_dev))

    # Clip the value to ensure it falls within the desired range
    value = np.clip(value, low, high - 1)

    return value

# TODO: For now use this function (later maybe take it directly again from act.utils.py)
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

# ----------------

# TODO: Apply this on the dataset containing all the tissues of our correction recording with the previous trained policy
# TODO: Also don't count the number of episodes in the base dataset (only for the corrections/finetuning dataset), as the number is fixed for each episode
### For DAgger
class DAggerSampler(Sampler):
    def __init__(
        self, all_indices, last_dataset_indices, batch_size, dagger_ratio, dataset_sizes
    ):
        self.other_indices, self.last_dataset_indices = self._flatten_indices(
            all_indices, last_dataset_indices, dataset_sizes
        )
        print(
            f"Len of data from the last dataset: {len(self.last_dataset_indices)}, Len of data from other datasets: {len(self.other_indices)}"
        )
        self.batch_size = batch_size
        self.dagger_ratio = dagger_ratio
        self.num_batches = len(all_indices) // self.batch_size

    @staticmethod
    def _flatten_indices(all_indices, last_dataset_indices, dataset_sizes):
        flat_other_indices = []
        flat_last_dataset_indices = []
        cumulative_size = 0

        for dataset_dir, size in dataset_sizes.items():
            for idx in range(size):
                if (dataset_dir, idx) in last_dataset_indices:
                    flat_last_dataset_indices.append(cumulative_size + idx)
                elif (dataset_dir, idx) in all_indices:
                    flat_other_indices.append(cumulative_size + idx)
            cumulative_size += size

        return flat_other_indices, flat_last_dataset_indices

    def __iter__(self):
        num_samples_last = int(self.batch_size * self.dagger_ratio)
        num_samples_other = self.batch_size - num_samples_last

        for _ in range(self.num_batches):
            batch_indices = []

            # TODO: Here chosing directly from the correction episodes (probably without stitching)
            if num_samples_last > 0 and self.last_dataset_indices:
                batch_indices.extend(
                    np.random.choice(
                        self.last_dataset_indices, num_samples_last, replace=True
                    )
                )

            # TODO: Here using the stitched episodes from base training
            if num_samples_other > 0 and self.other_indices:
                batch_indices.extend(
                    np.random.choice(
                        self.other_indices, num_samples_other, replace=True
                    )
                )

            np.random.shuffle(batch_indices)  # shuffle within each batch
            yield batch_indices

    def __len__(self):
        return self.num_batches