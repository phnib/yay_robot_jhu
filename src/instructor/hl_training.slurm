#!/bin/bash
SBATCH --job-name=hl_training        # Job name
# TODO: Later put the output, error files somewhere else + possible to add timestamp?!
#SBATCH --output=hl_training_output.txt           # Standard output file
#SBATCH --error=hl_training_error.txt             # Standard error file
#SBATCH --partition="express"    # Partition or queue name # TODO: Use later "a100"
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --cpus-per-task=1             # Number of CPU cores per task # TODO: How to request gpu nodes
#SBATCH --time=0:02:00                # Maximum runtime (D-HH:MM:SS) # TODO: Adjust the time
#SBATCH --mail-type=END               # Send email at job completion
#SBATCH --mail-user=phanse14@jh.edu    # Email address for notifications
#SBATCH -A akriege1_gpu                 # PI-userid_gpu
#SBATCH --gres=gpu:1                  # Number of GPUs

#Load necessary modules (if needed)
module load conda
# module load cuda/11.3 # TODO: Do I need to load CUDA or is this done manually?

# Activate the virtual environment
conda activate /home/phanse14/.conda/envs/yay

#Your job commands go here
python train_daVinci.py # TODO: Does it work with relative paths? Will it directly work with CUDA?!

#Optionally, you can include cleanup commands here (e.g., after the job finishes) 
#For example:
#rm some_temp_file.txt