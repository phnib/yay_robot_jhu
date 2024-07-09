#!/bin/bash
#SBATCH --job-name=hl_training        # Job name
#SBATCH --output=slurm_jobs_output/job_%j/hl_training_output.txt   # Standard output file in job_{job_id} folder
#SBATCH --error=slurm_jobs_output/job_%j/hl_training_error.txt     # Standard error file in job_{job_id} folder
#SBATCH --partition="a100"    # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --cpus-per-task=8             # Number of CPU cores per task
#SBATCH --time=0:01:00                # Maximum runtime (D-HH:MM:SS) # TODO: Adjust the time
#SBATCH --mail-type=END               # Send email at job completion
#SBATCH --mail-user=phanse14@jh.edu    # Email address for notifications
#SBATCH -A akriege1_gpu                 # PI-userid_gpu
#SBATCH --gres=gpu:1                  # Number of GPUs

# Unload all currently loaded modules to start with a clean environment
module purge 

#Load necessary modules (if needed)
module load anaconda
module load namd/2.14-cuda-smp # TODO: Does this line has an effect?

# Activate the virtual environment
conda activate /home/phanse14/.conda/envs/yay

# Set environment variables
export PATH_TO_YAY_ROBOT="/home/phanse14/scr4_akriege1/chole/yay_robot_jhu"
export PATH_TO_DATASET="/home/phanse14/scr4_akriege1/chole/chole_dataset"
export YOUR_CKPT_PATH="$PATH_TO_YAY_ROBOT/model_ckpts"
export WANDB_ENTITY="pascal-hansen" # TODO: Adjust to WandB cloud entity

#Your job commands go here
python train_daVinci.py \
    --task_name debugging \
    --ckpt_dir $YOUR_CKPT_PATH/hl/run_ckpt \
    --batch_size 128 \
    --num_epochs 1000 \
    --lr 1e-4 \
    --history_skip_frame 1 \
    --prediction_offset 0 \
    --history_len 0 \
    --seed 0 \
    --one_hot_flag \
    --validation_interval 1

#Optionally, you can include cleanup commands here (e.g., after the job finishes) 
#For example:
#rm some_temp_file.txt