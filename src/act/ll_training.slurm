#!/bin/bash
#SBATCH --job-name=ll_training        # Job name
#SBATCH --output=slurm_jobs_output/job_%j/ll_training_output.txt   # Standard output file in job_{job_id} folder
#SBATCH --error=slurm_jobs_output/job_%j/ll_training_error.txt     # Standard error file in job_{job_id} folder
#SBATCH --partition="a100"    # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --cpus-per-task=8             # Number of CPU cores per task
#SBATCH --time=0:01:00                # Maximum runtime (D-HH:MM:SS) # TODO: Adjust the time
#SBATCH --mail-type=END               # Send email at job completion
#SBATCH --mail-user=jchen396@jh.edu    # Email address for notifications
#SBATCH -A akriege1_gpu                 # PI-userid_gpu
#SBATCH --gres=gpu:1                  # Number of GPUs

# Unload all currently loaded modules to start with a clean environment
module purge 

#Load necessary modules (if needed)
module load anaconda
module load namd/2.14-cuda-smp # TODO: Does this line has an effect?

# Activate the virtual environment
conda activate /home/jchen396/.conda/envs/aloha

# Set environment variables
export PATH_TO_YAY_ROBOT="/home/jchen396/scr4_akriege1/chole/yay_robot_jhu"
export PATH_TO_DATASET="/home/jchen396/scr4_akriege1/chole/chole_dataset"
export YOUR_CKPT_PATH="$PATH_TO_YAY_ROBOT/model_ckpts"
# export WANDB_ENTITY="phansenjhu" # TODO: Adjust to WandB cloud entity # TODO: or rather "phansenjhu-johns-hopkins-university"
# export WANDB_API_KEY=$(cat ~/.wandb_api_key) # Read the API key from the secure file

# Login to W&B
# wandb login --relogin $WANDB_API_KEY

#Your job commands go here
python imitate_episodes.py \
    --task_name base_chole_clipping_cutting ... \
    --ckpt_dir $YOUR_CKPT_PATH/ll_ckpt/base_chole_clipping_cutting \
    --policy_class ACT --kl_weight 80 --chunk_size 100 --hidden_dim 512 --batch_size 16 --dim_feedforward 3200 \
    --use_language --language_encoder distilbert --max_skill_len 200 --num_epochs 30000  --lr 1e-4 \
    --image_encoder efficientnet_b3film --seed 42 

#Optionally, you can include cleanup commands here (e.g., after the job finishes) 
#For example:
#rm some_temp_file.txt