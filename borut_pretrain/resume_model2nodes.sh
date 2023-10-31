#!/bin/sh
#SBATCH --job-name=train_Borut
#SBATCH --output=./logs/train_Borut%j.out
#SBATCH --error=./logs/train_Borut%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --cpus-per-task=12
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu

#example salloc --nodes=1 --gres=gpu:2 --ntasks-per-node=2 --mem=0 --time=0-10:00:00 --cpus-per-task=12 --job-name=Interactive_GPU2 --partition=gpu 


       
source /ceph/hpc/data/st2207-pgp-users/ldragar/miniconda3/etc/profile.d/conda.sh
conda activate  /ceph/hpc/data/st2207-pgp-users/ldragar/pytorch_env


if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
    exit 1
fi


export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

export WANDB__SERVICE_WAIT=300

#script is made to run on 1 node with 1 gpu
# wandb agent $wandb_agent 

#!/bin/bash

project_name="borut_pretrain"
run_id=$1

read -ra args <<< $(python3 -c "import json, wandb; api = wandb.Api(); run = api.run('$project_name/$run_id'); metadata = run.file('wandb-metadata.json').download(replace=True); print(' '.join(json.load(open(metadata.name, 'r'))['args']))")
#remove file 
rm wandb-metadata.json

# # Use the model_name and args variables
# echo "Args: ${args[@]}"



srun --nodes=2 --cpus-per-task=12 --gpus-per-node=4 -p gpu --ntasks-per-node=4 --exclusive python3 pretrain_timm.py "${args[@]}" --resume_run_id $run_id