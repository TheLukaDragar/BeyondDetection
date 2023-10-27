#!/bin/sh
#SBATCH --job-name=predict_timm_dir_public
#SBATCH --output=predict_timm_dir_public%j.out
#SBATCH --error=predict_timm_dir_public%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --cpus-per-task=12
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu

#example salloc --nodes=1 --gres=gpu:2 --ntasks-per-node=2 --mem=0 --time=0-10:00:00 --cpus-per-task=12 --job-name=Interactive_GPU2 --partition=gpu 


       
source /ceph/hpc/data/st2207-pgp-users/ldragar/miniconda3/etc/profile.d/conda.sh
conda activate  /ceph/hpc/data/st2207-pgp-users/ldragar/pytorch_env

#    parser.add_argument('--model_name', type=str, default='swin_large_patch4_window12_384.ms_in22k_ft_in1k')           
#    parser.add_argument('--output_txt', type=str, default='./save_result/pred_swin55_public.txt')

srun python3 predict_timm_dir_public.py  --model_name eva02_large_patch14_448.mim_m38m_ft_in22k_in1k --output_txt ./save_result/pred_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k_public.txt 