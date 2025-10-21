#!/bin/bash
#SBATCH -N 1                     # 1 个节点
#SBATCH -p RTX3090               # 分区
#SBATCH -w node03                # 指定节点
#SBATCH --gres=gpu:8             # 8 张 GPU
#SBATCH -J MixGRPO               # 作业名称
#SBATCH -o logs/slurm-%j.out     # 输出日志
#SBATCH --mem=0

eval "$(conda shell.bash hook)"
conda activate MixGRPO
cd /cluster/home1/cyx/MixGRPO
mkdir -p logs
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY

bash scripts/finetune/finetune_flux_grpo_MixGRPO.sh