#!/bin/bash

function find_free_port() {
    # 端口搜索循环
    local start=24456
    local end=33456
    local free_port=24456
    for port in $(seq $start 100 $end)
    do
        # 使用lsof命令检查端口是否被占用，如果未被占用，那么将此端口号赋值给变量并退出搜索
        (echo >/dev/tcp/localhost/$port) >/dev/null 2>&1
        if [[ $? -eq 1 ]]; then
            free_port=$port
            break
        fi
    done
    echo $free_port
}

free_port=$(find_free_port)

# hostfile
hostfile="data/hosts/hostfile"

# wandb key
wandb_key="cd4c392a4259e60e98da6b73aef535bec6198ce8"

# network proxy settings and api url
image_reward_http_proxy=None
image_reward_https_proxy=None
pick_score_http_proxy=None
pick_score_https_proxy=None
unified_reward_url=None

# flux and reward models paths
model_path="./data/flux"
hps_path="./hps_ckpt/HPS_v2.1_compressed.pt"
hps_clip_path="./hps_ckpt/open_clip_pytorch_model.bin"
clip_score_path="hf-hub:apple/DFN5B-CLIP-ViT-H-14-384"
image_reward_path="./image_reward_ckpt/ImageReward.pt"
image_reward_med_config="./image_reward_ckpt/med_config.json"
unified_reward_default_question_type="semantic"
unified_reward_num_workers=1

# Dataset
data_json_path="data/rl_embeddings/prompt.json"

# MixGRPO Hyperparameters
experiment_name="1014_test"
reward_model="multi_reward" # "hpsv2", "clip_score" "image_reward", "pick_score", "unified_reward", "hpsv2_clip_score", "multi_reward"
seed=714
sampler_seed=7144
training_strategy="part" # "part", "all"
sampling_steps=25 # 图像生成采样步数
eta=0.7 #GRPO 采样噪声强度
kl_coeff=0.0
iters_per_group=25 #SDE窗口滑动间距（多久滑动一次）
group_size=4 #SDE窗口大小
sample_strategy="progressive"
prog_overlap_step=1 #窗口移动大小，即和上一窗口末尾重叠一步
trimmed_ratio=0.0
multi_reward_mix="advantage_aggr" # "reward_aggr", "advantage_aggr"
hps_weight=1.0
clip_score_weight=1.0
image_reward_weight=1.0
pick_score_weight=1.0
unified_reward_weight=1.0
gradient_accumulation_steps=12 #梯度积累步数

# DanceGRPO Sampling Parameters
timestep_fraction=0.6
frozen_init_timesteps=-1

# DPM
dpm_algorithm_type="null" # "null", "dpm-solver", "dpm-solver++"
dpm_apply_strategy="post"
dpm_post_compress_ratio=0.4
dpm_solver_order=2
dpm_solver_type="midpoint"

# Custom splitting nodes
nnodes_custom=1
nproc_per_node_custom=8
CHIEF_IP_custom=$(head -n 1 $hostfile)
INDEX_CUSTOME=0

cur_path=$(pwd)


#export WANDB_DISABLED=true ;
export WANDB_BASE_URL="https://api.wandb.ai" ;
export WANDB_MODE=online ; #offline
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "nnodes_custom=$nnodes_custom" ;
echo "nproc_per_node=$nproc_per_node_custom" ;
echo "INDEX_CUSTOME=$INDEX_CUSTOME" ;
echo "CHIEF_IP_custom=$CHIEF_IP_custom" ;
echo "free_port=$free_port" ;

torchrun --nnodes $nnodes_custom --nproc_per_node $nproc_per_node_custom --node_rank $INDEX_CUSTOME --master_addr $CHIEF_IP_custom --master_port $free_port \
    fastvideo/train_grpo_flux.py \
    --seed $seed \
    --pretrained_model_name_or_path $model_path \
    --vae_model_path $model_path \
    --cache_dir data/.cache \
    --data_json_path $data_json_path \
    --gradient_checkpointing \
    --train_batch_size 1 \
    --num_latent_t 1 \
    --sp_size 1 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --max_train_steps 300 \
    --learning_rate 1e-5 \
    --mixed_precision bf16 \
    --checkpointing_steps 50 \
    --allow_tf32 \
    --cfg 0.0 \
    --output_dir data/outputs \
    --h 720 \
    --w 720 \
    --t 1 \
    --sampling_steps $sampling_steps \
    --eta $eta \
    --lr_warmup_steps 0 \
    --sampler_seed $sampler_seed \
    --max_grad_norm 1.0 \
    --weight_decay 0.0001 \
    --num_generations 12 \
    --shift 3 \
    --use_group \
    --ignore_last \
    --timestep_fraction $timestep_fraction \
    --init_same_noise \
    --clip_range 1e-4 \
    --adv_clip_max 5.0 \
    --training_strategy $training_strategy \
    --experiment_name $experiment_name \
    --kl_coeff $kl_coeff \
    --iters_per_group $iters_per_group \
    --group_size $group_size \
    --sample_strategy $sample_strategy \
    --prog_overlap \
    --prog_overlap_step $prog_overlap_step \
    --max_iters_per_group 10 \
    --min_iters_per_group 1 \
    --roll_back \
    --trimmed_ratio $trimmed_ratio \
    --reward_model $reward_model \
    --hps_path $hps_path \
    --hps_clip_path $hps_clip_path \
    --clip_score_path $clip_score_path \
    --image_reward_path $image_reward_path \
    --image_reward_med_config $image_reward_med_config \
    --image_reward_http_proxy $image_reward_http_proxy \
    --image_reward_https_proxy $image_reward_https_proxy \
    --pick_score_http_proxy $pick_score_http_proxy \
    --pick_score_https_proxy $pick_score_https_proxy \
    --unified_reward_url $unified_reward_url \
    --unified_reward_default_question_type $unified_reward_default_question_type \
    --unified_reward_num_workers $unified_reward_num_workers \
    --multi_reward_mix $multi_reward_mix \
    --hps_weight $hps_weight \
    --clip_score_weight $clip_score_weight \
    --image_reward_weight $image_reward_weight \
    --pick_score_weight $pick_score_weight \
    --unified_reward_weight $unified_reward_weight \
    --dpm_algorithm_type $dpm_algorithm_type \
    --dpm_apply_strategy $dpm_apply_strategy \
    --dpm_post_compress_ratio $dpm_post_compress_ratio \
    --dpm_solver_order $dpm_solver_order \
    --dpm_solver_type $dpm_solver_type \
    --frozen_init_timesteps $frozen_init_timesteps \
    --wandb_key $wandb_key \
    --flow_grpo_sampling     
    # --drop_last_sample