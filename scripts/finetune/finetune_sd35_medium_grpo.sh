#!/bin/bash

# SD 3.5 Medium GRPO Training Script
# Configured to use stabilityai/stable-diffusion-3.5-medium with HuggingFace token

function find_free_port() {
    # Port search loop
    local start=24456
    local end=33456
    local free_port=24456
    for port in $(seq $start 100 $end)
    do
        (echo >/dev/tcp/localhost/$port) >/dev/null 2>&1
        if [[ $? -eq 1 ]]; then
            free_port=$port
            break
        fi
    done
    echo $free_port
}

free_port=$(find_free_port)

# Set HuggingFace token (must be set as environment variable before running)
# export HF_TOKEN="your_token_here"
# export HUGGING_FACE_HUB_TOKEN="your_token_here"
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set"
    echo "Please set it before running: export HF_TOKEN='your_hf_token'"
    exit 1
fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# hostfile
hostfile="data/hosts/hostfile"

# wandb key (set as environment variable before running)
# export WANDB_API_KEY="your_wandb_key_here"
if [ -z "$WANDB_API_KEY" ]; then
    echo "Warning: WANDB_API_KEY environment variable is not set"
    echo "WandB logging will be disabled or may fail"
    wandb_key=""
else
    wandb_key="$WANDB_API_KEY"
fi

# network proxy settings and api url
image_reward_http_proxy=None
image_reward_https_proxy=None
pick_score_http_proxy=None
pick_score_https_proxy=None
unified_reward_url=None

# SD 3.5 Medium model path - will download from HuggingFace
model_path="stabilityai/stable-diffusion-3.5-medium"

# Reward model paths
hps_path="hps_ckpt/HPS_v2.1_compressed.pt"
hps_clip_path="hps_ckpt/open_clip_pytorch_model.bin"
clip_score_path="hf-hub:apple/DFN5B-CLIP-ViT-H-14-384"
image_reward_path="image_reward_ckpt/ImageReward.pt"
image_reward_med_config="image_reward_ckpt/med_config.json"
unified_reward_default_question_type="semantic"
unified_reward_num_workers=1

# Dataset
data_json_path="data/rl_embeddings_sd35_medium/prompt.json"

# MixGRPO Hyperparameters for SD 3.5 Medium
experiment_name="sd35_medium_grpo_training"
reward_model="clip_score" # "hpsv2", "clip_score" "image_reward", "pick_score", "unified_reward", "hpsv2_clip_score", "multi_reward"
multi_reward_weights="1.0,1.0,1.0"
hpsv2_version=2.1
training_strategy="part"  # "part" for MixGRPO, "all" for DanceGRPO - matching Flux
kl_weight=0.0  # Match Flux kl_coeff
kl_warmup=150
dataloader_prefetch_factor=4
dataloader_persistent_workers=True
per_process_batch_size=1
gradient_accumulation_steps=8  # Match Flux: 4 GPUs × 1 batch × 12 gen × 8 accum = 384 images per step
random_seed=42
num_processes=4  # Use 4 GPUs to match Flux
run_validation=0.1
offline_rew=0.01
valid_data_json_path="data/rl_embeddings_sd3.5/valid_prompt.json"

# Checkpoint hyperparameters
checkpoint_dir="./sd35_medium_checkpoints"
num_train_steps=5000
num_checkpoint_steps=500
num_log_steps=100
num_valid_checkpoint_steps=500
eval_clip_score=False

# Model hyperparameters (optimized for SD 3.5 Medium)
mixed_precision="fp16"  # SD 3.5 Medium works well with fp16
train_learning_rate="8e-6"
train_adam_epsilon="1e-8"
train_adam_weight_decay="1e-2"

# SD 3.5 Medium specific settings
guidance_scale=4.5  # Lower guidance scale for Medium model
num_inference_steps=28  # Optimal for SD 3.5 Medium

echo "=========================================="
echo "Starting SD 3.5 Medium GRPO Training"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Model: $model_path"
echo "  HF Token: Set (${HF_TOKEN:0:10}...)"
echo "  Experiment: $experiment_name"
echo "  Reward Model: $reward_model"
echo "  Training Steps: $num_train_steps"
echo "  Batch Size: $per_process_batch_size"
echo "  Gradient Accumulation: $gradient_accumulation_steps"
echo "  Mixed Precision: $mixed_precision"
echo "  Checkpoint Directory: $checkpoint_dir"
echo ""

# Create checkpoint directory if it doesn't exist
mkdir -p $checkpoint_dir

# Launch training
torchrun --nproc_per_node=4 fastvideo/train_grpo_flux.py \
--experiment_name $experiment_name \
--wandb_key $wandb_key \
--pretrained_model_name_or_path $model_path \
--data_json_path $data_json_path \
--mixed_precision $mixed_precision \
--dataloader_num_workers 8 \
--train_batch_size $per_process_batch_size \
--gradient_accumulation_steps $gradient_accumulation_steps \
--kl_coeff $kl_weight \
--reward_model $reward_model \
--max_train_steps $num_train_steps \
--checkpointing_steps $num_checkpoint_steps \
--learning_rate $train_learning_rate \
--weight_decay $train_adam_weight_decay \
--output_dir $checkpoint_dir \
--seed $random_seed \
--hps_path $hps_path \
--hps_clip_path $hps_clip_path \
--image_reward_path $image_reward_path \
--image_reward_med_config $image_reward_med_config \
--clip_score_path $clip_score_path \
--image_reward_http_proxy $image_reward_http_proxy \
--image_reward_https_proxy $image_reward_https_proxy \
--pick_score_http_proxy $pick_score_http_proxy \
--pick_score_https_proxy $pick_score_https_proxy \
--unified_reward_url $unified_reward_url \
--unified_reward_default_question_type $unified_reward_default_question_type \
--unified_reward_num_workers $unified_reward_num_workers \
--training_strategy $training_strategy \
--h 1024 \
--w 1024 \
--t 1 \
--sampling_steps $num_inference_steps \
--num_generations 12 \
--shift 3.0 \
--cfg 4.5 \
--flow_grpo_sampling

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo ""
echo "Checkpoints saved in: $checkpoint_dir"
echo "To resume training, use the same command with --resume_from_checkpoint"