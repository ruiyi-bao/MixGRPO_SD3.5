#!/bin/bash

# SD 3.5 Medium Preprocessing Script
# Generates embeddings using stabilityai/stable-diffusion-3.5-medium

echo "=========================================="
echo "SD 3.5 Medium Embedding Preprocessing"
echo "=========================================="

# Set HuggingFace token (must be set as environment variable before running)
# export HF_TOKEN="your_token_here"
# export HUGGING_FACE_HUB_TOKEN="your_token_here"
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set"
    echo "Please set it before running: export HF_TOKEN='your_hf_token'"
    exit 1
fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# Configuration
model_name="stabilityai/stable-diffusion-3.5-medium"
variant="fp16"
data_root="data"
output_dir="data/rl_embeddings_sd35_medium"
prompt_file="data/rl_prompts.txt"
batch_size=1
num_workers=4

echo ""
echo "Configuration:"
echo "  Model: $model_name"
echo "  HF Token: Set (${HF_TOKEN:0:10}...)"
echo "  Data Root: $data_root"
echo "  Output Directory: $output_dir"
echo "  Prompt File: $prompt_file"
echo "  Batch Size: $batch_size"
echo "  Workers: $num_workers"
echo ""

# Create output directory
mkdir -p $output_dir

# Check if prompt file exists
if [ ! -f "$prompt_file" ]; then
    echo "Error: Prompt file not found at $prompt_file"
    echo "Please ensure the prompt file exists or update the path."
    exit 1
fi

echo "Starting preprocessing..."
echo ""

# Run preprocessing with SD 3.5 Medium
python fastvideo/data_preprocess/preprocess_flux_embedding.py \
    --model_path $model_name \
    --model_type sd3 \
    --train_batch_size $batch_size \
    --dataloader_num_workers $num_workers \
    --prompt_path $prompt_file \
    --output_dir $output_dir

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Preprocessing completed successfully!"
    echo "=========================================="
    echo ""
    echo "Embeddings saved in: $output_dir"
    echo ""
    echo "To verify the embeddings:"
    echo "  python -c \"import torch; data = torch.load('$output_dir/embeddings.pt'); print(f'Loaded {len(data)} embeddings')\""
    echo ""
    echo "To use these embeddings for training:"
    echo "  Update data_json_path in your training script to: $output_dir/prompt.json"
else
    echo ""
    echo "=========================================="
    echo "Preprocessing failed!"
    echo "=========================================="
    echo ""
    echo "Please check the error messages above."
    exit 1
fi