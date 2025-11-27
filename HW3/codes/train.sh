#!/bin/bash

# Training script for Transformer Language Model
# This script provides various training configurations and options

# Set default values
EXPERIMENT_NAME="tfmr_scratch_rmsnorm"
BATCH_SIZE=32
LEARNING_RATE=1e-4   
NUM_EPOCHS=20
MAXLEN=35
# DECODE_STRATEGY="random"
# TEMPERATURE=1.0
# TOP_P=1.0



# Create necessary directories
mkdir -p checkpoints
mkdir -p logs

# Set up logging
LOG_FILE="logs/training_${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "Starting training with the following configuration:"
echo "Experiment Name: $EXPERIMENT_NAME"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Number of Epochs: $NUM_EPOCHS"
echo "Max Length: $MAXLEN"
# echo "Decode Strategy: $DECODE_STRATEGY"
# echo "Temperature: $TEMPERATURE"
# echo "Top-p: $TOP_P"
echo "Log File: $LOG_FILE"
echo "=========================================="

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA is available. GPU will be used for training."
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
    echo "GPU Info: $GPU_INFO"
else
    echo "CUDA not available. Training will use CPU (slower)."
fi

# Start training
echo "Starting training at $(date)"
echo "=========================================="

python main.py \
    --name "$EXPERIMENT_NAME" \
    --model_config "./config.json" \
    --tokenizer_dir "./tokenizer" \
    --num_epochs "$NUM_EPOCHS" \
    --cpu_count 20 \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --data_dir "./data" \
    --train_dir "./checkpoints" \
    --maxlen "$MAXLEN" \
    2>&1 | tee "$LOG_FILE"
