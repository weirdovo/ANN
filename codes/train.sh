#!/bin/bash

# Training script for Transformer Language Model
# This script provides various training configurations and options

# Set default values
EXPERIMENT_NAME="transformer_finetune"
BATCH_SIZE=32
LEARNING_RATE=1e-5     # lower
NUM_EPOCHS=40
MAXLEN=35
DECODE_STRATEGY="top-p"
TEMPERATURE=1.0
TOP_P=1.0
PRETRAIN_DIR="./checkpoints"



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
echo "Decode Strategy: $DECODE_STRATEGY"
echo "Temperature: $TEMPERATURE"
echo "Top-p: $TOP_P"
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
    --decode_strategy "$DECODE_STRATEGY" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --pretrain_dir "$PRETRAIN_DIR" \
    2>&1 | tee "$LOG_FILE"

# Check if training completed successfully
# if [ ${PIPESTATUS[0]} -eq 0 ]; then
#     echo "=========================================="
#     echo "Training completed successfully at $(date)"
#     echo "Checkpoint saved in: ./checkpoints/checkpoint_${EXPERIMENT_NAME}.bin"
#     echo "Config saved in: ./checkpoints/config.json"
#     echo "Log saved in: $LOG_FILE"
    
#     # Run evaluation if training was successful
#     echo "=========================================="
#     echo "Running evaluation on test set..."
#     python main.py \
#         --test "$EXPERIMENT_NAME" \
#         --model_config "./config.json" \
#         --tokenizer_dir "./tokenizer" \
#         --batch_size "$BATCH_SIZE" \
#         --data_dir "./data" \
#         --train_dir "./checkpoints" \
#         --maxlen "$MAXLEN" \
#         --decode_strategy "$DECODE_STRATEGY" \
#         --temperature "$TEMPERATURE" \
#         --top_p "$TOP_P" \
#         2>&1 | tee "logs/evaluation_${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log"
    
#     echo "=========================================="
#     echo "Evaluation completed. Results saved in output_${DECODE_STRATEGY}.txt"
# else
#     echo "=========================================="
#     echo "Training failed. Check the log file: $LOG_FILE"
#     exit 1
# fi
