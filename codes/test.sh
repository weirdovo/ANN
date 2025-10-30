#!/bin/bash

# Training script for Transformer Language Model
# This script provides various training configurations and options

# Set default values
EXPERIMENT_NAME="tfmr_scratch_rmsnorm"
BATCH_SIZE=32
MAXLEN=35
DECODE_STRATEGY="random"
TEMPERATURE=0.7
TOP_P=1.0


# Run evaluation if training was successful
echo "=========================================="
echo "Running evaluation on test set..."
python main.py \
    --test "$EXPERIMENT_NAME" \
    --model_config "./config.json" \
    --tokenizer_dir "./tokenizer" \
    --batch_size "$BATCH_SIZE" \
    --data_dir "./data" \
    --train_dir "./checkpoints" \
    --maxlen "$MAXLEN" \
    --decode_strategy "$DECODE_STRATEGY" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    2>&1 | tee "logs/evaluation_${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "Evaluation completed. Results saved in output_${DECODE_STRATEGY}.txt"

