#!/bin/bash

# Training script for Transformer Language Model
# This script provides various training configurations and options

# Set default values
EXPERIMENT_NAME="transformer_finetune"
BATCH_SIZE=32
LEARNING_RATE=1e-4
NUM_EPOCHS=20
MAXLEN=35
DECODE_STRATEGY="top-p"
TEMPERATURE=0.9
TOP_P=0.9

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

