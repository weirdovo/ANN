#!/bin/bash

# CNN and MLP Model Training Script
# This script runs different model configurations for comparison

echo "=========================================="
echo "Starting Model Training Experiments"
echo "=========================================="

# Set common parameters
DATA_DIR="/home/alex/ANN/HW2/codes/cifar-10-batches-py"
BATCH_SIZE=100
NUM_EPOCHS=150
LEARNING_RATE=1e-3

# Create results directory

echo "Data directory: $DATA_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Number of epochs: $NUM_EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo ""

# Function to run experiment
run_experiment() {
    local model_type=$1
    local drop_rate=$2
    local experiment_name=$3
    local additional_args=$4
    
    echo "=========================================="
    echo "Running: $experiment_name"
    echo "Model: $model_type"
    echo "Dropout rate: $drop_rate"
    echo "=========================================="
    
    
    # Run the experiment
    python "$model_type/main.py" \
        --batch_size $BATCH_SIZE \
        --num_epochs $NUM_EPOCHS \
        --learning_rate $LEARNING_RATE \
        --drop_rate $drop_rate \
        --data_dir $DATA_DIR \
        --train_dir /home/alex/ANN/HW2/results/${model_type}_${experiment_name} \
        $additional_args \
        2>&1 | tee /home/alex/ANN/HW2/logs/${model_type}_${experiment_name}.log
    
    echo "Completed: $experiment_name"
    echo ""
}

# MLP Experiments
echo "Starting MLP experiments..."
run_experiment "mlp" 0.0 "0.0_dropout" ""
run_experiment "mlp" 0.1 "0.1_dropout" ""
run_experiment "mlp" 0.5 "0.5_dropout" ""
run_experiment "mlp" 0.8 "0.8_dropout" ""

echo "MLP experiments completed!"
echo ""

# CNN Experiments
echo "Starting CNN experiments..."
run_experiment "cnn" 0.0 "0.0_dropout" ""
run_experiment "cnn" 0.1 "0.1_dropout" ""
run_experiment "cnn" 0.5 "0.5_dropout" ""
run_experiment "cnn" 0.8 "0.8_dropout" ""

echo "CNN experiments completed!"
echo ""


echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Results saved in: ./results/"
echo "Logs saved in: ./logs/"
echo ""
echo "To compare models, check the log files for final accuracies."
