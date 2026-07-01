import matplotlib.pyplot as plt
import re
import os
# File path to the .log file
file_root = '/home/alex/ANN/HW2/logs'
log_file_path = ['cnn2_0.0_dropout.log', 'cnn2_0.1_dropout.log', 'cnn2_0.5_dropout.log', 'cnn2_0.8_dropout.log', 
                 'cnn_0.0_dropout.log', 'cnn_0.1_dropout.log', 'cnn_0.5_dropout.log', 'cnn_0.8_dropout.log', 
                 'mlp_0.0_dropout.log', 'mlp_0.1_dropout.log', 'mlp_0.5_dropout.log', 'mlp_0.8_dropout.log',
                 'mlp_woBN_0.5_dropout.log', 'cnn_woBN_0.5_dropout.log'
                 ]
for path in log_file_path:
    path = os.path.join(file_root, path)
    epochs = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    # Regular expressions to extract the required information from the log
    train_loss_pattern = r'training loss:\s+([0-9\.]+)'
    train_acc_pattern = r'training accuracy:\s+([0-9\.]+)'
    val_loss_pattern = r'validation loss:\s+([0-9\.]+)'
    val_acc_pattern = r'validation accuracy:\s+([0-9\.]+)'
    epoch_pattern = r'Epoch (\d+) of \d+'

    # Read the log file
    with open(path, 'r') as file:
        for line in file:
            # Skip lines containing 'best validation accuracy'
            if 'best validation accuracy' in line:
                continue

            # Extract information based on regex patterns
            epoch_match = re.search(epoch_pattern, line)
            if epoch_match:
                epochs.append(int(epoch_match.group(1)))
            
            train_loss_match = re.search(train_loss_pattern, line)
            if train_loss_match:
                train_loss.append(float(train_loss_match.group(1)))
            
            train_acc_match = re.search(train_acc_pattern, line)
            if train_acc_match:
                train_acc.append(float(train_acc_match.group(1)))
            
            val_loss_match = re.search(val_loss_pattern, line)
            if val_loss_match:
                val_loss.append(float(val_loss_match.group(1)))
            
            val_acc_match = re.search(val_acc_pattern, line)
            if val_acc_match:
                val_acc.append(float(val_acc_match.group(1)))

    # Check if all lists have the same length
    print(f"Epochs: {len(epochs)}")
    print(f"Train Loss: {len(train_loss)}")
    print(f"Train Accuracy: {len(train_acc)}")
    print(f"Validation Loss: {len(val_loss)}")
    print(f"Validation Accuracy: {len(val_acc)}")

    # # Ensure the lists are the same length, if not, truncate them
    # min_len = min(len(epochs), len(train_loss), len(train_acc), len(val_loss), len(val_acc))
    # epochs = epochs[:min_len]
    # train_loss = train_loss[:min_len]
    # train_acc = train_acc[:min_len]
    # val_loss = val_loss[:min_len]
    # val_acc = val_acc[:min_len]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Training Loss', color='blue')
    plt.plot(epochs, val_loss, label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Training Accuracy', color='blue')
    plt.plot(epochs, val_acc, label='Validation Accuracy', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')


    # Show the plots
    plt.tight_layout()
    plt.savefig(path+'.png')
    # plt.show()
