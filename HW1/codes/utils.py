from __future__ import division
from __future__ import print_function
import numpy as np
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import os

def onehot_encoding(label, max_num_class):
    encoding = np.eye(max_num_class)
    encoding = encoding[label]
    return encoding


def calculate_acc(output, label):
    correct = np.sum(np.argmax(output, axis=1) == label)
    return correct / len(label)


def LOG_INFO(msg):
    now = datetime.now()
    display_now = str(now).split(' ')[1][:-3]
    print(display_now + ' ' + msg)

def save_results_to_csv(loss_history, acc_history, filename="train_log.csv", meta=None, root = None):
    filename = os.path.join(root, filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        if meta is not None:
            writer.writerow(["# meta", meta])
        writer.writerow(["epochs", "loss", "acc"])
        for i, (l, a) in enumerate(zip(loss_history, acc_history), 1):
            writer.writerow([i, f"{l:.5f}", f"{a:.5f}"])
            

def plot_results_from_csv(filename="train_log.csv", plot_file="train_plot.png", root = None):
    filename = os.path.join(root,filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    iters, losses, accs = [], [], []

    with open(filename, mode="r") as f:
        first_line = f.readline()
        if first_line.startswith("#"):
            reader = csv.DictReader(f)
        else:
            f.seek(0)
            reader = csv.DictReader(f)
        for row in reader:
            iters.append(int(row["epochs"]))
            losses.append(float(row["loss"]))
            accs.append(float(row["acc"]))

    plt.figure(figsize=(10, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(iters, losses, label="Loss")
    plt.xlabel("Epoches")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(iters, accs, label="Accuracy", color="orange")
    plt.xlabel("Epoches")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(root, plot_file)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    

def plot_results_from_csv_new(
    train_file="train_log.csv", 
    test_file="test_log.csv", 
    plot_file="train_test_plot.png", 
    root=None,
    test_every=5
):
    train_path = os.path.join(root, train_file)
    test_path  = os.path.join(root, test_file)
    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    train_epochs, train_losses, train_accs = [], [], []
    with open(train_path, "r") as f:
        first = f.readline()
        if not first.startswith("#"):
            f.seek(0)
        reader = csv.DictReader(f)
        for row in reader:
            train_epochs.append(int(row["epochs"]))
            train_losses.append(float(row["loss"]))
            train_accs.append(float(row["acc"]))

    test_epochs, test_losses, test_accs = [], [], []
    with open(test_path, "r") as f:
        first = f.readline()
        if not first.startswith("#"):
            f.seek(0)
        reader = csv.DictReader(f)
        for row in reader:
            test_epochs.append(int(row["epochs"]))
            test_losses.append(float(row["loss"]))
            test_accs.append(float(row["acc"]))

    if test_every is not None and len(test_epochs) >= 2:
        if all((test_epochs[i+1] - test_epochs[i]) == 1 for i in range(len(test_epochs)-1)):
            test_epochs = [e * test_every for e in test_epochs]

    plt.figure(figsize=(10, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_epochs, train_losses, label="Train Loss")
    plt.plot(test_epochs,  test_losses,  "o--", label="Test Loss")
    plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.title("Train vs Test Loss"); plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_epochs, train_accs, label="Train Accuracy")
    plt.plot(test_epochs,  test_accs,  "o--", label="Test Accuracy")
    plt.xlabel("Epochs"); plt.ylabel("Accuracy"); plt.title("Train vs Test Accuracy"); plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(root, plot_file)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300)
    plt.close()