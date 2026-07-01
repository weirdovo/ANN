from network import Network
from utils import LOG_INFO, save_results_to_csv, plot_results_from_csv, plot_results_from_csv_new
from layers import Selu, HardSwish, Linear, Mish
from loss import BrierLoss, SoftmaxCrossEntropyLoss, HingeLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
import argparse

def get_activation(name: str):
    if name.lower() == "selu":
        return Selu("selu")
    elif name.lower() == "hardswish":
        return HardSwish("hardswish")
    elif name.lower() == "mish":
        return Mish("mish")
    else:
        raise ValueError(f"Unknown activation {name}")

def get_loss(name: str):
    if name.lower() == "sce":
        return SoftmaxCrossEntropyLoss("sce")
    elif name.lower() == "brier":
        return BrierLoss("brier")
    elif name.lower() == "hinge":
        return HingeLoss("hinge")
    else:
        raise ValueError(f"Unknown loss {name}")



def main(args):

    train_data, test_data, train_label, test_label = load_mnist_2d('codes/data')

    # Your model defintion here
    # You should explore different model architecture
    model = Network()
    model.add(Linear("fc1", 784, args.hidden1, 0.01))
    model.add(get_activation(args.activation))
    if args.hidden2 is not None:
        model.add(Linear("fc2", args.hidden1, args.hidden2, 0.01))
        model.add(get_activation(args.activation))
        model.add(Linear("fc3", args.hidden2, 10, 0.01))
    else:
        model.add(Linear("fc2", args.hidden1, 10, 0.01))
    
    loss = get_loss(args.loss)
    arch = f"{args.activation}_{args.loss}_h{args.hidden1}"
    # Training configuration
    # You should adjust these hyperparameters
    # NOTE: one iteration means model forward-backwards one batch of samples.
    #       one epoch means model has gone through all the training samples.
    #       'disp_freq' denotes number of iterations in one epoch to display information.

    config = {
        "learning_rate": args.lr,
        "weight_decay": args.wd,
        "momentum": args.mom,
        "batch_size": args.batch,
        "max_epoch": args.epoch,
        "disp_freq": 50,
        "test_epoch": 5,
    }

    trl,tracc,tel,teacc = [], [], [], []


    for epoch in range(config['max_epoch']):
        LOG_INFO('Training @ %d epoch...' % (epoch))
        train_loss, train_acc = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])
        trl.append(train_loss)
        tracc.append(train_acc)
        if (epoch + 1) % config['test_epoch'] == 0:
            LOG_INFO('Testing @ %d epoch...' % (epoch))
            test_loss, test_acc = test_net(model, loss, test_data, test_label, config['batch_size'])
            tel.append(test_loss)
            teacc.append(test_acc)

    meta_info = f"{arch}_lr={config['learning_rate']}_mom={config['momentum']}_wd={config['weight_decay']}"

    tag = f"_{meta_info}_ep{config['max_epoch']}"
    root = f"{args.trial}/{tag}"
    save_results_to_csv(trl, tracc, f"train{tag}.csv", meta_info, root)
    save_results_to_csv(tel, teacc, f"test{tag}.csv", meta_info, root)
    plot_results_from_csv(f"train{tag}.csv", f"train{tag}.png", root)
    plot_results_from_csv(f"test{tag}.csv", f"test{tag}.png", root)
    plot_results_from_csv_new(f"train{tag}.csv", f"test{tag}.csv", f"{tag}.png", root, config['test_epoch'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation", type=str, default="selu", help="Activation: selu|hardswish|mish")
    parser.add_argument("--loss", type=str, default="sce", help="Loss: sce|brier|hinge")
    parser.add_argument("--hidden1", type=int, default=128, help="Hidden layer 1 size")
    parser.add_argument("--hidden2", type=int, default=None, help="Hidden layer 2 size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.001, help="Weight decay")
    parser.add_argument("--mom", type=float, default=0.1, help="Momentum")
    parser.add_argument("--batch", type=int, default=100, help="Batch size")
    parser.add_argument("--epoch", type=int, default=100, help="Max epochs")
    parser.add_argument("--trial", type=str, default=None, help="Name of this trial")
    args = parser.parse_args()

    main(args)



