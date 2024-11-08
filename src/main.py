import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import os
import csv
import matplotlib.pyplot as plt
from datetime import datetime
from models import get_model_by_name
from data_handlers import CIFAR10, CIFAR100
from opacus import PrivacyEngine


import logging


"""
TODO: implement Parameter averaging using EMA
with https://github.com/lucidrains/ema-pytorch/tree/main/ema_pytorch
beta = 0.9999 as said in paper Unlocking High Accuracy
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a differentially private neural network"
    )
    parser.add_argument("--model", default="WideResNet")

    parser.add_argument(
        "--dataset", default="CIFAR10", help="available datasets: CIFAR10, CIFAR100"
    )

    parser.add_argument(
        "--algo",
        default="DPSGD",
        type=str,
        help="algorithm (ClipSGD, EFSGD, DPSGD, DiceSGD)",
    )

    # Clipping thresholds for DiceSGD
    parser.add_argument("--C", default=0.5, type=float, help="clipping threshold")
    parser.add_argument(
        "--C2",
        default=1.0,
        type=float,
        help="clipping threshold ration C2/C1",
    )

    parser.add_argument("--save_results", default=True)
    parser.add_argument("--optimizer", default="sgd")
    parser.add_argument(
        "--subset_size",
        default=50000,
        type=int,
        help="check optimizer and model capacity",
    )
    parser.add_argument("--dry_run", default=False, help="check a single sample")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096 * 2,
        help="input batch size for training (default: 16k)",
    )

    parser.add_argument(
        "--epsilon", type=float, default=3, help="epsilon = privacy budget (default: 1)"
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=10 ** (-5),
        help="delta for epsilon-delta DP (default: 10^-5)",
    )

    parser.add_argument(
        "--epochs", type=int, default=2, help="number of epochs to train (default: 2)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate (default: 0.001)"
    )
    parser.add_argument("--cpu", default=False, help="force CPU training")
    parser.add_argument(
        "--save_experiment",
        default=True,
        help="Save experiment details",
    )
    return parser.parse_args()


# -------------------------#
# Setup experiment params  #
# -------------------------#

args = parse_args()

device_str = "cpu"
if args.cpu == False:
    if torch.backends.mps.is_available():
        device_str = "mps"
    if torch.cuda.is_available():
        device_str = "cuda"

print("Using device", str(device_str))
device = torch.device(device_str)

# Load dataset
val_size = int(0.15 * args.subset_size)
classes = None
if args.dataset == "CIFAR100":
    CIFAR100_data = CIFAR100(
        val_size=val_size, batch_size=args.batch_size, subset_size=args.subset_size
    )
    train_dl = CIFAR100_data.train_dl
    test_dl = CIFAR100_data.val_dl
    classes = 100
elif args.dataset == "CIFAR10":
    CIFAR10_data = CIFAR10(
        val_size=val_size, batch_size=args.batch_size, subset_size=args.subset_size
    )
    train_dl = CIFAR10_data.train_dl
    test_dl = CIFAR10_data.val_dl
    classes = 10

model = get_model_by_name(args.model, classes=classes).to(device)

OPTIMIZERS = {
    "sgd": optim.SGD(model.parameters(), lr=args.lr),
    "sgdm": optim.SGD(model.parameters(), lr=args.lr, momentum=0.1),
    "adam": optim.Adam(model.parameters(), lr=args.lr),
    "rmsprop": optim.RMSprop(model.parameters(), lr=args.lr),
    "adagrad": optim.Adagrad(model.parameters(), lr=args.lr),
    "adamw": optim.AdamW(model.parameters(), lr=args.lr),
}

optimizer = OPTIMIZERS[args.optimizer]

loss_fn = nn.CrossEntropyLoss()

# -------------------------#
#       Training Loop      #
# -------------------------#

# Tracking loss
train_losses = []
test_losses = []

# Tracking accuracies
train_accuracies = []
test_accuracies = []


from opacus.utils.batch_memory_manager import BatchMemoryManager


def train(epoch, model, optimizer, dl, loss_fn, device, log_interval=1):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    num_iterations = 2500
    iteration_count = 0

    with BatchMemoryManager(
        data_loader=dl,
        max_physical_batch_size=4096,  # Physical batch size
        optimizer=optimizer,
    ) as memory_safe_data_loader:

        for batch_idx, (data, target) in enumerate(memory_safe_data_loader):
            if (
                iteration_count >= num_iterations
            ):  # Stop if we’ve reached the target iterations
                break

            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)
            loss = loss_fn(output, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

            if (batch_idx % log_interval) == 0:
                print(
                    f"Train ep {epoch} - batch {batch_idx} [{batch_idx * len(data)}/{len(dl.dataset)}] "
                    f"({100. * batch_idx / len(dl):.0f}%) \t loss: {loss.item()} \t "
                    f"accuracy: {100.0 * correct / total}"
                )

            iteration_count += 1  # Track the number of iterations (updates)

    avg_train_loss = running_loss / len(dl)
    train_losses.append(avg_train_loss)

    train_accuracy = 100.0 * correct / total
    train_accuracies.append(train_accuracy)


def evaluate(model, dl, loss_fn, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dl:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            running_loss += loss.item()

            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

    avg_test_loss = running_loss / len(dl)

    test_accuracy = 100.0 * correct / total

    print(f"Test loss: {avg_test_loss} test accuracy: {test_accuracy}")

    return avg_test_loss, test_accuracy


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    log_file = logging.getLogger(__name__)

    # Calculate delta
    # delta = 1 / (2 * len(train_dl.dataset))

    # Create directory for experiment
    if args.save_experiment:
        current_date = datetime.now().strftime("%Y-%b-%d %Hh%Mmin")
        experiment_dir = f"./experiments/{current_date}"
        os.makedirs(experiment_dir, exist_ok=True)

        # Save args to a file
        args_file = os.path.join(experiment_dir, "config.txt")
        with open(args_file, "w") as f:
            for arg, value in vars(args).items():
                f.write(f"{arg}={value}\n")

    # Initialize PrivacyEngine (if needed)
    if args.algo in ["DPSGD", "DiceSGD"]:
        privacy_engine = PrivacyEngine()

        model, optimizer, train_dl = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            # criterion=loss_fn, # this seems to be not compatible w/ opacus 1.5.0
            data_loader=train_dl,
            target_epsilon=args.epsilon,
            target_delta=args.delta,
            epochs=args.epochs,
            # noise_multiplier=1.1,
            max_grad_norm=1.0,
        )

    # Training algorithms

    if args.algo == "DiceSGD":
        from DiceSGD.trainers import DiceSGD

        DiceSGD(
            model,
            train_dl,
            test_dl,
            args.batch_size,
            args.epsilon,
            delta,
            args.subset_size,
            16,
            args.epochs,
            args.C,
            args.C2,
            device,
            args.lr / args.C,
            "sgd",
            log_file,
        )

    elif args.algo == "DPSGD":
        for epoch in range(1, args.epochs + 1):
            train(epoch, model, optimizer, train_dl, loss_fn, device)
            avg_test_loss, test_accuracy = evaluate(model, test_dl, loss_fn, device)
            test_losses.append(avg_test_loss)
            test_accuracies.append(test_accuracy)

            # Save model every 2 epochs
            if epoch % 2 == 0:
                model_path = os.path.join(experiment_dir, f"model_epoch_{epoch}.pth")
                torch.save(model.state_dict(), model_path)

    elif args.algo == "DynamicSGD":
        from DynamicSGD.trainers import DynamicSGD
        from ema_pytorch import EMA

        ema = EMA(
            model,
            beta=0.9999,  # exponential moving average factor
            update_after_step=100,  # only after this number of .update() calls will it start updating
            update_every=10,  # how often to actually update, to save on compute (updates every 10th .update() call)
        )

        dysgd = DynamicSGD(
            model=model,
            train_dl=train_dl,
            test_dl=test_dl,
            device=device,
            batch_size=args.batch_size,
            epsilon=args.epsilon,
            delta=args.delta,
            epochs=args.epochs,
            C=args.C,
            lr=args.lr,
            method=args.optimizer,
            decay_rate_sens=0.3,
            decay_rate_mu=0.8,
            ema=ema,
        )

        test_losses = dysgd.test_losses
        train_losses = dysgd.train_losses
        test_accuracies = dysgd.test_accuracies
        train_accuracies = dysgd.train_accuracies

    else:
        print("Algorithm doesn't exist")

    # Save losses to a CSV
    if args.save_experiment:
        loss_file = os.path.join(experiment_dir, "losses.csv")
        with open(loss_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["Epoch", "Train Loss", "Test Loss", "Train Accuracy", "Test Accuracy"]
            )
            for epoch, (train_loss, test_loss, train_acc, test_acc) in enumerate(
                zip(train_losses, test_losses, train_accuracies, test_accuracies), 1
            ):
                writer.writerow([epoch, train_loss, test_loss, train_acc, test_acc])

    # Plot and save the loss curves
    plt.plot(range(1, args.epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, args.epochs + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Testing Loss Curves")

    if args.save_experiment:
        plt.savefig(os.path.join(experiment_dir, "loss_curves.png"))

    plt.figure()
    plt.plot(range(1, args.epochs + 1), train_accuracies, label="Train Accuracy")
    plt.plot(range(1, args.epochs + 1), test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Training and Testing Accuracy Curves")

    if args.save_experiment:
        plt.savefig(os.path.join(experiment_dir, "accuracy_curves.png"))

    # Save model
    if args.save_experiment:
        model_path = os.path.join(experiment_dir, "model.pth")
        torch.save(model.state_dict(), model_path)

# eps=3, delta=10^-5
# experimente: DPSGD, momentum, nesterov momentum, unterschiedliche batch sizes, hessian approximation?
