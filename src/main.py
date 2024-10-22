import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import os
import csv
import matplotlib.pyplot as plt
from datetime import datetime
from models import get_model_by_name
from data_handlers import CIFAR100, CIFAR100
from opacus import PrivacyEngine
from DiceSGD.trainers import DiceSGD
from DynamicSGD.trainers import DynamicSGD
import logging
import timm
from opacus.validators import ModuleValidator

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
        "--algo",
        default="DPSGD",
        type=str,
        help="algorithm (ClipSGD, EFSGD, DPSGD, DiceSGD)",
    )

    # Clipping thresholds for DiceSGD
    parser.add_argument(
        "--C", default=0.5, nargs="+", type=float, help="clipping threshold"
    )
    parser.add_argument(
        "--C2",
        default=1.0,
        nargs="+",
        type=float,
        help="clipping threshold ration C2/C1",
    )

    parser.add_argument("--save_results", default=True)
    parser.add_argument("--optimizer", default="SGD")
    parser.add_argument(
        "--subset_size",
        default=None,
        type=int,
        help="check optimizer and model capacity",
    )
    parser.add_argument("--dry_run", default=False, help="check a single sample")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )

    parser.add_argument(
        "--epsilon", type=float, default=1, help="epsilon = privacy budget (default: 1)"
    )

    parser.add_argument(
        "--epochs", type=int, default=2, help="number of epochs to train (default: 2)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--cpu", action="store_true", default=False, help="force CPU training"
    )
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

device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

model = get_model_by_name(args.model).to(device)

OPTIMIZERS = {
    "SGD": optim.SGD(model.parameters(), lr=args.lr),
    "SGDM": optim.SGD(model.parameters(), lr=args.lr, momentum=0.1),
    "Adam": optim.Adam(model.parameters(), lr=args.lr),
}

optimizer = OPTIMIZERS[args.optimizer]

loss_fn = nn.CrossEntropyLoss()

# -------------------------#
#       Training Loop      #
# -------------------------#

# Tracking loss
train_losses = []
test_losses = []


def train(epoch, model, optimizer, dl, loss_fn, device, log_interval=5):
    model.train()
    running_loss = 0

    for batch_idx, (data, target) in enumerate(dl):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)
        loss = loss_fn(output, target)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx % log_interval) == 0:
            print(
                f"Training step: epoch {epoch} - batch {batch_idx} [{batch_idx * len(data)}/{len(dl.dataset)}] ({100. * batch_idx / len(dl):.0f}%) \t {loss.item()}"
            )

    avg_train_loss = running_loss / len(dl)
    train_losses.append(avg_train_loss)


def evaluate(model, dl, loss_fn, device):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for data, target in dl:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            running_loss += loss.item()
    avg_test_loss = running_loss / len(dl)
    return avg_test_loss


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    log_file = logging.getLogger(__name__)

    val_size = int(0.15 * args.subset_size)

    # Load dataset
    CIFAR100_data = CIFAR100(
        val_size=val_size, batch_size=args.batch_size, subset_size=args.subset_size
    )
    train_dl = CIFAR100_data.train_dl
    test_dl = CIFAR100_data.val_dl

    # Calculate delta
    delta = 1 / (2 * len(train_dl.dataset))

    # Initialize PrivacyEngine (if needed)
    if args.algo in ["DPSGD", "DiceSGD", "DynamicSGD"]:
        privacy_engine = PrivacyEngine()

        model, optimizer, train_dl = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_dl,
            noise_multiplier=1.1,
            max_grad_norm=1.0,
        )

    # Training algorithms
    if args.algo == "DiceSGD":
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
            avg_test_loss = evaluate(model, test_dl, loss_fn, device)
            test_losses.append(avg_test_loss)

    elif args.algo == "DynamicSGD":
        DynamicSGD(
            model=model,
            train_dl=train_dl,
            test_dl=test_dl,
            device=device,
            batch_size=args.batch_size,
            epsilon=args.epsilon,
            delta=delta,
            epochs=args.epochs,
            C=args.C,
            lr=args.lr,
            method="sgd",
            decay_rate_sens=0.3,
            decay_rate_mu=0.8,
        )
    else:
        print("Algorithm doesn't exist")

    # Create directory for experiment
    if args.save_experiment:
        current_date = datetime.now().strftime("%Y-%b-%d %Hh%Mmin")
        experiment_dir = f"./experiments/{current_date}"
        os.makedirs(experiment_dir, exist_ok=True)

        # Save args to a file
        args_file = os.path.join(experiment_dir, "config.txt")
        with open(args_file, "w") as f:
            for arg, value in vars(args).items():
                f.write(f"{arg}: {value}\n")

    # Save losses to a CSV
    if args.save_experiment:
        loss_file = os.path.join(experiment_dir, "losses.csv")
        with open(loss_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train Loss", "Test Loss"])
            for epoch, (train_loss, test_loss) in enumerate(
                zip(train_losses, test_losses), 1
            ):
                writer.writerow([epoch, train_loss, test_loss])

    # Plot and save the loss curves
    plt.plot(range(1, args.epochs + 1), train_losses, label="Train Loss")
    # plt.plot(range(1, args.epochs + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Testing Loss Curves")

    if args.save_experiment:
        plt.savefig(os.path.join(experiment_dir, "loss_curves.png"))

    # Save model
    if args.save_experiment:
        model_path = os.path.join(experiment_dir, "model.pth")
        torch.save(model.state_dict(), model_path)
