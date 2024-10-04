import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import os
import csv
import matplotlib.pyplot as plt
from datetime import datetime
from models import CNNModel
from data_handlers import CIFAR10


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a differentially private neural network"
    )
    parser.add_argument("--run_name", type=str, default="DDPM")
    parser.add_argument("--save_results", default=True)
    parser.add_argument("--model_snapshot_epochs", default=20)
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
        "--epochs", type=int, default=2, help="number of epochs to train (default: 5)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--cpu", action="store_true", default=False, help="force CPU training"
    )
    parser.add_argument(
        "--save_experiment",
        action="store_true",
        default=True,
        help="Save experiment details",
    )
    return parser.parse_args()


args = parse_args()

device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

model = CNNModel().to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)

loss_fn = nn.CrossEntropyLoss()

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


# Tracking loss
train_losses = []
test_losses = []


def train(epoch):
    model.train()
    running_loss = 0

    dl = CIFAR10(subset_size=args.subset_size).train_dl

    for batch_idx, (data, target) in enumerate(dl):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)
        loss = loss_fn(output, target)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if (batch_idx % 5) == 0:
            print(
                f"Training step: epoch {epoch} - batch {batch_idx} [{batch_idx * len(data)}/{len(dl.dataset)}] ({100. * batch_idx / len(dl):.0f}%) \t {loss.item()}"
            )

    avg_train_loss = running_loss / len(dl)
    train_losses.append(avg_train_loss)


def test():
    model.eval()
    test_loss = 0
    correct = 0

    dl = CIFAR10().val_dl

    with torch.no_grad():
        for data, target in dl:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()

            pred = output.argmax(dim=1, keepdims=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    avg_test_loss = test_loss / len(dl)
    test_losses.append(avg_test_loss)

    print(
        f"Test set: Average loss: {avg_test_loss}, Accuracy {correct}/{len(dl.dataset)} ({100. * correct / len(dl.dataset):.4f})\n"
    )


if __name__ == "__main__":

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test()

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
    plt.plot(range(1, args.epochs + 1), test_losses, label="Test Loss")
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
