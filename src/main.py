import argparse
import torch
import torch.optim as optim
import torch.nn as nn

from models import CNNModel
from data_handlers import CIFAR10

def parse_args():
    parser = argparse.ArgumentParser(description="Train a differentially private neural network")
    parser.add_argument("--run_name", type=str, default="DDPM")
    parser.add_argument("--save_results", default=True)
    parser.add_argument("--model_snapshot_epochs", default=20)
    parser.add_argument("--overfit", default=False, help="check optimizer and model capacity")
    parser.add_argument("--dry_run", default=False, help="check a single sample")
    parser.add_argument("--batch_size", type=int, default=64, help="input batch size for training (default: 64)")
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs to train (default: 5)")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate (default: 0.001)")
    parser.add_argument("--cpu", action="store_true", default=False, help="force CPU training")

    return parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CNNModel().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_fn = nn.CrossEntropyLoss()


def train(epoch):
    model.train()

    dl = CIFAR10().train_dl

    for batch_idx, (data, target) in enumerate(dl):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)
        loss = loss_fn(output, target)

        loss.backward()

        optimizer.step()

        if (batch_idx % 100) == 0:
            print(
                f"Train Epoch: {epoch}-{batch_idx} [{batch_idx * len(data)}/{len(dl.dataset)}] ({100. * batch_idx / len(dl):.0f}%) \t {loss.item()}")


def test():
    model.eval()

    test_loss = 0

    correct = 0

    with torch.no_grad():
        for data, target in CIFAR10().val_dl:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss *= loss_fn(output, target).item()

            pred = output.argmax(dim=1, keepdims=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(CIFAR10().val_dl.dataset)
    print(
        f"\nTest set: Average loss: {test_loss}, Accuracy {correct}/{len(CIFAR10().val_dl.dataset)} ({100. * correct / len(CIFAR10().val_dl.dataset):.4f})")


if __name__=='__main__':
    for epoch in range(1, 11):
        train(epoch)
        test()

    visualize = False

    if visualize:
        import matplotlib.pyplot as plt

        model.eval()
        data, target = CIFAR10().val_dl.dataset[1]
        data = data.unsqueeze(0).to(device)

        output = model(data)
        prediction = output.argmax(dim=1, keepdims=True).item()

        print(f"Prediction {prediction}")
        image = data.squeeze(0).squeeze(0).cpu().numpy()

        plt.imshow(image, cmap="gray")
        plt.show()