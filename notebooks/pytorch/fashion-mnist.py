from numpy import mod
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda


LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 5


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        # Flatten 2D 28x28 image into 1D
        self.flatten = nn.Flatten()
        # The layers
        self.layers = nn.Sequential(
            nn.Linear(in_features=28 * 28, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=10),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits


def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    train_data = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=ToTensor()
    )

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DEVICE: Using {device}")

    model = NeuralNet().to(device)
    print(f"MODEL: {model}")

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        test_loop(test_dataloader, model, loss_fn, device)
    print("Done!")


if __name__ == "__main__":
    main()
