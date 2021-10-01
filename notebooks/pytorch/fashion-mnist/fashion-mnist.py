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
            nn.Linear(in_features=28 * 28, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits


class Trainer:
    def __init__(
        self,
        train_data,
        test_data,
        learning_rate=1e-3,
        batch_size=64,
        epochs=10,
        model=NeuralNet(),
        optimizer=torch.optim.SGD,
        loss_fn=nn.CrossEntropyLoss(),
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        self.test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = loss_fn
        print(self)

    def train_loop(self, loader):
        size = len(loader.dataset)
        for batch, (X, y) in enumerate(self.train_loader):
            # Compute prediction and loss
            X = X.to(self.device)
            y = y.to(self.device)
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test_loop(self, loader):
        size = len(loader.dataset)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in loader:
                X = X.to(self.device)
                y = y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= size
        correct /= size
        print(
            f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )

    def run(self):
        for t in range(self.epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.train_loop(self.train_loader)
            self.test_loop(self.test_loader)
        print("Done!")

    def __repr__(self):
        return f"DEVICE: Using {self.device}\nMODEL: {self.model}"


def main():
    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=ToTensor()
    )

    trainer = Trainer(train_data=train_data, test_data=test_data)
    trainer.run()


if __name__ == "__main__":
    main()
