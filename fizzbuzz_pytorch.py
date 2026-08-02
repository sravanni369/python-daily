"""FizzBuzz, but a neural network learns the rules.

Instead of writing `if n % 15 == 0`, we let a small PyTorch classifier figure
out FizzBuzz from examples. It's a clean 4-class problem: number, Fizz, Buzz,
FizzBuzz. Train on 101-1023, test on the classic 1-100 it has never seen.

Run: python fizzbuzz_pytorch.py
"""

import torch
import torch.nn as nn

BITS = 10          # binary-encode integers (handles up to 1023)
CLASSES = ["num", "Fizz", "Buzz", "FizzBuzz"]


def encode(n):
    """Represent an integer as a BITS-long binary feature vector."""
    return [(n >> i) & 1 for i in range(BITS)]


def label(n):
    """The FizzBuzz class of n: 3=FizzBuzz, 2=Buzz, 1=Fizz, 0=plain number."""
    if n % 15 == 0:
        return 3
    if n % 5 == 0:
        return 2
    if n % 3 == 0:
        return 1
    return 0


def make(lo, hi):
    X = torch.tensor([encode(n) for n in range(lo, hi)], dtype=torch.float32)
    y = torch.tensor([label(n) for n in range(lo, hi)], dtype=torch.long)
    return X, y


class Net(nn.Module):
    """A tiny MLP: 10 input bits -> 100 hidden -> 4 classes."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(BITS, 128), nn.ReLU(), nn.Linear(128, 4))

    def forward(self, x):
        return self.net(x)


def train():
    torch.manual_seed(0)
    Xtr, ytr = make(101, 1024)          # train on 101..1023
    model = Net()
    opt = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, 3001):
        opt.zero_grad()
        loss = loss_fn(model(Xtr), ytr)
        loss.backward()
        opt.step()
        if epoch % 750 == 0:
            print(f"epoch {epoch:4d}  loss {loss.item():.4f}")
    return model


def fizzbuzz(model, n):
    """Ask the trained net for the FizzBuzz output of n."""
    x = torch.tensor([encode(n)], dtype=torch.float32)
    cls = model(x).argmax(1).item()
    return str(n) if cls == 0 else CLASSES[cls]


if __name__ == "__main__":
    model = train()

    # accuracy on the unseen 1..100
    Xte, yte = make(1, 101)
    with torch.no_grad():
        acc = (model(Xte).argmax(1) == yte).float().mean().item()
    print(f"\ntest accuracy on 1-100: {acc*100:.1f}%")

    # what it actually predicts for 1..15
    print("prediction 1-15:", [fizzbuzz(model, n) for n in range(1, 16)])
