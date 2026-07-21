"""Day 07 — PyTorch autograd: a linear-regression training loop from scratch.

Continues Day 05 (tensor basics). No nn.Module, no optimizer class — just
tensors with requires_grad, a manual forward pass, MSE loss, loss.backward(),
and hand-written SGD steps, so the mechanics of autograd are visible.

Requires: torch (pip install torch).
"""

import torch


def make_data(n: int = 200, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic data for y = 3.5*x1 - 2.0*x2 + 0.75 + noise."""
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, 2, generator=g) * 10          # features in [0, 10)
    true_w = torch.tensor([3.5, -2.0])
    true_b = 0.75
    noise = torch.randn(n, generator=g) * 0.3
    y = X @ true_w + true_b + noise
    return X, y


def train(X: torch.Tensor, y: torch.Tensor, lr: float = 0.01, epochs: int = 400):
    """Fit w, b by gradient descent using autograd only.

    Returns (w, b, history) where history is the loss every 50 epochs.
    """
    # Parameters start random; requires_grad=True tells autograd to track them.
    w = torch.randn(2, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    # Standardize features so one learning rate suits both weights.
    mean, std = X.mean(dim=0), X.std(dim=0)
    Xn = (X - mean) / std

    history = []
    for epoch in range(1, epochs + 1):
        y_hat = Xn @ w + b                  # forward pass
        loss = ((y_hat - y) ** 2).mean()    # MSE

        loss.backward()                     # autograd fills w.grad and b.grad

        with torch.no_grad():               # SGD step must not be tracked
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()                      # grads accumulate — reset each step
        b.grad.zero_()

        if epoch % 50 == 0 or epoch == 1:
            history.append((epoch, loss.item()))

    # Un-standardize: recover weights/bias in original feature units.
    with torch.no_grad():
        w_orig = w / std
        b_orig = b + (w * (-mean / std)).sum()
    return w_orig, b_orig.squeeze(), history


if __name__ == "__main__":
    X, y = make_data()
    w, b, history = train(X, y)

    print("loss curve (epoch, MSE):")
    for epoch, mse in history:
        print(f"  {epoch:>4}  {mse:10.4f}")

    print(f"\nlearned:  w = [{w[0].item():+.3f}, {w[1].item():+.3f}]  b = {b.item():+.3f}")
    print("true:     w = [+3.500, -2.000]  b = +0.750")

    # A second look at autograd on a scalar: d(x^3)/dx at x=2 is 12.
    x = torch.tensor(2.0, requires_grad=True)
    (x ** 3).backward()
    print(f"\nautograd sanity check: d(x^3)/dx at x=2 -> {x.grad.item():.1f} (expected 12.0)")
