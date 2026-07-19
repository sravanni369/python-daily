"""Day 05 - PyTorch tensor drills.

A guided tour of the PyTorch tensor: the single object that everything in deep
learning is built on. Concepts follow Chapter 1 of "Programming PyTorch for Deep
Learning" (Ian Pointer, O'Reilly) - creation, rank/shape, indexing, element-wise
math, pulling scalars out, the NumPy bridge, and moving work to the GPU.

A tensor is both a container for numbers and a set of rules for transforming them
- easiest to picture as a multidimensional array. Its *rank* is the number of
dimensions: a scalar is rank 0, a vector rank 1, a matrix rank 2, and so on.

Run:  python day05_pytorch_tensors.py   (works on CPU; uses GPU if available)

Author: Lakshmi Sravani Putta
Contact: sravannicareerv@gmail.com | linkedin.com/in/sravani-p-212899272
"""

from __future__ import annotations

import numpy as np
import torch


def creating_tensors() -> dict[str, torch.Tensor]:
    """Create tensors the four everyday ways and return them by name.

    - ``torch.rand`` : random values in [0, 1)
    - ``torch.tensor``: from a Python list (infers dtype)
    - ``torch.ones`` / ``torch.zeros``: filled constants
    """
    return {
        "random": torch.rand(2, 2),
        "from_list": torch.tensor([[0, 0, 1], [1, 1, 1], [0, 0, 0]]),
        "ones": torch.ones(1, 2),
        "zeros": torch.zeros(2, 2),
    }


def describe(t: torch.Tensor) -> dict[str, object]:
    """Return a tensor's rank, shape and dtype.

    Rank is ``t.dim()`` - the number of dimensions. It is the single most useful
    thing to know about a tensor when a shape error shows up.
    """
    return {"rank": t.dim(), "shape": tuple(t.shape), "dtype": str(t.dtype)}


def index_and_update(t: torch.Tensor) -> torch.Tensor:
    """Return a copy of ``t`` with element [0, 0] set to 5, via standard indexing."""
    out = t.clone()
    out[0][0] = 5
    return out


def elementwise_math() -> torch.Tensor:
    """Add two tensors element-wise: ones(1,2) + ones(1,2) == [[2., 2.]]."""
    return torch.ones(1, 2) + torch.ones(1, 2)

def to_python_scalar(value: float = 3.14) -> float:
    """Wrap a value in a rank-0 tensor, then extract it back with ``.item()``.

    ``.item()`` only works on a tensor holding exactly one element; it is how you
    pull a plain Python number out of a tensor (e.g. a loss value for logging).
    """
    rank0 = torch.tensor(value)
    return rank0.item()


def numpy_bridge(array: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    """Cross the NumPy <-> tensor bridge and back.

    ``torch.from_numpy`` and ``Tensor.numpy`` share the same underlying memory on
    CPU, so this is cheap - you keep your NumPy/pandas toolkit and just add GPU
    and gradients on top.
    """
    tensor = torch.from_numpy(array)
    back = tensor.numpy()
    return tensor, back


def pick_device() -> torch.device:
    """Return CUDA if a GPU is available, else CPU (the standard idiom)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autograd_demo() -> float:
    """Show why tensors matter: set ``requires_grad`` and differentiate.

    For y = x**2, dy/dx = 2x. At x = 3 that is 6. PyTorch records the operation
    and computes the gradient for us when we call ``.backward()`` - the exact
    mechanism a neural network uses to learn.
    """
    x = torch.tensor(3.0, requires_grad=True)
    y = x ** 2
    y.backward()
    return x.grad.item()


def _demo() -> None:
    print("=== 1. Creating tensors ===")
    for name, t in creating_tensors().items():
        print(f"{name:10s} -> {describe(t)}")

    print("\n=== 2. Indexing & update ([0,0] = 5) ===")
    base = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(index_and_update(base).tolist())

    print("\n=== 3. Element-wise math ===")
    print(elementwise_math().tolist())

    print("\n=== 4. Rank-0 tensor -> Python scalar ===")
    print(f".item() -> {to_python_scalar(3.14)}")

    print("\n=== 5. NumPy bridge ===")
    tensor, back = numpy_bridge(np.array([1, 2, 3]))
    print(f"from_numpy -> {tensor.tolist()} | back to numpy -> {back.tolist()}")

    print("\n=== 6. Device selection ===")
    device = pick_device()
    moved = torch.ones(2, 2).to(device)
    print(f"cuda available: {torch.cuda.is_available()} | tensor is on: {moved.device}")

    print("\n=== 7. Autograd: d/dx (x^2) at x=3 ===")
    print(f"gradient -> {autograd_demo()}  (expected 6.0)")


if __name__ == "__main__":
    _demo()
