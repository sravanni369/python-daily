"""Day 19b - all three paradigms from the guide, implemented and run.

Source: Zarnappa Earnoor, *Machine Learning: The 3 Core Paradigms*
        ("A Data Science Field Guide - Algorithms + Python code").
        Paradigm 1 - Supervised    - page 5
        Paradigm 2 - Unsupervised  - page 8
        Paradigm 3 - Deep Learning - page 10

The guide gives one runnable code block per paradigm and captions each with a
pattern to remember:

    p.5   "Pattern: create -> fit -> predict -> evaluate."
    p.8   "Pattern: no y - the model reads structure from X alone."
    p.10  "Pattern: define layers -> compile -> fit -> predict."

All three blocks run. All three print a confident-looking result. What this
file measures is that not one of the three ever tests the model on data it has
not already seen:

    Paradigm 1  scores accuracy on a single held-out row - two reachable values
    Paradigm 2  has no evaluation step at all; k=3 is asserted, never checked
    Paradigm 3  calls predict() on the exact four rows it just trained on

Paradigm 3 note: the book's import is `from tensorflow.keras.models import
Sequential`. TensorFlow is not installed in this interpreter (Python 3.13.5),
so that line raises ModuleNotFoundError here - the real traceback is printed
below rather than described. The architecture is then reproduced exactly in
PyTorch 2.9.1 (8-relu -> 4-relu -> 1-sigmoid, Adam, binary cross-entropy, 100
epochs), because the finding - predicting on your own training rows - does not
depend on which framework builds the layers.

Day 19's companion file, day19_accuracy_of_one_row.py, takes paradigm 1 apart
in depth. This file is the breadth pass across all three.

scikit-learn + pandas + numpy + torch. Deterministic. No network.
Run: python day19_three_paradigms.py
"""

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split

SEED = 42


def rule(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


# --------------------------------------------------------------------------
# Paradigm 1 - Supervised learning (book page 5)
# --------------------------------------------------------------------------
def paradigm_1_supervised():
    """The book's page-5 block, verbatim, then the size of what it evaluated."""
    X = [[1, 2], [2, 3], [3, 4], [4, 5]]
    y = [0, 0, 1, 1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("  Accuracy:", accuracy_score(y_test, y_pred))

    print()
    print(f"  rows in X                 : {len(X)}")
    print(f"  test_size=0.25 -> test rows: {len(y_test)}")
    print(f"  values that accuracy can take: {len(y_test) + 1} "
          f"(0 or 1 correct out of {len(y_test)})")
    print("  The 'evaluate' step scored one prediction.")
    print("  See day19_accuracy_of_one_row.py: over seeds 0-99 this same line")
    print("  prints 1.0 on 70 and 0.0 on 30, with the model never changing.")


# --------------------------------------------------------------------------
# Paradigm 2 - Unsupervised learning (book page 8)
# --------------------------------------------------------------------------
def paradigm_2_unsupervised():
    """The book's page-8 block, verbatim, then the check it never runs."""
    data = pd.DataFrame({
        "Feature1": [1, 2, 1.5, 5, 5.5, 6, 8, 8.5, 9],
        "Feature2": [1, 2, 1.0, 5, 5.2, 5.8, 8, 8.2, 9.5],
    })

    model = KMeans(n_clusters=3, random_state=42)
    model.fit(data)
    labels = model.labels_
    print("  Cluster Labels:", labels)

    print()
    print(f"  rows: {len(data)}   clusters requested: 3   (chosen in the code, "
          f"not measured)")
    print(f"  n_init actually used      : {model._n_init} "
          f"(the book never sets it; default is 'auto')")
    print(f"  inertia at k=3            : {model.inertia_:.4f}")
    print()
    print("  The block ends at .labels_. There is no evaluate step to leave out")
    print("  a row from - the pattern itself has no held-out anything. So the")
    print("  only honest question left is whether k=3 was the right call, and")
    print("  the book's code never asks it. Silhouette does (higher is better,")
    print("  range -1 to 1):")
    print()
    print(f"    {'k':>3} {'silhouette':>12} {'inertia':>12}")
    print("    " + "-" * 29)
    best_k, best_score = None, -2.0
    for k in range(2, 6):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(data)
        score = silhouette_score(data, km.labels_)
        if score > best_score:
            best_k, best_score = k, score
        print(f"    {k:>3} {score:>12.4f} {km.inertia_:>12.4f}")
    print()
    print(f"  Best silhouette is k={best_k} at {best_score:.4f}.")
    if best_k == 3:
        print("  The book's k=3 happens to be right on its own nine points - but")
        print("  the code prints nothing that would have told you either way.")
    else:
        print(f"  The book's k=3 is not the best split of its own nine points;")
        print(f"  k={best_k} scores higher, and the printed output says nothing.")


# --------------------------------------------------------------------------
# Paradigm 3 - Deep learning (book page 10)
# --------------------------------------------------------------------------
def paradigm_3_deep_learning():
    """The book's page-10 Keras import, then the same net in PyTorch."""
    print("  The book's first line, run as printed:")
    print("    >>> from tensorflow.keras.models import Sequential")
    try:
        from tensorflow.keras.models import Sequential  # noqa: F401
        print("    (imported - TensorFlow is present in this interpreter)")
    except ModuleNotFoundError as exc:
        print(f"    {type(exc).__name__}: {exc}")
        print("    TensorFlow is not installed here, so the block is reproduced")
        print("    in PyTorch with the same layer sizes and the same optimizer.")

    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.float32)
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y).unsqueeze(1)
    loss_fn = nn.BCELoss()                             # book: binary_crossentropy
    chance = math.log(2)

    def build(seed=SEED):
        """The book's p.10 architecture: 8-relu -> 4-relu -> 1-sigmoid."""
        torch.manual_seed(seed)
        return nn.Sequential(
            nn.Linear(2, 8), nn.ReLU(),      # Dense(8, relu, input_shape=(2,))
            nn.Linear(8, 4), nn.ReLU(),      # Dense(4, relu)
            nn.Linear(4, 1), nn.Sigmoid(),   # Dense(1, sigmoid)
        )

    def train(model, epochs, rows=None):
        """Adam at its default lr, full batch - the book's fit() call."""
        Xtr = Xt if rows is None else torch.from_numpy(X[rows])
        ytr = yt if rows is None else torch.from_numpy(y[rows]).unsqueeze(1)
        optimizer = torch.optim.Adam(model.parameters())  # book: 'adam'
        loss = None
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = loss_fn(model(Xtr), ytr)
            loss.backward()
            optimizer.step()
        return loss.item()

    def predict(model):
        with torch.no_grad():
            return (model(Xt) > 0.5).int().flatten().tolist()

    model = build()
    final_loss = train(model, 100)                     # book: epochs=100
    pred = predict(model)
    n_params = sum(p.numel() for p in model.parameters())

    print()
    print("  Predictions:", np.array(pred), "   <- the book prints [0 0 1 1]")
    print()
    print(f"  trainable parameters         : {n_params} = "
          f"(2*8+8) + (8*4+4) + (4*1+1) = 24 + 36 + 5")
    print(f"  rows trained on              : {len(X)}")
    print(f"  rows predicted on            : {len(X)}")
    print(f"  rows the model had never seen: 0")
    print(f"  final training loss          : {final_loss:.4f}   "
          f"(chance = ln 2 = {chance:.4f})")
    print()
    print("  Before the held-out question, a convergence one: at the book's 100")
    print("  epochs the loss is still at chance, and the printed labels are not")
    print("  the book's. epochs=100 with Adam's default lr=0.001 and 4 rows in one")
    print("  batch is 100 gradient steps, which is not enough to fit four points.")
    print()
    hits = 0
    losses = []
    for seed in range(20):
        m = build(seed)
        losses.append(train(m, 100))
        if predict(m) == [0, 0, 1, 1]:
            hits += 1
    print(f"  20 seeds at the book's 100 epochs: prints [0 0 1 1] on {hits}/20, "
          f"mean loss {np.mean(losses):.4f}")
    print()
    print(f"    {'epochs':>7} {'train loss':>11}  {'predictions':<14} matches book")
    print("    " + "-" * 50)
    for epochs in (100, 200, 500, 1000, 2000):
        m = build()
        loss = train(m, epochs)
        p = predict(m)
        print(f"    {epochs:>7} {loss:>11.4f}  {str(p):<14} "
              f"{str(p == [0, 0, 1, 1])}")
    print()
    print("  It takes about 500 epochs - 5x the book's number - before the block")
    print("  prints the output the book shows underneath it.")
    print()
    print("  Now the held-out question, asked only where the model has converged.")
    print("  predict(X) above ran on the same X that fit(X, y) trained on, so a")
    print("  matching [0 0 1 1] means the net can read back its own labels. Hold")
    print("  one row out instead - the smallest honest test this data allows:")
    print()
    print(f"    {'epochs':>7} {'mean train loss':>16} {'LOO accuracy':>14}  verdict")
    print("    " + "-" * 62)
    for epochs in (100, 500, 2000):
        correct, fold_losses = 0, []
        for i in range(len(X)):
            keep = [j for j in range(len(X)) if j != i]
            m = build()
            fold_losses.append(train(m, epochs, rows=keep))
            with torch.no_grad():
                p = int((m(torch.from_numpy(X[i : i + 1])) > 0.5).item())
            correct += p == int(y[i])
        mean_loss = float(np.mean(fold_losses))
        verdict = "underfit - says nothing" if mean_loss > 0.30 else "converged"
        print(f"    {epochs:>7} {mean_loss:>16.4f} "
              f"{f'{correct}/{len(X)} = {correct / len(X):.3f}':>14}  {verdict}")
    print()
    print("  The 0/4 at the book's 100 epochs is not a generalisation failure - the")
    print("  folds are sitting at a training loss of 0.64, so the net has not")
    print("  learned anything to generalise yet. Only the converged row counts, and")
    print("  it reports 3/4 on rows the model had not seen, against the 4/4 that")
    print("  predict(X) on the training set implies.")


if __name__ == "__main__":
    rule("PARADIGM 1 - SUPERVISED LEARNING (book p.5)")
    paradigm_1_supervised()

    rule("PARADIGM 2 - UNSUPERVISED LEARNING (book p.8)")
    paradigm_2_unsupervised()

    rule("PARADIGM 3 - DEEP LEARNING (book p.10)")
    paradigm_3_deep_learning()

    rule("VERDICT - three paradigms, three printed results")
    print("  Every block in the guide runs, and the algorithms it names are the")
    print("  right ones to learn. The gap is the last step of each pattern:")
    print()
    print("    p.5   evaluate  -> scored on 1 row, so 0.0 and 1.0 are the only")
    print("                       results it can ever print")
    print("    p.8   (none)    -> k=3 is a choice in the source, and nothing in")
    print("                       the output confirms or denies it")
    print("    p.10  predict   -> run on the same 4 rows it trained on, so the")
    print("                       printed [0 0 1 1] is memorised, not learned")
    print()
    print("  None of this is a bug and none of it raises. Three code blocks each")
    print("  print a clean result, and none of the three results is evidence that")
    print("  the model would work on a row it had not already been shown.")
    print()
    print("  The cheapest repair is one line per block: print len(y_test) beside")
    print("  the accuracy, print a silhouette beside the labels, and predict on")
    print("  rows that fit() never saw.")
