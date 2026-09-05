"""Hands-On Machine Learning (Aurelien Geron, O'Reilly, 1st ed), Chapter 3 - MNIST, p.113.

The chapter opens by fetching MNIST and building a "5-detector", to teach that accuracy is
the wrong metric when one class is rare. Run the printed listing today and you never reach
the lesson: three things break first, and the one that matters breaks quietly.

B24-1 blocking  `from sklearn.datasets import fetch_mldata`. Removed in scikit-learn 0.22
      after mldata.org went offline (changelog); the run below only shows it absent in 1.8.0.
B24-2 blocking  The replacement, fetch_openml, returns a pandas DataFrame by default
      (as_frame default changed in 0.24), so the book's `some_digit = X[36000]` is read as a
      COLUMN lookup -> KeyError.
B24-3 silent    fetch_openml returns labels as STRINGS. The book's `y_train == 5` compares
      str to int and yields 0 positives out of 60,000 - no error, no warning. It only
      surfaces later inside fit(), as a message about the model's class count, three lines
      from the line that caused it.

Verified 2026-09-05 on python 3.13.5, scikit-learn 1.8.0, numpy 2.2.6, pandas 2.3.1.
Data: fetch_openml('mnist_784', version=1) - 70,000 x 784, cached by sklearn in
~/scikit_learn_data. The first run needs network; later runs read that cache (~4s).
"""
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier; from sklearn.metrics import accuracy_score, precision_score, recall_score

def report(name, y_true, pred):
    a = accuracy_score(y_true, pred)
    print(f'    {name:<36} acc={a:.4f}  prec={precision_score(y_true, pred, zero_division=0):.4f}  '
          f'rec={recall_score(y_true, pred, zero_division=0):.4f}')
    return a

print('[1] B24-1  the line the chapter opens with, p.113')
try:
    from sklearn.datasets import fetch_mldata          # noqa: F401 - book listing, p.113
    print('    imported (unexpected)')
except ImportError as e:
    print(f'    ImportError: {e}')

print('[2] B24-2  the replacement, called with the defaults the book would have used')
Xdf, ys = fetch_openml('mnist_784', version=1, return_X_y=True)
print(f'    fetch_openml(...) returns a {type(Xdf).__name__}, so X[36000] names a column')
try:
    Xdf[36000]
except KeyError as e:
    print(f'    KeyError: {e} -- the book meant row 36000; pandas looked for a column')

X, y = Xdf.to_numpy(dtype=np.uint8), ys.to_numpy()
print(f'[3] B24-3  MNIST {X.shape}, label dtype={y.dtype}, first five {list(y[:5])}')
y_tr, y_te = y[:60000], y[60000:]
print(f'    (y_train == 5)   -> {(y_tr == 5).sum()} positives of {len(y_tr)}, and no warning is raised')
print(f"    (y_train == '5') -> {(y_tr == '5').sum()} positives -- the labels were strings all along")
try:
    SGDClassifier(random_state=42).fit(X[:2000], (y_tr == 5)[:2000])
except ValueError as e:
    print(f'    fit() objects at last, but blames the model: "{e}"')

print('[4] fixed labels, and the lesson the chapter was trying to teach')
y_tr5, y_te5 = (y_tr.astype(np.uint8) == 5), (y_te.astype(np.uint8) == 5)
print(f'    after astype(uint8): {y_tr5.sum()} fives in train, {y_te5.sum()} in test '
      f'({y_te5.mean() * 100:.2f}% of it) -- so never-5 scores {1 - y_te5.mean():.4f}')
base = report('never-5 baseline', y_te5, np.zeros_like(y_te5))
a42 = report('SGD, unscaled pixels, seed 42', y_te5, SGDClassifier(random_state=42).fit(X[:60000], y_tr5).predict(X[60000:]))
accs = sorted(accuracy_score(y_te5, SGDClassifier(random_state=s).fit(X[:60000], y_tr5).predict(X[60000:])) for s in range(10))
print(f'    same model, seeds 0-9: acc {accs[0]:.4f}-{accs[-1]:.4f} -- the worst seed loses to the baseline')
Xs = X.astype(np.float32) / 255
report('SGD, X/255, seed 42', y_te5, SGDClassifier(random_state=42).fit(Xs[:60000], y_tr5).predict(Xs[60000:]))

print(f'[5] Accuracy puts the seed-42 model {(a42 - base) * 100:.1f} points above the do-nothing baseline, which\n'
      f'    sounds decisive until you rerun it: across ten seeds on unscaled pixels the worst draw drops to\n'
      f'    {accs[0]:.4f} and loses outright. Recall never confuses the two - the baseline scores 0.0000 every\n'
      f'    time. That is the chapter\'s point, reached only after fixing three defects to get there.')
