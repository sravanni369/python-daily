"""AUC-PR is three different numbers - Practical ML for Computer Vision (Lakshmanan, Gorner, Gillard,
O'Reilly 2021), ch.8 "Metrics for Classification", pp.287-292. p.291: AUC from "a grid of two hundred
equally spaced thresholds"; p.292: use AUC-PR when classes are skewed. The p.292 listing is run with its
printed arguments on real predictions and compared with scikit-learn's exact values.
Dataset: Palmer Penguins (CC0), p.5 of "Data Science Public Datasets" (H. Ramchandani), saved as penguins.csv
from raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv
Positive class: Chinstrap, 68/342 = 19.9%. Features are deliberately weak (body mass, flipper length) so the
model is imperfect; with all four features every estimator prints 1.000 and there is nothing to compare.
"""
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, pandas as pd, sklearn, tensorflow as tf, keras
from sklearn.linear_model import LogisticRegression; from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc, precision_score, recall_score, accuracy_score
SEEDS, GRIDS, ALL4 = range(10), (10, 50, 200, 1000, 10000), ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]

def book_auc(y, p, n=200, curve="ROC"):                     # p.292 listing, arguments as printed
    m = tf.keras.metrics.AUC(num_thresholds=n, curve=curve, summation_method="interpolation", thresholds=None, multi_label=False)
    m.update_state(y, p); return float(m.result()), np.array(m.thresholds)

def book_pr(y, p):                                          # p.289 listing: Precision(), Recall() at the default 0.5
    P, R = tf.keras.metrics.Precision(), tf.keras.metrics.Recall(); P.update_state(y, p); R.update_state(y, p)
    return float(P.result()), float(R.result())

def splits(feats):                                          # one split is not a result
    for s in SEEDS:
        tr, te, a, b = train_test_split(d[feats].values, y, test_size=.2, random_state=s, stratify=y)
        yield s, b, LogisticRegression(max_iter=5000).fit(tr, a).predict_proba(te)[:, 1]

if __name__ == "__main__":
    d = pd.read_csv("penguins.csv"); n0 = len(d); d = d.dropna(subset=ALL4); y = (d.species == "Chinstrap").astype(int).values
    print(f"tf {tf.__version__} keras {keras.__version__} sklearn {sklearn.__version__} numpy {np.__version__} pandas {pd.__version__}")
    print(f"penguins {n0} rows, {n0 - len(d)} dropped (no measurements) -> {len(d)} | Chinstrap {y.sum()} = {y.mean():.3f} | majority-class acc {1 - y.mean():.4f}")
    th = book_auc(y[:2], np.array([.1, .9]))[1]; inside = int(((th >= 0) & (th <= 1)).sum())
    print(f"[1] p.292 num_thresholds=200 -> {len(th)} thresholds, {inside} inside [0,1]: {th[0]:.1e}, {th[1]:.6f}, {th[2]:.6f} ... {th[-2]:.6f}, {th[-1]:.7f}")
    roc_gap = {n: [] for n in GRIDS}; pr = {"AP": [], "trap": [], "k200": [], "k10000": []}
    print("[2] weak model (body_mass_g, flipper_length_mm) | 69-row stratified test, 14 positives | 10 seeds")
    for s, b, p in splits(["body_mass_g", "flipper_length_mm"]):
        ex, ap = roc_auc_score(b, p), average_precision_score(b, p); P, R, _ = precision_recall_curve(b, p); tp = auc(R, P)
        for n in GRIDS: roc_gap[n].append(book_auc(b, p, n)[0] - ex)
        k2, k1 = book_auc(b, p, 200, "PR")[0], book_auc(b, p, 10000, "PR")[0]
        for k, v in zip(pr, (ap, tp, k2, k1)): pr[k].append(v)
        if s == 0: kp, kr = book_pr(b, p); print(f"    p.289 at 0.5, seed0: keras P/R {kp:.4f}/{kr:.4f} = sklearn {precision_score(b, p >= .5):.4f}/{recall_score(b, p >= .5):.4f} | acc {accuracy_score(b, p >= .5):.4f} vs majority {1 - b.mean():.4f}")
        print(f"    seed{s} ROC exact {ex:.4f} grid-exact " + " ".join(f"n{n}:{roc_gap[n][-1]:+.4f}" for n in GRIDS) + f" | PR: AP {ap:.4f} trapezoid {tp:.4f} keras200 {k2:.4f} keras10000 {k1:.4f}")
    print("[3] ROC  mean |grid - exact| " + " ".join(f"n{n}:{np.mean(np.abs(roc_gap[n])):.4f}" for n in GRIDS) + " -> shrinks with the grid; 200 is a grid effect and a small one")
    A = {k: np.array(v) for k, v in pr.items()}
    print(f"[3] PR   AP {A['AP'].mean():.4f}+/-{A['AP'].std():.4f} | trapezoid-AP {(A['trap'] - A['AP']).mean():+.4f} | keras200-AP {(A['k200'] - A['AP']).mean():+.4f} | keras10000-AP {(A['k10000'] - A['AP']).mean():+.4f}, "
          f"below AP on {int((A['k10000'] < A['AP']).sum())}/10 seeds -> does not converge to AP: a different estimator, not a grid error")
    print("[4] control, all four features, seeds 0-2: " + " | ".join(f"AP {average_precision_score(b, p):.4f} keras200 {book_auc(b, p, 200, 'PR')[0]:.4f}" for s, b, p in splits(ALL4) if s < 3) + " -> a perfect model hides the gap")
