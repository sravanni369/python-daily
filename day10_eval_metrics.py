"""Day 10 - Classification evaluation metrics from scratch.

Precision, recall, F1, accuracy, and a confusion matrix built with the
standard library only. These are the numbers behind every "our model
scores 98% F1" claim, so it's worth knowing exactly how they're counted.

Definitions (binary case, "positive" = the class you care about):
    TP  predicted positive, actually positive
    FP  predicted positive, actually negative   (false alarm)
    FN  predicted negative, actually positive   (miss)
    TN  predicted negative, actually negative

    precision = TP / (TP + FP)   "when I say positive, how often am I right?"
    recall    = TP / (TP + FN)   "of the real positives, how many did I catch?"
    F1        = harmonic mean of precision and recall

Also includes macro averaging for multi-class labels: compute per-class
metrics treating each class as "positive" in turn, then average them.

Contact: sravannicareerv@gmail.com | linkedin.com/in/sravani-p-212899272
"""

from collections import Counter, defaultdict


def confusion_counts(y_true, y_pred, positive):
    """Return (tp, fp, fn, tn) for one class treated as positive."""
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        if p == positive and t == positive:
            tp += 1
        elif p == positive and t != positive:
            fp += 1
        elif p != positive and t == positive:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def precision_recall_f1(y_true, y_pred, positive):
    """Precision, recall and F1 for one class. Zero-division returns 0.0,
    matching scikit-learn's zero_division=0 behaviour."""
    tp, fp, fn, _ = confusion_counts(y_true, y_pred, positive)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return precision, recall, f1


def accuracy(y_true, y_pred):
    """Fraction of exact matches."""
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true) if y_true else 0.0


def macro_f1(y_true, y_pred):
    """Unweighted mean of per-class F1 across all classes seen in y_true.

    Macro averaging treats rare classes as equal citizens, which is why
    it drops sharply when a model ignores a minority class - exactly the
    failure accuracy hides."""
    classes = sorted(set(y_true))
    scores = [precision_recall_f1(y_true, y_pred, c)[2] for c in classes]
    return sum(scores) / len(scores) if scores else 0.0


def confusion_matrix(y_true, y_pred):
    """Nested dict matrix[true_label][pred_label] = count."""
    matrix = defaultdict(Counter)
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
    return {t: dict(row) for t, row in matrix.items()}


def print_report(y_true, y_pred):
    """Per-class table plus accuracy and macro F1, sklearn-report style."""
    classes = sorted(set(y_true))
    print(f"{'class':<10} {'precision':>9} {'recall':>9} {'f1':>9} {'support':>9}")
    for c in classes:
        p, r, f = precision_recall_f1(y_true, y_pred, c)
        support = sum(1 for t in y_true if t == c)
        print(f"{str(c):<10} {p:>9.3f} {r:>9.3f} {f:>9.3f} {support:>9}")
    print(f"\naccuracy: {accuracy(y_true, y_pred):.3f}")
    print(f"macro F1: {macro_f1(y_true, y_pred):.3f}")


if __name__ == "__main__":
    # Spam-filter style demo with a deliberately imbalanced dataset:
    # 12 ham, 4 spam. The model catches 3 of 4 spam with 1 false alarm.
    y_true = ["ham"] * 12 + ["spam"] * 4
    y_pred = ["ham"] * 11 + ["spam"] + ["spam"] * 3 + ["ham"]

    print("Confusion matrix (true -> pred):")
    for t, row in confusion_matrix(y_true, y_pred).items():
        print(f"  {t}: {row}")
    print()
    print_report(y_true, y_pred)

    # Why accuracy misleads: predict "ham" for everything.
    lazy = ["ham"] * 16
    print("\nLazy model that never predicts spam:")
    print(f"accuracy: {accuracy(y_true, lazy):.3f}  <- looks fine")
    print(f"macro F1: {macro_f1(y_true, lazy):.3f}  <- tells the truth")

