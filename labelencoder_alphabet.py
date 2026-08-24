"""Day 19 - LabelEncoder picks your positive class by the alphabet, and the book's fix does not fix it.

Source: "Low-Code AI: A Practical Project-Driven Introduction to Machine Learning"
(O'Reilly), ch. 6, PDF p. 247.

The book prints these four lines to turn string labels into the 1/0 that Keras needs:

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)
    le.inverse_transform([1])

then tells the reader: "In the output, you will see that Yes is being treated as the positive
class, or 1." That sentence is true, and it is an accident. LabelEncoder assigns integers by
SORTING the class strings, so "Yes" lands on 1 because Y comes after N - not because "Yes" is
the outcome anyone cares about.

sklearn's precision, recall and f1 default to pos_label=1, so they follow the alphabet too.
Rename the same two classes and the metric silently changes what it is measuring. Below, one
identical model on one identical dataset reports recall 0.55 or 0.95 depending only on what
the two strings are called.

The book half-sees this and offers an aside: "you can also ensure the order of the labels by
fitting the transformer on the set ['No','Yes']". THAT DOES NOT WORK. LabelEncoder sorts
whatever you fit it on, so the order you pass is discarded. It appears to work for No/Yes only
because 'No' already sorts before 'Yes'. Verified below on scikit-learn 1.8.0.

Standard library plus scikit-learn, which is the book's own dependency. Seeded, so the numbers
repeat.
"""

import random

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

EVENT_RATE = 0.08     # 8% of rows are the event of interest, e.g. fraud
DETECTION = 0.55      # the model catches 55% of real events
FALSE_ALARM = 0.05    # and wrongly flags 5% of non-events


def make_data(n, seed):
    """Truth and prediction as booleans. Deliberately label-free - the strings come later."""
    rng = random.Random(seed)
    truth, pred = [], []
    for _ in range(n):
        event = rng.random() < EVENT_RATE
        truth.append(event)
        pred.append(rng.random() < (DETECTION if event else FALSE_ALARM))
    return truth, pred


def as_strings(flags, names):
    """names = (word_for_non_event, word_for_event)."""
    return [names[1] if f else names[0] for f in flags]


def encode_and_score(truth, pred, names, fit_on=None, pos_label=None):
    """Run the book's four lines, then score with sklearn's defaults."""
    le = LabelEncoder()
    y_true_s, y_pred_s = as_strings(truth, names), as_strings(pred, names)
    le.fit(fit_on if fit_on is not None else y_true_s)
    y_true, y_pred = le.transform(y_true_s), le.transform(y_pred_s)

    scored = 1 if pos_label is None else int(le.transform([pos_label])[0])
    kw = dict(pos_label=scored, zero_division=0)
    return dict(
        classes=[str(c) for c in le.classes_],
        one_means=str(le.inverse_transform([1])[0]),
        scoring=str(le.inverse_transform([scored])[0]),
        precision=precision_score(y_true, y_pred, **kw),
        recall=recall_score(y_true, y_pred, **kw),
        f1=f1_score(y_true, y_pred, **kw),
    )


if __name__ == "__main__":
    N = 5000
    truth, pred = make_data(N, seed=19)
    events = sum(truth)
    print(f"One dataset, one model, {N} rows. {events} real events ({events / N:.1%}). "
          f"The model catches {DETECTION:.0%} of them.")
    print("Below, ONLY the two label strings change. The predictions never do.\n")

    hdr = f"  {'label pair':<22}{'le.classes_':<26}{'1 means':<11}{'precision':<11}{'recall':<9}f1"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    pairs = [
        (("No", "Yes"), "Yes"),
        (("N", "Y"), "Y"),
        (("negative", "positive"), "positive"),
        (("legit", "fraud"), "fraud"),
        (("normal", "abnormal"), "abnormal"),
        (("retained", "churned"), "churned"),
    ]
    for names, event in pairs:
        r = encode_and_score(truth, pred, names)
        warn = "" if r["one_means"] == event else "  <-- 1 is NOT the event"
        pair = f"{names[0]}/{names[1]}"
        cls = str(r["classes"])
        print(f"  {pair:<22}{cls:<26}{r['one_means']:<11}"
              f"{r['precision']:<11.3f}{r['recall']:<9.3f}{r['f1']:.3f}{warn}")

    yes = encode_and_score(truth, pred, ("No", "Yes"))
    fraud = encode_and_score(truth, pred, ("legit", "fraud"))
    print(f"\nSame model, same rows. Recall reads {yes['recall']:.3f} when the event word sorts")
    print(f"last, and {fraud['recall']:.3f} when it sorts first - a gap of "
          f"{fraud['recall'] - yes['recall']:+.3f} on a rename.")
    print(f"A fraud detector that actually finds {yes['recall']:.0%} of fraud would report "
          f"{fraud['recall']:.0%},")
    print("because it is quietly scoring how well it recognises legitimate rows.")

    print("\nThe book's suggested fix, tested:")
    for order in (["legit", "fraud"], ["fraud", "legit"]):
        r = encode_and_score(truth, pred, ("legit", "fraud"), fit_on=order)
        print(f"  le.fit({str(order)}) -> classes_ {str(r['classes']):<22} "
              f"1 means {r['one_means']!r}")
    print("  Both orders give the same answer. LabelEncoder SORTS whatever you fit it on,")
    print("  so passing an ordered list changes nothing. The book's aside works for 'No'/'Yes'")
    print("  only because 'No' already sorts first. It is a coincidence, not a fix.")

    print("\nWhat does work - name the positive class at the metric:")
    fixed = encode_and_score(truth, pred, ("legit", "fraud"), pos_label="fraud")
    print(f"  precision_score(..., pos_label=le.transform(['fraud'])[0])")
    print(f"  -> scoring {fixed['scoring']!r}: precision {fixed['precision']:.3f}, "
          f"recall {fixed['recall']:.3f}, f1 {fixed['f1']:.3f}")
    print("  That matches the No/Yes row, which is the point: the model was always the same.")

    print("\nFAILURE - inputs where the encoder does something worse than surprising:")
    three = LabelEncoder().fit(["Yes", "No", "Maybe"])
    print(f"  three classes  -> classes_ {[str(c) for c in three.classes_]}, "
          f"1 means {str(three.inverse_transform([1])[0])!r}")
    print("                    'Yes' is now 2, and a binary metric scores 'No' instead")
    case = LabelEncoder().fit(["yes", "Yes"])
    print(f"  'yes' vs 'Yes' -> classes_ {[str(c) for c in case.classes_]} - a casing typo")
    print("                    becomes a second class, and capital Y sorts before lowercase y")
    try:
        LabelEncoder().fit(["No", "Yes"]).transform(["Maybe"])
    except ValueError as exc:
        print(f"  unseen label   -> ValueError: {str(exc).split(':')[0]}")
