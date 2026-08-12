"""Day 11 - One-Hot Encoding From Scratch.

From Low-Code AI (Stripling & Abel, O'Reilly). The book's warning: storing car
colour as red=0, blue=1, silver=2 tells the model silver is twice blue, a
magnitude that does not exist. One column per category removes the fake order.
"""


def fit_categories(rows, column):
    """Sorted unique values, so column order is stable between train and serve."""
    return sorted({row[column] for row in rows})


def one_hot(rows, column, categories=None):
    """Expand one categorical column into a 0/1 column per known category."""
    categories = categories or fit_categories(rows, column)
    names = [f"{column}={value}" for value in categories]
    encoded = []
    for row in rows:
        wide = {key: value for key, value in row.items() if key != column}
        # A category unseen at training time encodes as all zeros, not a crash.
        wide.update({n: int(row[column] == c) for n, c in zip(names, categories)})
        encoded.append(wide)
    return encoded, names


if __name__ == "__main__":
    train = [{"colour": "red", "price": 21000}, {"colour": "blue", "price": 23500},
             {"colour": "silver", "price": 22750}]
    known = fit_categories(train, "colour")
    encoded, names = one_hot(train, "colour", known)
    print("ordinal trap: red=0 blue=1 silver=2 implies silver is twice blue")
    print("columns:", names)
    for row in encoded:
        print(row)
    serve, _ = one_hot([{"colour": "green", "price": 19000}], "colour", known)
    print("unseen colour ->", serve[0])
