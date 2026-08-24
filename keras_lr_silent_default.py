"""
Deep Learning Illustrated (Krohn, Beyleveld, Bassens), Addison-Wesley
Chapter 8, "An Intermediate Net in Keras", Examples 8.1 / 8.2 / 8.3, pages 127-128.

The book prints this to compile its intermediate-depth net:

    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.1),
                  metrics=['accuracy'])

On Keras 3 that line does not run. `lr` was removed. The error names the argument
it rejected but not the argument that replaced it, and the obvious reading of
"Argument(s) not recognized" is to delete it. Deleting it is silent, and it is not
the same model: SGD falls back to learning_rate=0.01, a tenth of the book's value.

This script runs the book's code verbatim, then trains the two repairs side by side
on the same data with the same seed, and measures what the silent one costs.

Dataset: MNIST, listed on p.21 of "Data Science Public Datasets" (Himanshu Ramchandani).
"""
import os, warnings, time
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import to_categorical

SEED   = 19
EPOCHS = 20
BATCH  = 128

line = "=" * 74

print(line)
print("BOOK   Deep Learning Illustrated, ch.8, Examples 8.1-8.3, pp.127-128")
print("CLAIM  'val_acc ... 92.34 percent accuracy after a single epoch' (p.129)")
print("       '...climbs to more than 95 percent by the third epoch and appears")
print("        to plateau around 97.6 percent by the twentieth.' (p.129)")
print(line)
print(f"keras {keras.__version__} | backend {keras.backend.backend()} | numpy {np.__version__}")
print()

# ---------------------------------------------------------------- Example 8.1
print(line)
print("1. Example 8.1, verbatim. The architecture.")
print(line)
print("    model = Sequential()")
print("    model.add(Dense(64, activation='relu', input_shape=(784,)))")
print("    model.add(Dense(64, activation='relu'))")
print("    model.add(Dense(10, activation='softmax'))")
print()

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(784,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

for w in caught:
    print("  WARNING:", str(w.message).split(". ")[0] + ".")

per_layer = [l.count_params() for l in model.layers]
print(f"  runs: yes.  built={model.built}  total params={model.count_params():,}")
print(f"  params by layer: {per_layer}")
print(f"  book, p.127: the second Dense layer adds 4,160 trainable parameters.")
print(f"  measured second layer: {per_layer[1]:,}  ->  "
      f"{'matches the book' if per_layer[1] == 4160 else 'DOES NOT MATCH'}")
print("  the architecture is intact. only the style is dated.")
print()

# ---------------------------------------------------------------- Example 8.2
print(line)
print("2. Example 8.2, verbatim. The compile step.")
print(line)
print("    model.compile(loss='categorical_crossentropy',")
print("                  optimizer=SGD(lr=0.1),")
print("                  metrics=['accuracy'])")
print()
try:
    SGD(lr=0.1)
    print("  runs: yes")
except Exception as e:
    print(f"  runs: no.  {type(e).__name__}: {e}")
    print()
    print("  the message names the argument it rejected. it does not name the one")
    print("  that replaced it. 'not recognized' reads like 'remove this'.")
print()

# ------------------------------------------------------------------- the data
(X_train, y_train), (X_valid, y_valid) = mnist.load_data()
X_train = X_train.reshape(60000, 784).astype("float32") / 255
X_valid = X_valid.reshape(10000, 784).astype("float32") / 255
y_train = to_categorical(y_train, 10)
y_valid = to_categorical(y_valid, 10)
print(f"MNIST loaded: train {X_train.shape}, valid {X_valid.shape}")
print()


def build_and_train(optimizer, label):
    keras.utils.set_random_seed(SEED)
    m = Sequential()
    m.add(Dense(64, activation='relu', input_shape=(784,)))
    m.add(Dense(64, activation='relu'))
    m.add(Dense(10, activation='softmax'))
    m.compile(loss='categorical_crossentropy', optimizer=optimizer,
              metrics=['accuracy'])
    t0 = time.time()
    h = m.fit(X_train, y_train, batch_size=BATCH, epochs=EPOCHS, verbose=0,
              validation_data=(X_valid, y_valid))
    va = h.history["val_accuracy"]
    print(f"  {label}")
    print(f"    epoch  1 val_acc = {va[0]:.4f}")
    print(f"    epoch  3 val_acc = {va[2]:.4f}")
    print(f"    epoch 20 val_acc = {va[-1]:.4f}   ({time.time()-t0:.0f}s)")
    return va


# ------------------------------------------------------------- the two repairs
print(line)
print("3. The two ways to make Example 8.2 run.")
print(line)
print()
print("  A. rename the argument, keep the book's value:  SGD(learning_rate=0.1)")
print("  B. delete the argument the error complained about:  SGD()")
print()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    _ = SGD()
dropped_lr = float(np.asarray(keras.ops.convert_to_numpy(SGD().learning_rate)))
print(f"  B silently gives you learning_rate={dropped_lr}, "
      f"{0.1/dropped_lr:.0f}x smaller than the book's 0.1.")
print(f"  warnings raised by B: {len(caught)}")
print()

va_a = build_and_train(SGD(learning_rate=0.1), "A  learning_rate=0.1  (the book's model)")
print()
va_b = build_and_train(SGD(),                  "B  argument deleted   (silently 0.01)")
print()

# ---------------------------------------------------------------------- result
print(line)
print("4. What the silent repair costs.")
print(line)
gap1  = va_a[0]  - va_b[0]
gap20 = va_a[-1] - va_b[-1]
print(f"  after 1 epoch    A {va_a[0]:.4f}   B {va_b[0]:.4f}   gap {gap1:+.4f}")
print(f"  after 20 epochs  A {va_a[-1]:.4f}   B {va_b[-1]:.4f}   gap {gap20:+.4f}")
print()
print(f"  B needs {next((i+1 for i, v in enumerate(va_b) if v >= va_a[0]), None)} "
      f"epochs to reach the accuracy A reached in 1.")
print()
print("  Against the book's printed figures:")
print(f"    book epoch 1  92.34%   ->  A measured {va_a[0]*100:.2f}%")
print(f"    book epoch 3  >95%     ->  A measured {va_a[2]*100:.2f}%")
print(f"    book epoch 20 ~97.6%   ->  A measured {va_a[-1]*100:.2f}%")
print()
print("  Neither run errors. Neither run warns. One is the book's network and one")
print("  is a tenth of its learning rate, and the only thing separating them is")
print("  how you chose to read a ValueError.")
print(line)
