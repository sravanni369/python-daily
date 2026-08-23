"""Run the book's data-cleaning block exactly as printed, on current pandas.

Source: "Data Engineering Made Simple: Your Friendly Guide to SQL, Python, and
PySpark", ch. 7 "Data Manipulation with Python", PDF p. 41 ("Cleaning the Patient
Data" / "Cleaning the Flight Data").

The book prints two cleaning lines per table. This runs them as written and checks
the result instead of trusting it.
"""

import warnings

import pandas as pd

patients = pd.DataFrame({
    "PatientID":   [1, 2, 3, 4, 4],
    "DateOfBirth": ["1980-04-02", "1975-11-30", "1992-01-17", "1968-07-08", "1968-07-08"],
    "Diagnosis":   ["Diabetes", None, "Asthma", None, None],
})

print(f"pandas {pd.__version__}")
print(f"rows: {len(patients)}   missing Diagnosis: {patients['Diagnosis'].isnull().sum()}")

print("\n--- the book's line, run verbatim ---")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    patients["Diagnosis"].fillna("Unknown", inplace=True)      # p. 41, as printed
for w in caught:
    print(f"  {w.category.__name__}: {str(w.message).splitlines()[0]}")
if not caught:
    print("  (no warning raised)")
print(f"  missing Diagnosis after the line: {patients['Diagnosis'].isnull().sum()}")

print("\n--- the assignment form ---")
patients["Diagnosis"] = patients["Diagnosis"].fillna("Unknown")
print(f"  missing Diagnosis after the line: {patients['Diagnosis'].isnull().sum()}")

print("\n--- the book's dedupe line, on the same table ---")
before = len(patients)
patients.drop_duplicates(inplace=True)
print(f"  rows {before} -> {len(patients)}")
print(patients.to_string(index=False))

print("\n--- the same book line under Copy-on-Write (the pandas 3.0 default) ---")
pd.options.mode.copy_on_write = True
cow = pd.DataFrame({"Diagnosis": ["Diabetes", None, "Asthma", None, None]})
print(f"  missing before: {cow['Diagnosis'].isnull().sum()}")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    cow["Diagnosis"].fillna("Unknown", inplace=True)          # p. 41, unchanged
for w in caught:
    print(f"  {w.category.__name__}: {str(w.message).splitlines()[0]}")
if not caught:
    print("  (no warning raised)")
print(f"  missing after:  {cow['Diagnosis'].isnull().sum()}")
