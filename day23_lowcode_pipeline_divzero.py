"""Day 23 - Low-Code AI (Stripling & Abel, O'Reilly 2023), ch.7 pp.229-230.
Convert the book's pandas block into the Pipeline it pitches, then check the conversion.
B23-1 p.229 blocking: .div(tenure, fill_values=0.0) - no such pandas kwarg -> TypeError.
B23-2 p.229 silent:   fill_value does not guard 0/0; 11 tenure==0 rows stay NaN and move 7030/7043 bins.
B23-3 p.230 fatal:    pd.cut in FunctionTransformer never fits -> unseen buckets at predict time.
python 3.13.5, pandas 2.3.1, sklearn 1.8.0, numpy 2.2.6 | IBM Telco Churn, 7043 rows.
Data: raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
      saved locally as telco_churn.csv
"""
import warnings, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer; from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression; from sklearn.metrics import accuracy_score, recall_score
from sklearn.pipeline import Pipeline; from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer, MinMaxScaler, OneHotEncoder
DROP = ['gender', 'StreamingTV', 'StreamingMovies', 'PhoneService', 'customerID']; SEEDS = (0, 1, 42, 2023, 7043); NUM = ['SeniorCitizen', 'tenure', 'MonthlyCharges']
CAT = ['Partner', 'Dependents', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'Contract', 'PaperlessBilling', 'PaymentMethod']

def diff(df, guard='work'):                                  # MonthlyCharges - AvgMonthlyCharge, p.229
    d = df.replace({'TotalCharges': {' ': 0.0}}).astype({'TotalCharges': 'float64'})
    a = d.TotalCharges.div(d.tenure, fill_value=0.0) if guard == 'book' else \
        d.TotalCharges.div(d.tenure).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return d.MonthlyCharges - a

def build(cut, unk='ignore', feat=True):                     # the low-code conversion, p.230
    def fn(df):
        o = df.drop(columns=DROP).assign(DiffCharges=diff(df)) if feat else df.drop(columns=DROP)
        return o.assign(DiffBuckets=pd.cut(o.DiffCharges, 5).astype(str)).drop(columns='DiffCharges') if feat and cut else o
    st = [('ohe', OneHotEncoder(drop='if_binary', handle_unknown=unk), CAT + (['DiffBuckets'] if feat and cut else [])),
          ('sca', MinMaxScaler(), NUM)] + ([('bkt', KBinsDiscretizer(n_bins=5, encode='onehot-dense',
          strategy='uniform'), ['DiffCharges'])] if feat and not cut else [])
    return Pipeline([('p', FunctionTransformer(fn)), ('c', ColumnTransformer(st)), ('m', LogisticRegression(max_iter=1000))])

def score(**kw):                                             # 5 stratified splits - one split is not a result
    A, R, I = [], [], []
    for s in SEEDS:
        tr, te, a, b = train_test_split(X, y, test_size=.2, random_state=s, stratify=y)
        p = build(**kw).fit(tr, a); q = p.predict(te); I += [int(p.named_steps['m'].n_iter_[0])]
        A += [accuracy_score(b, q)]; R += [recall_score(b, q)]
    return np.array(A), np.array(R), I

X = pd.read_csv('telco_churn.csv'); y = (X.pop('Churn') == 'Yes').astype(int)
print(f'Telco Churn {X.shape[0]}x{X.shape[1]+1} | churn={y.mean():.4f} | tenure==0 rows={(X.tenure==0).sum()} | dup rows on modelled cols={X[NUM+CAT].duplicated().sum()}')
try: X.TotalCharges.replace(' ', 0.0).astype('float64').div(X.tenure, fill_values=0.0)
except TypeError as e: print(f'[1] B23-1 p.229 as printed -> TypeError: {e}')
b, w = diff(X, 'book'), diff(X, 'work'); mv = pd.cut(b,5).cat.codes.values != pd.cut(w,5).cat.codes.values; nn = b.notna().values
print(f'[2] B23-2 NaN after fill_value=0.0: {b.isna().sum()} rows | DiffCharges range [{b.min():.2f},{b.max():.2f}] vs [{w.min():.2f},{w.max():.2f}]')
print(f'    rows those {b.isna().sum()} NaNs move into a different BIN: {mv.sum()}/{len(X)}  (excluding the 11 themselves: {(mv & nn).sum()}/{nn.sum()})')
tr, te, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
et, ee = ([f'{i.right:.2f}' for i in pd.cut(diff(d), 5).cat.categories] for d in (tr, te))
print(f'[3] B23-3 train edges {et}\n          test  edges {ee}  -> identical? {et == ee}')
try: build(True, unk='error').fit(tr, ytr).predict(te)
except ValueError as e: print(f"    book's OneHotEncoder(drop='if_binary') at predict -> ValueError: Found unknown categories "
                              f"{sorted(set(pd.cut(diff(te),5).astype(str)) - set(pd.cut(diff(tr),5).astype(str)))} in column 11")
print(f"    handle_unknown='ignore' instead: {(~pd.cut(diff(te),5).astype(str).isin(set(pd.cut(diff(tr),5).astype(str)))).sum()}"
      f'/{len(te)} test rows get an all-zero bucket - feature gone at serve time')
warnings.filterwarnings('ignore', category=UserWarning); print(f'[4] majority-class baseline              acc={max(1-y.mean(), y.mean()):.4f}  recall=0.0000')  # 5 repeats below of the defect [3] raised
for lbl, kw in [('no DiffCharges feature at all', dict(cut=True, feat=False)), ('working guard + pd.cut (p.230)',
                dict(cut=True)), ('working guard + KBinsDiscretizer', dict(cut=False))]:
    a, r, it = score(**kw); print(f'    {lbl:<36} acc={a.mean():.4f}+/-{a.std():.4f} recall={r.mean():.4f}+/-{r.std():.4f} n_iter={it}')
print('[5] The engineered feature buys nothing: dropping DiffCharges beats pd.cut on 5/5 seeds, and only\n'
      '    3/5 vs KBinsDiscretizer (+0.0013, inside the seed spread). pd.cut leads on recall solely because\n'
      '    every test row lost the feature (see [3]) - a shifted threshold, not a model. All beat 0.7346.')
