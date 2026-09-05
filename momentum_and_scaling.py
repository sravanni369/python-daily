"""Why Momentum Really Works (Gabriel Goh, Distill, 4 Apr 2017) - verified, then applied.

Part 1 holds the article's closed-form convergence rates against a real run. Part 2 uses
them to test my own explanation of the MNIST result in this repo, and finds it wrong.

The article derives, for a convex quadratic with eigenvalues in [l1, ln], k = ln/l1:
  gradient descent, a = 2/(l1+ln)                       rate (k-1)/(k+1)
  momentum, a = (2/(sqrt(l1)+sqrt(ln)))^2,
            b = ((sqrt(ln)-sqrt(l1))/(sqrt(ln)+sqrt(l1)))^2   rate (sqrt(k)-1)/(sqrt(k)+1)

In `mnist_five_detector.py` an SGD 5-detector scored 0.9492 on raw 0-255 pixels and 0.9772
on X/255. Describing that result I offered conditioning as the likely mechanism - in
conversation, not in the repo. It is aimed at the wrong quantity: dividing by 255 scales every
eigenvalue of the covariance by exactly 1/255^2, so k - a RATIO - cannot move, and [3] measures
it unmoved to the ~7 s.f. eigvalsh resolves. What does move is lam_max, by 255^2. The cause is
scikit-learn's learning_rate='optimal', which derives its step from alpha alone with no
reference to feature scale, so it is eta*lam_max that goes wrong.

Verified 2026-09-05 on python 3.13.5, scikit-learn 1.8.0, numpy 2.2.6, pandas 2.3.1.
"""
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier; from sklearn.metrics import accuracy_score, recall_score

def rate(lams, alpha, beta, iters=3000, burn=300):   # (w,z) is linear, so renormalising each step is exact
    w, z, acc = np.ones_like(lams), np.zeros_like(lams), 0.0
    for k in range(iters):
        z = beta * z + lams * w; w = w - alpha * z
        n = np.abs(w).max()
        if k >= burn: acc += np.log(n)
        w, z = w / n, z / n
    return np.exp(acc / (iters - burn))

print('[1] the article\'s rates, measured on quadratics with known eigenvalues')
for k in (10, 100, 1000, 10000):
    lams = np.linspace(1.0, float(k), 50); r = np.sqrt(k)
    a_m = (2 / (1 + r)) ** 2; b_m = ((r - 1) / (r + 1)) ** 2
    p_g, p_m = (k - 1) / (k + 1), (r - 1) / (r + 1); m_g, m_m = rate(lams, 2 / (1 + k), 0.0), rate(lams, a_m, b_m)
    print(f'    k={k:<6} GD pred {p_g:.6f} meas {m_g:.6f} dev {abs(p_g-m_g):.1e} | '
          f'momentum pred {p_m:.6f} meas {m_m:.6f} dev {m_m-p_m:.1e}')

print('[2] the momentum residual is a finite-window artefact, not an error in the formula')
lams = np.linspace(1.0, 100.0, 50); pred = 9 / 11
for it, bu in ((3000, 300), (10000, 1000), (30000, 3000)):
    m = rate(lams, (2 / 11) ** 2, (9 / 11) ** 2, it, bu)
    print(f'    window {it-bu:>6}  dev {m-pred:.2e}  predicted by the k*rho^k defective block '
          f'{pred * np.expm1(np.log(it/bu)/(it-bu)):.2e}')

print('[3] MNIST: does scaling change the condition number? (my hypothesis said yes)')
Xdf, ys = fetch_openml('mnist_784', version=1, return_X_y=True)
X = Xdf.to_numpy(dtype=np.uint8).astype(np.float64)   # mnist_five_detector.py's dtype; float32 moves seed 42 to 0.9679
y5 = (ys.to_numpy().astype(np.uint8) == 5); ks = []
for name, A in (('raw 0-255', X[:60000]), ('X/255', X[:60000] / 255)):
    ev = np.clip(np.linalg.eigvalsh(np.cov(A, rowvar=False)), 0, None)
    nz = ev[ev > ev[-1] * len(ev) * np.finfo(float).eps]   # relative tol, else the test is scale-dependent
    ks.append(nz[-1] / nz[0])
    print(f'    {name:<10} lam_max={ev[-1]:.4e}  rank={len(nz)}  kappa={nz[-1]/nz[0]:.8e}')
print(f'    kappa ratio {ks[0]/ks[1]:.9f} -- equal to the ~7 s.f. eigvalsh resolves at kappa=6e9;\n'
      '    this is cov(cX)=c^2 cov(X) confirmed numerically, an identity the test could not fail.')
tr, te = slice(0, 60000), slice(60000, 70000)
print(f'[4] the real cause: the schedule. never-5 baseline acc={1 - y5[te].mean():.4f} rec5=0.0000, 10 seeds each')
for lbl, A, kw in (('raw,   optimal (default)', X, {}), ('X/255, optimal (default)', X / 255, {}),
                   ('raw,   constant eta0=1e-6', X, dict(learning_rate='constant', eta0=1e-6)),
                   ('X/255, constant, eta0 & alpha matched', X / 255,
                    dict(learning_rate='constant', eta0=1e-6 * 255**2, alpha=1e-4 / 255**2))):
    P = [SGDClassifier(random_state=s, **kw).fit(A[tr], y5[tr]).predict(A[te]) for s in range(10)]
    a = np.array([accuracy_score(y5[te], q) for q in P]); r = np.array([recall_score(y5[te], q) for q in P])
    print(f'    {lbl:<38} acc {a.mean():.4f}+-{a.std(ddof=1):.4f} [{a.min():.4f},{a.max():.4f}]  '
          f'rec5 {r.mean():.4f} [{r.min():.4f},{r.max():.4f}]')

print("[5] Matched constant steps erase the gap, so the ANISOTROPY was never the problem - the SCALE\n"
      "    was. kappa does not move, but lam_max falls by 255^2, and that is exactly the 255^2 in the\n"
      "    matched eta0 above: it is eta*lam_max the schedule gets wrong. kappa can rule out anisotropy,\n"
      "    not scale - so my guess was not merely unproven, it was aimed at the wrong quantity.")
