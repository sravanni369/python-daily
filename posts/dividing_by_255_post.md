# "Dividing by 255 does not fix conditioning" — LinkedIn post

Attach: `dividing_by_255.png`
Numbers from `momentum_output.txt` (github.com/sravanni369/python-daily)
python 3.13.5 · scikit-learn 1.8.0 · numpy 2.2.6 · pandas 2.3.1 · 2026-09-05

---

## The post (paste-ready)

I said dividing MNIST by 255 helps because it improves conditioning. Then I measured it.

Condition number before: 6.34044150e+09. After: 6.34044144e+09. Of course it didn't move. Scaling every pixel by the same constant divides every eigenvalue by the square of that constant, and a ratio can't change.

Scaling does help. Accuracy across 10 seeds goes from 0.9550 ±0.0332 to 0.9757 ±0.0029, eleven times tighter. Just not for the reason I gave. λmax dropped by 255², and SGD's default schedule picks its learning rate without ever looking at your feature scale. It's the step size, not the conditioning.

The repo has the run log and the four-line check. It also has the first version, where I used an absolute eigenvalue cutoff and "proved" a 15× improvement that was never there.

github.com/sravanni369/python-daily

#MachineLearning #Python #ScikitLearn #DataScience #NumPy

---

## Verification of the 255² claim

Checked before posting. The script measures eigenvalues of the **covariance**:

```python
ev = np.clip(np.linalg.eigvalsh(np.cov(A, rowvar=False)), 0, None)
```

Covariance is quadratic in the data, so `cov(cX) = c² cov(X)` and every eigenvalue scales by
`c²`. Measured directly:

```
lam_max(raw)          = 3.3272e+05
lam_max(X/255)        = 5.1169e+00
ratio                 = 65025.00000000007      (255² = 65025)
```

Exponent 2 is correct as written. If a future run measures eigenvalues of something linear in
the data — singular values of `X` itself rather than eigenvalues of `XᵀX` or `cov(X)` — the
exponent becomes 1 and the sentence must change.

One wording note carried into the draft above: an earlier version read "divides every
eigenvalue by that constant". It is **the square of** that constant. The ratio argument is
unaffected, since `c²` cancels either way, but the statement itself would have been wrong.

## Notes for posting

- The hook is the admission, not the result. Someone correcting themselves is rarer on
  LinkedIn than another tip.
- The infographic carries the 4-line code block, so people can check their own matrices
  without leaving the feed. That is the part most likely to get saved and re-shared.
- Do not soften "I said" to "many people say". The specific admission is what makes it land,
  and it is what actually happened.
- If it needs to be shorter, cut the fourth paragraph and keep the link.
