#!/usr/bin/env python3
"""Validate the conditional-PPD pipeline on an exact linear-Gaussian model, where the
posterior-predictive p-value is provably uniform under the null.

Model: theta ~ N(0, I).  d = M theta + n,  n ~ N(0, C)  (C has cross-covariance
between the conditioning and predicted blocks).  Condition on d_cond, predict d_pred.
Repeat over many (theta*, data) draws and KS-test the p-values for uniformity.
Also check that an injected systematic in d_pred is detected (small p).

SCOPE / CAVEAT: this exercises the *linear-Gaussian* regime, where the posterior predictive
is exactly a Gaussian mixture with Gaussian components and the moment-matched statistic (and
thus both p_value and p_chi2) is uniform by construction. It does NOT test the robustness of
the analytic p_chi2 to a *non-Gaussian / multimodal* predictive -- for such cases only the
empirical replica-tail p_value is trustworthy (see conditional_ppd module docstring). For the
fixed-LCDM w(theta) application the predictive is unimodal (b_i^2 in a near-Gaussian b
posterior), which is why the Gaussian approximation is adequate there; other cases must be
checked before quoting p_chi2. Run: `python ppd/test_ppd_calibration.py` (numpy/scipy only).
"""
import os
import sys
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conditional_ppd as cppd

rng = np.random.default_rng(1)
nparam, ncond, npred = 5, 40, 30
ndata = ncond + npred
idx_c = np.arange(ncond)
idx_p = np.arange(ncond, ndata)

M = rng.standard_normal((ndata, nparam))
Mc, Mp = M[idx_c], M[idx_p]
Q = rng.standard_normal((ndata, ndata))
C = Q @ Q.T + ndata * np.eye(ndata)                 # PD data covariance with cross-cov
Ccc = C[np.ix_(idx_c, idx_c)]
inv_Ccc = np.linalg.inv(Ccc)

# --- precision-space vs covariance-space block algebra must agree to machine precision.
# This certifies the numerically robust path used for the real (ill-conditioned pm-marginalized)
# metric: Sigma_c = inv(Lambda_pp), A = -Sigma_c @ Lambda_pc  ==  covariance-block form.
_invC = np.linalg.inv(C)
_A_cov, _Sc_cov = cppd.block_algebra(C, idx_c, idx_p)
_A_prec, _Sc_prec = cppd.block_algebra(C, idx_c, idx_p, inv_cov=_invC)
_ba_ok = np.allclose(_A_cov, _A_prec, atol=1e-8) and np.allclose(_Sc_cov, _Sc_prec, atol=1e-8)
print(f"block algebra: precision-space vs covariance-space agree = {_ba_ok}  "
      f"(max|dA|={np.max(np.abs(_A_cov - _A_prec)):.1e}, "
      f"max|dSc|={np.max(np.abs(_Sc_cov - _Sc_prec)):.1e})\n")

# posterior p(theta | d_cond) for the linear-Gaussian model (prior N(0,I))
Sig_post = np.linalg.inv(np.eye(nparam) + Mc.T @ inv_Ccc @ Mc)
Lpost = np.linalg.cholesky(Sig_post)
Lc = np.linalg.cholesky(Ccc)
Lfull = np.linalg.cholesky(C)


def one_trial(seed, inject=None):
    r = np.random.default_rng(seed)
    theta_star = r.standard_normal(nparam)
    n_full = Lfull @ r.standard_normal(ndata)
    d = M @ theta_star + n_full
    d_c, d_p = d[idx_c], d[idx_p]
    if inject is not None:
        d_p = d_p + inject
    theta_hat = Sig_post @ (Mc.T @ inv_Ccc @ d_c)
    N = 1500
    thetas = theta_hat[None, :] + (r.standard_normal((N, nparam)) @ Lpost.T)
    m_cond = thetas @ Mc.T
    m_pred = thetas @ Mp.T
    res, *_ = cppd.run_conditional_ppd(C, idx_c, idx_p, m_cond, m_pred, d_c, d_p,
                                       seed=int(r.integers(1 << 30)))
    return res["p_value"], res["p_chi2"]


# ---- null calibration ---- #
out = np.array([one_trial(1000 + i) for i in range(500)])
pvals, pchi2 = out[:, 0], out[:, 1]
print("null calibration (linear-Gaussian, exact) -- NOISY data, both p-values should be uniform:")
for label, pv in [("replica tail", pvals), ("chi2 survival", pchi2)]:
    ks = stats.kstest(pv, "uniform")
    print(f"  {label:14s}: mean={pv.mean():.3f} (want ~0.5)  std={pv.std():.3f}  "
          f"KS_p={ks.pvalue:.3f} (want>0.05)  frac(<0.05)={ (pv<0.05).mean():.3f} (want ~0.05)")
ks_rep = stats.kstest(pvals, "uniform")
ks_chi = stats.kstest(pchi2, "uniform")

# ---- detection: inject a coherent systematic into d_pred ---- #
Cpp = C[np.ix_(idx_p, idx_p)]
sig = np.sqrt(np.diag(Cpp))
inj = 1.5 * sig                                      # ~1.5 sigma per-point coherent bump
det = np.array([one_trial(5000 + i, inject=inj) for i in range(60)])
p_det, pchi2_det = det[:, 0], det[:, 1]
print("\ndetection (coherent 1.5-sigma systematic in d_pred), both should be small:")
print(f"  replica tail : median p = {np.median(p_det):.4f}")
print(f"  chi2 survival: median p = {np.median(pchi2_det):.4g}")

ok = (_ba_ok and
      ks_rep.pvalue > 0.02 and 0.45 < pvals.mean() < 0.55 and
      ks_chi.pvalue > 0.02 and 0.45 < pchi2.mean() < 0.55 and
      abs((pvals < 0.05).mean() - 0.05) < 0.03 and abs((pchi2 < 0.05).mean() - 0.05) < 0.03 and
      np.median(p_det) < 0.1 and np.median(pchi2_det) < 0.1)
print("\nCALIBRATION", "PASS" if ok else "FAIL")
