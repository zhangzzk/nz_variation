"""Rendering helpers for the PPD internal-consistency results.

Loads the JSON summaries / npz arrays written by run_predict.py and produces the
p-value table, the conditional w(theta) predictive-band figure, and the PPD
discrepancy histograms. Kept out of the notebook so the logic is importable/testable.
"""
import json
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
def _out_dir(case):
    return Path(case["out_dir"])


def load_json(case, name):
    p = _out_dir(case) / name
    if not p.exists():
        return None
    with open(p) as fp:
        return json.load(fp)


def load_npz(case, name):
    p = _out_dir(case) / name
    return np.load(p) if p.exists() else None


# --------------------------------------------------------------------------- #
def _tension(r):
    """(T_obs, n, p_chi2, sigma_chi2) for a result dict, back-filling p_chi2 from T_obs
    for older GoF summaries that predate the analytic tension p-value."""
    from scipy import stats
    n = r["npred"]
    T = r.get("T_obs", r.get("T_obs_median", np.nan))
    p_chi2 = r.get("p_chi2")
    if p_chi2 is None and np.isfinite(T):
        p_chi2 = float(stats.chi2.sf(T, n))
    sig = r.get("sigma_chi2")
    if sig is None and p_chi2 is not None:
        sig = float(stats.norm.isf(p_chi2)) if 0 < p_chi2 < 1 else (np.inf if p_chi2 <= 0 else -np.inf)
    return T, n, p_chi2, sig


def pvalue_table(case, nsamples=2000, seed=1234, gof_seed=1234):
    """Combine GoF (per-probe), forward and reverse conditional PPD results into one table.

    Primary column is the tension p-value ``p_chi2`` (chi2_k survival of T_obs) and its
    sigma; the replica-tail ``p_value`` is kept for reference. For these noiseless
    injected-systematic DVs a clean/consistent case gives T_obs~0 -> p_chi2~1 (no tension);
    a detected systematic gives small p_chi2 / large sigma.
    """
    rows = []

    gof = load_json(case, f"gof_summary_N{nsamples}_s{gof_seed}.json")
    if gof:
        for variant in ("baseline", "model"):
            if variant in gof:
                for probe in ("gammat", "wtheta", "joint"):
                    r = gof[variant].get(probe)
                    if r:
                        T, n, p_chi2, sig = _tension(r)
                        rows.append({"test": "GoF", "conditioned_on": "-", "predicted": probe,
                                     "observed": variant, "n": n, "T_obs": T,
                                     "p_chi2": p_chi2, "sigma": sig, "replica_p": r["p_value"]})

    pred = load_json(case, f"predict_summary_N{nsamples}_s{seed}.json")
    if pred and "baseline" in pred:
        for variant in ("baseline", "model"):
            r = pred.get(variant)
            if r and "p_value" in r:
                T, n, p_chi2, sig = _tension(r)
                rows.append({"test": "cond fwd", "conditioned_on": pred["cond"],
                             "predicted": pred["pred"], "observed": variant, "n": n, "T_obs": T,
                             "p_chi2": p_chi2, "sigma": sig, "replica_p": r["p_value"]})

    prevr = load_json(case, f"predictrev_summary_N{nsamples}_s{seed}.json")
    if prevr:
        for name, obs_label in [("null", "baseline"), ("detect", "model")]:
            r = prevr.get(name)
            if r and "p_value" in r:
                T, n, p_chi2, sig = _tension(r)
                rows.append({"test": "cond rev", "conditioned_on": f"{prevr['cond']} ({name})",
                             "predicted": prevr["pred"], "observed": obs_label, "n": n, "T_obs": T,
                             "p_chi2": p_chi2, "sigma": sig, "replica_p": r["p_value"]})
    return rows


# --------------------------------------------------------------------------- #
def plot_wtheta_predictive(case, ax=None, nsamples=2000, seed=1234):
    """Conditional w(theta) predictive band per lens bin vs observed baseline & model."""
    import matplotlib.pyplot as plt
    arr = load_npz(case, f"predict_arrays_N{nsamples}_s{seed}.npz")
    if arr is None:
        raise FileNotFoundError("predict arrays not found; run `run_predict.py predict` first")

    angle = np.degrees(arr["angle_pred"]) * 60.0          # radians -> arcmin
    bin1 = arr["bin1_pred"]
    mu = arr["mu_baseline"]                                # (N, npred)
    mu_mean = mu.mean(0)
    Sig = arr["Sigma_c"]
    total_sd = np.sqrt(np.diag(Sig) + mu.var(0))          # predictive noise + parameter spread
    d_base = arr["obs_base_pred"]
    d_model = arr["obs_model_pred"]

    lens_bins = sorted(set(bin1.tolist()))
    if ax is None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
        axes = axes.ravel()
    else:
        axes = np.atleast_1d(ax)

    for k, lb in enumerate(lens_bins):
        a = axes[k]
        m = bin1 == lb
        order = np.argsort(angle[m])
        th = angle[m][order]
        a.fill_between(th, (mu_mean[m] - total_sd[m])[order], (mu_mean[m] + total_sd[m])[order],
                       color="#4477AA", alpha=0.25, label="γ_t→w(θ) predictive (68%)")
        a.plot(th, mu_mean[m][order], color="#4477AA", lw=1.5)
        a.plot(th, d_base[m][order], "o", ms=4, color="#222222", label="observed baseline")
        a.plot(th, d_model[m][order], "x", ms=6, color="#CC3311", label="observed contaminated")
        a.set_xscale("log")
        a.set_title(f"lens bin {lb}")
        if k == 0:
            a.legend(fontsize=9, frameon=False)
        a.set_xlabel("θ [arcmin]")
        a.set_ylabel("w(θ)")
    return axes


def plot_T_hist(case, which="conditional", variant="model", ax=None, nsamples=2000, seed=1234):
    """Histogram of the replica discrepancies T_rep with the scalar observed T_obs marked.

    The PPD p-value is the fraction of replicas with T_rep >= T_obs (area to the right).
    """
    import matplotlib.pyplot as plt
    arr = load_npz(case, f"predict_arrays_N{nsamples}_s{seed}.npz")
    if arr is None:
        raise FileNotFoundError("predict arrays not found")
    T_rep = np.asarray(arr[f"T_rep_{variant}"]).ravel()
    T_obs = float(np.asarray(arr[f"T_obs_{variant}"]))
    p = float(np.mean(T_rep >= T_obs))
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(T_rep, bins=40, density=True, color="#88CCEE", alpha=0.8,
            label=r"$T_{\rm rep}$ (replicas)")
    ax.axvline(T_obs, color="#CC3311", lw=2.2, label=fr"$T_{{\rm obs}}$  (p={p:.3f})")
    ax.set_xlabel("discrepancy $T$")
    ax.set_ylabel("density")
    ax.set_title(f"{which} PPD, observed = {variant}")
    ax.legend(fontsize=9, frameon=False)
    return ax
