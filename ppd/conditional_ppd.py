"""Conditional posterior-predictive-distribution (PPD) consistency test.

Split the data vector d = (d_cond, d_pred). Condition on d_cond; predict d_pred. For
posterior samples theta_i ~ p(theta | d_cond) with model vectors m_cond(theta_i),
m_pred(theta_i), the conditional Gaussian data-predictive of the predicted probe (given
the observed conditioning probe and theta_i) has

    mu_i    = m_pred(theta_i) + A (d_cond^obs - m_cond(theta_i)),  A = Sigma_pc Sigma_cc^{-1}
    Sigma_c = Sigma_pp - A Sigma_pc^T

so the posterior-predictive of d_pred is the mixture  sum_i N(mu_i, Sigma_c).

STATISTIC (see ``ppd_pvalue``). We do NOT use the Doux et al. (2021) *paired per-draw*
realized discrepancy T(d, theta_i) evaluated at each theta_i. Instead we moment-match the
mixture to a single Gaussian and use one global Mahalanobis distance in the full predictive
metric

    Sigma_ppd = Sigma_c + Cov_i(mu_i)          (measurement noise + parameter spread)
    m_bar     = mean_i(mu_i)
    T(d)      = (d - m_bar)^T Sigma_ppd^{-1} (d - m_bar)

The null ensemble is d_rep,i = mu_i + eps_i, eps_i ~ N(0, Sigma_c); the p-value is the
replica tail p = mean_i 1[T(d_rep,i) >= T(d_obs)]. This is a legitimate predictive check but
NOT identical to Doux's statistic (different power / interpretation); it is the form that
stays calibrated for these fixed-cosmology injected-systematic data vectors, where the
per-draw discrepancy on a *noiseless* injected DV degenerates (T_obs -> noncentrality, p->1).

The analytic tension value p_chi2 = chi2_k.sf(T_obs) ASSUMES the moment-matched Gaussian
describes the mixture well (unimodal, ~Gaussian). For the fixed-LCDM case the predicted
w(theta) ~ b_i^2 x fixed-shape is unimodal in a near-Gaussian b posterior, so this holds;
for multimodal / strongly non-Gaussian predictives it can fail and the empirical replica
tail (``p_value``) is the assumption-free number. Both are validated uniform under the null
on an exact linear-Gaussian model in ``test_ppd_calibration.py``.

Covariance convention: the data covariance is held fixed at a reference cosmology (only the
means m(theta_i) vary). It MUST be the effective pm-marginalized covariance the likelihood
used -- ``inv(2pt_inverse_covariance)``, NOT the raw ``2pt_covariance`` key (see theory_eval)
-- so the metric matches the probe-only chain.
"""
import numpy as np
from scipy import stats


def _psd_sqrt(S):
    """Symmetric PSD square root L with L @ L.T = S, via eigh. Robust where cholesky fails:
    the pm-marginalized covariance is near-singular along the point-mass modes (huge dynamic
    range), so inv(2pt_inverse_covariance) can carry tiny-negative round-off eigenvalues that
    make cholesky raise. Clipping to >=0 draws the (legitimately huge) marginalized modes."""
    w, V = np.linalg.eigh(0.5 * (S + S.T))
    return V * np.sqrt(np.clip(w, 0.0, None))


def _psd_inv(S, rcond=1e-12):
    """Symmetric (pseudo-)inverse via eigh, robust to ill-conditioning. Modes below
    ``rcond * max_eigenvalue`` are dropped -- for the pm-marginalized metric these are the
    point-mass directions with ~1e8x variance, which are marginalized and must not contribute
    to the discrepancy. For a well-conditioned Sigma_ppd (the forward conditional test) all
    modes are kept, so this equals the exact inverse to machine precision."""
    w, V = np.linalg.eigh(0.5 * (S + S.T))
    keep = w > w.max() * rcond
    inv_w = np.zeros_like(w)
    inv_w[keep] = 1.0 / w[keep]
    return (V * inv_w) @ V.T


def block_algebra(cov, idx_cond, idx_pred, inv_cov=None):
    """Return (A, Sigma_c) for conditioning idx_cond -> predicting idx_pred.

    If ``inv_cov`` (the joint PRECISION = inverse of the effective covariance) is supplied,
    use the numerically robust precision-block identities

        Sigma_c = inv(Lambda_pp),   A = -Sigma_c @ Lambda_pc

    which are mathematically identical to the covariance-block form below but invert only the
    predicted block Lambda_pp (69x69 and well-conditioned for the forward gamma_t->w test),
    NOT the pm-marginalized full covariance / conditioning block -- that is near-singular along
    the point-mass modes (condition number ~1e11, so inv(2pt_inverse_covariance) round-trips to
    ~1e3, not 0). Falls back to the covariance-block form when only ``cov`` is given. The two
    paths are checked equal to machine precision on an exact model in test_ppd_calibration.py.
    """
    if inv_cov is not None:
        L_pp = inv_cov[np.ix_(idx_pred, idx_pred)]
        L_pc = inv_cov[np.ix_(idx_pred, idx_cond)]
        Sigma_c = np.linalg.inv(L_pp)
        A = -Sigma_c @ L_pc
    else:
        S_cc = cov[np.ix_(idx_cond, idx_cond)]
        S_pp = cov[np.ix_(idx_pred, idx_pred)]
        S_pc = cov[np.ix_(idx_pred, idx_cond)]
        A = S_pc @ np.linalg.inv(S_cc)
        Sigma_c = S_pp - A @ S_pc.T
    Sigma_c = 0.5 * (Sigma_c + Sigma_c.T)          # symmetrize against round-off
    return A, Sigma_c


def conditional_means(m_cond, m_pred, d_cond_obs, A):
    """Conditional predictive means for every sample.

    m_cond : (N, ncond) model conditioning vectors per sample
    m_pred : (N, npred) model predicted vectors per sample
    d_cond_obs : (ncond,) observed conditioning vector
    A : (npred, ncond) from block_algebra
    returns mu : (N, npred)
    """
    m_cond = np.atleast_2d(m_cond)
    m_pred = np.atleast_2d(m_pred)
    resid = d_cond_obs[None, :] - m_cond               # (N, ncond)
    return m_pred + resid @ A.T                        # (N, npred)


def ppd_pvalue(mu, d_pred_obs, Sigma_noise, rng, n_rep=1):
    """Calibrated posterior-predictive p-value (moment-matched check; see module docstring).

    The posterior-predictive distribution of the predicted vector has replicas
    ``d_rep,i = mu_i + eps_i``, ``eps_i ~ N(0, Sigma_noise)``, so its spread is the
    parameter spread ``Cov(mu)`` plus the data noise ``Sigma_noise``. The single
    observed vector is compared to that ensemble via the Mahalanobis distance in the
    *full* predictive metric ``Sigma_ppd = Sigma_noise + Cov(mu)``; because the observed
    and the replicas are then exchangeable under the null, the p-value is uniform
    (validated by a synthetic Monte-Carlo calibration test).

    mu : (N, k) predictive means per posterior sample
         (conditional mean for the split test; theory m(theta_i) for goodness-of-fit).
    d_pred_obs : (k,) observed predicted vector.
    Sigma_noise : (k, k) data-noise covariance of the predicted probe
         (conditional cov for the split; marginal block cov for GoF).
    """
    mu = np.atleast_2d(mu)
    N, k = mu.shape
    L = _psd_sqrt(Sigma_noise)          # robust to the near-singular pm-marginalized metric

    # replica ensemble and full predictive covariance (parameter spread + data noise)
    reps = []
    for _ in range(n_rep):
        reps.append(mu + rng.standard_normal((N, k)) @ L.T)
    d_rep = np.concatenate(reps, axis=0)
    m_bar = mu.mean(axis=0)
    cov_mu = np.cov(mu, rowvar=False) if N > 1 else np.zeros((k, k))
    Sigma_ppd = Sigma_noise + np.atleast_2d(cov_mu)
    Sigma_ppd = 0.5 * (Sigma_ppd + Sigma_ppd.T)
    invP = _psd_inv(Sigma_ppd)          # eigh-based; drops marginalized point-mass null modes

    r_obs = d_pred_obs - m_bar
    T_obs = float(r_obs @ invP @ r_obs)
    r_rep = d_rep - m_bar[None, :]
    T_rep = np.einsum("ni,ij,nj->n", r_rep, invP, r_rep)
    p = float(np.mean(T_rep >= T_obs))

    # Analytic "tension" value: T_obs is the Mahalanobis distance of the observed vector from
    # the predictive mean in the full metric Sigma_ppd = Sigma_noise + Cov(mu). IF the
    # moment-matched Gaussian describes the mixture well (unimodal, ~Gaussian), a noisy null
    # gives d_obs - m_bar ~ N(0, Sigma_ppd) so T_obs ~ chi2_k and this survival p-value tracks
    # the replica tail -- but it resolves below the 1/N replica floor, so a precise sigma can
    # be quoted for strong detections. It is an APPROXIMATION resting on that Gaussianity; for
    # multimodal / non-Gaussian predictives prefer the empirical p_value (replica tail) above.
    p_chi2 = float(stats.chi2.sf(T_obs, k))
    if 0.0 < p_chi2 < 1.0:
        sigma_chi2 = float(stats.norm.isf(p_chi2))
    else:
        sigma_chi2 = np.inf if p_chi2 <= 0.0 else float(stats.norm.isf(1.0 - 1e-16))

    return {
        "p_value": p,
        "sigma_1sided": float(stats.norm.isf(p)) if 0.0 < p < 1.0 else (np.inf if p == 0.0 else -np.inf),
        "p_chi2": p_chi2,                     # analytic tension p-value (chi2_k survival of T_obs)
        "sigma_chi2": sigma_chi2,             # tension in sigma (primary for noiseless DVs)
        "npred": int(k),
        "n_samples": int(N),
        "T_obs": T_obs,                       # scalar: observed data vs the predictive ensemble
        "T_rep": T_rep,                       # (N*n_rep,) replica discrepancies
        "T_rep_median": float(np.median(T_rep)),
        "T_obs_median": T_obs,                # kept for summary compatibility
        "pred_spread_over_noise": float(np.median(np.sqrt(np.diag(cov_mu) / np.diag(Sigma_noise)))),
    }


def run_conditional_ppd(cov_ref, idx_cond, idx_pred, m_cond, m_pred,
                        d_cond_obs, d_pred_obs, seed=1234, n_rep=1, inv_cov=None):
    """End-to-end conditional PPD for one (conditioning, observed-predicted) combination.

    Pass ``inv_cov`` (the effective precision from theory_eval) to use the robust
    precision-space block algebra; ``cov_ref`` is then only needed as a fallback.
    """
    A, Sigma_c = block_algebra(cov_ref, idx_cond, idx_pred, inv_cov=inv_cov)
    mu = conditional_means(m_cond, m_pred, d_cond_obs, A)
    rng = np.random.default_rng(seed)
    res = ppd_pvalue(mu, d_pred_obs, Sigma_c, rng, n_rep=n_rep)
    res["mu_mean"] = mu.mean(axis=0)
    res["mu_std"] = mu.std(axis=0)
    res["Sigma_c_diag"] = np.diag(Sigma_c)
    return res, A, Sigma_c, mu


def gof_ppd(theory_samples, cov_ref, observed, probe_idx, seed=4321, n_rep=1):
    """Goodness-of-fit PPD from the joint posterior (per-probe + joint).

    For each probe the marginal covariance block is used; the conditional-mean
    correction is absent (mu_i = m(theta_i)). Because the same data both fits the
    posterior and is tested, this check is mildly conservative (double use of data),
    but it retains high power against a probe-localized systematic. Complements the
    unbiased conditional split test.

    theory_samples : (N, ndata) model vectors from the joint posterior
    cov_ref : (ndata, ndata) effective covariance (fixed reference)
    observed : (ndata,) observed data vector
    probe_idx : dict probe -> element indices
    """
    theory_samples = np.atleast_2d(theory_samples)
    rng = np.random.default_rng(seed)
    out = {}
    combos = list(probe_idx.items()) + [("joint", np.arange(observed.size))]
    for name, idx in combos:
        C = cov_ref[np.ix_(idx, idx)]
        mu = theory_samples[:, idx]
        out[name] = ppd_pvalue(mu, observed[idx], C, rng, n_rep=n_rep)
    return out
