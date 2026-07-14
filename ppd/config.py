"""Registry of PPD cases.

Each case names the ingredients the conditional PPD needs:

  - ``joint_ini``     : the production 2x2pt CosmoSIS ini (data_sets = gammat wtheta).
                        Used purely as the *theory evaluator* (predicts both probes and
                        exposes the pm-marginalized covariance). Theory is independent of
                        which DV FITS is fit, so one ini serves all observed variants.
  - ``baseline_ini``  : ini whose 2PT_FILE is the clean/baseline DV -> observed (uncontaminated).
  - ``model_ini``     : ini whose 2PT_FILE is the contaminated/model DV -> observed (contaminated).
  - ``cond_chain``    : the probe-only Nautilus chain giving p(theta | d_cond).
  - ``cond`` / ``pred``: which probe is conditioned on and which is predicted.

The data vector is ordered [gammat(380), wtheta(69)] after the 6/8 Mpc/h scale cuts
(verified from the datablock); ``PROBE_SLICES`` is filled at runtime from the evaluator.

Start case: fixed LSST-like LambdaCDM, condition on gamma_t, predict w(theta).
Extend by adding entries (KiDS/DES, relaxed, w0waCDM) that mirror this structure.
"""
from pathlib import Path

PROJECT_ROOT = Path("/project/ls-gruen/users/zekang.zhang/proj2_chains")

# fixed-h0/n_s LSST run that produced the joint chains + the DV FITS the chains fit
_LSST_FIXED = PROJECT_ROOT / "rerun_lsst_fixed_h0_ns_b23_may10"
_LSST_FIXED_STEM = "mcmc_gtwtheta_{variant}_lsst_tile3_b23_clean_fixed_h0_ns_cut68_20260510"
# the new gamma_t-only conditioning chain (forward direction: gamma_t -> w(theta))
_LSST_GTONLY = PROJECT_ROOT / "rerun_lsst_fixed_h0_ns_b23_gtonly_ppd"
# the new w(theta)-only conditioning chains (reverse direction: w(theta) -> gamma_t).
# TWO chains: one fit to the clean w(theta) (null) and one to the contaminated w(theta)
# (detection), because in the reverse direction the *conditioning* probe is the one that
# differs between baseline and model, so a single chain cannot serve both.
_LSST_WONLY = PROJECT_ROOT / "rerun_lsst_fixed_h0_ns_b23_wonly_ppd"
_WONLY_STEM = "mcmc_w_only_lsst_tile3_b23_fixed_h0_ns_cut8_{variant}_ppd"

CASES = {
    "lsst_fixed_lcdm_gt_to_w": {
        "label": "LSST-like, fixed h0/n_s, LambdaCDM: gamma_t -> w(theta)",
        "joint_ini":    _LSST_FIXED / "configs" / (_LSST_FIXED_STEM.format(variant="baseline") + ".ini"),
        "baseline_ini": _LSST_FIXED / "configs" / (_LSST_FIXED_STEM.format(variant="baseline") + ".ini"),
        "model_ini":    _LSST_FIXED / "configs" / (_LSST_FIXED_STEM.format(variant="model") + ".ini"),
        "cond_chain":   _LSST_GTONLY / "output" / "mcmc_gt_only_lsst_tile3_b23_fixed_h0_ns_cut6_ppd.txt",
        # joint 2x2pt chains (for the goodness-of-fit PPD complement + the chi2 gate)
        "joint_chain_baseline": _LSST_FIXED / "output" / (_LSST_FIXED_STEM.format(variant="baseline") + ".txt"),
        "joint_chain_model":    _LSST_FIXED / "output" / (_LSST_FIXED_STEM.format(variant="model") + ".txt"),
        "cond": "gammat",
        "pred": "wtheta",
        "out_dir": _LSST_GTONLY / "ppd_cache",
    },
    # reverse direction: condition on w(theta), predict gamma_t. For this contamination
    # (injected into w(theta) only) this is the higher-power diagnostic: the chain fit to
    # the contaminated w(theta) has a biased b_i/sigma8 posterior, and the *clean* observed
    # gamma_t then rejects the biased gamma_t prediction. The predicted probe (gamma_t) is
    # identical between baseline & model, so the observed gamma_t is the clean vector in both
    # tests; the difference is which conditioning chain / conditioning data is used.
    "lsst_fixed_lcdm_w_to_gt": {
        "label": "LSST-like, fixed h0/n_s, LambdaCDM: w(theta) -> gamma_t (reverse)",
        "joint_ini":    _LSST_FIXED / "configs" / (_LSST_FIXED_STEM.format(variant="baseline") + ".ini"),
        "baseline_ini": _LSST_FIXED / "configs" / (_LSST_FIXED_STEM.format(variant="baseline") + ".ini"),
        "model_ini":    _LSST_FIXED / "configs" / (_LSST_FIXED_STEM.format(variant="model") + ".ini"),
        # null test: chain fit to the clean w(theta); detection: chain fit to contaminated w(theta)
        "cond_chain_null":   _LSST_WONLY / "output" / (_WONLY_STEM.format(variant="baseline") + ".txt"),
        "cond_chain_detect": _LSST_WONLY / "output" / (_WONLY_STEM.format(variant="model") + ".txt"),
        "joint_chain_baseline": _LSST_FIXED / "output" / (_LSST_FIXED_STEM.format(variant="baseline") + ".txt"),
        "joint_chain_model":    _LSST_FIXED / "output" / (_LSST_FIXED_STEM.format(variant="model") + ".txt"),
        "cond": "wtheta",
        "pred": "gammat",
        "out_dir": _LSST_WONLY / "ppd_cache",
    },
}


def get_case(name):
    if name not in CASES:
        raise KeyError(f"unknown PPD case {name!r}; known: {list(CASES)}")
    return CASES[name]
