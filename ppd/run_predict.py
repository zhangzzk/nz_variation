#!/usr/bin/env python3
"""Driver for the clustering-vs-GGL PPD internal-consistency tests.

Subcommands
-----------
  gate     : evaluate theory at the joint-chain MAP and check the recomputed joint
             chi2 matches the chain-stored DATA_VECTOR--2PT_CHI2 (fidelity gate).
  gof      : goodness-of-fit PPD from the joint posterior (per-probe + joint), for the
             baseline (null) and model (contaminated) chains. High power; mild double-use.
  predict  : conditional PPD -- condition on gamma_t (probe-only chain), predict w(theta);
             compare predictive to baseline w (null) and model w (detection); power check.

Run in the py31 env after `source cosmosis-configure`, on a compute node
(`--constraint=x86-64-v3`). Heavy evals are parallelised across processes.

Examples
  srun ... python run_predict.py gate    --case lsst_fixed_lcdm_gt_to_w
  srun ... python run_predict.py gof     --case lsst_fixed_lcdm_gt_to_w --nsamples 2000 --nproc 32
  srun ... python run_predict.py predict --case lsst_fixed_lcdm_gt_to_w --nsamples 2000 --nproc 32
"""
import os

# Pin BLAS/OpenMP to a single thread BEFORE numpy/cosmosis/CAMB are imported. Each
# parallel worker runs one full CAMB pipeline; without this every worker spawns its own
# OpenMP pool and 16 workers x N threads thrash the cores (observed ~19 s/eval, slower
# than serial). With spawn, workers re-import this module, so this runs in every worker.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as ppd_config          # noqa: E402
import chain_io                      # noqa: E402
import conditional_ppd as cppd       # noqa: E402


# --------------------------------------------------------------------------- #
# parallel theory evaluation
# --------------------------------------------------------------------------- #
_WORKER = None


def _winit(ini_path):
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_v] = "1"
    global _WORKER
    from theory_eval import TheoryEvaluator
    _WORKER = TheoryEvaluator(ini_path)


def _weval(arg):
    i, vec = arg
    r = _WORKER.evaluate(vec)
    return i, np.asarray(r["theory"], float), float(r["chi2"])


def evaluate_samples(ini_path, vectors, nproc):
    """Evaluate theory at each ordered parameter vector; returns (theory[N,ndata], chi2[N])."""
    n = len(vectors)
    theory = [None] * n
    chi2 = np.full(n, np.nan)
    t0 = time.time()
    if nproc <= 1:
        _winit(ini_path)
        for i, vec in enumerate(vectors):
            _, th, c = _weval((i, vec))
            theory[i], chi2[i] = th, c
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{n}] {(time.time()-t0)/(i+1):.2f}s/eval", flush=True)
    else:
        import concurrent.futures as cf
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=nproc, mp_context=ctx,
                                    initializer=_winit, initargs=(ini_path,)) as ex:
            done = 0
            for i, th, c in ex.map(_weval, list(enumerate(vectors)), chunksize=4):
                theory[i], chi2[i] = th, c
                done += 1
                if done % 100 == 0:
                    print(f"  [{done}/{n}] {(time.time()-t0)/done:.2f}s/eval", flush=True)
    return np.array(theory, float), chi2


# --------------------------------------------------------------------------- #
def _load_reference_evaluator(case):
    from theory_eval import TheoryEvaluator
    ev = TheoryEvaluator(case["baseline_ini"])
    return ev


def _map_params(chain_path, short_names):
    entry = chain_io.load_chain(chain_path)
    row, idx = chain_io.map_row(entry)
    params = chain_io.sample_dict(entry, row, short_names)
    stored_chi2 = float(row[entry["col"]["chi2"]]) if "chi2" in entry["col"] else np.nan
    return params, stored_chi2, entry


# --------------------------------------------------------------------------- #
def cmd_gate(case, args):
    """Fidelity gate: recomputed joint chi2 == chain-stored 2pt_chi2 at the MAP."""
    from theory_eval import TheoryEvaluator
    print("=== chi2 reproduction gate ===", flush=True)
    results = {}
    for variant, ini_key, chain_key in [("baseline", "baseline_ini", "joint_chain_baseline"),
                                        ("model", "model_ini", "joint_chain_model")]:
        ev = TheoryEvaluator(case[ini_key])
        params, stored, _ = _map_params(case[chain_key], ev.short_names)
        out = ev.evaluate(params, want_cov=(variant == "baseline"))
        recomputed = out["chi2"]
        dchi = recomputed - stored
        print(f"  {variant:9s}: stored={stored:.5f}  recomputed={recomputed:.5f}  "
              f"diff={dchi:+.2e}  {'PASS' if abs(dchi) < 1e-2 else 'FAIL'}", flush=True)
        results[variant] = {"stored": stored, "recomputed": recomputed, "diff": dchi}
        if variant == "baseline":
            inv_cov = out["inv_cov"]                       # pm-marginalized precision the PPD will use
            cov_map = out["cov"]                           # its inverse = effective covariance
            # --- metric-consistency gate: the covariance the PPD conditions on must be the
            # SAME metric the likelihood scored the chain with. Recompute chi2 from *our*
            # precision and require it to match the stored 2pt_chi2; and require cov_map to
            # invert inv_cov cleanly. This catches the 2pt_covariance-vs-inverse_covariance
            # mismatch that the theory-ordering check above is structurally blind to.
            r = ev.observed - out["theory"]
            chi2_metric = float(r @ inv_cov @ r)
            dmetric = chi2_metric - recomputed
            ok_metric = abs(dmetric) < 1e-3
            print(f"  metric gate: chi2(PPD inv_cov)={chi2_metric:.5f} vs 2pt_chi2={recomputed:.5f}  "
                  f"diff={dmetric:+.2e}  {'PASS' if ok_metric else 'FAIL'}", flush=True)
            # informational: pm-marginalization makes the effective covariance near-singular
            # along the point-mass modes, so inv(inv_cov) is intrinsically ill-conditioned and
            # cov*inv_cov round-trips far from I. That is physical, not an error -- the conditional
            # PPD uses the precision directly (precision-space block algebra) and never inverts the
            # full covariance; the chi2 tie above is the metric-consistency check.
            k = inv_cov.shape[0]
            roundtrip = float(np.max(np.abs(cov_map @ inv_cov - np.eye(k))))
            cond = float(np.linalg.cond(inv_cov))
            print(f"  (info) effective-metric condition number = {cond:.2e}; "
                  f"cov*inv_cov round-trip max|Δ| = {roundtrip:.1e} (large by construction)", flush=True)
            results["metric_gate"] = {"chi2_ppd_metric": chi2_metric, "diff": dmetric,
                                      "roundtrip_max": roundtrip, "cond": cond, "pass": bool(ok_metric)}
            # covariance theta-dependence diagnostic. do_pm_sigcritinv makes the gammat-block
            # covariance depend on GEOMETRY (Sigma_crit_inv ~ distances), so perturb a geometry
            # parameter (Omega_m) -- NOT the amplitude A_s, which leaves the pm/sigcritinv metric
            # invariant (an A_s shift reports ~0, which is correct but uninformative). Measured on
            # the marginalized metric so the dependence is actually visible; this quantifies how
            # safe it is to freeze the covariance at the MAP across the posterior.
            gi = ev.probe_idx["gammat"]
            geo = next((n for n in ("omega_m", "omega_c", "omch2", "ommh2", "hubble", "h0")
                        if n in params), None)
            if geo is None:
                print("  cov theta-dep: no geometry parameter varied; skipping", flush=True)
            else:
                p2 = dict(params); p2[geo] = params[geo] * 1.05        # +5% geometry shift
                out2 = ev.evaluate(p2, want_cov=True)
                cov2 = out2["cov"]
                # self-check: confirm the shift actually moved the theory, so a ~0 covariance
                # change means "genuinely cosmology-invariant metric", not "perturbation no-op".
                dtheory = float(np.max(np.abs(out2["theory"][gi] - out["theory"][gi]) /
                                       (np.abs(out["theory"][gi]) + 1e-300)))
                frac = np.abs(cov2[np.ix_(gi, gi)] - cov_map[np.ix_(gi, gi)]) / np.abs(cov_map[np.ix_(gi, gi)] + 1e-300)
                print(f"  gammat-block cov frac. change for +5% {geo} (marginalized metric): "
                      f"median={np.median(frac):.3e} max={np.max(frac):.3e}  "
                      f"[sanity: gammat theory moved up to {dtheory:.1%}]", flush=True)
                results["cov_theta_dep"] = {"param": geo, "shift_frac": 0.05, "theory_change_max": dtheory,
                                            "median": float(np.median(frac)), "max": float(np.max(frac))}
    return results


# --------------------------------------------------------------------------- #
def cmd_gof(case, args):
    """Goodness-of-fit PPD from the joint posterior for baseline (null) & model chains."""
    from theory_eval import TheoryEvaluator, observed_vectors
    out_dir = Path(case["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    ev = _load_reference_evaluator(case)
    probe_idx = {k: v.tolist() for k, v in ev.probe_idx.items()}

    obs = {"baseline": ev.observed, "model": observed_vectors(case["model_ini"])["observed"]}
    summary = {"case": args.case, "mode": "gof", "nsamples": args.nsamples, "seed": args.seed,
               "probe_sizes": {k: len(v) for k, v in ev.probe_idx.items()}}

    for variant, chain_key in [("baseline", "joint_chain_baseline"), ("model", "joint_chain_model")]:
        chain = case[chain_key]
        print(f"\n=== GoF PPD: {variant} joint chain ===", flush=True)
        entry = chain_io.load_chain(chain)
        _, dicts, chosen = chain_io.thin_weighted(entry, ev.short_names, args.nsamples, seed=args.seed)
        vectors = [np.array([d[n] for n in ev.short_names], float) for d in dicts]

        cache = out_dir / f"theory_gof_{variant}_N{args.nsamples}_s{args.seed}.npz"
        if cache.exists() and not args.force:
            print(f"  loading cached theory {cache.name}", flush=True)
            theory = np.load(cache)["theory"]
        else:
            theory, _ = evaluate_samples(case["baseline_ini"], vectors, args.nproc)
            np.savez_compressed(cache, theory=theory, chosen=chosen)

        # reference covariance at this chain's MAP
        map_params, _, _ = _map_params(chain, ev.short_names)
        cov_ref = ev.evaluate(map_params, want_cov=True)["cov"]

        res = cppd.gof_ppd(theory, cov_ref, obs[variant], ev.probe_idx, seed=args.seed)
        summary[variant] = {p: {"p_value": r["p_value"], "sigma_1sided": r["sigma_1sided"],
                                "p_chi2": r["p_chi2"], "sigma_chi2": r["sigma_chi2"],
                                "npred": r["npred"], "T_obs": r["T_obs"], "T_obs_median": r["T_obs_median"]}
                            for p, r in res.items()}
        for p, r in res.items():
            print(f"  {p:8s}: T_obs={r['T_obs']:.1f}/{r['npred']}  "
                  f"tension p_chi2={r['p_chi2']:.3g} ({r['sigma_chi2']:+.2f}sigma)  "
                  f"replica_p={r['p_value']:.4f}", flush=True)

    with open(out_dir / f"gof_summary_N{args.nsamples}_s{args.seed}.json", "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\nwrote {out_dir / f'gof_summary_N{args.nsamples}_s{args.seed}.json'}", flush=True)
    return summary


# --------------------------------------------------------------------------- #
def cmd_predict(case, args):
    """Conditional PPD: condition on gamma_t (probe-only chain), predict w(theta)."""
    from theory_eval import observed_vectors
    out_dir = Path(case["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    ev = _load_reference_evaluator(case)
    cond, pred = case["cond"], case["pred"]
    ci, pi = ev.probe_idx[cond], ev.probe_idx[pred]

    obs_base = ev.observed
    obs_model = observed_vectors(case["model_ini"])["observed"]
    # gamma_t is unchanged by the contamination -- assert it
    d_cond_obs = obs_base[ci]
    assert np.allclose(obs_base[ci], obs_model[ci]), "conditioning probe differs between baseline/model!"

    cond_chain = case["cond_chain"]
    if not (os.path.exists(cond_chain) and os.path.getsize(cond_chain) > 0):
        print(f"conditioning chain not ready yet: {cond_chain}", flush=True)
        return {"status": "cond_chain_not_ready"}

    print(f"=== conditional PPD: {cond} -> {pred} ===", flush=True)
    entry = chain_io.load_chain(cond_chain)
    _, dicts, chosen = chain_io.thin_weighted(entry, ev.short_names, args.nsamples, seed=args.seed)
    vectors = [np.array([d[n] for n in ev.short_names], float) for d in dicts]

    cache = out_dir / f"theory_cond_N{args.nsamples}_s{args.seed}.npz"
    if cache.exists() and not args.force:
        print(f"  loading cached theory {cache.name}", flush=True)
        theory = np.load(cache)["theory"]
    else:
        theory, _ = evaluate_samples(case["baseline_ini"], vectors, args.nproc)
        np.savez_compressed(cache, theory=theory, chosen=chosen)

    m_cond = theory[:, ci]
    m_pred = theory[:, pi]

    # reference (pm-marginalized) metric at the conditioning-chain MAP. Pass the precision
    # (inv_cov) so the conditional uses the robust precision-space block algebra.
    map_params, _, _ = _map_params(cond_chain, ev.short_names)
    ref = ev.evaluate(map_params, want_cov=True)
    cov_ref, inv_ref = ref["cov"], ref["inv_cov"]

    summary = {"case": args.case, "mode": "predict", "cond": cond, "pred": pred,
               "nsamples": args.nsamples, "seed": args.seed,
               "npred": len(pi), "ncond": len(ci)}
    res_store = {}
    for variant, d_pred_obs in [("baseline", obs_base[pi]), ("model", obs_model[pi])]:
        res, A, Sigma_c, mu = cppd.run_conditional_ppd(
            cov_ref, ci, pi, m_cond, m_pred, d_cond_obs, d_pred_obs, seed=args.seed, inv_cov=inv_ref)
        summary[variant] = {"p_value": res["p_value"], "sigma_1sided": res["sigma_1sided"],
                            "p_chi2": res["p_chi2"], "sigma_chi2": res["sigma_chi2"],
                            "T_obs": res["T_obs"], "T_obs_median": res["T_obs_median"],
                            "npred": res["npred"]}
        res_store[variant] = (res, mu, Sigma_c)
        print(f"  vs {variant:8s} w({pred}): T_obs={res['T_obs']:.1f}/{res['npred']}  "
              f"tension p_chi2={res['p_chi2']:.3g} ({res['sigma_chi2']:+.2f}sigma)  "
              f"replica_p={res['p_value']:.4f}", flush=True)

    # ---- power check: injected signal vs conditional predictive noise ---- #
    _, _, Sigma_c = res_store["baseline"]
    contam = obs_model[pi] - obs_base[pi]                    # injected w(theta) systematic
    invSc = np.linalg.inv(Sigma_c)
    snr_data = float(contam @ invSc @ contam)                # chi2 of contamination vs data noise only
    mu_base = res_store["baseline"][1]                        # (N, npred) predictive means
    pred_std = mu_base.std(axis=0)                            # per-element predictive (parameter) spread
    total_var = np.diag(Sigma_c) + pred_std**2               # noise + predictive spread (diagonal proxy)
    snr_eff = float(np.sum(contam**2 / total_var))           # crude detectability including predictive spread
    print(f"  power check: contamination chi2 vs data-noise Sigma_c = {snr_data:.1f} (n={len(pi)}); "
          f"diag detectability incl. predictive spread ~ {snr_eff:.1f}", flush=True)
    summary["power_check"] = {"contam_chi2_datanoise": snr_data,
                              "contam_detectability_incl_spread": snr_eff,
                              "median_pred_std_over_datastd": float(np.median(pred_std / np.sqrt(np.diag(Sigma_c))))}

    # save arrays for the notebook
    np.savez_compressed(out_dir / f"predict_arrays_N{args.nsamples}_s{args.seed}.npz",
                        m_pred=m_pred, m_cond=m_cond,
                        mu_baseline=res_store["baseline"][1], mu_model=res_store["model"][1],
                        Sigma_c=res_store["baseline"][2],
                        obs_base_pred=obs_base[pi], obs_model_pred=obs_model[pi],
                        angle_pred=ev.angle[pi], bin1_pred=ev.bin1[pi], bin2_pred=ev.bin2[pi],
                        T_obs_baseline=res_store["baseline"][0]["T_obs"],
                        T_rep_baseline=res_store["baseline"][0]["T_rep"],
                        T_obs_model=res_store["model"][0]["T_obs"],
                        T_rep_model=res_store["model"][0]["T_rep"])
    with open(out_dir / f"predict_summary_N{args.nsamples}_s{args.seed}.json", "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\nwrote {out_dir / f'predict_summary_N{args.nsamples}_s{args.seed}.json'}", flush=True)
    return summary


# --------------------------------------------------------------------------- #
def cmd_predict_rev(case, args):
    """Reverse conditional PPD: condition on w(theta), predict gamma_t.

    Two tests, each with its own w(theta)-only chain:
      null   -- condition on the *clean* w(theta) (chain fit to it); predict gamma_t;
                compare to the observed gamma_t. Should pass.
      detect -- condition on the *contaminated* w(theta) (chain fit to it); predict
                gamma_t; compare to the *clean* observed gamma_t. The biased b_i/sigma8
                posterior propagates to a biased gamma_t prediction -> tension.

    gamma_t is unchanged by the contamination, so the observed predicted vector is the
    clean gamma_t in both tests (asserted).
    """
    from theory_eval import observed_vectors
    out_dir = Path(case["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    ev = _load_reference_evaluator(case)
    cond, pred = case["cond"], case["pred"]           # wtheta -> gammat
    ci, pi = ev.probe_idx[cond], ev.probe_idx[pred]

    obs_base = ev.observed
    obs_model = observed_vectors(case["model_ini"])["observed"]
    # predicted probe (gamma_t) is identical between baseline & model -- assert it
    assert np.allclose(obs_base[pi], obs_model[pi]), "predicted probe differs between baseline/model!"
    d_pred_obs = obs_base[pi]                          # clean observed gamma_t (== model)

    # (name, conditioning chain, observed conditioning w(theta) to condition on)
    tests = [
        ("null",   case["cond_chain_null"],   obs_base[ci]),   # clean w(theta)
        ("detect", case["cond_chain_detect"], obs_model[ci]),  # contaminated w(theta)
    ]

    summary = {"case": args.case, "mode": "predict_rev", "cond": cond, "pred": pred,
               "nsamples": args.nsamples, "seed": args.seed,
               "npred": len(pi), "ncond": len(ci)}
    save_arrays = {"angle_pred": ev.angle[pi], "bin1_pred": ev.bin1[pi], "bin2_pred": ev.bin2[pi],
                   "obs_pred": d_pred_obs}

    for name, chain, d_cond_obs in tests:
        if not (os.path.exists(chain) and os.path.getsize(chain) > 0):
            print(f"[{name}] conditioning chain not ready yet: {chain}", flush=True)
            summary[name] = {"status": "cond_chain_not_ready", "chain": str(chain)}
            continue

        print(f"\n=== reverse conditional PPD [{name}]: {cond} -> {pred} ===", flush=True)
        entry = chain_io.load_chain(chain)
        _, dicts, chosen = chain_io.thin_weighted(entry, ev.short_names, args.nsamples, seed=args.seed)
        vectors = [np.array([d[n] for n in ev.short_names], float) for d in dicts]

        cache = out_dir / f"theory_condrev_{name}_N{args.nsamples}_s{args.seed}.npz"
        if cache.exists() and not args.force:
            print(f"  loading cached theory {cache.name}", flush=True)
            theory = np.load(cache)["theory"]
        else:
            theory, _ = evaluate_samples(case["baseline_ini"], vectors, args.nproc)
            np.savez_compressed(cache, theory=theory, chosen=chosen)

        m_cond = theory[:, ci]                        # w(theta) at posterior samples
        m_pred = theory[:, pi]                        # gamma_t at posterior samples

        # reference (pm-marginalized) metric at this chain's MAP; pass the precision so the
        # conditional uses the robust precision-space block algebra (esp. important here: the
        # predicted probe is gamma_t, whose covariance block is near-singular under pm-marg).
        map_params, _, _ = _map_params(chain, ev.short_names)
        ref = ev.evaluate(map_params, want_cov=True)
        cov_ref, inv_ref = ref["cov"], ref["inv_cov"]

        res, A, Sigma_c, mu = cppd.run_conditional_ppd(
            cov_ref, ci, pi, m_cond, m_pred, d_cond_obs, d_pred_obs, seed=args.seed, inv_cov=inv_ref)
        summary[name] = {"p_value": res["p_value"], "sigma_1sided": res["sigma_1sided"],
                         "p_chi2": res["p_chi2"], "sigma_chi2": res["sigma_chi2"],
                         "T_obs": res["T_obs"], "T_obs_median": res["T_obs_median"],
                         "npred": res["npred"], "pred_spread_over_noise": res["pred_spread_over_noise"]}
        print(f"  [{name}] predict gamma_t vs clean observed: T_obs={res['T_obs']:.1f}/{res['npred']}  "
              f"tension p_chi2={res['p_chi2']:.3g} ({res['sigma_chi2']:+.2f}sigma)  "
              f"replica_p={res['p_value']:.4f}", flush=True)

        save_arrays[f"mu_{name}"] = mu
        save_arrays[f"Sigma_c_{name}"] = Sigma_c
        save_arrays[f"T_obs_{name}"] = res["T_obs"]
        save_arrays[f"T_rep_{name}"] = res["T_rep"]

    np.savez_compressed(out_dir / f"predictrev_arrays_N{args.nsamples}_s{args.seed}.npz", **save_arrays)
    with open(out_dir / f"predictrev_summary_N{args.nsamples}_s{args.seed}.json", "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\nwrote {out_dir / f'predictrev_summary_N{args.nsamples}_s{args.seed}.json'}", flush=True)
    return summary


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["gate", "gof", "predict", "predict_rev"])
    ap.add_argument("--case", default="lsst_fixed_lcdm_gt_to_w")
    ap.add_argument("--nsamples", type=int, default=2000)
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--force", action="store_true", help="ignore cached theory and recompute")
    args = ap.parse_args()
    case = ppd_config.get_case(args.case)
    {"gate": cmd_gate, "gof": cmd_gof, "predict": cmd_predict,
     "predict_rev": cmd_predict_rev}[args.mode](case, args)


if __name__ == "__main__":
    main()
