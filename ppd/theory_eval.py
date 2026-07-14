"""In-process CosmoSIS theory evaluator for the 2x2pt (gamma_t + w(theta)) pipeline.

Loads the production ``2pt_like`` pipeline once via ``LikelihoodPipeline`` and evaluates
the model data vector at arbitrary parameter points, reading straight from the datablock:

  data_vector/2pt_theory        -> model vector, order [gammat(380), wtheta(69)]
  data_vector/2pt_data          -> observed vector (the DV FITS this ini fits)
  data_vector/2pt_inverse_covariance -> effective (point-mass-marginalized) PRECISION the
                                        likelihood actually used; invert this for the metric.
  data_vector/2pt_{name,bin1,bin2,angle} -> per-element labels

NB: the ``2pt_covariance`` datablock key holds the ORIGINAL *unmarginalized* covariance
(GaussianLikelihood saves ``self.cov`` = ``2pt_point_mass.extract_covariance`` = ``cov_orig``),
which is a DIFFERENT metric from the pm-marginalized ``inv_cov`` the chi2/chains used. The PPD
must condition on the same metric as the likelihood, so we read ``2pt_inverse_covariance`` and
invert it -- never ``2pt_covariance``. (run_predict --gate asserts these agree via the chi2.)

This reuses the exact likelihood setup the chains used (scale cuts, pm-marginalization
with do_pm_sigcritinv, ordering), so the recomputed joint chi2 matches the chain-stored
DATA_VECTOR--2PT_CHI2 (validated in run_predict --gate).

Must run in the ``py31`` env after ``source cosmosis-configure`` (needs CAMB), on a
compute node (``--constraint=x86-64-v3``).
"""
import numpy as np

SECTION = "data_vector"


def _decode(arr):
    return np.array([x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
                     for x in np.atleast_1d(arr)])


class TheoryEvaluator:
    """Wrap a CosmoSIS 2pt_like ini as a callable theory(model) evaluator."""

    def __init__(self, ini_path):
        # imported lazily so the module can be imported without cosmosis on PATH
        from cosmosis.runtime.config import Inifile
        from cosmosis.runtime.pipeline import LikelihoodPipeline

        self.ini_path = str(ini_path)
        self.pipeline = LikelihoodPipeline(Inifile(self.ini_path))
        self.varied_names = [str(p) for p in self.pipeline.varied_params]
        # short canonical names: 'cosmological_parameters--omega_m' -> 'omega_m', 'bias_lens--b1' -> 'b1'
        self.short_names = [n.split("--")[-1].lower() for n in self.varied_names]

        # run once at the pipeline start point to capture ordering / labels / observed data
        res = self.pipeline.run_results(self.pipeline.start_vector())
        b = res.block
        self.name = _decode(b[SECTION, "2pt_name"])
        self.bin1 = np.asarray(b[SECTION, "2pt_bin1"], int)
        self.bin2 = np.asarray(b[SECTION, "2pt_bin2"], int)
        self.angle = np.asarray(b[SECTION, "2pt_angle"], float)   # radians
        self.observed = np.asarray(b[SECTION, "2pt_data"], float)
        self.ndata = self.observed.size
        self.probe_idx = {p: np.where(self.name == p)[0] for p in np.unique(self.name)}

    # ------------------------------------------------------------------ #
    def vector_from_dict(self, params):
        """Build the varied-parameter vector (pipeline order) from a dict of short names."""
        try:
            return np.array([float(params[s]) for s in self.short_names], float)
        except KeyError as e:
            raise KeyError(f"missing parameter {e} for varied set {self.short_names}")

    def evaluate(self, params, want_cov=False):
        """Evaluate theory at ``params`` (dict of short names or an ordered vector).

        Returns dict with theory vector, per-probe slices, chi2 (vs this ini's observed),
        and optionally the effective metric. When ``want_cov`` is set the dict carries
        ``inv_cov`` = the point-mass-marginalized precision the likelihood used
        (``data_vector/2pt_inverse_covariance``) and ``cov`` = its numerical inverse (the
        effective covariance the PPD conditions on). We deliberately do NOT read
        ``2pt_covariance`` -- that key holds the original *unmarginalized* covariance, a
        different metric from the one the chain chi2 was computed with.
        """
        vec = self.vector_from_dict(params) if isinstance(params, dict) else np.asarray(params, float)
        res = self.pipeline.run_results(vec)
        b = res.block
        theory = np.asarray(b[SECTION, "2pt_theory"], float)
        out = {
            "theory": theory,
            "chi2": float(b[SECTION, "2pt_chi2"]),
            "post": float(res.post),
            "like": float(res.like),
            "prior": float(res.prior),
        }
        for p, idx in self.probe_idx.items():
            out[p] = theory[idx]
        if want_cov:
            inv_cov = np.asarray(b[SECTION, "2pt_inverse_covariance"], float)
            inv_cov = 0.5 * (inv_cov + inv_cov.T)          # symmetrize against round-off
            out["inv_cov"] = inv_cov
            out["cov"] = np.linalg.inv(inv_cov)            # effective (pm-marginalized) covariance
        return out

    def reference_cov(self, params):
        """Effective (pm-marginalized) covariance at a reference parameter point.

        This is ``inv(2pt_inverse_covariance)`` -- the metric the likelihood used -- NOT
        the raw ``2pt_covariance`` key (which is unmarginalized). See ``evaluate``.
        """
        return self.evaluate(params, want_cov=True)["cov"]

    def fiducial_dict(self):
        """The pipeline start point as a short-name dict (fiducial / values-file centres)."""
        return dict(zip(self.short_names, self.pipeline.start_vector()))


def observed_vectors(ini_path):
    """Load just the observed data vector + labels for the DV FITS an ini fits.

    Cheap-ish (one pipeline setup + one eval); used to grab the baseline and model
    (contaminated) observed w(theta) with guaranteed-identical ordering / scale cuts.
    """
    ev = TheoryEvaluator(ini_path)
    return {
        "observed": ev.observed,
        "name": ev.name,
        "bin1": ev.bin1,
        "bin2": ev.bin2,
        "angle": ev.angle,
        "probe_idx": ev.probe_idx,
    }
