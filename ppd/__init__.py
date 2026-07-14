"""Posterior-predictive-distribution (PPD) internal-consistency tests between
galaxy clustering w(theta) and galaxy-galaxy lensing gamma_t for the 2x2pt chains.

Method: Doux et al. 2021 (DES-Y1, MNRAS 503, 2688). Condition on one probe,
predict the other via the conditional Gaussian data-predictive, and compare the
observed data to the posterior-predictive replicas with a calibrated p-value.

See ppd/config.py for the case registry and ppd/run_predict.py for the driver.
"""
