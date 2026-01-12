import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt
import pyccl as ccl

try:
    from . import utils
    from . import config
    from . import selection as sel
    from .clustering import ClusteringEnhancement
except ImportError:
    import utils
    import config
    import selection as sel
    from clustering import ClusteringEnhancement


# Constants from config
SYS_NSIDE = config.SIM_SETTINGS['sys_nside']
OUTPUT_PREDS = config.PATHS['output_preds']
N_POP_SAMPLE = config.SIM_SETTINGS['n_pop_sample']

def plot_selection_wtheta(result, theta_arcmin, output_dir):
    plt.figure(figsize=(8, 6))
    
    # Choose representative z-bins where global n(z) is significant
    # Indices where nbar is above a threshold
    nbar_peak = np.max(result.nbar)
    significant_indices = np.where(result.nbar > 0.05 * nbar_peak)[0]
    
    if len(significant_indices) > 5:
        # Pick 5 equidistant indices from significant ones
        indices = np.linspace(significant_indices[0], significant_indices[-1], 5, dtype=int)
        # Ensure peak is included if possible
        peak_idx = np.argmax(result.nbar)
        if peak_idx not in indices:
             # Find closest in indices and replace
             closest = np.argmin(np.abs(indices - peak_idx))
             indices[closest] = peak_idx
        indices = np.sort(np.unique(indices))
    else:
        indices = significant_indices

    for i in indices:
        z_val = result.z_mid[i]
        # result.w_selection is <delta N^2> (count fluctuation correlation)
        # Divide by dz^2 to get density correlation <delta n^2>
        if result.w_selection is not None:
             w_density = result.w_selection[i] / (result.dz[i] ** 2)
             plt.plot(theta_arcmin, w_density, linewidth=2, label=f"z={z_val:.2f}")

    plt.xlabel(r"$\theta$ [arcmin]")
    plt.ylabel(r"$\langle \delta n(z, \theta) \delta n(z, 0) \rangle$")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, alpha=0.3, which="both")
    plt.legend()
    plt.title(r"Spatial Variation of $n(z)$ Density")
    plt.savefig(os.path.join(output_dir, "selection_wtheta_z_data.png"))
    plt.close()


def plot_nz_variations_data(z_mid, n_maps, nbar, output_dir):
    """Plot n(z) variations for real data."""
    plt.figure(figsize=(8, 6))
    
    # n_maps is (nz, npix), we want to plot profiles for a subset of pixels
    # Transpose to (npix, nz) for easier iteration
    profiles = n_maps.T
    print(n_maps.shape)
    
    # Subsample pixels if there are too many
    n_plot = 50
    if profiles.shape[0] > n_plot:
        idx = np.random.choice(profiles.shape[0], n_plot, replace=False)
        profiles_subset = profiles[idx]
    else:
        profiles_subset = profiles
        
    for prof in profiles_subset:
        plt.plot(z_mid, prof, color="gray", alpha=0.1)
        
    plt.plot(z_mid, nbar, "k-", linewidth=2, label=r"Global $\bar{n}(z)$")
    plt.xlabel("Redshift z")
    plt.ylabel("n(z)")
    plt.title("n(z) Variations (Data)")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "nz_distribution_data.png"))
    plt.close()


def plot_model_vs_ccl(result, cosmo, nbar, z_mid, output_dir):
    """Plot Model w(theta) vs direct CCL calculation."""
    theta_arcmin = 60.0 * result.theta_deg
    plt.figure(figsize=(7, 5))
    plt.plot(theta_arcmin, theta_arcmin * result.w_model, "k--", linewidth=2, label=r"$w_{\rm model}$ (binned)")
    
    dndz_global = nbar / np.trapezoid(nbar, z_mid)
    gtracer = ccl.NumberCountsTracer(
        cosmo,
        has_rsd=False,
        dndz=(z_mid, dndz_global),
        bias=(z_mid, np.ones_like(z_mid)),
    )
    # Estimate lmax from theta_min
    lmax = 3000
    ell = np.arange(lmax + 1, dtype=int)
    cell = ccl.angular_cl(cosmo, gtracer, gtracer, ell)
    w_direct = ccl.correlation(
        cosmo,
        ell=ell,
        C_ell=cell,
        theta=result.theta_deg,
        type="NN",
        method="fftlog",
    )
    plt.plot(theta_arcmin, theta_arcmin * w_direct, "g-", linewidth=2, label=r"$w_{\rm total}$ (direct CCL)")
    
    plt.xlabel(r"$\theta$ [arcmin]")
    plt.ylabel(r"$\theta \cdot w(\theta)$ [arcmin]")
    plt.xscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(r"Model vs Direct CCL")
    plt.savefig(os.path.join(output_dir, "w_model_vs_ccl_data.png"))
    plt.close()


def plot_clustering_comparison(result, result_var, enhancement_factor, z_std_ratio, output_dir):
    """Plot comparison of clustering enhancement between different methods."""
    theta_arcmin = 60.0 * result.theta_deg
    fig_comp, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_top.plot(theta_arcmin, theta_arcmin * result.w_model, "k--", linewidth=2, label=r"$w_{\rm model}$")
    ax_top.plot(theta_arcmin, theta_arcmin * result.w_true, "r-", linewidth=2, label=r"$w_{\rm true}$ (wtheta)")
    ax_top.plot(theta_arcmin, theta_arcmin * result_var.w_true, "g:", linewidth=2, label=r"$w_{\rm true}$ (variance)")
    ax_top.set_ylabel(r"$\theta \cdot w(\theta)$ [arcmin]")
    ax_top.set_xscale("log")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(fontsize=12)
    header_text = f"Geo Enhanc: {enhancement_factor:.4f} | Ratio(std_z): {z_std_ratio:.4f}\nClust. Enhanc: {result.delta_w[0]/result.w_model[0]:.4f}"
    ax_top.set_title(header_text)

    w_abs = np.abs(result.w_model)
    thresh = 0.05 * np.nanmax(w_abs)
    mask = w_abs > thresh
    frac_diff = np.full_like(result.w_model, np.nan)
    frac_diff[mask] = (result.w_true[mask] - result.w_model[mask]) / result.w_model[mask]
    
    frac_diff_var = np.full_like(result.w_model, np.nan)
    frac_diff_var[mask] = (result_var.w_true[mask] - result.w_model[mask]) / result.w_model[mask]

    ax_bot.plot(theta_arcmin, frac_diff, "r-", linewidth=1.5, label="wtheta")
    ax_bot.plot(theta_arcmin, frac_diff_var, "g:", linewidth=1.5, label="variance")
    ax_bot.set_xlim(theta_arcmin.min(), theta_arcmin.max())
    ax_bot.set_ylabel(r"$\Delta w / w_{\rm model}$")
    ax_bot.set_xlabel(r"$\theta$ [arcmin]")
    ax_bot.set_xscale("log")
    ax_bot.grid(True, alpha=0.3, which="both")
    ax_bot.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "w_comparison_data.png"))
    plt.close()


def main() -> None:
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Regenerate Statistics from Predictions (consistent with new binning)
    print("Loading predictions from catalog to regenerate statistics...")
    preds_path = config.PATHS['output_preds']
    if not os.path.exists(preds_path):
        print(f"Error: Predictions file {preds_path} not found. Run selection.py first.")
        return

    print(f"Loading existing predictions from {OUTPUT_PREDS}...")
    cla_cat = pd.read_feather(OUTPUT_PREDS)
    
    # print("Processing loaded catalog (photo-z, cuts)...")
    # cla_cat = sel.process_classified_catalog(cla_cat)
    
    # Determine redshift bins from data
    print("Determining consistent redshift bins from loaded catalog...")
    z_mid, edges = utils.get_redshift_bins(cla_cat['redshift_input_p'])
    
    maps, seen_idx = sel.load_system_maps()
    psf_hp_map = maps[0]
    stats_dict = sel.generate_summary_statistics_from_cat(cla_cat, psf_hp_map, seen_idx, output_dir, z=z_mid, edges=edges)
    full_stats = stats_dict['full']

    n_maps = full_stats['dndzs'].T
    nbar = full_stats['dndz_det']
    # nbar = n_maps.mean(axis=1) # Use map mean for consistency with variance calculation
    
    # Redshift std ratio from stats
    z_std_ratio = full_stats.get('z_std_ratio', 1.0)

    # 2. Setup Cosmology and Clustering
    cosmo = ccl.Cosmology(
        Omega_c=config.COSMO_PARAMS['Omega_c'], 
        Omega_b=config.COSMO_PARAMS['Omega_b'], 
        h=config.COSMO_PARAMS['h'], 
        sigma8=config.COSMO_PARAMS['sigma8'], 
        n_s=config.COSMO_PARAMS['n_s']
    )

    ell_max = config.CLUSTERING_SETTINGS['ell_max']
    theta_deg = np.logspace(
        np.log10(config.CLUSTERING_SETTINGS['theta_min_deg']), 
        np.log10(config.CLUSTERING_SETTINGS['theta_max_deg']), 
        config.CLUSTERING_SETTINGS['theta_bins']
    )

    enhancer = ClusteringEnhancement(cosmo, ell_max=ell_max, ell_min=2)
    
    # 3. Compute Clustering Enhancement
    print("Computing clustering enhancement (wtheta mode)...")
    # Note: For real data, we must enable shot noise subtraction if not already handled.
    # variation.py passes n_samples to compute_enhancement_from_maps.
    result = enhancer.compute_enhancement_from_maps(
        n_maps=n_maps,
        nbar=nbar,
        z=z_mid,
        theta_deg=theta_deg,
        selection_mode="wtheta",
        nside=SYS_NSIDE,
        seen_idx=seen_idx,
        weights=full_stats['frac_pix'],
    )

    # 4. Compute Variance Mode Calculation (for comparison)
    # We need band-limited maps for fair variance comparison
    # to match the effective filtering occurring in wtheta mode (anafast).
    print("Computing clustering enhancement (variance mode)...")
    lmax_map = 3 * SYS_NSIDE - 1
    n_maps_bl = np.zeros_like(n_maps)
    npix = hp.nside2npix(SYS_NSIDE)
    
    for i in range(len(n_maps)):
        # Reconstruct full map for SHT
        m_full = np.zeros(npix)
        m_full[seen_idx] = n_maps[i]
        
        # Band-limit (smooth)
        alms = hp.map2alm(m_full, lmax=lmax_map)
        m_bl = hp.alm2map(alms, nside=SYS_NSIDE)
        
        # Extract seen pixels back
        n_maps_bl[i] = m_bl[seen_idx]
    
    result_var = enhancer.compute_enhancement_from_maps(
        n_maps=n_maps_bl, 
        nbar=nbar,
        z=z_mid,
        theta_deg=theta_deg,
        selection_mode="variance",
        nside=SYS_NSIDE,
        seen_idx=seen_idx,
        weights=full_stats['frac_pix'],
    )

    # 5. Calculate Geometric Enhancement
    geo_enhancement = utils.calculate_geometric_enhancement(
        z_mid, n_maps.T, nbar, frac_pix=full_stats['frac_pix']
    )

    # 6. Print Summary
    print(f"Geometric Enhancement Factor: {geo_enhancement:.6f}")
    print(f"Redshift-based std Ratio:   {z_std_ratio:.6f}")
    print(f"Measured Clust. Enhanc (theta_min): {1.0 + result.delta_w[0] / result.w_model[0]:.6f}")

    # Diagnostic: matrix weighted enhancement
    weights = nbar * result.dz
    w_mat0 = result.w_mat[:, :, 0]
    w_estim = np.einsum("i,j,ij->", weights, weights, w_mat0) / (np.sum(weights)**2)
    
    var_n_arr = np.average((n_maps - nbar[:, None]) ** 2, weights=full_stats['frac_pix'], axis=1)
    w_sel_density = var_n_arr * (result.dz**2)
    xi_diag0 = np.diagonal(w_mat0)
    dw_estim = np.sum(w_sel_density * xi_diag0)
    print(f"Binned Predicted Factor (theta_min): {1.0 + dw_estim / w_estim:.6f}")

    # Diagnostic: Check decoherence at theta_min
    peak_idx = np.argmax(result.nbar)
    if result.w_selection is not None:
        w_sel_0 = result.w_selection[peak_idx, 0] / (result.dz[peak_idx]**2)
        var_peak = np.average((n_maps[peak_idx] - nbar[peak_idx])**2, weights=full_stats['frac_pix'])
        print(f"Decoherence Factor (z={z_mid[peak_idx]:.2f}): w_sys(theta_min) / var_sys = {w_sel_0 / var_peak:.4f}")

    # 7. Plotting
    plot_nz_variations_data(z_mid, n_maps, nbar, output_dir)
    plot_model_vs_ccl(result, cosmo, nbar, z_mid, output_dir)
    plot_selection_wtheta(result, 60.0 * result.theta_deg, output_dir)
    plot_clustering_comparison(result, result_var, geo_enhancement, z_std_ratio, output_dir)
    utils.plot_geo_factor_z(z_mid, n_maps, nbar, output_dir, "geo_factor_z_data.png", frac_pix=full_stats['frac_pix'])

if __name__ == "__main__":
    main()
