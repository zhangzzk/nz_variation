import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pyccl as ccl

try:
    from . import utils
    from .clustering import ClusteringEnhancement
    from .selection import load_and_filter_catalog, smooth_nz_preserve_moments
    from . import config
except ImportError:
    import utils
    from clustering import ClusteringEnhancement
    from selection import load_and_filter_catalog, smooth_nz_preserve_moments
    import config


import numpy as np
import healpy as hp

def make_uncorrelated_nz_maps(
    z_edges: np.ndarray,
    nside: int,
    mean_z: float,
    sigma0: float,
    mean_scatter: float = 0.05,
    width_scatter: float = 0.0,
    noise_level: float = 0.0,
    seed: int = 42,
    n_plot: int = 50,
    chunk_size: int = 10000,
):
    np.random.seed(seed)
    nz = len(z_edges) - 1
    npix = hp.nside2npix(nside)

    z_fine = np.linspace(z_edges[0], z_edges[-1], 400)
    dz_fine = z_fine[1] - z_fine[0]
    z_2d = z_fine[None, :]

    # Mean and width variations (linear space)
    mean_pix = mean_z * (1 + mean_scatter * np.random.randn(npix))
    sigma_pix = sigma0 * np.clip(1 + width_scatter * np.random.randn(npix), 0.05, None)

    # Lognormal parameters per pixel
    sigma_ln2 = np.log1p((sigma_pix / mean_pix)**2)
    sigma_ln = np.sqrt(sigma_ln2)
    mu_ln = np.log(mean_pix) - 0.5 * sigma_ln2

    # Bin masks
    bin_masks = []
    for i in range(nz):
        lo, hi = z_edges[i], z_edges[i + 1]
        mask = (z_fine >= lo) & (z_fine < hi) if i < nz - 1 else (z_fine >= lo) & (z_fine <= hi)
        bin_masks.append(mask)

    weights = np.random.rand(npix)
    weight_sum = np.sum(weights)

    n_maps = np.zeros((nz, npix))
    nbar_fine_sum = np.zeros_like(z_fine)
    sigma_inv_local_sum = 0.0
    sum_std_local = 0.0

    eps = 1e-300  # avoid log(0)

    for start in range(0, npix, chunk_size):
        end = min(start + chunk_size, npix)
        idx_slice = slice(start, end)
        batch_weights = weights[idx_slice]

        mu = mu_ln[idx_slice][:, None]
        sig = sigma_ln[idx_slice][:, None]

        # Lognormal PDF (numerical normalization over support)
        profiles = np.exp(
            - (np.log(np.maximum(z_2d, eps)) - mu)**2 / (2 * sig**2)
        ) / (np.maximum(z_2d, eps) * sig * np.sqrt(2*np.pi))

        norms = np.trapezoid(profiles, z_fine, axis=1)
        norms[norms == 0] = 1e-12
        profiles /= norms[:, None]

        # Optional uncorrelated noise
        if noise_level > 0:
            profiles += noise_level * np.random.randn(*profiles.shape)
            profiles = np.maximum(profiles, 0)
            profiles /= np.trapezoid(profiles, z_fine, axis=1)[:, None]

        nbar_fine_sum += (profiles * batch_weights[:, None]).sum(axis=0)
        sigma_inv_local_sum += np.sum(np.trapezoid(profiles**2, z_fine, axis=1) * batch_weights)

        # Local std
        mu_l = np.trapezoid(z_2d * profiles, z_fine, axis=1)
        var_l = np.trapezoid((z_2d - mu_l[:, None])**2 * profiles, z_fine, axis=1)
        sum_std_local += np.sum(np.sqrt(np.maximum(var_l, 0)) * batch_weights)

        for i in range(nz):
            n_maps[i, start:end] = (
                np.sum(profiles[:, bin_masks[i]], axis=1) * dz_fine
            )

    # Convert to dn/dz
    n_maps /= np.diff(z_edges)[:, None]
    nbar = np.average(n_maps, weights=weights, axis=1)

    nbar_fine = nbar_fine_sum / weight_sum
    sigma_inv_local_mean = sigma_inv_local_sum / weight_sum
    mean_std_local = sum_std_local / weight_sum

    # Global width
    mu_g = np.trapezoid(z_fine * nbar_fine, z_fine)
    var_g = np.trapezoid((z_fine - mu_g)**2 * nbar_fine, z_fine)
    std_global = np.sqrt(np.maximum(var_g, 0))
    z_std_ratio = std_global / mean_std_local if mean_std_local > 0 else 1.0

    # Subset profiles for plotting
    subset = np.random.choice(npix, n_plot, replace=False)
    mu_s = mu_ln[subset][:, None]
    sg_s = sigma_ln[subset][:, None]

    profiles_subset = np.exp(
        - (np.log(np.maximum(z_2d, eps)) - mu_s)**2 / (2 * sg_s**2)
    ) / (np.maximum(z_2d, eps) * sg_s * np.sqrt(2*np.pi))
    profiles_subset /= np.trapezoid(profiles_subset, z_fine, axis=1)[:, None]

    return (
        n_maps,
        nbar,
        z_fine,
        profiles_subset,
        nbar_fine,
        sigma_inv_local_mean,
        z_std_ratio,
        weights,
    )




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
        # result.w_selection is now (nz, nz, ntheta)
        # We plot the auto-correlation for each bin i
        w_density = result.w_selection[i, i] / (result.dz[i] ** 2)
        
        plt.plot(theta_arcmin, w_density, linewidth=2, label=f"z={z_val:.2f}")

    plt.xlabel(r"$\theta$ [arcmin]")
    plt.ylabel(r"$\langle \delta n(z, \theta) \delta n(z, 0) \rangle$")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, alpha=0.3, which="both")
    plt.legend()
    plt.title(r"Spatial Variation of $n(z)$ Density")
    plt.savefig(os.path.join(output_dir, "selection_wtheta_z_toy.png"))
    plt.close()


def plot_nz_variations(z_fine, profiles, nbar_fine, z_edges, mean_scatter, output_dir):
    """Plot n(z) variations."""
    plt.figure(figsize=(8, 6))
    for prof in profiles:
        plt.plot(z_fine, prof, color="gray", alpha=0.1)
    plt.plot(z_fine, nbar_fine, "k-", linewidth=2, label=r"Global $\bar{n}(z)$")
    plt.axvline(z_edges[0], color="r", linestyle=":", alpha=0.2)
    plt.axvline(z_edges[-1], color="r", linestyle=":", alpha=0.2)
    plt.xlabel("Redshift z")
    plt.ylabel("n(z)")
    plt.title(f"n(z) Variations (mean_scatter={mean_scatter})")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "nz_distribution_toy.png"))
    plt.close()


def plot_model_vs_ccl(result, cosmo, nbar_fine, z_fine, lmax_map, output_dir):
    """Plot Model w(theta) vs direct CCL calculation."""
    theta_arcmin = 60.0 * result.theta_deg
    plt.figure(figsize=(7, 5))
    plt.plot(theta_arcmin, theta_arcmin * result.w_model, "k--", linewidth=2, label=r"$w_{\rm model}$ (binned)")
    
    dndz_global = nbar_fine / np.trapezoid(nbar_fine, z_fine)
    gtracer = ccl.NumberCountsTracer(
        cosmo,
        has_rsd=False,
        dndz=(z_fine, dndz_global),
        bias=(z_fine, np.ones_like(z_fine)),
    )
    ell = np.arange(lmax_map + 1, dtype=int)
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
    plt.savefig(os.path.join(output_dir, "w_model_vs_ccl_toy.png"))
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
    plt.savefig(os.path.join(output_dir, "w_comparison_toy.png"))
    plt.close()


def main() -> None:
    nside = config.SIM_SETTINGS['sys_nside']
    ell_max = config.CLUSTERING_SETTINGS['ell_max']
    theta_deg = np.logspace(
        np.log10(config.CLUSTERING_SETTINGS['theta_min_deg']), 
        np.log10(config.CLUSTERING_SETTINGS['theta_max_deg']), 
        config.CLUSTERING_SETTINGS['theta_bins']
    )

    # Match real data peaks
    # Load catalog just to get representative redshift distribution
    
    gal_cat, _ = load_and_filter_catalog()
    z_mid, z_edges = utils.get_redshift_bins(gal_cat['redshift'])
    
    mean_z = 0.6
    sigma0 = 0.32
    mean_scatter = 0.05
    width_scatter = 0.2
    noise_level = 0.

    cosmo = ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.67,
        sigma8=0.8,
        n_s=0.965,
    )

    n_maps, nbar, z_fine, profiles, nbar_fine, sigma_inv_local_mean, z_std_ratio, weights = make_uncorrelated_nz_maps(
        z_edges,
        nside,
        mean_z,
        sigma0,
        mean_scatter=mean_scatter,
        width_scatter=width_scatter,
        noise_level=noise_level,
        seed=42,
    )

    # Create a footprint similar to real data (~2.5% of sky)
    npix = hp.nside2npix(nside)
    center_vec = hp.ang2vec(0, 0, lonlat=True)
    radius_rad = np.deg2rad(10) # 5 degree radius is ~2.4% fsky at nside 128
    seen_idx = hp.query_disc(nside, center_vec, radius_rad)
    n_maps_partial = n_maps[:, seen_idx]
    weights_partial = weights[seen_idx]

    # Z-Smoothing (Consistent with Data)
    if config.ANALYSIS_SETTINGS['smooth_nz']:
        print(f"Smoothing {n_maps_partial.shape[1]} distributions along z...")
        sigma_dz = config.ANALYSIS_SETTINGS['smoothing_sigma_dz']
        # smooth_nz_preserve_moments expects (N, Nz)
        n_maps_partial = smooth_nz_preserve_moments(z_mid, n_maps_partial.T, sigma_dz=sigma_dz).T
    
    nbar_partial = np.average(n_maps_partial, weights=weights_partial, axis=1)

    enhancer = ClusteringEnhancement(cosmo, ell_max=ell_max, ell_min=2)
    result = enhancer.compute_enhancement_from_maps(
        n_maps=n_maps_partial,
        nbar=nbar_partial,
        z=z_mid,
        theta_deg=theta_deg,
        selection_mode="wtheta",
        nside=nside,
        seen_idx=seen_idx,
        weights=weights_partial,
    )

    # For fair comparison in variance mode, we need to use band-limited maps
    # to match the effective filtering occurring in wtheta mode (anafast).
    print("Computing clustering enhancement (variance mode)...")
    lmax_map = min(ell_max, 3 * nside - 1)
    n_maps_bl = np.zeros((len(n_maps_partial), len(seen_idx)))
    for i in range(len(n_maps_partial)):
        # Reconstruct full map for SHT (zero-pad outside mask to match data)
        m_full = np.zeros(npix)
        m_full[seen_idx] = n_maps_partial[i]
        alms = hp.map2alm(m_full, lmax=lmax_map)
        m_bl = hp.alm2map(alms, nside=nside)
        n_maps_bl[i] = m_bl[seen_idx]

    result_var = enhancer.compute_enhancement_from_maps(
        n_maps=n_maps_bl,
        nbar=nbar_partial,
        z=z_mid,
        theta_deg=theta_deg,
        selection_mode="variance",
        nside=nside,
        seen_idx=seen_idx,
        weights=weights_partial,
    )
    
    # Compare Delta w at smallest theta
    dw_wtheta = result.delta_w[0]
    dw_var = result_var.delta_w[0]
    print(f"Sanity Check (Delta w at theta={theta_deg[0]:.4f} deg):")
    print(f"  Via w(theta) integration: {dw_wtheta:.6e}")
    print(f"  Via variance integration: {dw_var:.6e}")
    print(f"  Ratio (wtheta/variance):  {dw_wtheta / dw_var:.4f}")
    
    # Calculate Geometric factor using binned results matches selection.py/variation.py logic
    enhancement_factor = utils.calculate_geometric_enhancement(
        z_mid, n_maps_partial.T, nbar_partial, frac_pix=weights_partial
    )

    # Diagnostic: matrix weighted enhancement
    # User's logic: w_tot = Sum_{i,j} n_i n_j w_mat_ij
    # selection enhancement adds: Sum_i w_sel_i * w_mat_ii
    nz_weights = nbar_partial * result.dz
    w_mat0 = result.w_mat[:, :, 0]
    w_estim = np.einsum("i,j,ij->", nz_weights, nz_weights, w_mat0) / (np.sum(nz_weights)**2)
    
    var_n_arr = np.average((n_maps_partial - nbar_partial[:, None]) ** 2, weights=weights_partial, axis=1)
    w_sel_density = var_n_arr * (result.dz**2)
    xi_diag0 = np.diagonal(w_mat0)
    dw_estim = np.sum(w_sel_density * xi_diag0)
    
    print(f"Geometric Enhancement Factor: {enhancement_factor:.6f}")
    print(f"Redshift-based std Ratio:   {z_std_ratio:.6f}")
    print(f"Measured Clust. Enhanc (theta_min): {1.0 + result.delta_w[0] / result.w_model[0]:.6f}")
    print(f"Binned Predicted Factor (theta_min): {1.0 + dw_estim / w_estim:.6f}")

    # Diagnostic: Check decoherence at theta_min
    peak_idx = np.argmax(result.nbar)
    # result.w_selection is (nz, nz, ntheta)
    w_sel_0 = result.w_selection[peak_idx, peak_idx, 0] / (result.dz[peak_idx]**2)
    var_peak = np.average((n_maps_partial[peak_idx] - nbar_partial[peak_idx])**2, weights=weights_partial)
    print(f"Decoherence Factor (z={z_mid[peak_idx]:.2f}): w_sys(theta_min) / var_sys = {float(w_sel_0 / var_peak):.4f}")

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    plot_nz_variations(z_fine, profiles, nbar_fine, z_edges, mean_scatter, output_dir)
    plot_model_vs_ccl(result, cosmo, nbar_fine, z_fine, lmax_map, output_dir)
    plot_selection_wtheta(result, 60.0 * result.theta_deg, output_dir)
    plot_clustering_comparison(result, result_var, enhancement_factor, z_std_ratio, output_dir)
    utils.plot_geo_factor_z(result.z_mid, n_maps_partial, result.nbar, output_dir, "geo_factor_z_toy.png", frac_pix=weights_partial)


if __name__ == "__main__":
    main()
