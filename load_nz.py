import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import pyccl as ccl
from clustering_enhancement import ClusteringEnhancement

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

def plot_nz(z, nbar_z, nz_maps, output_dir, enhancement_factor):

    print(nz_maps.shape)
    plt.figure(figsize=(8, 6))
    for i in range(nz_maps.shape[1]):
        plt.plot(z, nz_maps[:, i], color="gray", alpha=0.1)
    plt.plot(z, nbar_z, "k-", linewidth=2, label=r"Global $\bar{n}(z)$")
    plt.xlabel("Redshift z")
    plt.ylabel("n(z)")
    plt.title(f"<sigma_inv_local> / sigma_inv_global: {enhancement_factor:.6f}")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "nz_distribution.png"))
    plt.close()

def plot_w_comparison(result, theta_arcmin, output_dir, enhancement_factor):

    fig_comp, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(8, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )

    ax_top.plot(theta_arcmin, theta_arcmin * result.w_model, "k--", linewidth=2, label=r"$w_{\rm model}$")
    ax_top.plot(theta_arcmin, theta_arcmin * result.w_true, "r-", linewidth=2, label=r"$w_{\rm true}$")
    ax_top.set_ylabel(r"$\theta \cdot w(\theta)$ [arcmin]")
    ax_top.set_xscale("log")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(fontsize=12)
    ax_top.set_title(f"<sigma_inv_local> / sigma_inv_global: {enhancement_factor:.6f}")

    w_abs = np.abs(result.w_model)
    thresh = 0.05 * np.nanmax(w_abs)
    mask = w_abs > thresh
    frac_diff = np.full_like(result.w_model, np.nan)
    frac_diff[mask] = (result.w_true[mask] - result.w_model[mask]) / result.w_model[mask]

    ax_bot.plot(theta_arcmin, frac_diff, "r-", linewidth=1.5)
    ax_bot.set_xlim(theta_arcmin.min(), theta_arcmin.max())
    ax_bot.set_ylabel(r"$\Delta w / w_{\rm model}$")
    ax_bot.set_xlabel(r"$\theta$ [arcmin]")
    ax_bot.set_xscale("log")
    ax_bot.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "w_comparison.png"))
    plt.close()

def main():

    sys_nside = 128
    n_pop_sample = 10_000
    cat_path = '/project/ls-gruen/users/zekang.zhang/proj2_sims/'

    with fits.open(cat_path + f"/sys_preds/sys_{sys_nside}_{n_pop_sample}_nz.fits") as hdul_read:
        z = hdul_read['Z'].data
        dndzs = hdul_read['DNDZS'].data
        dndz_det = hdul_read['DNDZ_DET'].data
        sm_dndzs = hdul_read['SM_DNDZS'].data
        sm_dndz_det = hdul_read['SM_DNDZ_DET'].data
        SEEN = hdul_read['SEEN_IDX'].data

    dndz = dndz_det
    dndzs = dndzs.T
    delta_nz = (dndzs - dndz[:, None])/dndz[:, None]

    sigmas_inv_local = []
    for i in range(dndzs.shape[1]):
        sigma_inv_local = np.trapezoid(dndzs[:,i] ** 2, z)
        sigmas_inv_local.append(sigma_inv_local)
    sigma_inv_local_mean = np.mean(sigmas_inv_local)
    sigma_inv_global = np.trapezoid(dndz ** 2, z)
    enhancement_factor = sigma_inv_local_mean / sigma_inv_global

    plot_nz(z, dndz, dndzs, output_dir, enhancement_factor)

    cosmo = ccl.Cosmology(
        Omega_c=0.25, Omega_b=0.05, h=0.67, sigma8=0.8, n_s=0.965
    )

    ell_max = 2048
    theta_deg = np.logspace(-2, 1.0, 30)

    enhancer = ClusteringEnhancement(cosmo, ell_max=ell_max, ell_min=2)
    result = enhancer.compute_enhancement_from_maps(
            n_maps=dndzs,
            nbar=dndz,
            z=z,
            theta_deg=theta_deg,
            nbar_z=(z, dndz),
            selection_mode="wtheta",
            nside=sys_nside,
            seen_idx=SEEN,
        )

    # Diagnostic: xi_m weighted enhancement
    xi0 = result.xi_m[:, 0]
    var_n_arr = np.mean((dndzs - dndz[:, None]) ** 2, axis=1)
    dz = result.dz
    dw_estim = np.sum(var_n_arr * (dz**2) * xi0)
    w_estim = np.sum((dndz**2) * (dz**2) * xi0)

    print(f"Geometric Enhancement Factor: {enhancement_factor:.6f}")
    print(f"Measured Clust. Enhanc (theta_min): {1.0 + result.delta_w[0] / result.w_model[0]:.6f}")
    print(f"xi_m-weighted Predicted Factor: {1.0 + dw_estim / w_estim:.6f}")

    plot_w_comparison(result, 60.0 * theta_deg, output_dir, enhancement_factor)

if __name__ == "__main__":
    main()
