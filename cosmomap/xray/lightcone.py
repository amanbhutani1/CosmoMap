"""Assemble predicted (n_e, T) volumes into a mock X-ray surface-brightness
lightcone and write it to a WCS-tagged FITS image.

This realises the dissertation's lightcone stage (§3.5.2–3.5.3): per redshift it
computes the bremsstrahlung emissivity, projects it to a surface-brightness map
with (1+z)^-4 dimming, maps the comoving box to an angular footprint via the
flat-ΛCDM angular-diameter distance D_A(z), trilinearly interpolates between
redshift slices, and applies optional PSF blurring, background, and a flux limit.
``astropy`` provides the cosmological distances and the FITS/WCS machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cosmomap.xray.emissivity import compute_emissivity, surface_brightness


@dataclass
class LightconeConfig:
    """Geometry, cosmology and instrument settings for the mock lightcone."""

    redshifts: list[float] = field(default_factory=lambda: [0.0, 0.5, 1.0, 1.5, 2.0])
    box_size_mpc_h: float = 25.0
    h: float = 0.6711
    omega_m: float = 0.3
    field_of_view_deg: float = 1.0
    npix: int = 256
    n_interp_slices: int = 0  # extra trilinear-interpolated redshift planes between slices
    psf_fwhm_arcmin: float = 0.0  # 0 disables PSF convolution
    background_level: float = 0.0  # additive background (Eq 25)
    flux_limit: float = 0.0  # zero pixels below this surface brightness (Eq 23)


def _cosmology(config: LightconeConfig):
    from astropy.cosmology import FlatLambdaCDM

    return FlatLambdaCDM(H0=config.h * 100.0, Om0=config.omega_m)


def comoving_distance(z: float, config: LightconeConfig) -> float:
    """Line-of-sight comoving distance χ(z) [Mpc] (Eq 20)."""
    return float(_cosmology(config).comoving_distance(z).value)


def angular_diameter_distance(z: float, config: LightconeConfig) -> float:
    """Angular-diameter distance D_A(z) = χ/(1+z) [Mpc] (Eq 21)."""
    return float(_cosmology(config).angular_diameter_distance(z).value)


def _resample(image: np.ndarray, npix: int) -> np.ndarray:
    from scipy.ndimage import zoom

    if image.shape == (npix, npix):
        return image
    return zoom(image, (npix / image.shape[0], npix / image.shape[1]), order=1)


def _angular_footprint_fraction(z: float, config: LightconeConfig) -> float:
    """Fraction of the FoV subtended by the comoving box at redshift z (Eq 22).

    θ = r_perp / D_A, with r_perp the proper transverse box size. At z=0 the
    footprint is clamped to the full field of view.
    """
    if z <= 0:
        return 1.0
    d_a = angular_diameter_distance(z, config)  # Mpc
    proper_box = (config.box_size_mpc_h / config.h) / (1.0 + z)  # proper Mpc
    theta_deg = np.degrees(proper_box / max(d_a, 1e-6))
    return float(min(theta_deg / config.field_of_view_deg, 1.0))


def _project_to_fov(ne: np.ndarray, temperature: np.ndarray, z: float, config: LightconeConfig):
    """Project one (n_e, T) volume to a surface-brightness map on the FoV grid."""
    emissivity = compute_emissivity(ne, temperature)
    voxel_depth = config.box_size_mpc_h / ne.shape[-1]
    sb = surface_brightness(emissivity, axis=-1, redshift=z, voxel_depth=voxel_depth)
    # Map the box's angular footprint into the common FoV grid.
    frac = _angular_footprint_fraction(z, config)
    sub = max(int(round(config.npix * frac)), 1)
    sb = _resample(sb, sub)
    canvas = np.zeros((config.npix, config.npix), dtype=np.float64)
    off = (config.npix - sub) // 2
    canvas[off : off + sub, off : off + sub] = sb
    return canvas


def _interpolate_slice(slices: dict[float, np.ndarray], zs: list[float], z: float) -> np.ndarray:
    """Trilinear (linear-in-z) interpolation between adjacent projected slices."""
    if z <= zs[0]:
        return slices[zs[0]]
    if z >= zs[-1]:
        return slices[zs[-1]]
    hi = next(i for i, zz in enumerate(zs) if zz >= z)
    z0, z1 = zs[hi - 1], zs[hi]
    w = (z - z0) / (z1 - z0)
    return (1 - w) * slices[z0] + w * slices[z1]


def _apply_psf(image: np.ndarray, config: LightconeConfig) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    pix_arcmin = (config.field_of_view_deg * 60.0) / config.npix
    sigma_pix = (config.psf_fwhm_arcmin / pix_arcmin) / 2.355  # FWHM -> sigma
    return gaussian_filter(image, sigma_pix) if sigma_pix > 0 else image


def build_lightcone(
    volumes_ne: dict[float, np.ndarray],
    volumes_temperature: dict[float, np.ndarray],
    config: LightconeConfig,
) -> np.ndarray:
    """Stack per-redshift (n_e, T) volumes into one surface-brightness map."""
    zs = sorted(z for z in config.redshifts if z in volumes_ne and z in volumes_temperature)
    if not zs:
        raise ValueError("no matching redshift volumes were provided")
    slices = {z: _project_to_fov(volumes_ne[z], volumes_temperature[z], z, config) for z in zs}

    if config.n_interp_slices > 0 and len(zs) > 1:
        z_samples = np.linspace(zs[0], zs[-1], (len(zs) - 1) * (config.n_interp_slices + 1) + 1)
    else:
        z_samples = np.asarray(zs, dtype=np.float64)

    total = np.zeros((config.npix, config.npix), dtype=np.float64)
    for z in z_samples:
        total += _interpolate_slice(slices, zs, float(z))

    if config.psf_fwhm_arcmin > 0:
        total = _apply_psf(total, config)
    total = total + config.background_level
    if config.flux_limit > 0:
        total = np.where(total >= config.flux_limit, total, 0.0)
    return total


def _build_wcs(config: LightconeConfig):
    from astropy.wcs import WCS

    scale = config.field_of_view_deg / config.npix  # deg/pixel
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [config.npix / 2, config.npix / 2]
    wcs.wcs.cdelt = [-scale, scale]
    wcs.wcs.crval = [0.0, 0.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


def save_fits(sb_map: np.ndarray, path: str | Path, config: LightconeConfig) -> None:
    """Write a surface-brightness map to FITS with a tangent-plane WCS header."""
    from astropy.io import fits

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = _build_wcs(config).to_header()
    header["BUNIT"] = "erg/s/cm^2/arcmin^2"
    header["CONTENT"] = "CosmoMap mock X-ray surface brightness"
    fits.PrimaryHDU(data=np.asarray(sb_map, dtype=np.float32), header=header).writeto(
        path, overwrite=True
    )
