"""Render a gallery of result figures from a trained CosmoMap model.

Produces PNGs into an output directory (default ``docs/figures/``) for the README:

* ``learning_curve.png``  - training/validation loss and R^2 vs epoch
* ``fields.png``          - inputs, true/predicted T and n_e, and residuals (a central slice)
* ``uncertainty.png``     - predicted n_e and its aleatoric 1-sigma map
* ``power_spectrum.png``  - true vs predicted 3D power spectra
* ``pred_vs_true.png``    - voxel-wise predicted-vs-true density with R^2
* ``lightcone.png``       - mock X-ray surface-brightness lightcone

Usage::

    python scripts/make_figures.py --config outputs/gpu/config.yaml \
        --model outputs/gpu/final_model.keras --synthetic --out docs/figures
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from tensorflow import keras

import cosmomap.models  # noqa: F401  -- import registers FiLM3D so saved models load
from cosmomap.config import Config
from cosmomap.data import make_synthetic_camels
from cosmomap.data.camels import load_camels
from cosmomap.data.normalization import denormalize_field, normalize_field
from cosmomap.metrics import power_spectrum_3d, r2_score
from cosmomap.xray import LightconeConfig, build_lightcone

plt.switch_backend("Agg")
plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "savefig.bbox": "tight"})

_FIELD_CMAP = {"Mcdm": "viridis", "Vcdm": "cividis", "T": "inferno", "ne": "magma"}


def _mid(vol: np.ndarray) -> np.ndarray:
    """Central slice along the last spatial axis of a 3D volume."""
    return np.asarray(vol)[:, :, vol.shape[2] // 2]


def _show(ax, img, cmap, title):
    lo, hi = np.percentile(img, [2, 98])
    im = ax.imshow(img.T, origin="lower", cmap=cmap, vmin=lo, vmax=hi)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _center_slice(resolution: int, patch: int):
    s = (resolution - patch) // 2
    return (slice(s, s + patch),) * 3


def predict_sample(model, data, config, sim=0, z_index=0):
    """Return (inputs_norm, true_norm, mean_norm, logvar_norm) for one sample."""
    sl = _center_slice(data.resolution, config.data.patch_size)
    idx = sim * len(config.data.redshifts) + z_index
    eps = config.data.log_epsilon

    inputs = np.stack(
        [normalize_field(data.fields[f][idx][sl], f, eps)[0] for f in config.data.fields_in],
        axis=-1,
    )
    true = np.stack(
        [normalize_field(data.fields[f][idx][sl], f, eps)[0] for f in config.data.fields_out],
        axis=-1,
    )
    pred = np.asarray(
        model({"image": inputs[None].astype("float32"), "cosmo": data.cond[idx][None].astype("float32")},
              training=False)
    )[0]
    n = config.n_out
    return inputs, true, pred[..., :n], pred[..., n:]


def fig_learning_curve(csv_path: Path, out: Path):
    if not csv_path.exists():
        return
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return
    ep = [int(r["epoch"]) for r in rows]

    def col(name):
        return [float(r[name]) for r in rows] if name in rows[0] else None

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ep, col("loss"), label="train")
    if col("val_loss"):
        a1.plot(ep, col("val_loss"), label="val")
    a1.set(title="Loss", xlabel="epoch", ylabel="loss")
    a1.legend()
    a1.grid(alpha=0.3)
    if col("r2"):
        a2.plot(ep, col("r2"), label="train")
        if col("val_r2"):
            a2.plot(ep, col("val_r2"), label="val")
        a2.set(title="R$^2$", xlabel="epoch", ylabel="R$^2$")
        a2.legend()
        a2.grid(alpha=0.3)
    fig.suptitle("Training history (RTX 4070, synthetic data)")
    fig.savefig(out / "learning_curve.png")
    plt.close(fig)


def fig_fields(inputs, true, mean, config, out):
    fin, fout = config.data.fields_in, config.data.fields_out
    fig, axes = plt.subplots(3, 3, figsize=(11, 11))
    _show(axes[0, 0], _mid(inputs[..., 0]), _FIELD_CMAP.get(fin[0], "viridis"), f"input: {fin[0]} (DM density)")
    _show(axes[0, 1], _mid(inputs[..., 1]), _FIELD_CMAP.get(fin[1], "cividis"), f"input: {fin[1]} (DM velocity)")
    axes[0, 2].axis("off")
    for r, name in enumerate(fout):
        _show(axes[r + 1, 0], _mid(true[..., r]), _FIELD_CMAP.get(name, "viridis"), f"{name}: truth")
        _show(axes[r + 1, 1], _mid(mean[..., r]), _FIELD_CMAP.get(name, "viridis"), f"{name}: prediction")
        _show(axes[r + 1, 2], _mid(np.abs(true[..., r] - mean[..., r])), "cividis", f"{name}: |residual|")
    fig.suptitle("Inputs, predictions and residuals (central slice, normalised units)")
    fig.savefig(out / "fields.png")
    plt.close(fig)


def fig_uncertainty(mean, logvar, config, out):
    if "ne" not in config.data.fields_out:
        return
    c = config.data.fields_out.index("ne")
    sigma = np.exp(0.5 * logvar[..., c])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))
    _show(a1, _mid(mean[..., c]), "magma", "n$_e$: predicted mean")
    _show(a2, _mid(sigma), "cividis", "n$_e$: aleatoric 1$\\sigma$")
    fig.suptitle("Heteroscedastic uncertainty (the model knows where it is unsure)")
    fig.savefig(out / "uncertainty.png")
    plt.close(fig)


def fig_power_spectrum(true, mean, config, out):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.8, config.n_out))
    for c, name in enumerate(config.data.fields_out):
        k, pk_t = power_spectrum_3d(true[..., c], box_size=config.data.box_size_mpc_h)
        _, pk_p = power_spectrum_3d(mean[..., c], box_size=config.data.box_size_mpc_h)
        m = (k > 0) & (pk_t > 0)
        ax.loglog(k[m], pk_t[m], "-", color=colors[c], label=f"{name} truth")
        ax.loglog(k[m], pk_p[m], "--", color=colors[c], label=f"{name} prediction")
    ax.set(xlabel="k [h/Mpc]", ylabel="P(k)", title="Power spectrum: truth vs prediction")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.savefig(out / "power_spectrum.png")
    plt.close(fig)


def fig_pred_vs_true(true, mean, config, out):
    fig, axes = plt.subplots(1, config.n_out, figsize=(5.2 * config.n_out, 5))
    axes = np.atleast_1d(axes)
    for c, name in enumerate(config.data.fields_out):
        t, p = true[..., c].ravel(), mean[..., c].ravel()
        ax = axes[c]
        ax.hexbin(t, p, gridsize=50, cmap="inferno", bins="log", mincnt=1)
        lim = [min(t.min(), p.min()), max(t.max(), p.max())]
        ax.plot(lim, lim, "w--", lw=1)
        ax.set(xlabel=f"true {name}", ylabel=f"predicted {name}",
               title=f"{name}:  R$^2$ = {r2_score(t, p):.3f}")
    fig.suptitle("Predicted vs true (per voxel, normalised units)")
    fig.savefig(out / "pred_vs_true.png")
    plt.close(fig)


def fig_lightcone(model, data, config, out, sim=0):
    sl = _center_slice(data.resolution, config.data.patch_size)
    eps = config.data.log_epsilon
    ne_vol, t_vol = {}, {}
    for zi, z in enumerate(config.data.redshifts):
        idx = sim * len(config.data.redshifts) + zi
        xin = np.stack([normalize_field(data.fields[f][idx][sl], f, eps)[0]
                        for f in config.data.fields_in], axis=-1)[None].astype("float32")
        pred = np.asarray(model({"image": xin, "cosmo": data.cond[idx][None].astype("float32")},
                                training=False))[0]
        for ci, name in enumerate(config.data.fields_out):
            _, stats = normalize_field(data.fields[name][idx][sl], name, eps)
            phys = denormalize_field(pred[..., ci], stats)
            (ne_vol if name == "ne" else t_vol)[z] = phys
    lc = LightconeConfig(redshifts=config.data.redshifts,
                         box_size_mpc_h=config.data.box_size_mpc_h, npix=192)
    sb = build_lightcone(ne_vol, t_vol, lc)
    sb = np.maximum(sb, sb[sb > 0].min() if np.any(sb > 0) else 1e-30)
    fig, ax = plt.subplots(figsize=(6, 5.4))
    im = ax.imshow(sb, origin="lower", cmap="hot", norm=LogNorm())
    ax.set(title="Mock X-ray surface brightness (predicted lightcone)", xticks=[], yticks=[])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="erg s$^{-1}$ cm$^{-2}$ arcmin$^{-2}$")
    fig.savefig(out / "lightcone.png")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--n-sims", type=int, default=48)
    parser.add_argument("--out", default="docs/figures")
    args = parser.parse_args(argv)

    config = Config.from_yaml(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = keras.models.load_model(args.model, compile=False)
    data = (
        make_synthetic_camels(config, n_sims=args.n_sims, seed=config.train.seed)
        if args.synthetic
        else load_camels(config)
    )

    inputs, true, mean, logvar = predict_sample(model, data, config)
    fig_learning_curve(Path(args.model).parent / "training_log.csv", out)
    fig_fields(inputs, true, mean, config, out)
    fig_uncertainty(mean, logvar, config, out)
    fig_power_spectrum(true, mean, config, out)
    fig_pred_vs_true(true, mean, config, out)
    fig_lightcone(model, data, config, out)
    print(f"Wrote figures to {out.resolve()}:")
    for p in sorted(out.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
