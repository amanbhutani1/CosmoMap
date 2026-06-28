# Getting the CAMELS data

This project trains on the [CAMELS](https://camels.readthedocs.io/) (Cosmology and
Astrophysics with MachinE Learning Simulations) suite — specifically the
**IllustrisTNG Latin-Hypercube (LH)** set, which spans 1,000 simulations varying two
cosmological parameters (`Omega_m`, `sigma_8`) and four feedback parameters
(`A_SN1`, `A_AGN1`, `A_SN2`, `A_AGN2`).

## 1. Register and download

CAMELS data products are public, hosted on the
[CAMELS data release](https://camels.readthedocs.io/en/latest/data_access.html)
(Globus / direct download). You need, per redshift snapshot:

- the 3D fields for `Mcdm` (dark-matter density), `Vcdm` (velocity), `T`
  (gas temperature) and `ne` (electron number density);
- the parameter file `CosmoAstroSeed_IllustrisTNG_*.txt`.

## 2. Arrange the files

Place everything under the directory referenced by `data.data_root` in your config
(default `data/camels/`), named so the loader can find them:

```
data/camels/
  CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt   # one row per simulation
  Mcdm_z=0.0.npy        # shape (n_sims, R, R, R), float32
  Vcdm_z=0.0.npy
  T_z=0.0.npy
  ne_z=0.0.npy
  Mcdm_z=0.5.npy
  ...                   # one set per redshift in data.redshifts
```

The loader (`cosmomap/data/camels.py`) also accepts the `samples_3d_<field>_z=<z>.npy`
naming convention. Use `load_camels(config, max_sims=N)` to load a subset while
developing — the full LH set is large and was preloaded into >150 GB of RAM in the
original work.

## 3. Train

```bash
cosmomap-train --config configs/default.yaml
```

> No data yet? Everything also runs on a built-in synthetic generator:
> `cosmomap-train --config configs/synthetic.yaml --synthetic`
