"""CosmoMap: field-level deep learning for baryon prediction on CAMELS.

A 3D FiLM-conditioned, heteroscedastic U-Net that maps dark-matter fields
(density + velocity) to baryonic fields (gas temperature + electron number
density), conditioned on the six CAMELS cosmological/astrophysical parameters,
followed by an X-ray emissivity + lightcone forward model.

MSci dissertation project, University of Nottingham (see ``docs/METHOD.md``).
"""

from cosmomap.config import Config

__version__ = "0.1.0"
__all__ = ["Config", "__version__"]
