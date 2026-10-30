"""Build any named model (envmeshnet + 9 baselines)."""
from __future__ import annotations
def build(name: str, seed: int = 42, **kw):
    """Return unfitted classifier/module for name. Strict imports."""
    from src.models.baselines import SKLEARN_NAMES, TORCH_NAMES, sklearn_baseline, SeqClf
    if name in SKLEARN_NAMES:
        return sklearn_baseline(name, seed)
    if name in TORCH_NAMES:
        return SeqClf(kw.get("d", 71), kw.get("n_cls", 6), name)
    if name == "envmeshnet":
        from src.models.env_meshnet import EnvMeshNet
        return EnvMeshNet(e_dim=kw.get("e_dim", 15), h=kw.get("h", 64),
                          n_blocks=kw.get("n_blocks", 2), drop=kw.get("drop", 0.3))
    raise ValueError(f"unknown model {name}")
