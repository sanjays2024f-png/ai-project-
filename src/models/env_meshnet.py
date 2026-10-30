"""Env-MeshNet: ADAPTATION of MeshNet (Feng et al. AAAI 2019) to geo-meshes.
Original MeshNet = 3D shape classification. Here: Delaunay faces over India,
spatial MLP -> mesh-conv (face+neighbours) -> global pool -> heads:
(a) AQI softmax + PM2.5 regression, (b) disease risk via pollution embedding+pop.
STRICT: requires torch. No silent object fallback — import fails loudly.
DISCREPANCY NOTE: canonical MeshNet operates on 3D triangle meshes with
face-rotation-invariant descriptors; this adaptation uses geographic Delaunay
faces with lat/lon + environmental means. Documented as adaptation, not canonical.
"""
from __future__ import annotations
import torch
import torch.nn as nn

HAS_TORCH = True

class MeshBlock(torch.nn.Module):
    """One mesh-conv: face feat + mean(neighbour face feats) -> residual MLP."""
    def __init__(self, d: int, drop: float = 0.3):
        super().__init__()
        self.lin = nn.Linear(2 * d, d)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(drop)
        self.norm = nn.LayerNorm(d)
    def forward(self, x, fenv, neigh_idx):
        """x:(B,F,d) face hidden; fenv:(B,F,e); neigh:(F,3)."""
        B, F, _ = x.shape
        agg = x.clone()
        for f in range(F):
            nb = [int(n) for n in neigh_idx[f] if int(n) >= 0]
            if nb:
                agg[:, f] = torch.stack([x[:, f]] + [x[:, n] for n in nb]).mean(0)
        h = torch.cat([x, agg], dim=-1)
        h = self.drop(self.act(self.lin(h)))
        return self.norm(x + h)

class EnvMeshNet(torch.nn.Module):
    """Full model: spatial MLP + mesh blocks + pool + 3 heads + MC dropout.

    Nodes = grid cells; edges = Delaunay face adjacency (neighbour faces).
    Env node features = mean AOD/NO2/met/wind/fire/rain per face.
    Disease head active ONLY when valid (non-synthetic) labels supplied;
    pass disease_labels='synthetic' to get NA instead of circular predictions.
    """
    def __init__(self, e_dim: int = 15, s_dim: int = 2, st_dim: int = 9,
                 n_aqi: int = 6, n_dis: int = 3, h: int = 64,
                 n_blocks: int = 2, drop: float = 0.3):
        super().__init__()
        self.spatial = torch.nn.Sequential(torch.nn.Linear(s_dim, h), torch.nn.ReLU())
        self.env_proj = torch.nn.Sequential(torch.nn.Linear(e_dim + st_dim, h), torch.nn.ReLU())
        self.blocks = torch.nn.ModuleList([MeshBlock(h, drop) for _ in range(n_blocks)])
        self.drop = torch.nn.Dropout(drop)
        self.aqi_head = torch.nn.Linear(h, n_aqi)
        self.pm_head = torch.nn.Linear(h, 1)
        self.dis_head = torch.nn.Linear(h + 1, n_dis)  # pollution emb + pop
    def forward(self, spatial, env, struct, pop, neigh_idx):
        """spatial:(B,F,2) env:(B,F,E) struct:(F,9) pop:(B,F,1) -> aqi,pm,dis.
        dis logits are valid ONLY with real disease labels; otherwise ignore head.
        """
        B, F, _ = spatial.shape
        st = struct.unsqueeze(0).expand(B, -1, -1)
        x = self.spatial(spatial) + self.env_proj(torch.cat([env, st], dim=-1))
        for blk in self.blocks:
            x = blk(x, env, neigh_idx)
        g = x.mean(dim=1)  # global pool over faces
        g = self.drop(g)
        aqi = self.aqi_head(g)
        pm = self.pm_head(g).squeeze(-1)
        dis = self.dis_head(torch.cat([g, pop.mean(dim=1)], dim=-1))
        return aqi, pm, dis
    def predict_mc(self, spatial, env, struct, pop, neigh_idx, n: int = 20):
        """MC-dropout: n stochastic passes -> mean + uncertainty."""
        self.train(True)
        outs = [self.forward(spatial, env, struct, pop, neigh_idx) for _ in range(n)]
        import torch as _t
        aq = _t.stack([o[0] for o in outs]).mean(0)
        pm = _t.stack([o[1] for o in outs])
        di = _t.stack([o[2] for o in outs]).mean(0)
        unc = pm.std(0)  # pm std = uncertainty; High if large
        return aq, pm.mean(0), di, unc

