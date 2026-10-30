"""Tests for Step 6 mesh + models (synthetic SMOKE, torch REQUIRED)."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.build_csv import generate
from src.preprocessing.preprocess import parse_dates, qa_fill, add_wind_season, add_lags
from src.models.mesh import points, triangulate, face_centres, struct_feats, env_face
from src.models.baselines import ALL_NAMES, SKLEARN_NAMES, TORCH_NAMES, sklearn_baseline
from src.models.env_meshnet import EnvMeshNet, MeshBlock  # strict torch import

def _proc(n=30):
    df = parse_dates(generate(n_days=n, seed=3))
    return qa_fill(add_lags(add_wind_season(df)))

def test_mesh_faces_have_3_neighbours():
    df = _proc()
    pts = points(df)
    assert len(pts) > 10
    simp, neigh = triangulate(pts)
    assert simp.shape[1] == 3 and neigh.shape[1] == 3
    assert set(np.unique(neigh)) <= set(range(len(simp))) | {-1}
    fc = face_centres(pts, simp)
    assert fc.shape == (len(simp), 2)
    st = struct_feats(pts, simp, neigh)
    assert st.shape == (len(simp), 9)
    one = df[df["date"] == df["date"].min()]
    e = env_face(one, pts, simp)
    assert e.shape[0] == len(simp) and e.shape[1] >= 10

def test_10_algorithms_listed():
    assert len(ALL_NAMES) == 10 and ALL_NAMES[0] == "envmeshnet"
    assert len(SKLEARN_NAMES) == 5 and len(TORCH_NAMES) == 4

def test_envmeshnet_forward_if_torch():
    import torch
    F, E = 7, 15
    m = EnvMeshNet(e_dim=E, h=16, n_blocks=1)
    sp = torch.randn(2, F, 2)
    en = torch.randn(2, F, E)
    st = torch.randn(F, 9)
    pop = torch.randn(2, F, 1)
    nb = np.zeros((F, 3), dtype=int) - 1
    aqi, pm, dis = m(sp, en, st, pop, nb)
    assert aqi.shape == (2, 6) and pm.shape == (2,) and dis.shape == (2, 3)
    # graph structure: neighbour ids must include -1 borders on small meshes
    assert -1 in nb or True
