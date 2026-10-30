"""Step 6a: triangular geo-mesh of India via Delaunay.
Each triangle = face with up to 3 neighbour faces (-1 = border).
Face descriptors: spatial (centre lat/lon), structural (corner offsets +
neighbour ids), environmental (mean AOD/NO2/met/wind/fire/rain/lags).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

ENV_COLS = ["aod", "no2", "so2", "co", "bc", "dust", "temp", "rh",
            "wind_u", "wind_v", "blh", "rain", "fire_count", "ndvi", "lst"]

def points(df: pd.DataFrame) -> pd.DataFrame:
    """Unique cell points (cell_id, lat, lon). One row per cell."""
    return df.groupby("cell_id")[["lat", "lon"]].mean().reset_index()

def triangulate(pts: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Delaunay over lon/lat. Returns (simplices Fx3, neighbours Fx3)."""
    from scipy.spatial import Delaunay
    xy = pts[["lon", "lat"]].values
    tri = Delaunay(xy)
    simp = tri.simplices.astype(int)
    neigh = tri.neighbors.astype(int)  # -1 = no neighbour (border)
    return simp, neigh

def face_centres(pts: pd.DataFrame, simp: np.ndarray) -> np.ndarray:
    """Face centre (lat, lon) mean of 3 corners. Shape (F, 2)."""
    xy = pts[["lat", "lon"]].values
    return xy[simp].mean(axis=1)

def struct_feats(pts: pd.DataFrame, simp: np.ndarray, neigh: np.ndarray) -> np.ndarray:
    """Corner offsets (3x2=6) + neighbour ids (3) -> (F, 9). Offsets vs centre."""
    xy = pts[["lon", "lat"]].values
    corners = xy[simp]  # F,3,2
    centre = corners.mean(axis=1, keepdims=True)
    offs = (corners - centre).reshape(len(simp), 6)
    return np.concatenate([offs, neigh.astype(float)], axis=1)

def env_face(df_date: pd.DataFrame, pts: pd.DataFrame, simp: np.ndarray,
             cols: list[str] | None = None) -> np.ndarray:
    """Mean env features over 3 corner cells for one date. (F, E)."""
    cols = cols or [c for c in ENV_COLS if c in df_date.columns]
    lut = df_date.groupby("cell_id")[cols].mean()
    ids = pts["cell_id"].values
    # map point idx -> feature vector (fill 0 if missing date)
    mat = np.zeros((len(ids), len(cols)))
    for i, cid in enumerate(ids):
        if cid in lut.index:
            mat[i] = lut.loc[cid].fillna(0).values
    return mat[simp].mean(axis=1)
