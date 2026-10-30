"""Step 2a: NASA downloader for whole India via earthaccess.
Reads .env (EARTHDATA_USERNAME/PASSWORD, FIRMS_MAP_KEY). No hard-coded creds.
Real files -> data/raw/nasa/. Synthetic generator output lives in data/raw/synthetic/. Use --dry-run without login to preview queries.
Docs: https://earthaccess.readthedocs.io
Login: https://urs.earthdata.nasa.gov | Search: https://search.earthdata.nasa.gov
NOTE: TEMPO is North-America-only -> NOT used for India.
"""
from __future__ import annotations
import argparse, os, sys
from dataclasses import dataclass
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.require import need
try:
    from dotenv import load_dotenv
except ImportError:  # --dry-run only; real download needs python-dotenv
    def load_dotenv(*a: object, **k: object) -> bool:  # type: ignore
        print("WARNING: python-dotenv missing -> .env not loaded (dry-run OK; real download needs it)")
        return False


BBOX = (68.0, 6.0, 97.5, 37.5)  # W,S,E,N whole India
@dataclass
class DS:
    """One NASA dataset entry."""
    key: str; short: str; ver: str; note: str
# short names map to search.earthdata concepts; versions pinned per spec.
DATASETS: list[DS] = [
 DS("maiac_aod","MCD19A2","061","MAIAC AOD 1km; Himalaya snow/cloud gaps"),
 DS("merra_aer","M2T1NXAER","5.12.4","MERRA-2 aerosol; BC/dust/wind fill"),
 DS("modis_db","MOD04_L2","","DeepBlue AOD; Thar bright-surface fill"),
 DS("omi_no2","OMNO2d","003","OMI NO2/SO2/O3; mask row-anomaly via QA flags"),
 DS("mopitt_co","MOPITT02J","9","MOPITT CO total column"),
 DS("merra_met","M2T1NXSLV","5.12.4","MERRA-2 met: T,RH,U/V,BLH"),
 DS("gpm_rain","GPM_3IMERGDF","07","GPM IMERG daily rain; monsoon cloud fill"),
 DS("modis_ndvi","MOD13A2","061","NDVI 1km; temporal composite in monsoon"),
 DS("modis_lst","MOD11A1","061","LST daily 1km"),
 DS("nasa_dem","NASADEM_HGT","001","elevation static"),
]
def _login(dry: bool) -> bool:
    """Login with earthaccess unless dry-run. Requires earthaccess for real runs."""
    if dry: return False
    need("earthaccess")
    load_dotenv()
    u, p = os.getenv("EARTHDATA_USERNAME"), os.getenv("EARTHDATA_PASSWORD")
    if not u or not p: raise RuntimeError("Set EARTHDATA_* in .env (see .env.example)")
    import earthaccess  # import here so --dry-run needs no deps
    earthaccess.login(strategy="environment")
    return True
def plan(start: str, end: str) -> list[str]:
    """Human-readable query plan for README/logs."""
    w,s,e,n = BBOX
    return [f"{d.key}:{d.short} v{d.ver or '?'} bbox={w},{s},{e},{n} {start}..{end} # {d.note}" for d in DATASETS]
def search_download(start: str, end: str, outdir: Path, dry: bool) -> list[Path]:
    """Search+download each dataset via earthaccess. Returns local files."""
    files: list[Path] = []
    if dry:
        for line in plan(start, end): print("DRY:", line)
        return files
    import earthaccess
    _login(False)
    w,s,e,n = BBOX
    outdir.mkdir(parents=True, exist_ok=True)
    for d in DATASETS:
        kw: dict = dict(short_name=d.short, bounding_box=(w,s,e,n),
                        temporal=(start, end))
        if d.ver: kw["version"] = d.ver
        try:
            res = earthaccess.search_data(**kw)
        except Exception as ex:
            print(f"SKIP {d.key}: search failed ({ex})"); continue
        print(f"{d.key}: {len(res)} granules")
        try:
            got = earthaccess.download(res, outdir=str(outdir / d.key))
            for g in got: files.append(Path(g))
        except Exception as ex:
            print(f"SKIP {d.key}: download failed ({ex})")
    return files
def firms_fires(day: str, out: Path, key: str = "") -> Path:
    """FIRMS VIIRS NRT fires over India bbox. Needs FIRMS_MAP_KEY."""
    import requests
    key = key or os.getenv("FIRMS_MAP_KEY", "")
    if not key: raise RuntimeError("Set FIRMS_MAP_KEY in .env")
    w,s,e,n = BBOX
    url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/VIIRS_SNPP_NRT/"
           f"{w},{s},{e},{n}/1/{day}")
    r = requests.get(url, timeout=60); r.raise_for_status()
    r.encoding = "utf-8"  # force UTF-8 for FIRMS CSV (prevents mojibake in fire counts)
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(r.text, encoding="utf-8")
    print(f"FIRMS {day}: {len(r.text.splitlines())-1} hotspots -> {out}")
    return out
def main() -> None:
    """CLI: python src/data/download_data.py --start 2024-08-01 --end 2024-08-05 --dry-run"""
    ap = argparse.ArgumentParser(description="Download NASA data for India")
    ap.add_argument("--start", default="2024-08-01"); ap.add_argument("--end", default="2024-08-05")
    ap.add_argument("--out", default="data/raw/nasa"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    load_dotenv()
    if a.dry_run:
        search_download(a.start, a.end, Path(a.out), dry=True)
    else:
        search_download(a.start, a.end, Path(a.out), dry=False)
if __name__ == "__main__":
    main()

