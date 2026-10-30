"""Nationwide India geographic index (Phase 2).

Merges structured reference files in data/geography/ with the existing
station registry and Vellore seed. No model, pipeline, or dataset logic
changes here — lookup only.

Sources:
- data/geography/india_states_uts.csv (28 states + 8 UTs)
- data/geography/india_districts.csv (district centres)
- data/geography/india_localities.csv (cities/towns/villages + Vellore seed)
- data/external/stations.csv (monitoring registry)
- data/external/vellore_places.csv (verified Vellore seed)

All coordinates are reference centres (city/district/state centres from
OpenStreetMap Nominatim or project registry), NOT boundaries or
measurements. Pollution/model availability is decided downstream.
"""
from __future__ import annotations
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEO_DIR = ROOT / "data" / "geography"


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[-,()]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_states() -> list[dict]:
    rows = _read_csv(GEO_DIR / "india_states_uts.csv")
    out = []
    for r in rows:
        try:
            out.append({
                "place_id": r.get("place_id", ""),
                "place": r.get("place", ""),
                "aliases": [a.strip() for a in (r.get("aliases") or "").split("|") if a.strip()],
                "district": r.get("district", ""),
                "state": r.get("state", ""),
                "place_type": r.get("place_type", "state"),
                "lat": float(r["lat"]), "lon": float(r["lon"]),
                "pin_code": r.get("pin_code", ""),
                "parent_id": r.get("parent_id", ""),
                "source": r.get("source", ""),
                "source_date": r.get("source_date", ""),
            })
        except Exception:
            continue
    return out


def _state_centres() -> dict[str, tuple[float, float]]:
    centres: dict[str, tuple[float, float]] = {}
    for r in load_states():
        try:
            centres[_norm(r.get("state", ""))] = (float(r["lat"]), float(r["lon"]))
        except Exception:
            continue
    return centres


def _read_district_name_files() -> list[tuple[str, str]]:
    names: list[tuple[str, str]] = []
    for fname in ["districts_north.csv", "districts_west.csv",
                  "districts_up_bihar.csv", "districts_east.csv",
                  "districts_central.csv", "districts_ka_ap_tg.csv",
                  "districts_tn_kl.csv", "districts_ne.csv",
                  "districts_islands.csv"]:
        for r in _read_csv(GEO_DIR / fname):
            d = (r.get("district") or "").strip()
            s = (r.get("state") or "").strip()
            if d and s:
                names.append((d, s))
    return names


def load_districts() -> list[dict]:
    """District reference: centres where available + name-only entries.

    - india_districts.csv rows have lat/lon (reference centres).
    - districts_*.csv files provide comprehensive district NAMES (all
      states/UTs, Census/LGD names). lat/lon are None intentionally —
      no fake coordinates. Search matches names; assessment needs
      coordinates via offline centres or live geocoding.
    """
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for r in _read_csv(GEO_DIR / "india_districts.csv"):
        d = (r.get("district") or "").strip()
        s = (r.get("state") or "").strip()
        if not d or not s:
            continue
        try:
            lat = float(r["lat"])
            lon = float(r["lon"])
        except Exception:
            continue
        seen.add((_norm(d), _norm(s)))
        out.append({
            "place_id": r.get("district_id") or f"{d}",
            "place": d,
            "aliases": [],
            "district": d,
            "state": s,
            "place_type": "district",
            "lat": lat, "lon": lon,
            "has_coordinates": True,
            "pin_code": "",
            "parent_id": "",
            "source": r.get("source", ""),
            "source_date": r.get("source_date", ""),
        })
    centres = _state_centres()
    for d, s in _read_district_name_files():
        key = (_norm(d), _norm(s))
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "place_id": f"NAME-{s}-{d}",
            "place": d,
            "aliases": [],
            "district": d,
            "state": s,
            "place_type": "district",
            # No fake coordinates: None means "name indexed, coords need geocoding"
            "lat": None, "lon": None,
            "has_coordinates": False,
            "pin_code": "",
            "parent_id": "",
            "source": "Census of India / LGD district names (reference list, no coords)",
            "source_date": "2024-2026",
        })
    return out


def load_localities() -> list[dict]:
    rows = _read_csv(GEO_DIR / "india_localities.csv")
    out = []
    for r in rows:
        try:
            out.append({
                "place_id": r.get("place_id", r.get("place", "")),
                "place": r.get("place", ""),
                "aliases": [a.strip() for a in (r.get("aliases") or "").split("|") if a.strip()],
                "district": r.get("district", ""),
                "state": r.get("state", ""),
                "place_type": r.get("place_type", "locality"),
                "lat": float(r["lat"]), "lon": float(r["lon"]),
                "pin_code": (r.get("pin_code") or "").strip(),
                "parent_id": r.get("parent_id", ""),
                "source": r.get("source", ""),
                "source_date": r.get("source_date", ""),
            })
        except Exception:
            continue
    return out


def load_combined_index() -> list[dict]:
    """Combined offline index: states + districts + localities.

    Duplicate (place, district, state) triples are MERGED (aliases/PIN union,
    coordinates preferred) so city aliases like bombay->Mumbai are never lost
    behind a district stub with the same name.
    """
    merged: dict[tuple[str, str, str], dict] = {}
    states = load_states()
    districts = sorted(load_districts(),
                       key=lambda r: (r.get("lat") is None, r.get("place", "")))
    localities = load_localities()
    for rec in states + districts + localities:
        key = (_norm(rec.get("place", "")), _norm(rec.get("district", "")),
               _norm(rec.get("state", "")))
        if key not in merged:
            merged[key] = {**rec, "aliases": list(rec.get("aliases", []))}
            continue
        prev = merged[key]
        # union aliases
        seen_a = {_norm(a) for a in prev.get("aliases", [])}
        for a in rec.get("aliases", []):
            if _norm(a) not in seen_a:
                prev["aliases"].append(a)
                seen_a.add(_norm(a))
        # prefer coordinates + pin + place_type detail
        if prev.get("lat") is None and rec.get("lat") is not None:
            for k in ("lat", "lon", "place_id", "place_type", "source",
                      "source_date", "parent_id", "has_coordinates"):
                prev[k] = rec.get(k, prev.get(k))
        if not prev.get("pin_code") and rec.get("pin_code"):
            prev["pin_code"] = rec["pin_code"]
    return list(merged.values())


def _score_candidate(nq: str, rec: dict) -> tuple[int, str] | None:
    name = _norm(rec.get("place", ""))
    aliases = [_norm(a) for a in rec.get("aliases", [])]
    pin = (rec.get("pin_code") or "").strip()
    if not nq:
        return None
    if nq == name:
        return 100, "exact"
    if nq in aliases:
        return 95, "alias"
    if pin and nq == pin:
        return 100, "pin"
    if name.startswith(nq) and len(nq) >= 3:
        return 85, "prefix"
    if nq in name and len(nq) >= 3:
        return 78, "partial"
    for a in aliases:
        if nq in a and len(nq) >= 3:
            return 74, "alias-partial"
    return None


def search_offline(query: str, district: str = "", state: str = "",
                    limit: int = 8) -> list[dict]:
    """Offline nationwide search.

    Supports exact, case-insensitive, partial, alias, PIN, plus optional
    district/state filters. Returns ranked candidates with district/state
    so callers can disambiguate duplicates. Never invents coordinates.
    """
    nq = _norm(query)
    if not nq:
        return []
    nd = _norm(district)
    ns = _norm(state)
    # PIN fast path
    results: list[tuple[int, dict]] = []
    for rec in load_combined_index():
        if nd and _norm(rec.get("district", "")) != nd:
            # allow district filter to also match place name (district-as-place)
            if _norm(rec.get("place", "")) != nd:
                continue
        if ns and _norm(rec.get("state", "")) != ns:
            continue
        scored = _score_candidate(nq, rec)
        if scored:
            score, mtype = scored
            results.append((score, {**rec, "match_type": mtype,
                                   "score": score}))
    # fuzzy fallback for typos (offline, no network)
    if len(results) < limit:
        try:
            from rapidfuzz import fuzz as _fuzz
            have_fuzz = True
        except Exception:
            have_fuzz = False
        existing_keys = {(r["place"], r["district"], r["state"]) for _, r in results}
        fuzzy_hits: list[tuple[int, dict]] = []
        for rec in load_combined_index():
            if (rec["place"], rec["district"], rec["state"]) in existing_keys:
                continue
            if nd and _norm(rec.get("district", "")) != nd and \
                    _norm(rec.get("place", "")) != nd:
                continue
            if ns and _norm(rec.get("state", "")) != ns:
                continue
            if have_fuzz:
                s = max([_fuzz.ratio(nq, _norm(rec.get("place", "")))] +
                        [_fuzz.ratio(nq, a) for a in
                         [_norm(x) for x in rec.get("aliases", [])]] or [0])
            else:
                import difflib
                s = int(difflib.SequenceMatcher(
                    None, nq, _norm(rec.get("place", ""))).ratio() * 100)
            if s >= 80:
                fuzzy_hits.append((s, {**rec, "match_type": "fuzzy",
                                       "score": s}))
        fuzzy_hits.sort(key=lambda t: -t[0])
        results.extend(fuzzy_hits[: max(0, limit - len(results))])
    # Rank: entries WITH coordinates first (actionable for assessment),
    # then score desc, then state/district/place for stability.
    def _rank_key(t: tuple[int, dict]) -> tuple:
        score, rec = t
        has_coords = 1 if rec.get("lat") is not None and rec.get("lon") is not None else 0
        return (-has_coords, -score, rec.get("state", ""),
                rec.get("district", ""), rec.get("place", ""))
    results.sort(key=_rank_key)
    # de-duplicate identical display triples, keep best score
    out: list[dict] = []
    seen_keys = set()
    for score, rec in results:
        key = (rec["place"], rec["district"], rec["state"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out.append(rec)
        if len(out) >= limit:
            break
    return out


def district_name_counts() -> dict:
    """Counts for honest coverage reporting (names vs with-coordinates)."""
    districts = load_districts()
    with_coords = sum(1 for r in districts if r.get("lat") is not None)
    per_state: dict[str, int] = {}
    per_state_coords: dict[str, int] = {}
    for r in districts:
        s = r.get("state", "")
        per_state[s] = per_state.get(s, 0) + 1
        if r.get("lat") is not None:
            per_state_coords[s] = per_state_coords.get(s, 0) + 1
    return {
        "district_names_indexed": len(districts),
        "districts_with_coordinates": with_coords,
        "districts_name_only": len(districts) - with_coords,
        "per_state_total": per_state,
        "per_state_with_coordinates": per_state_coords,
    }


def station_coverage() -> dict:
    """Ground-monitoring coverage: stations per state/district from registry."""
    import csv as _csv
    p = ROOT / "data" / "external" / "stations.csv"
    states: set[str] = set()
    districts: set[tuple[str, str]] = set()
    count = 0
    if p.exists():
        with p.open(encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                name = (row.get("name") or "").strip()
                if name.startswith("#") or not name:
                    continue
                count += 1
                if row.get("state"):
                    states.add(row["state"].strip())
                if row.get("district") and row.get("state"):
                    districts.add((row["state"].strip(), row["district"].strip()))
    return {"stations_indexed": count,
            "states_with_stations": len(states),
            "districts_with_stations": len(districts)}


def model_input_coverage() -> dict:
    """Env-MeshNet inference input coverage from synthetic dataset index.

    The web assessment route is not wired to model inference; this reports
    which districts have dataset rows that could feed the pipeline.
    """
    import csv as _csv
    p = ROOT / "data" / "raw" / "synthetic" / "india_dataset.csv"
    districts: set[tuple[str, str]] = set()
    states: set[str] = set()
    if p.exists():
        with p.open(encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                if row.get("state"):
                    states.add(row["state"])
                if row.get("district") and row.get("state"):
                    districts.add((row["state"], row["district"]))
    return {"states_with_dataset_rows": len(states),
            "districts_with_dataset_rows": len(districts)}


def coverage_summary() -> dict:
    states = load_states()
    districts = load_districts()
    localities = load_localities()
    state_names = sorted({r["state"] for r in states if r.get("state")})
    district_keys = sorted({(r["state"], r["district"])
                            for r in districts if r.get("district")})
    with_coords = sum(1 for r in districts if r.get("lat") is not None)
    by_type: dict[str, int] = {}
    for r in localities:
        by_type[r.get("place_type", "locality")] = \
            by_type.get(r.get("place_type", "locality"), 0) + 1
    return {
        "states_uts": len(state_names),
        "state_names": state_names,
        "districts_indexed": len(district_keys),
        "districts_with_coordinates": with_coords,
        "districts_name_only": len(district_keys) - with_coords,
        "localities_indexed": len(localities),
        "locality_by_type": by_type,
        "stations": station_coverage(),
        "model_inputs": model_input_coverage(),
        "sources": sorted({r.get("source", "") for r in
                           states + districts + localities if r.get("source")}),
    }
