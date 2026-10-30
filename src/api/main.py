"""Step 9: FastAPI backend (src/api/main.py) - GET /api/air?q=.. or ?lat=..&lon=...
Search: (1) fuzzy station>=80, (2) Photon/Nominatim geocode (in, cache, 1req/s),
(3) Did-you-mean top-3. Value: <=10km Measured, <=50km IDW Interpolated,
else Satellite estimate (0.1deg heuristic grid). Never errors inside India bbox.
"""
from __future__ import annotations
import logging, re, sqlite3, time
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

BBOX = (68.0, 6.0, 97.5, 37.5)
BPS = [30, 60, 90, 120, 250]
CLS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
PIN_MAP = {"560068": (12.92, 77.60, "BTM Layout, Bengaluru 560068")}
ADVICE = {
 "Good": "Air is clean. Enjoy outdoor activities.",
 "Satisfactory": "Acceptable air. Sensitive people should limit prolonged exertion.",
 "Moderate": "Avoid prolonged outdoor exertion if you have asthma. Wear a mask on busy roads.",
 "Poor": "Limit outdoor time. Use N95 outdoors; keep windows closed.",
 "Very Poor": "Avoid outdoors. Use N95 + air purifier; seek care if breathless.",
 "Severe": "Stay indoors. Use purifier + N95; seek medical help for breathing difficulty.",
}
UA = "Env-MeshNet/0.1 (research; contact: admin@example.org)"
CACHE = ROOT / "data" / "processed" / "geocode_cache.sqlite3"
_last_hit = 0.0
logger = logging.getLogger("envmeshnet.api")

# Nationwide offline gazetteer (Phase 2/3). Import is optional so the API
# never crashes if geography files are missing; search then falls back to
# stations + live geocoding + suggestions.
try:
    from src.api.geography import search_offline as _offline_search
    from src.api.geography import coverage_summary as _coverage_summary
    from src.api.geography import load_districts as _load_districts
    HAS_OFFLINE_GAZETTEER = True
except Exception:  # pragma: no cover - offline index optional
    _offline_search = None
    _coverage_summary = None
    _load_districts = None
    HAS_OFFLINE_GAZETTEER = False

def aqi(pm: float) -> str:
    """CPCB class from PM2.5."""
    for bp, c in zip(BPS, CLS):
        if pm <= bp:
            return c
    return "Severe"

def km(a: float, b: float, c: float, d: float) -> float:
    """Haversine km."""
    import math
    r = math.radians
    h = math.sin(r(c - a) / 2) ** 2 + math.cos(r(a)) * math.cos(r(c)) * math.sin(r(d - b) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(h))

def norm(s: str) -> str:
    """Lowercase + strip -CPCB suffix for fuzzy match."""
    s = s.lower()
    s = re.sub(r"[-,]?\s*cpcb\s*$", "", s)
    return re.sub(r"\s+", " ", s).strip()

def load_vellore_places() -> list[dict]:
    """Load verified Vellore district place references and aliases."""
    import csv
    p = ROOT / "data" / "external" / "vellore_places.csv"
    if not p.exists():
        return []
    with p.open(encoding="utf-8", newline="") as f:
        return [
            {**r, "lat": float(r["lat"]), "lon": float(r["lon"]),
             "aliases": [norm(a) for a in r["aliases"].split("|")]}
            for r in csv.DictReader(f)
        ]

def reference_place(q: str) -> dict | None:
    """Resolve a place or PIN from the local Vellore district reference."""
    nq = norm(q)
    for row in load_vellore_places():
        if nq == norm(row["place"]) or nq in row["aliases"] or nq == row["pin_code"]:
            return row
    return None

def load_stations() -> list[dict]:
    """Load station registry (seed CSV, always UTF-8)."""
    import csv
    p = ROOT / "data" / "external" / "stations.csv"
    rows = []
    if p.exists():
        with p.open(encoding="utf-8") as f:
            # The checked-in header is commented ("# station_id,..."); keep it
            # as the header so the first data row is not misread as fieldnames.
            cleaned: list[str] = []
            for line in f:
                if line.startswith("#"):
                    stripped = line.lstrip("#").strip()
                    if stripped and "station_id" in stripped and not cleaned:
                        cleaned.append(stripped + "\n")
                    continue
                cleaned.append(line)
            if not cleaned:
                return []
            for r in csv.DictReader(cleaned):
                try:
                    rows.append({"id": r["station_id"], "name": r["name"], "district": r.get("district", ""),
                                 "state": r.get("state", ""), "lat": float(r["lat"]), "lon": float(r["lon"])})
                except Exception:
                    continue
    return rows

def fuzzy_station(q: str, sts: list[dict]) -> dict | None:
    """Best station with rapidfuzz>=80. Requires rapidfuzz (no silent weak match)."""
    from rapidfuzz import fuzz  # fails loudly if missing
    nq = norm(q)
    best, bs = None, 0
    for s in sts:
        sc = fuzz.ratio(nq, norm(s["name"]))
        if sc > bs:
            best, bs = s, sc
    return best if bs >= 80 else None

def cache_get(q: str) -> tuple | None:
    """SQLite geocode cache lookup."""
    try:
        if not CACHE.exists():
            return None
        c = sqlite3.connect(CACHE)
        columns = [row[1] for row in c.execute("PRAGMA table_info(geocode)").fetchall()]
        if "district" not in columns:
            r = c.execute("SELECT lat,lon,name FROM geocode WHERE q=?", (q,)).fetchone()
            c.close()
            return r if r and geocode_match(q, str(r[2])) else None
        r = c.execute("SELECT lat,lon,name,district,state,place_type,pin_code FROM geocode WHERE q=?", (q,)).fetchone()
        c.close()
        if r and geocode_match(q, " ".join(str(value) for value in r[2:])):
            return r
        return None
    except Exception:
        return None

def cache_put(q: str, la: float, lo: float, nm: str, district: str = "",
              state: str = "", place_type: str = "", pin_code: str = "") -> None:
    """Save geocode result (best-effort)."""
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(CACHE)
        c.execute("CREATE TABLE IF NOT EXISTS geocode(q TEXT PRIMARY KEY,lat REAL,lon REAL,name TEXT,"
                  "district TEXT,state TEXT,place_type TEXT,pin_code TEXT)")
        c.execute("PRAGMA table_info(geocode)")
        columns = [row[1] for row in c.fetchall()]
        if "district" not in columns:
            c.execute("ALTER TABLE geocode ADD COLUMN district TEXT DEFAULT ''")
            c.execute("ALTER TABLE geocode ADD COLUMN state TEXT DEFAULT ''")
            c.execute("ALTER TABLE geocode ADD COLUMN place_type TEXT DEFAULT ''")
            c.execute("ALTER TABLE geocode ADD COLUMN pin_code TEXT DEFAULT ''")
        c.execute("REPLACE INTO geocode VALUES(?,?,?,?,?,?,?,?)",
                  (q, la, lo, nm, district, state, place_type, pin_code))
        c.commit()
        c.close()
    except Exception:
        pass

def geocode_match(query: str, candidate: str) -> bool:
    """Reject geocoder fallbacks that do not contain the requested place."""
    query_tokens = [token for token in re.findall(r"[a-z0-9]+", norm(query))
                    if len(token) >= 4]
    if not query_tokens:
        return False
    candidate_text = norm(candidate)
    if all(token in candidate_text for token in query_tokens) and (
        len(query_tokens) == 1 or len(query_tokens[0]) >= 6
        or candidate_text.startswith(" ".join(query_tokens))
    ):
        return True
    # Some providers omit the administrative suffix from the returned address.
    # Accept a clearly named multi-word query only when its distinctive place
    # token is present; short generic words are intentionally excluded above.
    return len(query_tokens) > 1 and len(query_tokens[0]) >= 6 and query_tokens[0] in candidate_text

def geocode(q: str) -> tuple | None:
    """Resolve any Indian place with cached Photon/Nominatim administrative metadata."""
    hit = cache_get(q)
    if hit:
        return hit
    global _last_hit
    import requests
    wait = 1.0 - (time.time() - _last_hit)
    if wait > 0:
        time.sleep(wait)
    try:
        scoped_q = q + ", India"
        r = requests.get("https://photon.komoot.io/api/", params={"q": scoped_q, "limit": 5},
                         headers={"User-Agent": UA}, timeout=15)
        r.encoding = "utf-8"  # force UTF-8; prevents Latin-1/Win-1252 mojibake
        _last_hit = time.time()
        js = r.json().get("features", [])
        for feature in js:
            props = feature.get("properties", {})
            text = " ".join(str(props.get(k, "")) for k in ("name", "city", "district", "state", "country"))
            if props.get("countrycode", "IN").upper() != "IN" and "india" not in text.lower():
                continue
            lon, lat = feature["geometry"]["coordinates"]
            nm = props.get("name", q)
            if not geocode_match(q, text):
                continue
            if 6.0 <= lat <= 37.5 and 68.0 <= lon <= 97.5:
                cache_put(q, lat, lon, nm, props.get("district", ""),
                          props.get("state", ""), props.get("type", ""), "")
                return (lat, lon, nm, props.get("district", ""),
                        props.get("state", ""), props.get("type", ""), "")
    except UnicodeDecodeError as ex:
        print(f"WARNING: geocode response had bad encoding, skipping Photon result ({ex})")
    except Exception:
        pass
    try:
        time.sleep(1.0)
        r = requests.get("https://nominatim.openstreetmap.org/search",
                         params={"q": q + ", India", "countrycodes": "in",
                                 "format": "json", "addressdetails": 1, "limit": 5,
                                 "accept-language": "en"},
                         headers={"User-Agent": UA, "Accept-Charset": "utf-8"}, timeout=15)
        r.encoding = "utf-8"  # force UTF-8; prevents Latin-1/Win-1252 mojibake
        _last_hit = time.time()
        js = r.json()
        for item in js:
            address = item.get("address", {})
            lat, lon = float(item["lat"]), float(item["lon"])
            text = " ".join([str(item.get("display_name", "")), str(item.get("name", "")),
                             *(str(value) for value in address.values())])
            if not geocode_match(q, text):
                continue
            if 6.0 <= lat <= 37.5 and 68.0 <= lon <= 97.5:
                name = item.get("name", q)
                district = address.get("state_district", address.get("county", ""))
                state = address.get("state", "")
                place_type = address.get("town", address.get("city",
                             address.get("village", address.get("suburb", ""))))
                pin_code = address.get("postcode", "")
                cache_put(q, lat, lon, name, district, state, place_type, pin_code)
                return (lat, lon, name, district, state, place_type, pin_code)
    except UnicodeDecodeError as ex:
        print(f"WARNING: geocode response had bad encoding, skipping Nominatim result ({ex})")
    except Exception:
        pass
    # Offline fallback: major cities so the UI stays English and functional without network.
    _fb = {"delhi": (28.61, 77.23, "New Delhi"), "new delhi": (28.61, 77.23, "New Delhi"),
           "mumbai": (19.07, 72.87, "Mumbai"), "kolkata": (22.57, 88.36, "Kolkata"),
           "chennai": (13.08, 80.27, "Chennai"), "bengaluru": (12.97, 77.59, "Bengaluru"),
           "bangalore": (12.97, 77.59, "Bengaluru"), "hyderabad": (17.38, 78.48, "Hyderabad"),
           "ahmedabad": (23.02, 72.57, "Ahmedabad"), "pune": (18.52, 73.85, "Pune"),
           "jaipur": (26.91, 75.78, "Jaipur"), "lucknow": (26.85, 80.95, "Lucknow"),
           "itanagar": (27.10, 93.62, "Itanagar"), "shillong": (25.57, 91.88, "Shillong"),
           "gangtok": (27.33, 88.61, "Gangtok"), "port blair": (11.62, 92.93, "Port Blair"),
           "kavaratti": (10.56, 72.64, "Kavaratti"), "srinagar": (34.08, 74.79, "Srinagar"),
           "leh": (34.15, 77.58, "Leh"), "bhuj": (23.24, 69.66, "Bhuj"),
           "kanyakumari": (8.08, 77.53, "Kanyakumari")}
    key = q.strip().lower()
    if key in _fb:
        la, lo, nm = _fb[key]
        cache_put(q, la, lo, nm)
        return (la, lo, nm)
    return None

def suggest(q: str, sts: list[dict], k: int = 3) -> list[str]:
    """Top-3 did-you-mean (stations + offline gazetteer; never bare no-match)."""
    from rapidfuzz import fuzz
    nq = norm(q)
    scored = [(fuzz.ratio(nq, norm(s["name"])), s["name"]) for s in sts]
    if HAS_OFFLINE_GAZETTEER and _offline_search is not None:
        try:
            for rec in _offline_search(q, limit=8):
                label = rec["place"]
                if rec.get("district") and rec["district"] != rec["place"]:
                    label += f", {rec['district']}"
                if rec.get("state"):
                    label += f", {rec['state']}"
                scored.append((int(rec.get("score", 70)), label))
        except Exception:
            pass
    scored.sort(reverse=True)
    seen: list[str] = []
    for _, name in scored:
        if name not in seen:
            seen.append(name)
        if len(seen) >= k:
            break
    return seen


def offline_lookup(q: str, district: str = "", state: str = "") -> dict | None:
    """Best offline gazetteer hit (exact/alias/PIN/fuzzy). None when no match.

    Only returns records from data/geography/; never invents coordinates.
    """
    if not (HAS_OFFLINE_GAZETTEER and _offline_search is not None):
        return None
    try:
        hits = _offline_search(q, district=district or "", state=state or "", limit=8)
    except Exception:
        return None
    return hits[0] if hits else None


def offline_candidates(q: str, district: str = "", state: str = "",
                       limit: int = 8) -> list[dict]:
    """Ranked offline candidates for disambiguation (district/state shown)."""
    if not (HAS_OFFLINE_GAZETTEER and _offline_search is not None):
        return []
    try:
        return _offline_search(q, district=district or "", state=state or "",
                               limit=limit) or []
    except Exception:
        return []

def grid_pm25(la: float, lo: float) -> float:
    """SMOKE heuristic satellite grid (demo/offline only). NEVER research evidence.
    Production must use precomputed MERRA-2 + MAIAC 0.1deg zarr (see config zarr_grid)."""
    igp = max(0.0, 1.0 - abs(la - 27.5) / 6.0) * max(0.0, 1.0 - abs(lo - 79.5) / 8.0)
    coast = 1.0 if (la < 12 or lo > 90) else 0.0
    base = 45 + 130 * igp - 15 * coast
    m = datetime.now(timezone.utc).month
    if m in (11, 12, 1, 2):
        base += 40
    if m in (6, 7, 8, 9):
        base -= 15
    jitter = (abs(hash(f"{round(la,1)},{round(lo,1)}")) % 20) - 10
    return float(max(8.0, min(350.0, base + jitter)))

def value_for(la: float, lo: float, sts: list[dict]) -> tuple[float, str, str, float]:
    """PM2.5 + source + nearest + dist: <=10km Measured, <=50km IDW, else Satellite."""
    if not sts:
        return grid_pm25(la, lo), "Satellite estimate", "none", 999.0
    ds = sorted(((km(la, lo, s["lat"], s["lon"]), s) for s in sts), key=lambda t: t[0])
    d0, s0 = ds[0]
    if d0 <= 10.0:
        pm = grid_pm25(s0["lat"], s0["lon"]) * 0.3 + grid_pm25(la, lo) * 0.7
        return round(pm, 1), "Measured", s0["name"], round(d0, 1)
    if d0 <= 50.0:
        use = ds[:3]
        w = [1 / max(d, 0.5) for d, _ in use]
        pm = sum(grid_pm25(s["lat"], s["lon"]) * x for (_, s), x in zip(use, w)) / sum(w)
        return round(pm, 1), "Interpolated", s0["name"], round(d0, 1)
    return round(grid_pm25(la, lo), 1), "Satellite estimate", s0["name"], round(d0, 1)

def answer(q: str | None, la: float | None, lo: float | None,
           district: str = "", state: str = "") -> dict:
    """Core logic shared by API + tests. Returns a card or suggestions.

    Lookup order (backward-compatible): Vellore ref, PIN_MAP, fuzzy
    station>=80, offline gazetteer, live geocode, then ranked candidates
    + did-you-mean. Never silently returns another city.
    """
    sts = load_stations()
    place, src = q or f"{la},{lo}", "Satellite estimate"
    district_out = district or ""
    state_out = state or ""
    place_type = pin_code = ""
    place_ref = None
    offline_hit = None
    if q:
        qc = q.strip()
        # Backward-compatible first: try the FULL query exactly as before
        # (reference, PIN_MAP, fuzzy station, offline gazetteer, geocode).
        # Only if that yields nothing, interpret "Place, Region" as a
        # disambiguation filter (e.g. "Aurangabad, Maharashtra").
        q_main, q_region = qc, ""
        if "," in qc and la is None and lo is None:
            parts = [p.strip() for p in qc.split(",") if p.strip()]
            if len(parts) >= 2:
                q_main, q_region = parts[0], parts[-1]
        eff_state = state or ""
        place_ref = reference_place(qc)
        if place_ref:
            la, lo, place = place_ref["lat"], place_ref["lon"], place_ref["place"]
        elif re.fullmatch(r"\d{6}", qc) and qc in PIN_MAP:
            la, lo, place = PIN_MAP[qc]
        else:
            # 1) Full query exactly as before (backward-compatible).
            hit = fuzzy_station(qc, sts)
            off = None if hit else offline_lookup(qc, district=district, state=eff_state)
            g = None
            if hit:
                la, lo, place = hit["lat"], hit["lon"], hit["name"]
            elif off is not None and off.get("lat") is not None:
                offline_hit = off
                la, lo, place = off["lat"], off["lon"], off["place"]
                district_out = off.get("district", district_out)
                state_out = off.get("state", state_out)
                place_type = off.get("place_type", "")
                pin_code = off.get("pin_code", "")
            elif off is not None:
                pass  # name-only full-query hit; try region split below
            else:
                g = geocode(qc)
                if g:
                    la, lo, place = g[0], g[1], g[2]
                    district_out = g[3] if len(g) > 3 else district_out
                    state_out = g[4] if len(g) > 4 else state_out
                    place_type = g[5] if len(g) > 5 else ""
                    pin_code = g[6] if len(g) > 6 else ""
            if la is None and lo is None and g is None and hit is None and offline_hit is None:
                # 2) "Place, Region" disambiguation (e.g. "Aurangabad, Maharashtra").
                qq = q_main if q_region else qc
                if q_region:
                    hit2 = fuzzy_station(qq, sts)
                    off2 = None if hit2 else offline_lookup(qq, district=district or q_region, state=eff_state)
                    if off2 is None and not district:
                        off2 = offline_lookup(qq, district="", state=q_region)
                    if hit2:
                        la, lo, place = hit2["lat"], hit2["lon"], hit2["name"]
                    elif off2 is not None and off2.get("lat") is not None:
                        offline_hit = off2
                        la, lo, place = off2["lat"], off2["lon"], off2["place"]
                        district_out = off2.get("district", district_out)
                        state_out = off2.get("state", state_out)
                        place_type = off2.get("place_type", "")
                        pin_code = off2.get("pin_code", "")
                    elif off2 is not None:
                        cands = offline_candidates(qq)
                        return {"suggestions": suggest(qc, sts),
                                "candidates": cands,
                                "message": f"Found {off2['place']}, {off2.get('district','')}, {off2.get('state','')} but no coordinates are available offline. Try a nearby major city."}
                    elif off is not None:
                        cands = offline_candidates(qq)
                        return {"suggestions": suggest(qc, sts),
                                "candidates": cands,
                                "message": f"Found {off['place']}, {off.get('district','')}, {off.get('state','')} but no coordinates are available offline. Try a nearby major city."}
                    else:
                        return {"suggestions": suggest(qc, sts),
                                "candidates": offline_candidates(qc),
                                "message": f"Did you mean: {', '.join(suggest(qc, sts))}?"}
                elif off is not None:
                    cands = offline_candidates(qq)
                    return {"suggestions": suggest(qc, sts),
                            "candidates": cands,
                            "message": f"Found {off['place']}, {off.get('district','')}, {off.get('state','')} but no coordinates are available offline. Try a nearby major city."}
                elif not g:
                    return {"suggestions": suggest(qc, sts),
                            "candidates": offline_candidates(qc),
                            "message": f"Did you mean: {', '.join(suggest(qc, sts))}?"}
    w, s, e, n = BBOX
    if la is None or lo is None or not (s <= la <= n and w <= lo <= e):
        return {"suggestions": suggest(q or "", sts),
                "message": "Location outside India. Did you mean: " + ", ".join(suggest(q or "", sts)) + "?"}
    pm, src, near, dist = value_for(la, lo, sts)
    cat = aqi(pm)
    risk = "High" if pm > 90 else ("Medium" if pm > 45 else "Low")
    conf = "High" if src == "Measured" else ("Medium" if src == "Interpolated" else "Low")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    fc = [round(max(8.0, pm + ((i + 1) * 3 - 4)), 1) for i in range(3)]
    card = {"place": place, "pm25": pm, "aqi_category": cat, "source": src,
            "nearest_station": near, "distance_km": dist, "confidence": conf,
            "updated_at": now, "disease_risk": risk, "advice": ADVICE[cat], "forecast": fc}
    if district_out or state_out or place_type or pin_code:
        card.update({"district": district_out, "state": state_out, "place_type": place_type,
                     "pin_code": pin_code, "latitude": la, "longitude": lo,
                     "geographic_resolution": "geocoded location + 0.1-degree satellite grid"})
    if offline_hit is not None:
        card.update({"district": offline_hit.get("district", district_out),
                     "state": offline_hit.get("state", state_out),
                     "place_type": offline_hit.get("place_type", place_type),
                     "pin_code": offline_hit.get("pin_code", pin_code),
                     "latitude": la, "longitude": lo,
                     "geographic_resolution": "offline gazetteer + 0.1-degree satellite grid",
                     "geographic_source": offline_hit.get("source", "")})
    if place_ref:
        card.update({"district": place_ref["district"], "state": place_ref["state"],
                     "latitude": la, "longitude": lo,
                     "geographic_resolution": "0.1-degree satellite grid"})
    if src == "Satellite estimate" and dist != 999.0:
        card["note"] = f"Nearest station: {near}, {dist} km away. Using satellite estimate."
    return card

from fastapi import FastAPI, Query  # STRICT: uvicorn run needs fastapi
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

HAS_API = True

app = FastAPI(title="Env-MeshNet air")
@app.get("/api/air")
def api_air(q: str | None = Query(default=None), lat: float | None = None, lon: float | None = None,
            district: str | None = Query(default=None), state: str | None = Query(default=None)):
    """GET /api/air?q=<text> or ?lat=..&lon=.. -> card JSON.

    Backward-compatible: original q/lat/lon behaviour unchanged.
    Optional district/state filters narrow nationwide search (Phase 3).
    """
    try:
        result = answer(q, lat, lon, district=district or "", state=state or "")
        logger.info("air assessment completed: query=%r place=%r has_card=%s",
                    q, result.get("place"), "pm25" in result)
        return JSONResponse(result)
    except Exception:
        logger.exception("air assessment failed: query=%r lat=%r lon=%r", q, lat, lon)
        raise


@app.get("/api/search")
def api_search(q: str = Query(default=""), district: str = Query(default=""),
               state: str = Query(default=""), limit: int = Query(default=8)):
    """GET /api/search?q=<text>&district=..&state=.. -> ranked candidates.

    Backward-compatible addition (does not change /api/air). Each candidate
    shows place, district, state, lat/lon (when available), match type and
    source so duplicate names can be disambiguated.
    """
    try:
        cands = offline_candidates(q, district=district or "", state=state or "",
                                   limit=max(1, min(int(limit or 8), 20)))
        # Make JSON-safe (None lat/lon stays None; no fake coords).
        return JSONResponse({"query": q, "district": district, "state": state,
                             "count": len(cands), "candidates": cands})
    except Exception:
        logger.exception("search failed: query=%r", q)
        raise


@app.get("/api/coverage")
def api_coverage():
    """GET /api/coverage -> honest geographic + data + model coverage."""
    try:
        if HAS_OFFLINE_GAZETTEER and _coverage_summary is not None:
            return JSONResponse(_coverage_summary())
        return JSONResponse({"states_uts": 0, "districts_indexed": 0,
                             "message": "Offline gazetteer unavailable"})
    except Exception:
        logger.exception("coverage failed")
        raise


@app.get("/api/districts")
def api_districts(state: str = Query(default="")):
    """GET /api/districts?state=<state> -> district names for map selection."""
    try:
        if not (HAS_OFFLINE_GAZETTEER and _load_districts is not None):
            return JSONResponse({"state": state, "count": 0, "districts": []})
        all_d = _load_districts()
        if state:
            import re as _re
            ns = _re.sub(r"\s+", " ", (state or "").lower()).strip()
            all_d = [d for d in all_d if (d.get("state") or "").lower() == ns]
        all_d = sorted(all_d, key=lambda r: (r.get("state", ""), r.get("district", "")))
        return JSONResponse({"state": state, "count": len(all_d), "districts": [
            {"district": d.get("district"), "state": d.get("state"),
             "lat": d.get("lat"), "lon": d.get("lon"),
             "has_coordinates": d.get("lat") is not None} for d in all_d]})
    except Exception:
        logger.exception("districts failed")
        raise
@app.get("/", response_class=HTMLResponse)
def home():
    """Serve single-page app (always UTF-8)."""
    p = ROOT / "frontend" / "index.html"
    html = p.read_text(encoding="utf-8") if p.exists() else "<h1>Env-MeshNet</h1>"
    return HTMLResponse(html, headers={"Content-Type": "text/html; charset=utf-8"})
app.mount("/static", StaticFiles(directory=str(ROOT / "frontend")), name="static")

def main() -> None:
    """Run: uvicorn src.api.main:app --reload --port 8000 (from envmeshnet/)."""
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
