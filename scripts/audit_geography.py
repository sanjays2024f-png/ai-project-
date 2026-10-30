"""Write an evidence-based geographic coverage report for the current workspace."""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_dataset() -> tuple[set[str], set[tuple[str, str]]]:
    path = ROOT / "data" / "raw" / "synthetic" / "india_dataset.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    states = {row["state"] for row in rows if row.get("state")}
    districts = {(row["state"], row["district"]) for row in rows
                 if row.get("state") and row.get("district")}
    return states, districts


def read_places() -> Counter[str]:
    path = ROOT / "data" / "external" / "vellore_places.csv"
    counts: Counter[str] = Counter()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            counts[row["place_type"]] += 1
    return counts


def offline_coverage() -> dict:
    from src.api.geography import coverage_summary
    try:
        return coverage_summary()
    except Exception as ex:
        return {"FAILED": str(ex)}


def main() -> None:
    states, districts = read_dataset()
    place_counts = read_places()
    cov = offline_coverage()
    report = ROOT / "results" / "reports" / "geographic_coverage.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    state_lines = "\n".join(f"- {state}" for state in sorted(states))
    district_lines = "\n".join(f"- {district}, {state}"
                               for state, district in sorted(districts))
    place_lines = "\n".join(f"- {kind}: {count}" for kind, count in sorted(place_counts.items()))
    off_lines = ""
    if "FAILED" not in cov:
        names = "\n".join(f"- {s}" for s in cov.get("state_names", []))
        off_lines = (
            "\n## Offline gazetteer coverage (data/geography + registry)\n\n"
            f"- States/UTs indexed: {cov.get('states_uts')}\n"
            f"- District names indexed: {cov.get('districts_indexed')} "
            f"({cov.get('districts_with_coordinates')} with coordinates, "
            f"{cov.get('districts_name_only')} name-only, no fake coords)\n"
            f"- Localities indexed: {cov.get('localities_indexed')} "
            f"({cov.get('locality_by_type')})\n"
            f"- Ground stations: {cov.get('stations', {}).get('stations_indexed')} stations, "
            f"{cov.get('stations', {}).get('states_with_stations')} states, "
            f"{cov.get('stations', {}).get('districts_with_stations')} districts\n"
            f"- Districts with dataset rows (model-input proxy): "
            f"{cov.get('model_inputs', {}).get('districts_with_dataset_rows')}\n"
            f"- Gazetteer sources: {'; '.join(cov.get('sources', []))}\n\n"
            "### Indexed state/UT names\n\n" + names + "\n"
        )
    else:
        off_lines = f"\n## Offline gazetteer coverage\n\nFAILED: {cov['FAILED']}\n"
    report.write_text(
        "# Env-MeshNet geographic coverage audit\n\n"
        "Generated from the files present in this workspace; this is not a claim "
        "of complete India coverage.\n\n"
        "## Indexed administrative coverage\n\n"
        f"- States/union territories represented in pollution dataset: {len(states)}\n"
        f"- Districts represented in pollution dataset: {len(districts)}\n"
        f"- Local reference places: {sum(place_counts.values())}\n\n"
        "### States and union territories found\n\n" + state_lines + "\n\n"
        "### Districts found\n\n" + district_lines + "\n\n"
        "### Local reference place counts\n\n" + place_lines + "\n"
        + off_lines +
        "\n## Data-source coverage\n\n"
        "- `data/raw/synthetic/india_dataset.csv`: synthetic smoke data for "
        f"{len(districts)} district/state combinations; it is not nationwide "
        "observational evidence.\n"
        "- `data/external/stations.csv`: the checked-in monitoring registry "
        "contains station records (ground observations where available).\n"
        "- `data/geography/`: offline reference index (28 states + 8 UTs, "
        "700+ district names, major localities). District names without "
        "coordinates are name-only by design — no fake coordinates.\n"
        "- Photon/Nominatim: runtime geocoding fallback, cached per query; "
        "availability depends on network access and service limits.\n"
        "- Satellite estimate: existing 0.1-degree heuristic path used only "
        "when a coordinate resolves and no qualifying station is available. "
        "Labelled as Satellite estimate, never as ground PM2.5.\n\n"
        "## Model coverage\n\n"
        "- Env-MeshNet model inference is not wired into the web assessment "
        "route. No model prediction is claimed by this report.\n"
        "- Disease-risk values in the existing synthetic pipeline are circular "
        "smoke labels and are not validated clinical predictions.\n\n"
        "## Missing coverage and limitations\n\n"
        "- District-boundary GeoJSON is not bundled; map uses marker points "
        "and live tiles with progressive loading.\n"
        "- Name-only districts need live geocoding or a nearby major city "
        "for assessment; search never silently returns another city.\n"
        "- PIN-code coverage is limited to the local reference seed plus "
        "checked-in station/PIN entries.\n"
        "- No complete list of missing Indian districts can be inferred from "
        "the synthetic dataset alone.\n",
        encoding="utf-8",
    )
    print(report)


if __name__ == "__main__":
    main()

