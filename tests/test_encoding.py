"""Encoding regression test: user-facing files must be clean UTF-8 English.

Fails on mojibake tokens (double-encoded UTF-8) in RENDERED text, while allowing
legitimate Unicode (em-dash, micro, superscript, middot, degree).
Run: pytest tests/test_encoding.py -v
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BAD_TOKENS = ["Ã©", "Ã³", "�", "â€"]
TEXT_GLOBS = ["frontend/index.html", "config/settings.yaml",
              "data/external/stations.csv", "data/external/vellore_places.csv"]

def test_source_files_are_utf8_and_mojibake_free():
    problems: list[str] = []
    for rel in TEXT_GLOBS:
        p = ROOT / rel
        if not p.exists():
            continue
        raw = p.read_bytes()
        try:
            txt = raw.decode("utf-8")
        except UnicodeDecodeError as ex:
            problems.append(f"{rel}: not valid UTF-8 ({ex})")
            continue
        for tok in BAD_TOKENS:
            if tok in txt:
                problems.append(f"{rel}: mojibake token {tok!r} found")
    assert not problems, "encoding artifacts: " + "; ".join(problems)

def test_html_declares_utf8_and_required_labels():
    html = (ROOT / "frontend/index.html").read_text(encoding="utf-8")
    assert '<meta charset="UTF-8"/>' in html
    assert "<title>" in html and "Env-MeshNet" in html and "India Air Quality" in html
    # static labels baked into the page
    for label in ["Enter a city, district, or state",
                  "Search",
                  "Nearest monitoring station", "Last updated", "Respiratory Risk",
                  "Estimated air quality and associated health risk"]:
        assert label in html, f"missing dashboard label: {label}"
    # units render as single-encoded UTF-8, not mojibake
    assert "g/m" in html and "Â" not in html.split("<script>")[0]
    # dynamic labels come from the backend (not hardcoded in static HTML by design):
    # source + confidence are rendered from /api/air JSON at runtime.
    backend = (ROOT / "src/api/main.py").read_text(encoding="utf-8")
    assert '"Satellite estimate"' in backend
    assert "confidence" in html  # template suffix `${j.confidence} confidence`
    assert '"Low confidence"' not in html  # must NOT be hardcoded; comes from API
    import sys
    sys.path.insert(0, str(ROOT))
    from src.api.main import answer
    r = answer("Kanyakumari", None, None)
    assert r.get("source") == "Satellite estimate"
    assert r.get("confidence") in ("High", "Medium", "Low")

def test_frontend_search_rejects_stale_and_invalid_results():
    html = (ROOT / "frontend/index.html").read_text(encoding="utf-8")
    assert "searchRequestId" in html
    assert "if(requestId!==searchRequestId)return" in html
    assert "if(!r.ok)" in html
    assert "Location response is missing required assessment fields" in html
    assert "location service is unreachable" in html

def test_backend_is_valid_utf8_python():
    """src/api/main.py must decode as UTF-8 and compile (ASCII-only comments by policy)."""
    raw = (ROOT / "src/api/main.py").read_bytes()
    txt = raw.decode("utf-8")
    compile(txt, "src/api/main.py", "exec")
    non_ascii = sorted({c for c in txt if ord(c) > 127})
    assert not non_ascii, f"src/api/main.py should be ASCII-only, found: {non_ascii}"
