"""Dependency/config guard test — fails loudly if env incomplete."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def test_require_helper_errors_loudly():
    from src.utils.require import need
    try:
        need("definitely_missing_module_xyz")
        assert False, "need() must raise"
    except RuntimeError as ex:
        assert "FAILED" in str(ex) and "pip install" in str(ex)

def test_config_loads():
    import yaml
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "settings.yaml").read_text())
    assert cfg["seed"] == 42
    assert cfg["split"]["n_folds"] == 10
    assert cfg["bbox"] == {"west": 68.0, "south": 6.0, "east": 97.5, "north": 37.5}
    assert cfg["smote"]["apply_on_train_only"] is True
