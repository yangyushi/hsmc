from pathlib import Path
import sys

sys.path.append("../lib")

from hsmc import _assets


def test_source_assets_resolve_from_workflow_directory():
    simulate_slit = _assets._source_asset_path("simulate_slit")
    simulate_slit_v2 = _assets._source_asset_path("simulate_slit_v2")

    assert simulate_slit.is_dir()
    assert simulate_slit.name == "simulate_slit"
    assert simulate_slit.parent.name == "workflow"

    assert simulate_slit_v2.is_dir()
    assert simulate_slit_v2.name == "simulate_slit_v2"
    assert simulate_slit_v2.parent.name == "workflow"


def test_copy_asset_preserves_workflow_templates(tmp_path):
    target = _assets.copy_simulate_slit_v2(tmp_path)

    assert target == tmp_path / "simulate_slit_v2"
    assert target.is_dir()
    assert (target / "workflow.py").is_file()
    assert (target / "scripts" / "validate_workflow.py").is_file()
    assert (target / "common" / "workflow_support.py").is_file()
