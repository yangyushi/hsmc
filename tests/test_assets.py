import pytest

from hsmc import _assets


@pytest.mark.parametrize(
    ("workflow", "expected_file"),
    (
        ("simulate_slit", "configure.ini"),
        ("simulate_slit_v2", "workflow.py"),
    ),
)
def test_copy_workflow_copies_registered_bundle(tmp_path, workflow, expected_file):
    target = _assets.copy_workflow(workflow, tmp_path)

    assert target == tmp_path / workflow
    assert target.is_dir()
    assert (target / expected_file).is_file()
