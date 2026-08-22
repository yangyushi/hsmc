import pytest

from hsmc import _assets


def test_shipped_registry_is_consistent_with_bundled_workflows():
    workflows = _assets._load_workflow_specs()
    names = [workflow.name for workflow in workflows]
    bundle_dirs = [workflow.bundle_dir for workflow in workflows]
    copy_dirs = [workflow.copy_dir for workflow in workflows]
    bundled = {
        path.name
        for path in _assets.workflow_root().iterdir()
        if path.is_dir()
    }

    assert len(names) == len(set(names))
    assert len(bundle_dirs) == len(set(bundle_dirs))
    assert len(copy_dirs) == len(set(copy_dirs))
    for workflow in _assets._load_workflow_specs():
        assert workflow.name
        assert workflow.bundle_dir
        assert workflow.copy_dir
    assert set(bundle_dirs) == bundled


def _assert_registry_error(
    tmp_path,
    registry_json: str,
    expected_message: str,
    bundle_names: tuple[str, ...] = ()
) -> None:
    root = tmp_path / "workflow"
    for name in bundle_names:
        bundle = root / name
        bundle.mkdir(parents=True)
    root.mkdir(parents=True, exist_ok=True)
    (root / "workflows.json").write_text(registry_json)

    with pytest.raises(ValueError) as exc:
        _assets._load_workflow_specs(root)
    assert expected_message in str(exc.value)
