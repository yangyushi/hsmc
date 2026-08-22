import pytest

from hsmc import _assets


@pytest.mark.parametrize(
    ("argv", "expected_dir", "expected_file"),
    (
        (["create", "simulate_slit"], "simulate_slit", "configure.ini"),
        (["create", "simulate_slit_v2"], "simulate_slit_v2", "workflow.py"),
    ),
)
def test_workflow_cli_create_copies_requested_workflow(monkeypatch, tmp_path, argv, expected_dir, expected_file):
    monkeypatch.chdir(tmp_path)

    assert _assets.workflow_cli_main(argv) == 0
    assert (tmp_path / expected_dir).is_dir()
    assert (tmp_path / expected_dir / expected_file).is_file()


def test_workflow_cli_create_accepts_explicit_destination(tmp_path):
    assert _assets.workflow_cli_main(["create", "simulate_slit_v2", str(tmp_path)]) == 0
    assert (tmp_path / "simulate_slit_v2").is_dir()
    assert (tmp_path / "simulate_slit_v2" / "workflow.py").is_file()


def test_workflow_cli_list_prints_slugs(capsys):
    assert _assets.workflow_cli_main(["list"]) == 0
    assert len(capsys.readouterr().out.splitlines()) > 0


def test_workflow_cli_rejects_unknown_workflow(capsys):
    with pytest.raises(SystemExit) as exc:
        _assets.workflow_cli_main(["create", "does-not-exist"])

    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_workflow_cli_rejects_target_collision(capsys, tmp_path):
    (tmp_path / "simulate_slit").mkdir()

    with pytest.raises(SystemExit) as exc:
        _assets.workflow_cli_main(["create", "simulate_slit", str(tmp_path)])

    assert exc.value.code == 2
    assert "Destination already exists" in capsys.readouterr().err
