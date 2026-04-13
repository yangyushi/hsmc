from pathlib import Path

from hsmc import _assets


def test_create_slit_main_uses_current_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    assert _assets.create_slit_main() == 0
    assert (tmp_path / "simulate_slit").is_dir()
    assert (tmp_path / "simulate_slit" / "configure.ini").is_file()


def test_create_slit_v2_main_uses_current_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    assert _assets.create_slit_v2_main() == 0
    assert (tmp_path / "simulate_slit_v2").is_dir()
    assert (tmp_path / "simulate_slit_v2" / "workflow.py").is_file()
