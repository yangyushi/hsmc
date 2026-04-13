from __future__ import annotations

import shutil
import sys
from importlib import resources
from pathlib import Path


def _source_asset_path(asset_name: str) -> Path:
    package_asset = resources.files("hsmc").joinpath("workflow", asset_name)
    if package_asset.is_dir():
        return Path(package_asset)

    repo_asset = Path(__file__).resolve().parents[2] / "workflow" / asset_name
    if repo_asset.is_dir():
        return repo_asset

    raise FileNotFoundError(f"Unable to locate bundled asset: {asset_name}")


def _copy_asset(asset_name: str, destination: str | Path = ".") -> Path:
    target_root = Path(destination).resolve()
    target = target_root / asset_name

    if target.exists():
        raise FileExistsError(f"Destination already exists: {target}")

    source = _source_asset_path(asset_name)
    shutil.copytree(source, target)
    return target


def copy_simulate_slit(destination: str | Path = ".") -> Path:
    return _copy_asset("simulate_slit", destination)


def copy_simulate_slit_v2(destination: str | Path = ".") -> Path:
    return _copy_asset("simulate_slit_v2", destination)


def create_slit_main() -> int:
    copy_simulate_slit(Path.cwd())
    return 0


def create_slit_v2_main() -> int:
    copy_simulate_slit_v2(Path.cwd())
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit(
            "Usage: python -m hsmc._assets {simulate_slit|simulate_slit_v2} [destination]"
        )

    artifact = argv.pop(0)
    destination = argv.pop(0) if argv else "."

    if artifact == "simulate_slit":
        copy_simulate_slit(destination)
        return 0
    if artifact == "simulate_slit_v2":
        copy_simulate_slit_v2(destination)
        return 0

    else:
        raise SystemExit(f"Unknown artifact: {artifact}")


if __name__ == "__main__":
    raise SystemExit(main())
