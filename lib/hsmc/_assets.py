from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path


@dataclass(frozen=True)
class WorkflowSpec:
    name: str
    bundle_dir: str
    copy_dir: str


def workflow_root() -> Path:
    package_root = resources.files("hsmc").joinpath("workflow")
    if package_root.is_dir():
        return Path(package_root)

    repo_root = Path(__file__).resolve().parents[2] / "workflow"
    if repo_root.is_dir():
        return repo_root

    raise FileNotFoundError("Unable to locate bundled workflows.")


def workflow_registry_path(root: Path | None = None) -> Path:
    return (workflow_root() if root is None else root) / "workflows.json"


def _build_workflow_spec(entry: object) -> WorkflowSpec:
    if not isinstance(entry, dict):
        raise ValueError("Each workflow entry must be a JSON object.")

    required_keys = ("name", "bundle_dir", "copy_dir")
    missing = [key for key in required_keys if key not in entry]
    if missing:
        raise ValueError(
            f"Workflow entry is missing required fields: {', '.join(missing)}"
        )

    fields: dict[str, str] = {}
    for key in ("name", "bundle_dir", "copy_dir"):
        value = entry[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Workflow field {key!r} must be a non-empty string.")
        fields[key] = value

    return WorkflowSpec(
        name=fields["name"],
        bundle_dir=fields["bundle_dir"],
        copy_dir=fields["copy_dir"],
    )


def _validate_workflow_specs(workflows: tuple[WorkflowSpec, ...], root: Path) -> tuple[WorkflowSpec, ...]:
    if not workflows:
        raise ValueError("Workflow registry must declare at least one workflow.")

    for field in ("name", "bundle_dir", "copy_dir"):
        values = [getattr(workflow, field) for workflow in workflows]
        if len(values) != len(set(values)):
            raise ValueError(f"Workflow registry contains duplicate {field} values.")

    registered = {workflow.bundle_dir for workflow in workflows}
    bundled = {path.name for path in root.iterdir() if path.is_dir()}
    if registered != bundled:
        missing = sorted(bundled - registered)
        extra = sorted(registered - bundled)
        details = []
        if missing:
            details.append(f"unregistered bundle directories: {', '.join(missing)}")
        if extra:
            details.append(f"missing bundle directories: {', '.join(extra)}")
        raise ValueError("Workflow registry does not match bundled workflows: " + "; ".join(details))

    return workflows


def _load_workflow_specs(root: Path | None = None) -> tuple[WorkflowSpec, ...]:
    root = workflow_root() if root is None else root
    registry_path = workflow_registry_path(root)
    try:
        payload = json.loads(registry_path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Workflow registry not found: {registry_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Workflow registry is not valid JSON: {registry_path}") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("workflows"), list):
        raise ValueError("Workflow registry must be a JSON object with a workflows list.")

    workflows = tuple(_build_workflow_spec(entry) for entry in payload["workflows"])
    return _validate_workflow_specs(workflows, root)


@lru_cache(maxsize=1)
def _workflow_specs() -> tuple[WorkflowSpec, ...]:
    return _load_workflow_specs()


def list_workflows() -> tuple[str, ...]:
    return tuple(workflow.name for workflow in _workflow_specs())


def get_workflow(name: str) -> WorkflowSpec:
    for workflow in _workflow_specs():
        if workflow.name == name:
            return workflow
    raise KeyError(f"Unknown workflow: {name}")


def _source_asset_path(name: str) -> Path:
    workflow = get_workflow(name)
    source = workflow_root() / workflow.bundle_dir
    if source.is_dir():
        return source
    raise FileNotFoundError(f"Unable to locate bundled workflow: {name}")


def copy_workflow(name: str, destination: str | Path = ".") -> Path:
    workflow = get_workflow(name)
    target_root = Path(destination).resolve()
    target = target_root / workflow.copy_dir

    if target.exists():
        raise FileExistsError(f"Destination already exists: {target}")

    shutil.copytree(_source_asset_path(name), target)
    return target


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hsmc-workflow")
    subparsers = parser.add_subparsers(dest="command")

    create_parser = subparsers.add_parser("create", help="copy a bundled workflow")
    create_parser.add_argument("workflow", choices=list_workflows())
    create_parser.add_argument("destination", nargs="?", default=".")

    subparsers.add_parser("list", help="list available workflows")
    subparsers.add_parser("help", help="show help")

    return parser


def workflow_cli_main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if args.command in (None, "help"):
        parser.print_help()
        return 0

    if args.command == "list":
        for workflow in list_workflows():
            print(workflow)
        return 0

    if args.command == "create":
        try:
            copy_workflow(args.workflow, args.destination)
        except FileExistsError as exc:
            parser.exit(2, f"{parser.prog}: error: {exc}\n")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def main(argv: list[str] | None = None) -> int:
    return workflow_cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
