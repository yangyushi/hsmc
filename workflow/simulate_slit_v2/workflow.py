#!/usr/bin/env python3
import argparse
import subprocess

from common.workflow_support import (
    CONFIG_PATH,
    ROOT_DIR,
    config_uses_auto,
    compute_workflow_metadata,
    ensure_output_directories,
    ensure_workflow_metadata,
    generated_paths,
    group_generated_paths_by_uuid,
    isf_arrays_path,
    isf_metadata_path,
    isf_ready,
    read_json,
    remove_generated_outputs,
    root_metadata_path,
    run_stage,
    write_workflow_log,
)
from scripts.validate_workflow import validate_environment


def start_workflow() -> int:
    ensure_output_directories()
    try:
        run_stage("scripts.validate_workflow", "validate_workflow")
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    current_uuid = ensure_workflow_metadata()["workflow_uuid"]
    write_workflow_log("workflow_start", "workflow", current_uuid=current_uuid)

    if config_uses_auto():
        if isf_ready(current_uuid):
            write_workflow_log(
                "reuse_existing_output",
                "isf",
                current_uuid=current_uuid,
                metadata_file=isf_metadata_path(current_uuid).name,
                array_file=isf_arrays_path(current_uuid).name,
            )
        else:
            try:
                run_stage("scripts.get_relax_time", "isf", current_uuid=current_uuid)
            except subprocess.CalledProcessError as exc:
                return exc.returncode

    for module_name, stage in (
        ("scripts.simulate", "simulate"),
        ("scripts.bulk", "bulk"),
        ("scripts.analysis", "analysis"),
        ("scripts.plot", "plot"),
    ):
        try:
            run_stage(module_name, stage, current_uuid=current_uuid)
        except subprocess.CalledProcessError as exc:
            return exc.returncode
    write_workflow_log("workflow_end", "workflow", current_uuid=current_uuid)
    return 0


def check_workflow() -> int:
    status, metadata = validate_environment(write_artifacts=False, emit_output=False)
    if status != 0:
        return status
    print(f"Configuration and environment look valid for workflow_uuid={metadata['workflow_uuid']}")
    print(f"Configuration file: {CONFIG_PATH.name}")
    return 0


def validate_results() -> int:
    if not CONFIG_PATH.is_file():
        print(
            f"Missing configuration file: {CONFIG_PATH.name}. "
            "Validation against the current configuration is not possible."
        )
        return 1

    current = compute_workflow_metadata()
    current_uuid = current["workflow_uuid"]
    grouped, untagged = group_generated_paths_by_uuid()
    current_paths = grouped.get(current_uuid, [])

    print(f"Current workflow_uuid: {current_uuid}")
    if not grouped:
        print("No generated UUID-tagged artifacts found.")
        return 0

    if current_paths:
        print("Artifacts matching the current code and configuration:")
        for path in sorted(current_paths):
            print(f"- {path.relative_to(ROOT_DIR)}")
    else:
        print("Warning: no generated artifacts match the current code and configuration.")

    stale_uuids = sorted(uuid for uuid in grouped if uuid != current_uuid)
    if stale_uuids:
        print("Warning: found artifacts from different code/config UUIDs:")
        for artifact_uuid in stale_uuids:
            print(f"- {artifact_uuid}")

    if untagged:
        print("Warning: found untagged generated files:")
        for path in untagged:
            print(f"- {path}")

    metadata_path = root_metadata_path(current_uuid)
    if metadata_path.is_file():
        stored_metadata = read_json(metadata_path)
        if (
            stored_metadata.get("workflow_uuid") != current_uuid
            or stored_metadata.get("config_sha256") != current["config_sha256"]
            or stored_metadata.get("hashed_files") != current["hashed_files"]
        ):
            print("Warning: stored workflow metadata does not match the current code/config hashes.")
            return 1
    elif current_paths:
        print("Warning: current UUID artifacts exist but root metadata file is missing.")
        return 1

    current_isf_metadata = isf_metadata_path(current_uuid)
    if current_isf_metadata.is_file():
        isf_metadata = read_json(current_isf_metadata)
        if isf_metadata.get("workflow_uuid") != current_uuid:
            print("Warning: ISF metadata UUID does not match the current workflow UUID.")
            return 1

    print("Validation completed.")
    return 0


def clean_workflow(force: bool = False) -> int:
    paths = generated_paths()
    if not paths:
        print("No generated outputs found.")
        return 0

    if not force:
        print("This will remove generated outputs under result/, figure/, tcc/, and root workflow logs/metadata.")
        try:
            reply = input("Continue? [y/N] ").strip().lower()
        except EOFError:
            print("Aborted.")
            return 1
        if reply not in {"y", "yes"}:
            print("Aborted.")
            return 1

    remove_generated_outputs()
    print("Generated outputs removed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Slit workflow entrypoint")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("start", help="start the simulation workflow")
    subparsers.add_parser("check", help="check configuration and environment")
    subparsers.add_parser("validate", help="validate generated artifacts against the current code and configuration")
    clean_parser = subparsers.add_parser("clean", help="remove generated outputs")
    clean_parser.add_argument("--yes", action="store_true", help="skip confirmation prompt")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "start"

    if command == "start":
        return start_workflow()
    if command == "check":
        return check_workflow()
    if command == "validate":
        return validate_results()
    if command == "clean":
        return clean_workflow(force=args.yes)

    parser.error(f"Unknown command: {command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
