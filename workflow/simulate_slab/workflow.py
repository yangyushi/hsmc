#!/usr/bin/env python3
import argparse
import subprocess

from common.workflow_support import (
    CONFIG_PATH,
    ROOT_DIR,
    config_uses_auto,
    ensure_output_directories,
    generated_paths,
    group_generated_paths_by_uuid,
    isf_arrays_path,
    isf_metadata_path,
    isf_ready,
    get_workflow_logger,
    read_json,
    remove_generated_outputs,
    run_stage,
    workflow_uuid,
)
from scripts.validate_workflow import validate_environment


def start_workflow(no_preview: bool = False) -> int:
    ensure_output_directories()
    try:
        run_stage("scripts.validate_workflow", "validate_workflow")
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    current_uuid = workflow_uuid()
    logger = get_workflow_logger("workflow", current_uuid)
    logger.info("Starting workflow")

    if not no_preview:
        try:
            run_stage("scripts.preview", "preview", current_uuid=current_uuid)
        except subprocess.CalledProcessError as exc:
            return exc.returncode

    if config_uses_auto():
        if isf_ready(current_uuid):
            logger.info(
                "Reusing existing ISF output: metadata=%s, arrays=%s",
                isf_metadata_path(current_uuid).name,
                isf_arrays_path(current_uuid).name,
            )
        else:
            try:
                run_stage(
                    "scripts.get_relax_time", "isf", current_uuid=current_uuid
                )
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
    logger.info("Completed workflow")
    return 0


def check_workflow() -> int:
    status, current_uuid = validate_environment(
        write_artifacts=False, emit_output=False
    )
    if status != 0:
        return status
    print(
        f"Configuration and environment look valid for {current_uuid}"
    )
    print(
        f"Configuration file: {CONFIG_PATH.name}"
    )
    return 0


def validate_results() -> int:
    if not CONFIG_PATH.is_file():
        print(
            f"Missing configuration file: {CONFIG_PATH.name}. "
            "Validation against the current configuration is not possible."
        )
        return 1

    current_uuid = workflow_uuid()
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
        print("Warning: no artifact match the current code and configuration.")

    stale_uuids = sorted(uuid for uuid in grouped if uuid != current_uuid)
    if stale_uuids:
        print("Warning: found artifacts from different code/config UUIDs:")
        for artifact_uuid in stale_uuids:
            print(f"- {artifact_uuid}")

    if untagged:
        print("Warning: found untagged generated files:")
        for path in untagged:
            print(f"- {path}")

    current_isf_metadata = isf_metadata_path(current_uuid)
    if current_isf_metadata.is_file():
        isf_metadata = read_json(current_isf_metadata)
        if isf_metadata.get("workflow_uuid") != current_uuid:
            print("Warning: ISF UUID does not match the workflow UUID.")
            return 1

    print("Validation completed.")
    return 0


def clean_workflow(force: bool = False) -> int:
    paths = generated_paths()
    if not paths:
        print("No generated outputs found.")
        return 0

    if not force:
        print(
            "This will remove generated outputs under "
            "result/, figure/, tcc/, and root workflow logs."
        )
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
    parser = argparse.ArgumentParser(description="Slab workflow entrypoint")
    subparsers = parser.add_subparsers(dest="command")

    start_parser = subparsers.add_parser(
        "start", help="start the simulation workflow"
    )
    start_parser.add_argument(
        "--no-preview",
        action="store_true",
        help="skip the interactive geometry preview stage",
    )
    subparsers.add_parser(
        "check", help="check configuration and environment"
    )
    subparsers.add_parser(
        "validate",
        help="validate outputs against the current code and configuration"
    )
    clean_parser = subparsers.add_parser(
        "clean", help="remove generated outputs"
    )
    clean_parser.add_argument(
        "--yes", action="store_true", help="skip confirmation prompt"
    )

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command

    if command is None:
        return start_workflow(no_preview=False)
    if command == "start":
        return start_workflow(no_preview=args.no_preview)
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
