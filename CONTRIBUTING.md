# Contributing

## Adding A Bundled Workflow

Bundled workflows live under `workflow/` and are exposed through the
`hsmc-workflow` CLI. The workflow registry in `workflow/workflows.json` is the
single source of truth for what gets shipped.

To add a new bundled workflow:

1. Create the workflow directory under `workflow/`.
2. Add one entry to `workflow/workflows.json`.
3. Set the required metadata in that entry:
   - `name`: public workflow name used by `hsmc-workflow create`
   - `bundle_dir`: source directory under `workflow/`
   - `copy_dir`: directory name created in the destination
4. Update user-facing documentation in `Readme.md` if the new workflow should be advertised there.
5. Run `pytest -q` before merging.

## Workflow Integrity Rules

The test suite enforces these invariants:

- every bundled workflow directory is registered exactly once
- workflow names, bundle directories, and copy directories are unique
- every workflow declares the metadata required for discovery and copying
