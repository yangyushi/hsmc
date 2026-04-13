# Slit Simulation Workflow

This template is meant to be copied into a standalone experiment directory by `hsmc-workflow create simulate_slit_v2`, configured locally, and then run through one command:

```bash
python3 workflow.py start
```

## Intention

The slit workflow studies how confinement and wall structure change local structure, density, and cluster populations relative to a matched bulk reference. The simulation engine stays in `hsmc`; this template owns experiment policy such as orchestration, sampling cadence, dependency validation, caching, logging, and result layout.

## Layout

- `workflow.py`: the only workflow entrypoint
- `common/`: shared local helpers for UUID computation, artifact naming, logging, and slit setup
- `scripts/`: runnable workflow stages
- `configure.ini.example`: tracked template configuration
- `configure.ini`: user-created local configuration, intentionally not versioned

The stage order is controlled by `workflow.py`:

1. validate the environment and configuration
2. run ISF only when `auto` dump frequency requires it
3. run the slit simulation
4. derive the bulk reference
5. run TCC analysis
6. render plots

Available subcommands:

```bash
python3 workflow.py check
python3 workflow.py validate
python3 workflow.py start
python3 workflow.py clean
```

## Configuration

Start from:

```bash
cp configure.ini.example configure.ini
```

Then fill in the experiment-specific values.

Both of these fields accept either an integer or `auto`:

- `[Run] dump_frequency`
- `[Run] dump_frequency_bulk`

When set to `auto`, the workflow estimates the ISF relaxation time and snaps the fitted sweep count upward to the next power of ten.

## Integrity and Outputs

Every run is tagged by one deterministic workflow ID, stored as a hex digest over all `.py` files under `common/` and `scripts/` plus the active `configure.ini`. The ID is written into:

- root-level log filenames such as `workflow.<uuid>.log`
- result artifact filenames under `result/`
- figure filenames under `figure/`

This lets the copied directory serve as a self-contained experiment record whose code and configuration integrity can be re-verified later.

Array-heavy outputs use NumPy formats:

- `.npy` for single-array outputs
- `.npz` for multi-array outputs

Scalar metadata stays in JSON where that is simpler to inspect.

## Dependencies

The workflow checks these before simulation starts:

- `numpy`
- `matplotlib`
- `scipy`
- `hsmc`
- `tcc`
- the `tcc` executable on `PATH`

If the environment is not usable, validation fails before any simulation stage runs.
