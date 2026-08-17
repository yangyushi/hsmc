# Crystal-Slab Simulation Workflow

This template is copied into each run directory by `launch.py`. It is configured locally and run through:

```bash
python3 workflow.py start
```

## Intention

Study how a fixed crystalline slab embedded in a bulk fluid changes local structure, density, and cluster populations relative to a matched bulk reference. The simulation cell is fully periodic in all three directions. The simulation engine stays in `hsmc`; this template owns experiment policy such as orchestration, sampling cadence, dependency validation, caching, logging, and result layout.

## Layout

- `workflow.py`: the only workflow entrypoint
- `common/`: shared local helpers for UUID computation, artifact naming, logging, and slab setup
- `scripts/`: runnable workflow stages
- `lib/`: vendored `crystal-slab` generator code
- `configure.ini.example`: tracked template configuration
- `configure.ini`: user-created local configuration, intentionally not versioned

The stage order is controlled by `workflow.py`:

1. validate the environment and configuration
2. generate an interactive 3D preview and 2D geometry schematic
3. run ISF only when `auto` dump frequency requires it
4. run the slab simulation
5. derive the bulk reference
6. run TCC analysis
7. render plots

Available subcommands:

```bash
python3 workflow.py check
python3 workflow.py validate
python3 workflow.py start
python3 workflow.py start --no-preview
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

Every run is tagged by one deterministic workflow ID, stored as a hex digest over all `.py` files under `common/`, `scripts/`, and `lib/` plus the active `configure.ini`. The ID is written into:

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
- `lib.crystal_slab` (vendored)
- the `tcc` executable on `PATH`

If the environment is not usable, validation fails before any simulation stage runs.
