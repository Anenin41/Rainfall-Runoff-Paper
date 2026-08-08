# Rainfall–Runoff Shallow Water Moment Solver

A research-oriented **one-dimensional finite-volume solver** for shallow-water moment equations (SWME). The code resolves vertically varying horizontal velocity profiles through moment coefficients and includes an extension for **rainfall, infiltration, exfiltration, and mixing-induced friction**.

## Capabilities

- Classical **SWE / SWME** and hyperbolic **HSWME** transport models.
- Moment orders from the depth-averaged case \(N=0\) upward in the core SWME implementation.
- Classical, spatially adaptive, smoothed-adaptive, interpolated-adaptive, and micro–macro simulation modes.
- Path-conservative finite-volume solvers: PRICE, Lax–Friedrichs, Roe, and Osher.
- Explicit Euler, implicit Euler, and exact source integration options.
- Recharge extension (`RechargeSWME1D`) for uniform rainfall, Horton or constant infiltration, exfiltration, and admissible rainfall/bed mixing friction. Recharge source terms are currently implemented for \(N=0,1,2\).
- Optional solution-history and hyperbolicity diagnostics exported as CSV files.

## Quick start

1. Create a Python virtual environment through `python3 -m venv .venv`.
2. Activate it with `source .venv/bin/activate`.
3. Install requirements via `pip install -r requirements.txt`.
4. Configure a case in `config/config.txt`.
5. Select the PDE model, moment order, mesh resolution, numerical flux, time integrator, boundary conditions, and end time.
6. Run:

```bash
python3 main.py
```

For recharge cases, set `pde_type = RechargeSWME1D`. Results are written under `Data-processing/Results/Recharge/`.

## Repository structure

```text
main.py                    # Configuration-driven entry point and output handling
pde.py                     # PDE interface; SWME, HSWME, vegetation, and Hermite models
simulation.py              # Classical and adaptive simulation drivers; CFL time loop
mesh.py                    # Uniform one-dimensional mesh definition
spatialDiscretization.py   # Path-conservative finite-volume flux/fluctuation solvers
timeIntegration.py         # Source-term integrators
plotting.py                # Post-processing plots
config/config.txt          # Runtime case configuration

recharge/
├── recharge_pde.py        # RechargeSWME1D model wrapper
├── source_terms.py        # Rainfall/infiltration and friction source vectors
├── initial_conditions.py  # Custom initial conditions for the RechargeSWME1D wrapper
├── laws.py                # Horton, constant infiltration, mixing-friction closures
└── context.py             # Time-, timestep-, and cell-local source context

Data-processing/           # Generated CSV output and figures
Makefile                   # Repository utilities for repeatable development tasks
```

## Solver architecture

The execution path is deliberately modular:

```text
config → main.py → PDE model + mesh + numerical method
       → simulation driver → finite-volume transport update
       → source-term integration → post-processing / CSV / plots
```

PDE classes provide the system matrix, source terms, initial conditions, primitive-variable conversion, and wave-speed estimate. Simulation classes own ghost cells, CFL stepping, boundary updates, finite-volume fluctuations, source integration, and optional diagnostics. Recharge physics remains isolated from the core SWME transport structure through the `recharge/` layer.

## Developer utilities

The `Makefile` that is present here should be the standard entry point for routine project tasks. For now, it only contains a data sanitation command as `make purge`. Running this command deletes *every* file stored under `Data-processing/Results/Recharge/`. In future versions of the solver, it should be updated with a `make help` command, shortly explaining the present utilities. In the case of migrating to a compiled language, move this `Makefile` preferably inside the `Data-processing/` folder.

## Extending the solver

- Add a new physical model by implementing the `PDE` interface in `pde.py`.
- Add rainfall/infiltration closures in `recharge/laws.py` and connect them through `recharge/source_terms.py`.
- Add a finite-volume method in `spatialDiscretization.py` or a time integrator in `timeIntegration.py`.
- Keep model physics, numerical methods, runtime configuration, and post-processing separate.

## Current scope

The main entry point presently builds a **uniform 1D mesh**. The recharge model is designed for the classical 1D driver and currently supports moment orders \(N=0,1,2\). Dry-state handling and fully unstructured/variable-grid gradients are outside the present solver scope.
