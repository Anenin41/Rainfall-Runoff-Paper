# API reference

Generated from the docstrings in the source. This page is for reading the code, not for
learning the model; start from [The model](model.md) for that.

## swme.coefficients

All mathematics for the shifted Legendre basis and the projection tensors it produces.
Everything here depends only on the moment order, never on the flow, so results are
computed once per order and cached.

::: swme.coefficients
    options:
      members:
        - Coefficients
        - get_coefficients
        - eval_phi

## swme.pde

The models themselves: the system matrix, the source terms, initial conditions, and
conversion from the conserved state to primitives.

::: swme.pde
    options:
      members:
        - PDE
        - SWME1D

## swme.source_terms

Navier-slip friction, as a vector for explicit integration and as a matrix for the
implicit path.

::: swme.source_terms

## swme.topography

Bed profiles and the settings object that carries one.

::: swme.topography

## swme.wetdry

The single rule for reading primitives out of a state, so that no two parts of the
solver can disagree about what a nearly dry cell means.

::: swme.wetdry

## swme.mesh

The uniform 1D mesh, including bed elevation sampling and its ghost cells.

::: swme.mesh

## swme.spatialDiscretization

The path-conservative schemes. They differ only in how the viscosity matrix is built.

::: swme.spatialDiscretization
    options:
      members:
        - SpatialDiscretization
        - PVM
        - PRICE
        - LF
        - Roe
        - Osher

## swme.timeIntegration

::: swme.timeIntegration

## swme.simulation

The time loop, the positivity limiter, and the optional diagnostics.

::: swme.simulation
    options:
      members:
        - ClassicalSimulation1D

## swme.cli

Config loading and validation, the object builders, and output writing.

::: swme.cli
    options:
      members:
        - load_config
        - resolve_config
        - shipped_config_names
        - build_topography
        - build_wet_dry
        - build_pde
        - build_scheme
        - build_time_integration
        - build_mesh
        - build_simulation
        - build_run_metadata
        - write_outputs
        - run
        - main

## swme.report

The PDF report generator. See [Run reports](reports.md) for how to use it.

::: swme.report.data
    options:
      members:
        - RunData
        - RunMetadata
        - SchemeCounters
        - HyperbolicityData
        - from_simulation
        - from_directory
        - vertical_velocity_profile

::: swme.report.history
    options:
      members:
        - FieldSnapshots
        - select_steps
        - stream_field_history
        - snapshots_from_history_list

::: swme.report.assemble
    options:
      members:
        - PageSpec
        - ReportResult
        - selected_pages
        - iter_figures
        - build_report

::: swme.report.hyperbolicity
    options:
      members:
        - recompute_summary
        - first_nonhyperbolic_time
        - worst_time

The page layout: font metrics, wrapping, captions that take their space out of their own
axes, and the colour helpers. See [Run reports](reports.md#how-the-pages-are-laid-out).

::: swme.report.style
    options:
      members:
        - ReportStyle
        - Prose

## swme.purge

The output-directory listing and cleanup tool behind `uv run purge`.

::: swme.purge
    options:
      members:
        - find_runs
        - list_runs
        - delete_runs

## recharge.laws

Infiltration models and the mixing friction closure.

::: recharge.laws

## recharge.recharge_pde

::: recharge.recharge_pde

## recharge.source_terms

The recharge mass source and the combined friction: Navier-slip from the base model plus
the mixing friction this extension contributes.

::: recharge.source_terms

## recharge.initial_conditions

The three recharge benchmarks, all generic and nested in the moment order. The class
docstring records what the seeded moment ratio is for and what it does not promise.

::: recharge.initial_conditions
    options:
      members:
        - RechargeSWME1D_CustomIC
