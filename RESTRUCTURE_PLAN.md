# SWME / RechargeSWME Restructure Plan

Design record and progress log for restructuring the 1D finite-volume solver behind the
thesis *"A Rainfall-Runoff Extension of the Shallow Water Moment Framework"*: arbitrary
moment order N, well-balanced bottom topography, wet-dry treatment, and `uv` packaging.

**This file is the single source of truth for the restructure.** Read it before making
structural changes; the design decisions below are already made and recorded with their
justification, so don't re-derive them. Keep it in sync as work lands — if a decision
changes, update the relevant section rather than silently diverging from it.

> Package names throughout are the current ones: `swme` (core solver) and `recharge`
> (sibling extension package). Sections 1–4 were originally written against the old
> `moment_sw` name; they have been updated in place.

---

## Status

**Steps 0–8 complete. Step 8.5 is under way: Phase 1 (producer) done, Phases 2–3 (the report
itself) next.** 632 tests pass.

| Step | Scope | Status |
|:--|:--|:--|
| 0 | Scaffolding: `pyproject.toml`, `src/` layout, `uv sync` | Done |
| 1 | Generic-N coefficient engine (`swme/coefficients.py`) | Done |
| 1.5 | Package rename `moment_sw` → `swme` + `recharge`; config format | Done |
| 2 | Regression tests written against the legacy hardcoded code | Done |
| 3 | Swap in the generic engine, delete the per-order blocks | Done |
| 4 | Delete out-of-scope models (HME, vegetation, adaptive) | Done |
| 4.5 | Baseline validation against the thesis's own Chapter 5 results | Done |
| 5 | Bottom topography + well-balancing | Done |
| 5.5 | Hyperbolicity audit: SWME vs. HSWME | Done |
| 6 | Wet-dry treatment | Done |
| 7 | CLI rewrite (`cli.py`, YAML config) | Done |
| 8 | Cleanup (dead code, stale docs, `purge` tool) | Done |
| 8.5 | In-package post-processing suite (PDF run reports) | Phase 1 done |
| 9 | Documentation site (mkdocs) | Pending |

**Where the code stands.** The solver runs entirely on the arbitrary-N generic engine —
every hardcoded `if order == N` block is gone and the `RechargeSWME1D` N ∈ {0,1,2} cap is
lifted. Out-of-scope models are removed: `pde.py` 5034 → ~1000 lines, `simulation.py`
2891 → ~750, `plotting.py` 466 → 165, `recharge/source_terms.py` 933 → 102. Topography
with a well-balanced augmented-path coupling is in; hyperbolicity is audited and reported
at runtime; a dam break onto a dry bed now runs, where before any zero height crashed the
solver on its first step.

**Everything so far has stayed strictly additive.** This document twice predicted that
would end — at Step 5, then at Step 6 — and it did not. Topography is opt-in and a flat
bed keeps the original code path; Step 5.5's only numerical change is provably an
identity; Step 6's desingularization routes `h >= h_wet` through the literal pre-Step-6
expressions, and its positivity limiter never binds for a CFL-limited wet flow. The
reference run still reproduces `total mass = 242.1436312896249` bit for bit, and thesis
configs still reproduce their CSVs byte for byte.

**The refactor is validated against the thesis.** Step 4.5 reproduced Chapter 5 end to
end: three equations check out numerically to high precision (§5.1 eq. 5.5, §5.2 eqs.
5.9–5.10, §5.6 eq. 5.17), §5.1's figure matches visually, and the user has confirmed the
regenerated §5.3–§5.5 figures match the thesis.

**Seven real defects were found and fixed on the way** — see [Findings and
limitations](#6-findings-and-limitations) for the ones that still constrain how the
solver should be used.

---

## Context: why this restructure

The solver worked, but two structural problems blocked further use:

1. **Every moment order was hand-hardcoded.** `compute_system_matrix` had explicit
   `if order == 0/1/.../6:` blocks of fully-expanded coefficients, transcribed by hand
   from a symbolic derivation, and the pattern repeated across `compute_system_matrix_diff`,
   the friction source, a ~1700-line `_compute_source_matrix_inverse`, and
   `compute_vertical_velocity_profile` — capping the base model at N=6. The recharge
   extension was worse: separate hand-written functions per order and a hard
   `NotImplementedError` for `order not in (0, 1, 2)`. Thesis Appendix C gives the general
   formulas for arbitrary N, so the hardcoded blocks were replaced by a generic
   tensor-contraction engine.
2. **No topography, no wet-dry treatment, no well-balancing.** Confirmed by repo-wide
   grep: zero references to bed elevation, hydrostatic reconstruction or dry-cell handling.
   The solver hard-crashed the instant any height went non-positive. The `matlab/`
   reference code had none of it either — a flat-bed-only Roe-averaged scheme with
   unguarded `h`-divisions that would itself NaN on a dry cell. So this was new
   numerical-methods design, not a port.

Additionally, only **SWME, HSWME** (the `hyperbolic=True` flag on `SWME1D`) and
**RechargeSWME1D** matter going forward; `HermiteMomentEquations` (Boltzmann/BGK — different
physics), `VegetationSWME1D`, and the spatially-adaptive / micro-macro machinery were
deleted. The package also had no build system, just a `requirements.txt` and a manual
`python3 main.py`; `uv` packaging was introduced alongside.

### Assets reused rather than rebuilt

- **`symbolic_math/symbo.py`** already computed `A_ijk`, `B_ijk`, `C_ij`, `r_i`, `s_i`
  generically in N with exact rationals. Its math was absorbed into `swme/coefficients.py`
  and the directory deleted.
- **Closed forms** (thesis Appendix B.1/B.4, no sympy needed): `r_i = (-1)^i`, `s_i = 1`;
  `E` lower-triangular with `E_ii = i/(2i+1)`, `E_ij = (-1)^(i+j)` for `j<i`, `0` for `j>i`;
  `F` lower-triangular with `F_ii = i/(2i+1)`, `F_ij = 1` for `j<i`, `0` for `j>i`.
- **Basis convention:** `phi_i(z) = P_i(1 - 2z)` (shifted Legendre on `[0,1]`), so
  `phi_i(1) = (-1)^i` and `phi_i(0) = 1`.
- **Cross-validated finding — do not re-derive.** The base system matrix, the base
  friction source, the recharge mass source `S_{R,I}` and its friction block `P(U)` all
  reduce to the *same* generic tensor contractions. `RechargeSWME1D` is not separate
  physics: it is the same `P(U)` with `f_R, f_I` possibly nonzero, plus an additive
  `S_{R,I}(U)` that vanishes at `R = I = 0`. That is why one engine serves all three models.
- `recharge/laws.py` and `recharge/context.py` have no order-dependence and were left alone.

---

## Locked-in decisions

Settled with the user; do not re-litigate.

1. **Well-balanced scheme:** augmented-path-conservative topography, not the simpler
   central-difference fallback. See [§2](#2-bottom-topography-and-well-balancing).
2. **`symbolic_math/`:** absorb into `swme/coefficients.py`, then delete the directory.
   The user keeps their own copy; no in-repo wrapper CLI needed.
3. **Batch parameter-sweep scripts** (`main_HME_errorChecks.py`, `main_SWME_errorData.py`):
   delete, no replacement. Arbitrary-N plus a clean CLI makes ad-hoc sweeps easy to script.
4. **`matlab/`:** delete entirely. Its transport math, scheme and friction closure are all
   superseded, and it has no topography/wet-dry content — the one thing that would have
   justified keeping it as reference.
5. **Package layout:** `swme` (core) with `recharge` as a **sibling** top-level package,
   not nested. `recharge` imports from `swme` as a normal cross-package import
   (`from swme.pde import SWME1D`).
6. **Config format: real YAML**, not INI. **Executed in Step 7**: all 24 shipped configs
   are `.yaml`, `main.py` and every `.ini` are gone, and `cli.py` validates sections and
   keys rather than silently defaulting. `config/example.yaml` (an unconsumed stub) was
   deleted along with them — the shipped configs are the examples now.

---

## 1. Generic-N coefficient engine

`swme/coefficients.py` owns *all* math for the shifted-Legendre moment basis and its
projection tensors — ported from `symbo.py` (basis construction, `compute_A` via Wigner-3j,
`compute_B` via the `J_j(z)` antiderivative table) plus the closed forms above.

```python
@dataclass(frozen=True)
class Coefficients:
    N: int
    A: np.ndarray         # (N+1, N+1, N+1)  symmetric in the last two indices
    B: np.ndarray         # (N+1, N+1, N+1)
    C: np.ndarray         # (N+1, N+1)
    E: np.ndarray         # (N+1, N+1)
    F: np.ndarray         # (N+1, N+1)
    r: np.ndarray         # (N+1,)  = (-1)^i  for i >= 1
    s: np.ndarray         # (N+1,)  = 1       for i >= 1
    phi_at_1: np.ndarray  # (N+1,)  = (-1)^i
    phi_at_0: np.ndarray  # (N+1,)  = 1

@lru_cache(maxsize=None)
def get_coefficients(N: int) -> Coefficients: ...

def eval_phi(N: int, z: np.ndarray) -> np.ndarray:
    """(N+1, len(z)) shifted-Legendre phi_i(z) by Bonnet recurrence.
    Post-processing only (velocity-profile reconstruction), never the hot path."""
```

**Caching:** in-memory `lru_cache` keyed by N, no persisted file cache.
`get_coefficients(N)` is called once per PDE construction, not per cell or timestep;
realistic N is 0–10 and the sympy cost is sub-second. The extension point is documented
(swap the cache layer behind the same signature) but deliberately not built.

The dataclass also carries precomputed hot-path slices added in Step 3 — `A_m`,
`transport_m` (`= 2*A_m + B_m`), `C_m`, `E_m`, `F_m`, `r_m`, `s_m`, `phi1_m`, `phi0_m`,
`two_i_plus_1`, `inv_two_i_plus_1` — so the per-call code never re-slices.

### Generic formulas

State `U = (h, h*u_m, h*a_1, ..., h*a_N)`, primitives `u_m = U[1]/h`, `alpha = U[2:]/h`.
Surface and bed velocities: `u_s = u_m + sum(alpha * phi_at_1[1:])`,
`u_b = u_m + sum(alpha * phi_at_0[1:])`. Tensors are sliced `[1:,...]` to align 1-based
moment indices with the 0-based `alpha` array.

**System matrix `A(U)`:**

```
Amat[0,1]  = 1
Amat[1,0]  = g*h - u_m**2 - sum(alpha**2 / (2*i+1) for i in 1..N)
Amat[1,1]  = 2*u_m
Amat[1,2:] = 2*alpha / (2*i+1)
Amat[2:,0] = -2*u_m*alpha - einsum('ijk,j,k->i', A_t, alpha, alpha)
Amat[2:,1] = 2*alpha
Amat[2:,2:] = u_m*eye(N) + einsum('ilk,k->il', 2*A_t + B_t, alpha)
```

Expanding this by hand for N=1 reproduces the legacy hardcoded entries exactly (since
`A_111 = B_111 = 0` by Wigner-3j parity).

**Generalized friction `P(U)`** — base SWME uses `f_R = f_I = 0`; `RechargeSWME1D` passes
nonzero `f_R, f_I` from its mixing-friction closure:

```
P[0]   = 0
P[1]   = f_R*u_s + (f_I + nu/lambda)*u_b
P[i+2] = (2i+1)*(f_R*phi_i(1)*u_s + (f_I + nu/lambda)*phi_i(0)*u_b)
         + (2i+1)*(nu/h**2) * sum_j C[i,j]*U[j+2]
```

**Recharge mass source `S_{R,I}(U)`** — zero when `R = I = 0`:

```
S[0]   = R - I
S[1]   = R*u_s - I*u_b
S[i+2] = (2i+1)*R*(phi_i(1)*u_s - u_m*r_i - sum_j E[i,j]*alpha_j)
         + (2i+1)*I*(-phi_i(0)*u_b + u_m*s_i + sum_j F[i,j]*alpha_j)
```

> **Sign correction (Step 2).** The last term was originally documented here with a
> **minus** sign. That version failed the regression tests against the legacy N=1/N=2 code
> and was traced to a transcription error against the thesis formula — not a legacy bug.
> It is `+ sum_j F[i,j]*alpha_j`. Do not reintroduce the minus.

**Bed-slope source** `B[1] = g*h*dZ/dx`, all other entries zero — but it is *not*
discretized as an ordinary cell-centered term; see [§2](#2-bottom-topography-and-well-balancing).

### Module boundary: what lives in `swme` vs. `recharge`

The original draft put one combined `compute_generalized_friction(..., f_R, f_I)` in
`swme/source_terms.py`. That is mathematically fine but **violates the package boundary**:
`swme/` should hold only the base model's own physics — transport *and* its own Navier-slip
friction, which exist with or without recharge — while `recharge/` holds only what the
extension adds. Corrected split, implemented in Step 2:

**`swme/source_terms.py`**

```python
reconstruct_boundary_velocities(order, values, thresholds)   # -> h, u_m, alpha, u_s, u_b
compute_navier_slip_friction(order, values, viscosity, slip_length, thresholds)  # P_slip(U)
compute_friction_operator_matrix(order, h, viscosity, slip_length, thresholds)   # S_slip(h)
```

`SWME1D.compute_source_term` returns `-compute_navier_slip_friction(...)`.
`reconstruct_boundary_velocities` is shared with `recharge` deliberately: reading the
surface/bed velocity off the moment coefficients is a kinematic fact about the
representation, not friction physics, so sharing it does not blur the boundary.

**`recharge/source_terms.py`**

```python
compute_recharge_mass_source(order, values, R, I, thresholds)          # S_{R,I}(U)
compute_mixing_friction(order, values, f_R, f_I, thresholds)           # P_mix(U)
compute_total_friction(order, values, f_R, f_I, viscosity, slip_length, thresholds)
```

`RechargeSWME1D.compute_source_term` evaluates `R, I, f_R, f_I` from its closures and
returns `compute_recharge_mass_source(...) - compute_total_friction(...)`.

**`_compute_source_matrix_inverse` was replaced, not genericized.** The ~1700-line
Mathematica-derived per-order implicit-Euler matrix became: build the small
`(N+2)x(N+2)` friction operator `S(h)` generically from the `C` tensor, then invert
`(I - dt*S)` numerically at runtime. One small solve per cell per step, negligible next to
the 5-point Gauss quadrature already done per interface. **This single change removed
~1700 of `pde.py`'s ~4970 lines** and was the highest-value simplification in the
restructure.

> The recharge extension has no implicit/linear-source path — `RechargeSWME1D` always uses
> the explicit vector form regardless of `linear_source`. A recharge-side friction operator
> would be new functionality, not a port. **This matters in practice:** see the friction
> stiffness finding in [§6](#6-findings-and-limitations).

---

## 2. Bottom topography and well-balancing

New engineering; no reference implementation existed anywhere in the repo.

### 2.1 Where `Z(x)` lives

On the **mesh**, not the PDE — PDE objects are deliberately stateless with respect to grid
and config, and stay that way.

```python
mesh.set_bed_elevation(z_of_x, boundary_condition) -> None
# populates mesh.bed_elevation, shape (resolution+2,) including ghost cells
```

Ghost cells are filled with the *same* boundary condition the simulation uses (PERIODIC
wraps, everything else is zero-gradient), because the two edge interfaces would otherwise
see an inconsistent `(U, Z)` pair. The mesh records which condition it used and
`run_simulation` refuses to start on a mismatch.

`has_topography` is derived from whether the sampled bed is actually non-zero — not from
`set_bed_elevation` having been called — so `bed_profile = flat` leaves the solver on its
original, non-augmented path rather than a numerically-equivalent-but-different one.

### 2.2 The augmented `(U, Z)` path

The existing scheme is a Castro–Parés path-conservative fluctuation solver, not a
conservative flux-difference solver: `PVM.compute_fluctuation` computes
`∫₀¹ A(psi(s)) ds · (U_R - U_L)` by 5-point Gauss-Legendre quadrature along the linear
path `psi(s) = (1-s)U_L + s·U_R`. So topography can ride the *same* machinery by widening
the state:

```
Ã(W) = [ A(U)   +g*h*e_momentum ]      (n+1)x(n+1),  n = order+2,  W = (U, Z)
       [  0            0        ]
```

One extra column, nonzero only in the momentum row, and one all-zero row because `Z` does
not evolve. The quadrature loop is unchanged on `(n+1)`-vectors; only the first `n` rows of
the resulting fluctuation are kept.

> **Sign correction (Step 5).** This entry originally read `-g*h`. That is the sign the
> bed-slope term carries as a *right-hand-side source*; moved into the transport matrix on
> the left, where the augmented formulation puts it, it is **`+g*h`**. The wrong sign is
> not a small error — it makes the momentum row of `Ã·ΔW` at a lake at rest equal
> `g*h*(Δh - ΔZ) = 2*g*h*Δh` instead of zero, i.e. maximally *anti*-balanced. A test pins
> it by flipping that one entry and asserting the C-property breaks.

**Why this recovers the C-property.** At a lake at rest (`h + Z` constant, `u_m = alpha = 0`)
the linear path stays a lake at rest at every quadrature node, and each row of `Ã·ΔW`
vanishes identically: mass because `Δ(h·u_m) = 0`, momentum because
`g*h(s)*Δh + g*h(s)*ΔZ = g*h(s)*Δ(h+Z) = 0`, and the moment rows because every coefficient
is zero at rest. No Audusse-style hydrostatic reconstruction is needed.

**But that is only the central part of the fluctuation.** The full fluctuation is
`½(Ã ± Q)ΔW`, so the numerical viscosity `Q` must annihilate the equilibrium jump too.
Writing `Q = P(Ã)` for the scheme's viscosity polynomial, and noting the jump lies in
`ker(Ã)`, we get `Q·ΔW = P(0)·ΔW`:

> **A PVM scheme is well balanced exactly when `P(0) = 0`.**

That holds for **Roe** (`P(x) = |x|`) and **Osher**, and fails for **LF** (`P(x) = Δx/Δt`)
and **PRICE** (`P(x) = Δx/2Δt + Δt·x²/2Δx`), whose constant terms leave an
`O(Δx/Δt · Δh)` residual at rest. Not fixable by tuning the coupling — it is structural to
those schemes. Recorded as a `well_balanced` class flag; `ClassicalSimulation1D` warns when
topography meets a scheme that lacks it. Not a practical restriction: all 20 thesis configs
already use Roe.

---

## 3. Wet-dry treatment

Three layers, replacing the old `if h <= 0: raise RuntimeError`.

### 3.1 Two-tier threshold convention

`swme/wetdry.py` owns the single rule for reading primitives out of a state, so the
transport matrix, wave speeds, friction and recharge sources cannot disagree about what a
nearly-dry cell means.

| Threshold | Default | Meaning |
|:--|:--|:--|
| `eps_div` | 1e-14 | Machine-precision division guard; not physical |
| `h_dry` | 1e-4 | At/below: dry — no moments, velocity driven to zero |
| `h_wet` | 1e-3 | At/above: ordinary wet flow, no regularization at all |

Absolute depths, not ratios — sized for the thesis's `h ~ O(1)` and configurable per case
via a `[wet_dry]` section. **Choosing `h_dry` is a real trade-off**; see
[§6](#6-findings-and-limitations).

### 3.2 Desingularized primitive reconstruction

Applied *only* where primitives are extracted for closure evaluation — never to the
conserved state array itself.

```
u_m   = 2*h*q / (h**2 + max(h, h_dry)**2)          # identically q/h for h >= h_dry
ramp  = clip((h - h_dry) / (h_wet - h_dry), 0, 1)
alpha = alpha * ramp                                # -> plug flow as h -> h_dry
```

The moment ramp is physically motivated — a vanishing film has no meaningful vertical
velocity profile — and numerically stabilizing, since it stops `alpha_i/h` terms blowing up
in the friction closure.

> **Deviation from the original design.** This section originally wrote the
> Kurganov–Petrova denominator with `max(h, eps_div)` *and* separately demanded
> `u_m = 0 exactly` below `h_dry`. Those do not fit together: with `eps_div ~ 1e-14` the
> desingularization never activates above machine noise, so the only thing shaping the
> velocity would be the hard cut at `h_dry` — a jump of size `q/h_dry` sitting exactly at
> the wetting front, the worst possible place for a discontinuity. Using the *physical*
> threshold in the denominator gives the intended behaviour continuously and with no
> special case: identically `q/h` for `h >= h_dry` (the denominator is exactly `2h²` there),
> decaying quadratically to zero below. `eps_div` keeps its original job as the 0/0 guard.

For `h >= h_wet` the code takes the literal pre-Step-6 expressions, which is what keeps wet
runs bit-identical. Two places needed care to preserve that: `compute_max_wavespeed` keeps
its old expression verbatim behind a fully-wet fast path (`(a*a)/(h*h)` and `(a/h)**2`
agree mathematically but not to the last bit, and the wave speed sets `delta_t`), and the
`nu/h²` friction term feeds on the conserved moments rather than `h*alpha` above `h_wet`.

### 3.3 Positivity-preserving timestep limiter

The mass update of cell `i` is `h_i^{n+1} = h_i - dt/dx * (F⁺_{i-1,0} + F⁻_{i,0})`, so
wherever that net outflow is positive it must not exceed the water the cell holds. `dt` is
capped pre-emptively, replacing the old crash-after-the-fact. For a wet, CFL-limited flow
the cap is far looser than the CFL condition and never binds — which is why existing
results are untouched.

Cells already at or below `h_dry` are **excluded** from the limiter: they hold nothing to
protect, and a wet-dry interface can hand one a tiny spurious positive outflow for which
the only admissible timestep is exactly zero, deadlocking the run over a quantity smaller
than `h_dry`. Their round-off excursions are clamped instead, and the mass that creates is
accumulated in `mass_created_by_clamping` and warned about if it becomes non-negligible —
turning "we clamp and hope" into a checkable claim.

Schemes whose viscosity depends on `delta_t` (LF, PRICE) re-derive the limit up to
`_MAX_POSITIVITY_ITERATIONS` times; Roe and Osher need zero, since `|A|` ignores `delta_t`.
The old `RuntimeError` survives as the assertion that the limiter worked — it now fires
only on a genuinely *negative* height, since `h == 0` is a legitimate dry cell.

---

## 4. Package layout

```
recharge-paper/                    # repo root is the uv project root
├── pyproject.toml                 # hatchling; packages = ["src/swme", "src/recharge"]
├── README.md
├── RESTRUCTURE_PLAN.md            # this file
├── src/
│   ├── swme/                      # core solver
│   │   ├── coefficients.py        # §1 — absorbs symbolic_math/symbo.py
│   │   ├── source_terms.py        # §1 — base Navier-slip friction only
│   │   ├── topography.py          # §2 — bed profiles Z(x) + settings
│   │   ├── wetdry.py              # §3 — thresholds + primitive extraction
│   │   ├── mesh.py                # + bed_elevation
│   │   ├── pde.py                 # SWME1D only, genericized
│   │   ├── spatialDiscretization.py
│   │   ├── timeIntegration.py
│   │   ├── simulation.py          # Simulation (ABC) + ClassicalSimulation1D
│   │   ├── plotting.py            # Plotting (ABC) + SWME1DPlotClassical
│   │   ├── cli.py                 # YAML config -> solver; the moment-sw entry point
│   │   └── config/                # shipped .yaml cases (thesis + benchmarks)
│   └── recharge/                  # sibling package, NOT nested inside swme/
│       ├── context.py, laws.py    # order-independent, untouched
│       ├── source_terms.py        # mass source + mixing friction
│       ├── initial_conditions.py
│       └── recharge_pde.py        # `from swme.pde import SWME1D`
├── scripts/                       # run_thesis_configs.sh
├── processing/                    # thesis figure scripts (not installed)
├── results/                       # solver output (gitignored)
└── tests/
```

The distribution name and console script stay `moment-sw` for continuity; only the
*importable* package names changed. Install and run with `uv sync` then
`uv run moment-sw ...`.

---

## 5. Step log

Work through pending steps in order; each should be a separately reviewable and testable
unit. Do not skip the "write the test against the old code first" pattern — it is what made
the hardcoded-block deletions safe.

### Steps 0–4 — genericize and delete

Scaffolding, the coefficient engine, the package rename, regression tests against the
legacy code, the engine swap, and deletion of out-of-scope models. Complete; the details
live in git history. What still matters:

- **Two legacy bugs found.** The N=6 hardcoded system-matrix block disagreed with the
  generic implementation at **22 of 64 entries** — not just the one obvious
  `alpha5`/`alpha6` copy-paste slip. Ruled out as a generic-code error by an independent,
  legacy-free consistency check: since `A_ijk`/`B_ijk` do not depend on the truncation
  order, the order-N matrix with `alpha_N = 0` must reproduce the order-(N-1) matrix, and
  it does for every order 1–6. The N=6 block was simply deleted rather than preserved.
- **Golden fixtures — do not regenerate.** Before deleting the legacy code, its outputs
  were captured to `tests/data/legacy_golden.npz` (126 arrays) and
  `legacy_recharge_golden.npz` (30 arrays), and both regression suites rewritten against
  those. The safety net therefore survives the deletion permanently instead of evaporating
  with it. **N=6 system-matrix goldens came from the *generic* implementation**, since the
  legacy N=6 block is the buggy one; everything else came from the legacy code. Both test
  files record this provenance.
- **Performance cost, accepted.** Replacing inlined scalar arithmetic with numpy array
  expressions took the reference run 36s → 94s → **60s** after optimizing (precomputed
  slices on `Coefficients`, BLAS `@` instead of `einsum`, flat-stride diagonal update). The
  residual ~1.7x is per-call numpy dispatch overhead on tiny arrays, not algorithmic, and
  shrinks in relative terms as N grows. The real fix is batching the state over cells — see
  [Open items](#7-open-items).
- **Process lesson.** A helper script that located methods by matching `    def <name>(`
  also matched method-listing lines inside class docstrings, silently deleting a closing
  `"""` and breaking `pde.py`. Caught by an `ast.parse` check, not by tests. For bulk
  surgery on this codebase, prefer explicit line ranges or a real AST tool, and `ast.parse`
  after every structural edit.

### Step 4.5 — baseline validation against the thesis

A deliberate pause, at the user's request, to validate the refactor against the thesis's
own results *before* later steps changed numerics on purpose.

No work was needed to make the code *run* — it already did. What was missing was
convenience and two small gaps: `compute_vertical_velocity_profile` was still hardcoded
through N=6 and would **silently drop higher modes** for N≥7 (a silent-wrong-answer path
that Step 3 had made reachable), and this document contained a false bug report about that
same method's indexing, which was a misreading of the post-processed array layout and has
been removed.

Delivered: a `--config` / `--output-dir` / `--list-configs` CLI (pulled forward from Step
7), **20 Chapter 5 configs** transcribed from the thesis's runtime-parameter tables,
`scripts/run_thesis_configs.sh` to run them all into a `results/` layout, and all 7
`processing/*.py` figure scripts repointed from the original author's absolute paths.

| Script | Thesis section | Reads |
|:--|:--|:--|
| `ersoy_alpha_comparison.py`, `plotter.py`'s `[comparison]` | §5.1 | `Ersoy/ErsoyData{0,1,2}/` |
| `plotter.py` (single run) | §5.2 | `5p2_Horton_At_Rest/` |
| `non_wrapping_pulse_model_comparison.py`, `plot_non_wrapping_zoom_profiles.py` | §5.3 | `Non_Wrapping_Pulse/` |
| `smooth_pulse_model_comparison_cases.py` | §5.4 | `Smooth_Pulse/` |
| `inflow_outflow_comparison.py` | §5.5 | `Smooth_Pulse_Inflow_Outflow/` |
| `dry_wet_ablation_comparison.py`, `zoomed_dry_wet_comparison.py` | §5.6 | `Dry_Wet_Test/` |

> `Dry`/`Wet` there means source-free/source-active — a pre-existing naming choice with
> nothing to do with the actual wet-dry treatment of §3.

**Result:** all 20 configs succeeded, all 8 processing scripts ran clean, and three thesis
equations were confirmed numerically — §5.1 eq. (5.5) on all three `alpha_R` branches,
§5.2 eqs. (5.9)–(5.10) (t\* = 510.8 vs. ≈511; h(1800) = 1.1694 vs. 1.1694, rel. err
3.2e-6), and §5.6 eq. (5.17) (growth rate 0.200000 vs. 0.2).

One tradeoff was made here: output filenames briefly gained a config-name prefix to fix a
real collision, then **reverted**, because the pre-existing comparison scripts expect the
plain legacy naming inside a per-case subfolder. Uniqueness is now the caller's
responsibility via `--output-dir`, which `run_thesis_configs.sh` enforces.

### Step 5 — topography and well-balancing

Design in [§2](#2-bottom-topography-and-well-balancing); both sign and viscosity
corrections recorded there were *found by* the mandatory validation, exactly as this
document anticipated they might be.

New: `swme/topography.py` (seven bed profiles behind `get_bed_profile`, plus the frozen
`TopographySettings`), `mesh.bed_elevation` with boundary-consistent ghost cells, the
augmented matrix in `pde.py`, `lakeAtRest` / `perturbedLakeAtRest` initial conditions, a
`well_balanced` flag per scheme, a `[topography]` config section, and two runnable configs.

**Validation — `tests/test_topography.py`, 123 tests**, layered cheapest-first: the profile
library and ghost-cell filling; the structure of `Ã`; the algebraic C-property
(`Ã(W(s))·ΔW = 0` at all five Gauss nodes, N=0–6 × hyperbolic on/off × 20 random jumps, to
`< 1e-13`) with a negative control so it cannot pass vacuously; scheme-level fluctuations
(`< 1e-13` for Roe/Osher, `> 1e-3` for LF/PRICE — the limitation asserted, not hidden); and
the mandatory end-to-end lake-at-rest runs over six beds × N=0,1,2, plus HSWME, a viscous
run, and RechargeSWME.

Drift is genuinely zero rather than merely small. Measured on deliberately long runs
(N=2, 200 cells, `t_end = 5.0`, ~1400 steps):

| Bed | max free-surface error | max velocity | max moments |
|:--|:--|:--|:--|
| Parabolic bump | 2.2e-16 (one ULP of H=2) | 5.8e-16 | 0.0 exactly |
| Discontinuous step | **0.0 exactly** | 1.8e-15 | 0.0 exactly |

The fluctuations are zero to round-off at *every* step rather than small-and-accumulating,
which is why 1400 steps are no worse than one.

### Step 5.5 — hyperbolicity audit

Inserted at the user's request after they noticed a `ComplexWarning` during Step 5
verification. Complex eigenvalues of the transport matrix are the signature of
hyperbolicity loss, which for a moment model is the difference between a well-posed problem
and plausible-looking garbage.

**The warning turned out to be a red herring, and that was itself the bug.** numpy 2.x no
longer down-casts `np.linalg.eig` output to real when the spectrum happens to be real — it
returns `complex128` unconditionally. So the warning fired on *every* Roe step of *every*
run regardless of hyperbolicity: a 100% false-positive rate, which is exactly why it read
as harmless background noise. The real defect was the opposite of the one it appeared to
report — **a genuine loss of hyperbolicity was indistinguishable from normal operation.**

Fixed by making the viscosity explicitly real (bit-identical: numpy's implicit cast was
already doing this, and a real matrix's complex eigenvalues come in conjugate pairs with
equal moduli, so their contributions sum to something real either way) and adding a real
detector in its place, built from eigenvalues the scheme has *already* computed.

The physics that came out of it is in [§6](#6-findings-and-limitations).
**`tests/test_hyperbolicity.py`, 99 tests.**

### Step 6 — wet-dry treatment

Design in [§3](#3-wet-dry-treatment). **A dam break onto a dry bed now runs**, where before
any zero height raised `RuntimeError` on the first step. New `swme/wetdry.py`, a
`[wet_dry]` config section, `damBreak_dryBed` / `damBreak_dryBed_partial` initial
conditions, and a `wetdry_dam_break` config. **`tests/test_wetdry.py`, 62 tests.**

Mass conservation on the dry dam break is **exact** — 0.0e+00 at every resolution tested.

**Also fixed here: rain could not wet dry ground.** `RechargeSWME1D.compute_source_term`
returned an all-zero source for any cell at or below its dry tolerance, which made it
impossible for rainfall to ever wet a dry cell — in a rainfall-runoff model. The mass row
`S[0] = R - I` needs no primitives at all, and every other row vanishes on its own once the
wet-dry rule zeroes the velocities and moments, so the short circuit was removed rather
than special-cased. Infiltration out of a dry cell stays bounded because the closures
already cap `I` at the available `h/dt`.

**And a correction to Step 5.5's own detector**, which a wet-dry front exposed — see
[§6](#6-findings-and-limitations).

### Step 7 — CLI rewrite

`swme/cli.py` replaces `main.py`, and every shipped config is now YAML. The numerics are
untouched: the same objects are built with the same values, so thesis configs reproduce
their CSVs **byte for byte** through the new entry point.

The migration was scripted rather than hand-typed — a converter walked each `.ini`,
emitted numbers verbatim from the source text (never round-tripped through `float`), and a
check then compared every INI-derived value against its YAML twin: **24/24 identical, all
24 building a complete solver end to end.** Six keys were dropped as dead, verified against
`main.py` as read by nothing: `frictionModel`, `start_order`, `structuredGrid`, `1D`,
`resolutionY`, `y1boundary`/`y2boundary` — leftovers from models deleted in Step 4.

**The format change was the means, not the point.** INI silently absorbed mistakes:
`configparser` returned strings, so every use site needed `getfloat`/`getboolean`/`getint`,
and a misspelled key fell through to a default rather than failing. `cli.py` validates the
section set and each section's key set, so `viscosty:` now names itself instead of quietly
running with the default viscosity. Values are checked by type at the boundary, and a
quoted `"false"` — truthy in any naive loader — is rejected rather than silently enabling
the implicit source path.

**One YAML trap worth knowing.** PyYAML implements YAML 1.1, whose float pattern requires a
*signed* exponent: `1.0e-3` is a float but `1.0e3` is a **string**. The shipped configs
happened to use signed exponents throughout, but the converter normalises the form anyway
and `cli.py` coerces numeric-looking strings with an error message that names the rule.

**Three latent defects fixed while rewriting:**

- Unsupported values used to `print` and continue — `'PDE_type is not implemented yet'` was
  followed by a `NameError` on the undefined `_pde`, and the pvm and time-integrator
  dispatches had the same shape. They raise now.
- `linear_source: true` returns a *matrix* that only `ImplicitEuler` applies. Pairing it
  with any other integrator was a silent wrong answer, prevented only by how `main.py`
  happened to compose them. It is now a startup error.
- **Defect D1 from Step 8.5 resolved early.** All CSV output sat inside the
  `RechargeSWME1D` branch, so a plain SWME1D/HSWME1D run — including the shipped topography
  and wet-dry cases — produced no numerical artifact at all. Output is model-agnostic now
  (`swme_N1_final.csv` and friends); recharge filenames are unchanged, because
  `processing/*.py` matches them exactly.

**Two deliberate behaviour changes**, neither touching numerics: plotting is now opt-in via
`--plot`, since a solver that blocks on an interactive window at the end of every run is
wrong for scripted use — and was the reason `run_thesis_configs.sh` has to set
`MPLBACKEND=Agg`. And `main()` returns `None`: the console-script wrapper calls
`sys.exit(main())`, so returning the data array made `sys.exit` print it to stderr and exit
**1**, which looks exactly like a crash even though the run and its CSVs were fine. Callers
who want the array should use `cli.run()`, which returns it.

**Validation.** `tests/test_cli.py`, 33 tests (suite 587 → 620): every shipped config loads
*and builds*; thesis parameters spot-checked against the thesis tables directly; no config
value silently parsed as a string; and one test per failure mode the rewrite exists to
catch. Byte-identity re-confirmed against `results/` after the swap.

### Step 8 — cleanup

Wider than the one-line scope this document originally gave it, because Steps 4–7 left more
debris than `requirements.txt` and the `Makefile`. **No numerics moved** — every change is
either a deletion of provably-unused code or a change to what a run prints / where it writes
by default. Byte-identity re-confirmed against `results/` afterwards.

**Deleted.** `src/swme/Makefile` (its `purge` rule was broken as written: CWD-relative paths
that resolved to a directory which never existed); `src/swme/requirements.txt` (stale pins,
and missing `sympy` and `pyyaml`, both hard runtime imports); the empty `Data-processing/`
tree; `recharge/laws.py`'s three `horton_test*` demos and `__main__` block (**244 lines**,
imported by nothing, with a default argument pointing at the deleted `config.txt` via a path
that never resolved); `mesh.py`'s `UniformRectangularMesh2D` (fully orphaned — one repo-wide
hit, its own `class` statement); and `RechargeSWME1D`'s write-only `eps_dry` parameter.

**Quieted.** `run_simulation` printed four lines *per timestep* — 1.9 MB of log for one
thesis case. Now behind `ClassicalSimulation1D.verbose` (default off) and `moment-sw
--verbose`. Measured: a `wetdry_dam_break` run went from thousands of stdout lines to **6**.
Relatedly, `cli.py` imported `swme.plotting` — and therefore `pyplot` — at module scope, so
every scripted run paid for a matplotlib import it never used; that is now inside the
`if plot:` branch, which also retires the last reason `run_thesis_configs.sh` needs
`MPLBACKEND=Agg`.

**Output directory.** The default was still `Data-processing/Results/Recharge`, a path from
before `results/` was the convention. It is now `results/<config-name>/`. That is not
cosmetic: runs previously shared one directory *and* one set of filenames, so a second run
silently overwrote the first — the exact hazard Step 4.5 hit, and the reason a `purge` tool
was wanted in the first place. Segregating by config name removes it at the source.

**`purge` re-homed as a curation tool, not an `rm -rf`.** `src/swme/purge.py`, registered so
`uv run purge` works. Its stated original purpose was keeping track of which runs were worth
keeping, which is a *seeing* problem — so listing is the default (name, size, file count,
last modified, newest first), deletion needs `--delete` plus a pattern, and it confirms
before removing anything.

**Docs.** Both package READMEs described a layout that no longer exists — `main.py`,
`config/config.txt`, `recharge/` nested inside `swme/`, `Data-processing/`, and the deleted
vegetation / Hermite / adaptive / micro-macro models; `src/recharge/README.md` told you to
`pip install -r requirements.txt`, a file that never existed there. Both are now short
pointers at the root `README.md` and this document, which is the only way they stop drifting.
Also fixed: a stale `main.py` reference in `run_thesis_configs.sh` that was inside the range
`--help` prints, a `.ini` filename in a config comment, and `.gitignore`'s no-op `data/` rule
(no such directory) plus missing build/cache entries.

**No dependency was removable** — all seven are genuinely imported. Worth recording anyway:
`scipy`'s only use is `spopt.newton` in `timeIntegration.py`, reachable solely through
`ImplicitEuler`, which **no shipped config selects** (tests only).

### Remaining steps

- [x] **Step 7 — CLI rewrite.** Done; see the entry above.
- [x] **Step 8 — cleanup.** Done; see the entry above.
- [ ] **Step 8.5 — post-processing suite.** Phase 1 done; see the entry below and the design
      that follows it.
- [ ] **Step 9 — documentation site.** `mkdocs` + `mkdocs-material` + `mkdocstrings` (the
      `docs` dependency group already exists). `mkdocs.yml` at repo root and a `docs/` tree:
      index, quick start, model overview (adapted from §1–§3), configuration reference, and
      an API reference generated from docstrings. Do this *after* Steps 7–8.5 so the docs
      describe the final structure.

#### Step 8.5 Phase 1 — the producer: diagnostics defects, and a metadata sidecar

Step 8.5 lands in three phases: the producer first, then the report skeleton, then
hyperbolicity. Defects before consumers, because **these code paths had never executed** —
`store_hyperbolicity` is false in all 20 shipped configs and every `*_hyperbolicity_*.csv` on
disk was 1 byte — so building the report on them first would have meant debugging first-run
producer code and new rendering code simultaneously.

**D4 was worse than the plan estimated, and the measurement is the point.** The first real
`store_hyperbolicity = True` run in this project's history — a dam break onto a dry bed, N=1,
200 cells — would have reported **100 of 200 cells non-hyperbolic at t = 0**, falling to 65 by
t = 0.195. Every one of them was a *dry* cell. N=1 SWME is unconditionally hyperbolic (§6: the
spectrum is exactly `{u_m, u_m ± sqrt(g·h + alpha_1²)}`), so that is a **100% false positive
rate**, and it is the same failure mode as the `ComplexWarning` of Step 5.5 and the
interface-counter warning of Step 6 — the third time this project has caught a diagnostic
blaming the model for the wet-dry treatment. The summary now separates
`num_nonhyperbolic_cells` (0 here, correctly) from `num_dry_cells`, `num_failed_cells` and
`num_evaluated_cells`, and `fraction_nonhyperbolic_cells` divides by cells actually evaluated —
otherwise a 90%-dry run reports a reassuringly small fraction for the few wet cells that
genuinely broke.

**D3 confirmed on the same run.** `max_abs_imag_eig` was NaN at every step and
`worst_cell_index` was 199 — the last dry cell in index order — because
`if np.isnan(max_abs_imag) or ...` latched the running maximum to NaN, after which every
comparison was False. It now tracks the worst *finite* spectrum: 0.0 exactly, at cell 0, a wet
cell. Per-cell rows carry NaN rather than 0 in `is_hyperbolic` for cells that were never
evaluated, so the two files agree about what was skipped.

**D6, found during the audit.** The two hyperbolicity CSVs carried a hardcoded `recharge_`
stem, ignoring the run's `prefix` unlike the other three files — so a plain SWME run wrote
files claiming to be recharge output, and two models sharing an output directory overwrote each
other's. Also fixed: the end-of-run warning called `spectra_examined` a count of "interfaces",
which is wrong by 5× for Osher (one line of D5; the rest is a presentation requirement for the
report).

**`{prefix}_run.json`, a metadata sidecar.** The CSVs carry only `[x, h, u_m, a_1..a_N]`, which
leaves a report rebuilt from disk unable to name the flux scheme or draw the bed. Neither gap is
cosmetic: the interface counters are uninterpretable without the scheme (Roe records one path
average per interface, Osher five weight-scaled node matrices, LF/PRICE none), and the
topography page has no bed at all. Widening the CSVs would have broken byte-comparison against
every result in `results/`, so this sits *beside* them. It records the run rather than copying
the config — a config can be edited afterwards, and `--output-dir` means the config-to-directory
map cannot be inverted anyway. The bed is stored as a profile name plus parameters, not a
sampled array: `TopographySettings` carries a closure that will not serialise, and
`get_bed_profile(name, **params)` at the CSV's own `x` reproduces the mesh's sampling to
round-off (~3e-16; `x` itself loses ~1 ULP through the CSV).

**`build_simulation` split out of `run()`**, so a caller can get a real simulation object
without also acquiring an output directory, CSV writing and console printing.

**Validation.** `tests/test_diagnostics.py`, 12 tests (suite 620 → **632**). Byte-identity
re-confirmed against `results/` on both a recharge case (`Ersoy/ErsoyData0`) and the wet-dry
thesis case (`Dry_Wet_Test/Dry_N1`) — all CSVs identical, which is what makes this safe: every
path touched is either skipped entirely when `store_hyperbolicity` is false, or writes a file
that was empty on disk.

#### Step 8.5 design — an in-package post-processing suite

**Goal.** A `swme/report/` subpackage that renders a multi-page PDF about a single run,
covering the fields, time histories, **vertical velocity profiles** and **hyperbolicity
diagnostics**. Runs after Step 7 (the CLI is what invokes it) and before Step 9 (the docs
should describe it).

**The capability mostly exists already — in the wrong place.** `processing/plotter.py`
(1,560 lines) already has a config schema, multi-format saving, column validation,
N-agnostic moment auto-detection, velocity profiles and *seven* hyperbolicity plots. So this
step is a **promotion and hardening of its generic core**, not a build from scratch. Three
things make the promotion worth doing rather than just running the script:

- It is unusable as a library: `CONFIG_FILE` is a relative path read at *import* time, which
  also `mkdir`s the output directory, so it only works with `cwd = processing/`. Every
  plotting function reads a module-level `CFG` instead of taking arguments, and it writes one
  file per figure — there is no multi-page PDF anywhere in the repo.
- Its hyperbolicity plots have **never once run**: every toggle is false and every
  `recharge_hyperbolicity_*.csv` on disk is empty, because `store_hyperbolicity = False` in
  all 20 shipped configs. Those seven code paths are unexercised and need first-time testing.
- The profile maths is hand-rolled in **six** copies across `processing/`, each with its own
  `phi_1`/`phi_2` and **capped at `a2`** (only `plotter.py` reaches `a3`). That is the same
  silent-truncation defect Step 4.5 fixed inside the package, still live outside it. Use
  `SWME1D.compute_vertical_velocity_profile` (backed by `coefficients.eval_phi`) instead.

**Shape.** `swme/report/` with `data.py` (a `RunData` dataclass plus two loaders),
`pages.py` (one function per page, each taking explicit arguments and returning a `Figure`),
`style.py`, and `cli.py`. No module-level config, no import-time side effects, no `cwd`
dependence — the specific lesson from `plotter.py`.

`RunData` is the single intermediate representation, so pages never know where the data came
from: `from_simulation(...)` reads a finished run in-process, `from_directory(...)` rebuilds
from CSVs on disk. Every field except the final state is optional, which is what lets the
report degrade gracefully on a run that captured less.

**The directory loader must not read `field_history.csv` naively.** The largest on disk is
185 MB / 1.54M rows and `results/` totals 1.4 GB. Take time series from the small
`summary_history.csv` and stream the field history with `chunksize`, keeping only a bounded,
evenly-spaced set of snapshot steps. Peak memory must be independent of run length.

**Pages:** cover/run summary; final state (N-agnostic, no N≤6 cap); time histories;
`h(x,t)` and `u_m(x,t)` space-time maps; vertical velocity profiles; hyperbolicity;
wet-dry (only if any cell went below `h_wet`); topography with the `h+Z−H` residual (only
if `mesh.has_topography`).

**The hyperbolicity page must not undo §6.** The scheme counters measure the path-averaged
*interface* matrix, not `A(U)` — and a wet-dry front trips them at N=0, plain shallow water.
A report printing one "hyperbolicity" number would resurrect exactly the false-positive
failure mode Steps 5.5 and 6 fixed. Keep two clearly separated panels: the always-available
scheme-level counters, captioned as *not* evidence of model loss on their own; and the
model-level per-cell view, shown only when `store_hyperbolicity = True` and replaced by an
explicit "not captured, enable it with…" placeholder otherwise — never a blank, never an
implied all-clear.

**Wiring:** `moment-sw` gains `--report / --no-report` (default on) writing
`<output-dir>/report.pdf`; a new `moment-sw-report <dir>` console script rebuilds one from
CSVs, so the 20 existing thesis runs get reports without re-running anything.

**Six defects sit in the data the report would consume**, and plotting them as-is yields a
report that looks authoritative and is wrong. D1 and D2 were fixed early, in Step 7; D3, D4
and D6 are fixed in Phase 1 of this step; D5 is a presentation requirement rather than a code
change, and lands with the hyperbolicity page.

- **D1** *(fixed in Step 7)* — `main.py` set `store_history`/`store_hyperbolicity` and wrote
  *all* CSVs only inside the `RechargeSWME1D` branch, so the three non-recharge configs
  (`topography_lake_at_rest`, `topography_perturbed_lake`, `wetdry_dam_break`) captured
  nothing at all.
- **D2** *(fixed in Step 7)* — the hyperbolicity CSVs were written *nested inside*
  `if len(history) > 0`, so `store_hyperbolicity=True` with `store_history=False` wrote
  nothing, while the shipped default wrote two empty files on every run. Each is now gated on
  its own list.
- **D3** — `simulation.py`'s worst-cell tracker read
  `if np.isnan(max_abs_imag) or max_abs_imag > max_abs_imag_global`. Once a dry or failed
  cell set the running max to NaN it latched: `max_abs_imag_eig` stayed NaN for that step and
  `worst_cell_index`/`worst_x` pointed at the last NaN cell, not the worst spectrum.
- **D4** — dry cells got `is_hyperbolic = 0`, so they inflated `num_nonhyperbolic_cells`.
  On a wet-dry run that conflates "dry" with "ill-posed", the two things Step 6 worked
  hardest to separate.
- **D5** (presentation, not a code fix) — `Osher` records 5× per interface on weight-scaled
  single-node matrices rather than the path average, and `LF`/`PRICE` never eigendecompose
  at all, so their counters stay 0 — a vacuous zero, not a reassuring one. Osher's
  `max_abs_imaginary_eigenvalue` is scaled by its quadrature weights (0.118–0.284), so it is
  not comparable to Roe's either, and the fixed `1e-10` threshold is up to 8.4× stricter in
  `A`-units. The report must name the scheme and must never render "0 of 0" as an all-clear.
  One line *was* a code fix: the end-of-run warning called `spectra_examined` a count of
  "interfaces", which is wrong by 5× for Osher.
- **D6** — the two hyperbolicity CSVs were written with a hardcoded `recharge_` stem,
  ignoring the run's `prefix` unlike the other three files. A plain SWME run produced files
  claiming to be recharge output, and two models sharing an output directory collided on them.

**Validation.** `tests/test_report.py`, plus the first `tests/conftest.py` (forcing
`matplotlib.use("Agg")` — there is currently no matplotlib anywhere in `tests/`). Cover: a
non-empty PDF with the expected page count; graceful degradation with each optional data
source missing; both loaders agreeing for the same run; the profile page matching
`compute_vertical_velocity_profile` directly; loader memory independent of history length;
and all seven ported hyperbolicity plots rendering against a real
`store_hyperbolicity = True` run. End-to-end, `--config wetdry_dam_break` must produce a
report — that doubles as the D1 regression.

**Scope guard.** Single-run reports only. `processing/` and `swme/plotting.py` are *not*
modified; the thesis Chapter 5 multi-run comparison figures stay where they are. Two
adjacent items are noted but not undertaken: once this lands `processing/plotter.py` is
fully superseded and could be retired along with the ~30–35% of near-verbatim helper
duplication across the six comparison scripts; and `run_simulation` prints four lines per
timestep (a 1.9 MB `run.log` for one case) while `main.py` calls a blocking `plt.show()`
unconditionally — both hostile to batch reporting.

---

## 6. Findings and limitations

Measured, not assumed. Each is pinned by a test so it stays documented rather than being
rediscovered as a mystery.

### Hyperbolicity: SWME vs. HSWME

Hyperbolicity depends **only** on the scaled moments `alpha_i / sqrt(g*h)` — verified
Galilean invariant (shifting `u_m` leaves the imaginary parts bit-for-bit unchanged) and
exactly linear under `h → c²h, alpha → c·alpha`. So the maps below are universal.

- **N=0 and N=1 SWME are unconditionally hyperbolic.** The N=1 spectrum is exactly
  `{u_m, u_m ± sqrt(g*h + alpha_1²)}`, verified against the closed form to 7e-14 out to
  `|alpha_1| = 50`. At those orders HSWME *is* SWME, so no N≤1 run can ever lose
  hyperbolicity whatever the forcing does.
- **From N=2 up, SWME loses it**, and the affected fraction of state space grows fast
  (uniform sampling, `h ∈ [0.2,4]`, `|alpha| ≤ 3`, 20k states per order):

  | N | SWME non-hyperbolic | HSWME |
  |--:|--:|--:|
  | 1 | 0.00 % | 0.00 % |
  | 2 | 3.07 % | 0.00 % |
  | 3 | 11.24 % | 0.00 % |
  | 4 | 21.06 % | 0.00 % |
  | 5 | 34.59 % | 0.00 % |
  | 6 | 48.03 % | 0.00 % |

- **The N=2 unstable set is a narrow wedge, not a magnitude threshold** — counter-intuitive,
  and it matters for choosing a safety criterion. It is confined to slopes
  `|alpha_2/alpha_1| ∈ [1.14, 1.40]`; every ray outside that range stays hyperbolic at every
  magnitude probed (to `|alpha|/sqrt(g*h) = 60`). Concretely `alpha = (1.5, 1.8)` is
  non-hyperbolic while the strictly larger `(2.0, 3.0)` is fine. **"Keep the moments small"
  is the wrong criterion; the ratio is what matters.**
- **HSWME never loses hyperbolicity** at any order tested (0 of 20 000 states at each of
  N=1..6, `max|Im| = 0.0` exactly), and repairs every sampled state that breaks SWME. The
  mechanism is visible in the spectrum: zeroing `alpha_2..alpha_N` in the transport matrix
  makes the eigenvalues independent of those moments entirely, while preserving the outer
  wave speeds `u_m ± sqrt(g*h + alpha_1²)`.

**The thesis runs are clean, with structural margin.** Every state the Chapter 5
simulations visited was replayed through both closures straight from the validated CSVs:
**14 417 060 states across all 20 runs, zero hyperbolicity loss**, SWME and HSWME alike.
Not luck — the largest scaled moment reached anywhere is `|alpha_1|/sqrt(g*h) = 0.71`, and
the initial conditions set `alpha_2 = -0.5·alpha_1`, a ray outside the unstable wedge and
therefore hyperbolic **at any magnitude** (confirmed to `|alpha_1|/sqrt(g*h) = 200`).

### The runtime hyperbolicity warning measures the *scheme*, not the model

The always-on counter eigendecomposes the **path-averaged interface matrix**
`Σ_k w_k A(psi(s_k))`, not `A(U)` at any state. `A(U)` is nonlinear in `U`, so that average
is not `A(anything)` and need not be hyperbolic even when every matrix being averaged is.

A dam break onto a dry bed trips it at **78 of 12 462 interfaces at N=0** — plain shallow
water, unconditionally hyperbolic, with no moments to destabilize — and the count is
*identical* at N=1, N=2 and HSWME, which is what gives it away. Meanwhile `A(U)` itself was
checked at all **62 310** states those runs visited: `max|Im| = 0.0` exactly.

The warning text says so explicitly and points at `store_hyperbolicity = True` for the
cell-by-cell model-level spectrum. **Had this not been caught, the detector added in Step
5.5 would have blamed the model for every wet-dry run — the same failure mode as the
`ComplexWarning` it replaced.**

### `h_dry` is a modelling decision, and a vacuum front is where it bites

At a dry front the *exact* solution contains arbitrarily small depths — Ritter's `h`
vanishes quadratically — so the leading edge sits below any fixed `h_dry` and is damped.
The computed front comes out too slow, and **refining the mesh does not help**, because it
only resolves more of the truncated region. Measured on the standard dry dam break at 800
cells against an exact front speed of 2.0:

| `h_dry` | Front speed | Error |
|:--|--:|--:|
| 1e-4 (default) | 1.758 | 12 % |
| 1e-8 | 1.934 | 3 % |
| 1e-12 | 1.984 | 0.8 % |

The trade-off runs the other way too: `h_dry` also floors the `nu/h²` friction term, so
shrinking it without bound is not free — at `h_dry = 1e-10` that term reaches `nu·1e20`,
which is finite (so no `isfinite` check catches it) and meaningless.

> **Rule of thumb:** put `h_dry` well below the smallest depth the problem must resolve,
> then check that `nu/h_dry²` is still a sane number. An inviscid problem has no lower
> limit; a viscous one does.

### Viscous drying needs the implicit source path

Navier-slip friction carries `nu/h²`, which near a drying front is stiff however `h_dry` is
chosen — the floor only caps it at `nu/h_dry²`, still ~1e6 at the default. Explicit
integration goes unstable: **the moment blows up first and drags the height negative**, so
the symptom points at the positivity limiter, which is not at fault.

Measured: fails for viscosities from 1e-4 up and for `h_dry` from 1e-4 to 1e-2 alike — so
it is genuinely stiffness, not threshold tuning — and succeeds at every setting with
`linear_source = True` + `ImplicitEuler`, which is exactly what that path exists for. The
`RuntimeError` names this cause when it fires on a viscous explicit run.

### Runs overshoot their configured `t_end`

`run_simulation`'s loop is `while t < t_end` with no final partial step, so a run ends up to
one timestep *past* the requested time, by a resolution-dependent amount. Pre-existing and
unrelated to wet-dry.

**Deliberately not fixed.** Correcting it changes every validated result, and the thesis
comparisons are the thing this restructure exists to preserve. It is recorded here because
it matters whenever a result is read at a specific time — comparing against a closed-form
solution, or comparing two resolutions to each other.

---

## 7. Open items

Not blocking; decide at implementation time.

- **Batch/vectorize the state over cells.** The solver calls `compute_system_matrix` once
  per interface per quadrature point per timestep on arrays of size `N+2`. At small N the
  per-call numpy dispatch overhead dominates the arithmetic by orders of magnitude — this
  is the whole ~1.7x cost from Step 3. Evaluating the whole grid in one batch would remove
  it entirely and would have sped up the legacy structure too. A sizeable refactor of the
  time loop, deliberately not bundled into this restructure, but it is the correct answer
  to the performance question and worth doing before any large production runs.
- **Soft upper bound on N** (~10–12 estimated, where the moment closure itself becomes
  physically questionable, independent of engine performance) — worth documenting once
  arbitrary N is exercised, not enforcing as a hard cap.
- **`recharge/initial_conditions.py`** has small per-order-capped hardcoding (effectively
  N≤2 for one profile), trivially loop-generalizable. Low priority: test-IC construction,
  not core physics.
