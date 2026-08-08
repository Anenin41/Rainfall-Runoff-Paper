# `swme`/`recharge` Restructure Plan — Arbitrary-N, Well-Balanced, Wet-Dry SWME/HSWME/RechargeSWME Solver

*(Originally titled the "`moment_sw` Restructure Plan" — the package has since been
renamed to `swme` with `recharge` as a sibling package, per Step 1.5 below. Left the
historical title context in the note above rather than rewriting every prose mention of
`moment_sw` throughout §1-4, which describe design decisions made before the rename and
are unaffected by it — see decision #5 for the mapping.)*

**Status: IMPLEMENTATION IN PROGRESS — Steps 0, 1, 1.5, and 2 complete: scaffolding,
coefficients engine, package rename/config cleanup, and the full regression-test suite
(185 tests) proving generic-N implementations of the system matrix, base Navier-slip
friction, and recharge mass-source/mixing-friction match the legacy hardcoded code (with
two real bugs found and fixed along the way — see Step 2's entry below). The generic
implementations exist and are tested but are NOT YET WIRED IN — `pde.py`/
`recharge/recharge_pde.py` still run their original hardcoded code paths unchanged (the
end-to-end solver run is byte-for-byte identical to before Step 2). Step 3 (actually
wiring the generic engine in and deleting the hardcoded blocks) is next.**
This file is the single source of truth for this restructure. Any agent picking up this
work should read this file first, update the checkboxes/status notes as work lands, and
avoid re-deriving the design decisions below (they've already been made and are recorded
with justification). Keep this file in sync with reality as the restructure proceeds —
if a decision changes during implementation, update the relevant section here, don't just
silently diverge from it.

## Context: why this restructure

`moment_sw/` is a 1D finite-volume research solver for the Shallow Water Moment Equations
(SWME), their hyperbolic-regularized variant (HSWME), and a rainfall-runoff extension
(`RechargeSWME1D`) implementing the model derived in the accompanying thesis
(*"A Rainfall-Runoff Extension of the Shallow Water Moment Framework"*). The solver works
correctly but has two structural problems that block further use of it:

1. **Every moment order N is hand-hardcoded.** `pde.py`'s `SWME1D.compute_system_matrix`
   has explicit `if order == 0/1/.../6:` blocks with fully-expanded closed-form
   coefficients (clearly transcribed by hand from a symbolic derivation), and this pattern
   repeats across `compute_system_matrix_diff`, the friction source term, a ~1700-line
   `_compute_source_matrix_inverse`, and `compute_vertical_velocity_profile`, capping the
   base model at N=6. The `RechargeSWME1D` extension is worse: `recharge/source_terms.py`
   has fully separate hand-written functions per order and hard-raises
   `NotImplementedError` for any `order not in (0, 1, 2)`. Appendix C of the thesis gives
   the fully general formulas for arbitrary N — this restructure replaces the hardcoded
   blocks with a generic tensor-contraction engine implementing those formulas.
2. **No topography, no wet-dry treatment, no well-balancing.** Confirmed by full-repo
   grep: zero references to bed elevation, hydrostatic reconstruction, or dry-cell
   handling anywhere in `moment_sw/`. The solver currently hard-crashes
   (`RuntimeError`) the instant any cell height goes non-positive. The `matlab/`
   reference code the user provided was checked and **also has none of this** — it's a
   flat-bed-only Roe-averaged Rusanov-type scheme with unguarded `h`-divisions that would
   itself NaN on a dry cell. So this part of the restructure is new numerical-methods
   design, not a port.

Additionally: the important models going forward are **SWME, HSWME (the `hyperbolic=True`
flag on the same `SWME1D` class), and RechargeSWME1D** only. The `HermiteMomentEquations`
(Boltzmann/BGK model — different physics entirely), `VegetationSWME1D` (vegetation drag
extension), and all spatially-adaptive/micro-macro simulation machinery are out of scope
and should be deleted to debloat the codebase, per explicit instruction.

Finally, the package currently has no `pyproject.toml`/build system — just a
`moment_sw/requirements.txt` and a manual `python3 main.py` entry point reading a single
hardcoded `config/config.txt`. This restructure also introduces `uv`-based packaging.

## Key existing assets discovered during research (reuse these, don't rebuild)

- **`symbolic_math/symbo.py`** already implements, generically in N, exact-rational
  (`sympy.Rational`) computation of the tensors `A_ijk`, `B_ijk`, `C_ij` (via
  `build_shifted_legendre_basis`, `compute_A` — using a Wigner-3j closed form,
  `compute_B`, `compute_E_F_C`) and vectors `r_i, s_i`. It currently only pretty-prints to
  stdout via a CLI (`python3 symbo.py --A --N 5`) — no return-value API. **This is the
  mathematical engine to build the new `moment_sw/coefficients.py` module from.**
- Closed forms (already proven, thesis Appendix B.1/B.4 — no sympy needed for these four):
  `r_i = (-1)^i`, `s_i = 1`, `E` is lower-triangular with `E_ii = i/(2i+1)`,
  `E_ij = (-1)^(i+j)` for `j<i`, `0` for `j>i`; `F` is lower-triangular with
  `F_ii = i/(2i+1)`, `F_ij = 1` for `j<i`, `0` for `j>i`.
- Basis convention used throughout: `phi_i(z) = P_i(1 - 2z)` (shifted Legendre on `[0,1]`),
  `phi_i(1) = (-1)^i`, `phi_i(0) = 1`.
- **Cross-validated finding (do not re-derive):** the base SWME/HSWME system matrix, the
  base friction source, `RechargeSWME1D`'s mass source `S_{R,I}`, and its friction block
  `P(U)` **all reduce to the same generic tensor-contraction formulas** (given in full
  below). `RechargeSWME1D` is not separate physics from `SWME1D` — it's the same generic
  `P(U)` with `f_R, f_I` possibly nonzero, plus an additive `S_{R,I}(U)` term with
  `R=I=0` reproducing plain SWME. This is why one shared engine can serve all three models.
- `compute_number_of_variables` (`= order+2`), `compute_max_wavespeed`, and
  `convert_to_primitive` in `pde.py` are **already generic** loops over arbitrary order —
  keep these as templates for style/structure.
- `recharge/laws.py` and `recharge/context.py` have **zero order-dependence already** —
  leave them essentially unchanged.

## User decisions locked in (from clarifying questions — do not re-litigate)

1. **Well-balanced scheme: augmented-path-conservative topography** (not the simpler
   central-difference fallback). See §3 below for the full design.
2. **`symbolic_math/`**: absorb its math into `moment_sw/coefficients.py`, then **delete
   the `symbolic_math/` directory** from the repo (user has their own copy elsewhere; no
   need to keep a wrapper CLI in-repo).
3. **Batch parameter-sweep scripts** (`main_HME_errorChecks.py`,
   `main_SWME_errorData.py`): **delete, do not replace.** Arbitrary-N support plus a clean
   CLI makes ad-hoc sweeps easy to script later if ever needed; out of scope now.
4. **`matlab/`: delete in its entirety** (added after Steps 0-1 landed). Nothing in it is
   needed — its transport-matrix math, numerical scheme, and friction closure are all
   superseded by the Python solver, and it has zero topography/wet-dry content (the one
   thing that would have justified keeping it as reference for Steps 5-6). See Step 4.
5. **Package layout: rename `moment_sw` → `swme`, and un-nest `recharge` as a sibling
   top-level package** (added after Steps 0-1 landed, so Steps 0-1 were implemented under
   the old `src/moment_sw/{*, recharge/}` name; **executed in Step 1.5, done** — the repo
   on disk is now `src/swme/` + `src/recharge/`). Every `moment_sw.*` reference elsewhere
   in this document (§1-4 below, written before this decision) means what is now
   `swme.*`; `recharge/*` is the sibling top-level package `src/recharge/`, not nested
   inside `swme/`. `recharge` imports from `swme` as a normal cross-package import (e.g.
   `from swme.pde import SWME1D`), not a relative parent-package import.
6. **Config format: real YAML**, not `configparser`/INI (added after Steps 0-1 landed;
   Step 0's `config/config.txt` was fixed to *run*, not redesigned). `pyyaml` is now a
   dependency (added in Step 1.5). Step 1.5 renamed `config.txt` → `config.ini` (same
   `configparser`-readable content, `main.py` still reads it) and added
   `config/example.yaml` as an unconsumed stub sketching the target schema. The actual
   loader swap — `cli.py` (Step 7) using `yaml.safe_load` and native YAML types instead
   of `configparser`'s `getboolean()`/`getfloat()` string coercion — is still pending,
   deliberately deferred to Step 7 since `main.py` gets fully rewritten there anyway.

---

## 1. Generic-N coefficient engine — `moment_sw/coefficients.py`

New module. Owns *all* math related to the shifted-Legendre moment basis and its
projection tensors. Ported from `symbolic_math/symbo.py` (basis construction, `compute_A`
via Wigner-3j, `compute_B` via the `J_j(z)` antiderivative table) plus the closed forms for
`r, s, E, F` listed above (validate the closed forms once, in a test, against
`symbo.py`-style sympy integration for N=0..8, then trust them at zero runtime cost).

```python
@dataclass(frozen=True)
class Coefficients:
    N: int
    A: np.ndarray        # (N+1,N+1,N+1) float64
    B: np.ndarray        # (N+1,N+1,N+1) float64
    C: np.ndarray        # (N+1,N+1)     float64
    E: np.ndarray        # (N+1,N+1)     float64
    F: np.ndarray        # (N+1,N+1)     float64
    r: np.ndarray        # (N+1,)        float64, = (-1)^i
    s: np.ndarray        # (N+1,)        float64, = 1
    phi_at_1: np.ndarray # (N+1,) = (-1)^i
    phi_at_0: np.ndarray # (N+1,) = 1

@lru_cache(maxsize=None)
def get_coefficients(N: int) -> Coefficients: ...

def eval_phi(N: int, z: np.ndarray) -> np.ndarray:
    """(N+1, len(z)) float64, shifted-Legendre phi_i(z) via float-domain Bonnet
    recurrence. Used only for post-processing (vertical velocity profile
    reconstruction), never on the hot simulation path."""
```

**Caching strategy: in-memory `functools.lru_cache` keyed by N, no persisted file cache.**
`get_coefficients(N)` is called once per PDE construction (order fixed per run), not per
cell/timestep. Domain-realistic N is 0–10ish (moment closure itself becomes physically
questionable well before the coefficient engine becomes a bottleneck); sympy cost is
`O((N+1)^3)` closed-form rational evaluations, sub-second in practice. Don't build a file
cache until profiling actually shows this matters — document the extension point
(swap the cache layer behind the same `get_coefficients` signature) but don't build it now.

### Generic formulas to implement (replaces every `if order == N` block)

State: `U = (h, h*u_m, h*a_1, ..., h*a_N)`. Primitives `u_m = U[1]/h`, `alpha = U[2:]/h`.
Surface/bed velocities `u_s = u_m + sum(alpha * phi_at_1[1:])`,
`u_b = u_m + sum(alpha * phi_at_0[1:])`.

System matrix `A(U)` (index tensors `A, B, C, E, F` sliced `[1:,1:,1:]`/`[1:,1:]` to align
1-based moment indices with 0-based `alpha` array; `A` symmetric in its last two indices):

```
Amat[0,1] = 1
Amat[1,0] = g*h - u_m**2 - sum(alpha**2 / (2*i+1) for i in 1..N)
Amat[1,1] = 2*u_m
Amat[1,2:] = 2*alpha / (2*i+1)                                          # i = 1..N
Amat[2:,0] = -2*u_m*alpha - einsum('ijk,j,k->i', A_t, alpha, alpha)
Amat[2:,1] = 2*alpha
Amat[2:,2:] = u_m*eye(N) + einsum('ilk,k->il', 2*A_t + B_t, alpha)
```

Cross-check performed: expanding this by hand for N=1 reproduces the existing hardcoded
`A[2][0]=-2*u_m*alpha1, A[2][1]=2*alpha1, A[2][2]=u_m` exactly (since `A_111=B_111=0` by
Wigner-3j parity). Trust it for general N, but still write the regression test in §4.

Generalized friction `P(U)` (base SWME uses `f_R=f_I=0`; `RechargeSWME1D` passes nonzero
`f_R, f_I` from its mixing-friction closure):

```
P[0] = 0
P[1] = f_R*u_s + (f_I + nu/lambda)*u_b
P[i+2] = (2i+1)*(f_R*phi_i(1)*u_s + (f_I+nu/lambda)*phi_i(0)*u_b)
         + (2i+1)*(nu/h**2) * sum_j C[i,j]*U[j+2]                        # i = 1..N
```

Recharge mass source `S_{R,I}(U)` (zero when `R=I=0`, i.e. plain SWME/HSWME):

```
S[0] = R - I
S[1] = R*u_s - I*u_b
S[i+2] = (2i+1)*R*(phi_i(1)*u_s - u_m*r_i - sum_j E[i,j]*alpha_j)
         + (2i+1)*I*(-phi_i(0)*u_b + u_m*s_i + sum_j F[i,j]*alpha_j)     # i = 1..N
# CORRECTED during Step 2 (was documented with a minus sign here originally;
# the minus-sign version failed regression tests against the legacy N=1/N=2
# code and was traced to a transcription error against the thesis formula,
# not a legacy bug - see tests/test_recharge_regression.py).
```

Bed-slope source `B(U,Z)`: `B[1] = g*h*dZ/dx`, all other entries 0 (see §3 for how this is
actually discretized — it is NOT just tacked on as an ordinary cell-centered term, see the
augmented-path design).

### Module consumption

**Superseded/refined during Step 2 by a package-boundary correction — this is the
current design.** The original draft below put a single combined
`compute_generalized_friction(order, values, f_R, f_I, ...)` function in one new
`moment_sw/source_terms.py` module, parametrized by `f_R, f_I` (which default to 0 for
plain SWME). That is mathematically fine (and was cross-validated correct — see §0) but
**violates the swme/recharge package boundary**: `swme/` should contain only the base
SWME/HSWME model's own physics (transport *and* its own Navier-slip friction, since that
exists with or without the recharge extension), while `recharge/` should contain only what
the recharge extension actually *adds* — the rainfall/infiltration mass source `S_{R,I}(U)`
and the mixing-friction contribution `P_mix(U)` (the `f_R, f_I`-dependent terms, which are
meaningless without recharge). Corrected split (implemented in Step 2):

- `swme/source_terms.py`:
  ```python
  def reconstruct_boundary_velocities(order, values, eps_div=1e-12): ...  # -> (h, u_m, alpha, u_s, u_b)
  def compute_navier_slip_friction(order, values, viscosity, slip_length, eps_div=1e-12) -> np.ndarray: ...  # P_slip(U)
  def compute_friction_operator_matrix(order, h, viscosity, slip_length) -> np.ndarray: ...  # S_slip(h)
  ```
  `SWME1D.compute_source_term` (plain SWME/HSWME, no recharge) returns
  `-compute_navier_slip_friction(...)`.
- `recharge/source_terms.py` (kept — NOT deleted; its N=0/1/2 hardcoded functions are what
  get replaced/removed in Step 3/4, but the file itself stays as the home for the
  recharge-specific generic functions):
  ```python
  def compute_recharge_mass_source(order, values, R, I, eps_div=1e-14) -> np.ndarray: ...  # S_{R,I}(U)
  def compute_mixing_friction(order, values, f_R, f_I, eps_div=1e-14) -> np.ndarray: ...  # P_mix(U)
  def compute_total_friction(order, values, f_R, f_I, viscosity, slip_length, eps_div=1e-14) -> np.ndarray: ...
      # = swme.source_terms.compute_navier_slip_friction(...) + compute_mixing_friction(...)
  ```
  `RechargeSWME1D.compute_source_term` evaluates `R, I, f_R, f_I` from its closures, then
  returns `compute_recharge_mass_source(...) - compute_total_friction(...)`.
  `recharge/source_terms.py`'s five order-specific functions (`compute_recharge_source_n0/
  n1/n2`, `compute_friction_matrix_n0/n1/n2`) and the three dispatchers
  (`compute_recharge_source`, `compute_friction_matrix`, `compute_total_source`) are deleted
  in Step 3/4 once the generic functions are wired in — see Step 2's regression suite below,
  which validates the generic functions against exactly these before that deletion.
  `RechargeSWME1D`'s `NotImplementedError` for `order not in (0,1,2)` disappears entirely —
  arbitrary N is now a free consequence.
  `pde.py`'s `SWME1D.compute_system_matrix` becomes ~15 lines calling
  `coefficients.get_coefficients(order)` and doing the einsum contractions given earlier in
  this section, replacing the ~300-line hardcoded block (implemented in Step 2 as the
  staged, tested-but-not-yet-wired `_compute_system_matrix_generic`; Step 3 does the actual
  wiring/deletion). The `hyperbolic` (HSWME) flag stays a boolean constructor arg; formalize
  it as a single private helper `_closure_alpha(order, alpha)` that zeroes `alpha[1:]` (i.e.
  `alpha_2..alpha_N`, never `alpha_1`) when `hyperbolic=True`, called once inside
  `compute_system_matrix`. Do not turn this into a strategy-object hierarchy — it's a
  one-line conditional (this is exactly what `_compute_system_matrix_generic`'s `hyperbolic`
  parameter already does).
- `_compute_source_matrix_inverse` (~1700 lines, Mathematica-derived per-order closed-form
  implicit-Euler matrix) is **replaced**, not genericized-in-place: build the small
  `(N+1)x(N+1)` friction operator matrix `S(h)` generically (linear in the conserved state at
  fixed h, so it's literally the Jacobian of `compute_navier_slip_friction`'s linear map,
  built from the `C` tensor — implemented and regression-tested in Step 2 as
  `swme.source_terms.compute_friction_operator_matrix`) and invert `(I - dt*S)` numerically
  via `np.linalg.inv`/`solve` at runtime — one small linear solve per cell per timestep,
  negligible next to the existing 5-point Gauss quadrature per interface. **This single
  change removes ~1700 of `pde.py`'s ~4970 lines** and is the highest-value simplification in
  the whole restructure. (Note: the *recharge* extension currently has no implicit/linear-
  source path at all in the legacy code — `RechargeSWME1D.compute_source_term` always uses
  the explicit vector form regardless of `linear_source` — so there is nothing to
  regression-test there; a `recharge`-side friction operator matrix, combining
  `swme`'s Navier-slip operator with an analogous linear mixing-friction operator, would be
  new functionality, not a Step 2/3 requirement, if ever wanted.)
- Fix the `PDE.compute_source_term` abstract signature to match reality: every concrete
  override and every call site (`simulation.py`) already passes `(order, values, delta_t)`
  — the ABC currently only declares `(order, values)`. Fix the ABC declaration to include
  `delta_t`. Document (docstring + constructor-time assertion in `SWME1D.__init__`) the
  existing implicit invariant that `linear_source=True` must be paired with
  `ImplicitEuler` — currently only enforced by composition in `main.py`, make it explicit.
- `compute_vertical_velocity_profile`'s hardcoded per-order polynomial list is replaced
  with `coefficients.eval_phi(N, z_points)` contracted against `alpha`:
  `u(z) = u_m + sum(alpha_i * phi_i(z))`. (Also fix a pre-existing off-by-one: the
  `if order >= 0:` branch references `values[i,2]` = `alpha_1` unconditionally, out of
  bounds for `order=0` — should be `if order >= 1:`. Harmless today only because it's
  never called with `order=0`; fix while genericizing.)

---

## 2. Bottom topography, well-balancing, wet-dry treatment

New engineering — no existing reference implementation anywhere in this repo. **Chosen
approach (per user decision): augmented-path-conservative topography coupling.**

### 2.1 Where `Z(x)` lives

On the **mesh**, not the PDE (PDE objects are deliberately stateless w.r.t. order/config,
keep them that way). Add to `mesh.py`:
```python
def set_bed_elevation(self, z_of_x: Callable[[np.ndarray], np.ndarray]) -> None:
    """Populates self.bed_elevation, shape (resolution+2,) including ghost cells
    (zero-gradient extrapolation for INFLOW_OUTFLOW, wraps like state for PERIODIC)."""
```
Defaults to all-zeros (flat bed) if never called — **every existing config/test that
doesn't mention topography behaves exactly as today**; this must be a strictly additive
feature, verify with a regression run of an existing config before/after.

### 2.2 Well-balanced treatment: augmented-state path-conservative bed slope

The existing scheme is a Castro–Parés-style path-conservative fluctuation solver, not a
conservative flux-difference Riemann solver: `spatialDiscretization.py`'s
`PVM.compute_fluctuation` computes `∫_0^1 A(psi(s)) ds · (U_R - U_L)` via 5-point
Gauss-Legendre quadrature along the linear path `psi(s) = (1-s)*U_L + s*U_R`.

Design: augment the state to `W = (U, Z)` and the system matrix to
```
Ã(W) = [ A(U)   -g*h*e_momentum ]     # (n+1)x(n+1), n = order+2
        [  0          0         ]
```
i.e. `A(U)` plus one extra column (nonzero only at `row=momentum(1), col=n`, value
`-g*h`) and one extra all-zero row (`Z` is a passive/frozen path coordinate,
`dZ/dt = 0`). Change `PVM.compute_fluctuation` /
`compute_generalized_roe_and_viscosity` to accept the augmented `(U, Z)` pair and the
augmented matrix callable — the same 5-point quadrature loop applies unchanged on
`(n+1)`-vectors; take only the first `n` rows of the resulting fluctuation as the actual
state update (the `Z`-row is zero by construction). `ClassicalSimulation1D.run_simulation`
must build `system_matrix` closures over `(U, Z)` pairs and pass `mesh.bed_elevation[i]`
alongside `values[i,:]` at every interface.

Why this recovers the C-property: for a lake-at-rest state (`h+Z = const`,
`u_m = alpha_i = 0` everywhere), the fluctuation becomes exactly the discrete gradient of
`h+Z` along the same path/quadrature that produces the moment-transport viscosity — which
is zero by construction. No separate Audusse-style hydrostatic reconstruction step is
needed because the topography coupling goes through the *same* non-conservative-product
machinery already used for the moment transport, per the standard Castro–Parés path-
conservative treatment of source terms with topography.

**Mandatory validation before trusting this in production:** implement a dedicated
lake-at-rest regression test (flat free surface over non-flat, non-trivial `Z(x)`, zero
initial velocity/moments, run forward in time, assert the state stays at machine-precision
rest) as part of the same PR that adds this feature. This is flagged as a genuinely
research-level numerical question in the design research — budget real validation time,
don't just assume the formula above is bug-free on first implementation.

### 2.3 Wet-dry treatment

Three layers, replacing the current `if h <= 0: raise RuntimeError`:

**(a) Two-tier epsilon convention.** Keep the existing `eps_div ~ 1e-14`-scale
machine-precision division guard (already used in `recharge/source_terms.py`'s
`_evaluate_rainfall_and_infiltration`, fold into the new shared `source_terms.py`).
Introduce a **separate, larger, configurable physical dry threshold** `h_dry` (expose as a
required/documented config field on `SWME1D`/simulation construction, not a silently
hardcoded constant — pick a value relative to the actual `h` scale of the test case being
run; the thesis's test cases use `h ~ O(1)`, so something like `h_dry = 1e-4`,
`h_wet = 1e-3` is a reasonable starting default, but this should be tunable per config).

**(b) Desingularized primitive reconstruction** (Kurganov–Petrova style), applied only
where primitives are extracted for closure evaluation (`compute_system_matrix`,
`compute_max_wavespeed`, `reconstruct_boundary_velocities`/`compute_navier_slip_friction`,
`compute_recharge_mass_source`/`compute_mixing_friction`) —
**never** applied to the conserved state array itself:
```
h_reg = max(h, eps_div)
u_m   = 2*h*q / (h**2 + max(h, eps_div)**2)         # smooth as h -> 0
ramp  = clip((h - h_dry) / (h_wet - h_dry), 0, 1)   # h_wet > h_dry
alpha_eff = alpha * ramp                             # moments -> plug flow as h -> h_dry
```
Below `h_dry`: treat as fully dry, `u_m = alpha = 0` exactly, contributes `0` (not `NaN`)
to `compute_max_wavespeed`'s reduction, skip in hyperbolicity diagnostics (reuse the
existing "skip dry states" pattern already present in `simulation.py`'s
`_store_hyperbolicity_snapshot`, ~line 292, as the template). The `ramp` on moments is
physically motivated (a vanishing film has no meaningful vertical velocity profile) and
numerically stabilizing (prevents `alpha_i/h`-type terms from blowing up in the friction
closure as `h -> h_dry`).

**(c) Positivity-preserving timestep limiter.** (b) prevents blow-up but does not by
itself guarantee `h_i^{n+1} >= 0` near a wetting front under the explicit FV update. Add a
drying-timestep restriction (Bollermann/Kurganov/Noelle-style): after computing
fluctuations but before applying the update, compute per-cell a local admissible `dt_i`
such that outgoing mass flux cannot exceed water currently available in the cell, and cap
`delta_t = min(delta_t_CFL, min_i(delta_t_i))`. This replaces the current post-hoc
`if values[i,0] <= 0: raise RuntimeError` (in `simulation.py`, `run_simulation`) with a
**pre-emptive** cap applied before the update — keep the `RuntimeError` as a debug-mode
assertion (should never fire once the limiter is correct) rather than deleting it.

### 2.4 Affected modules summary

- `mesh.py`: `bed_elevation` field + `set_bed_elevation`.
- `pde.py`: `eps_div`/`h_dry`/`h_wet` constructor params on `SWME1D`; desingularized
  primitive-extraction helper reused across system matrix / wavespeed / source calls; no
  more crash-on-`h<=0` inside `compute_system_matrix` (replaced by the ramp).
- `spatialDiscretization.py`: augmented `(U,Z)` path support in `PVM` fluctuation methods.
- `simulation.py`: `ClassicalSimulation1D.run_simulation` builds augmented system-matrix
  closures with `mesh.bed_elevation`; adds the drying-timestep limiter; demotes the
  hard-crash `RuntimeError` on non-positive height to a debug assertion.

---

## 3. Deletion plan

Cross-checked with repo-wide grep during design research — nothing outside the files being
deleted references these symbols except other files also being deleted, or `main.py`,
which is being rewritten anyway (see §5).

| Location | What to delete | Why it's safe |
|---|---|---|
| `pde.py` | `HermiteMomentEquations` class (~1080 lines) | Different physics (Boltzmann/BGK), zero references outside deleted `main_HME_errorChecks.py`/`plotting.py`'s HME plot classes. |
| `pde.py` | `VegetationSWME1D` class (~780 lines) | Zero references outside deleted `main_*_errorChecks.py` scripts. |
| `pde.py` | `compute_system_matrix_diff` (base `SWME1D`, ~130 lines) + ABC/Vegetation declarations | **Confirmed dead code** — grep across the whole repo finds only its 3 definition sites, never called anywhere, including by the adaptive simulation classes also being deleted. |
| `pde.py` | `compute_source_term_lastentry` + `compute_breakdown_criteria_full` | Only ever used by adaptive order-switching (`SpatiallyAdaptiveSimulation1D`, being deleted). Delete alongside their ABC declarations. |
| `simulation.py` | `SpatiallyAdaptiveSimulation1D` (ABC) and all subclasses (`NonConservativeAdaptiveSimulation1D`, `ConservativeAdaptiveSimulation1D`, `InterpolatedAdaptiveSimulation1D`, `SmoothedSubdomainReconstruction`, `SmoothedConsAdaptiveSimulation1D`, `SmoothedNonConsAdaptiveSimulation1D`) — ~1820 lines | Only referenced from `main.py`/deleted `main_*_errorChecks.py`/deleted `plotting.py` adaptive classes. |
| `simulation.py` | `Micro_macro` | Same — only referenced from files being deleted. |
| `plotting.py` | `SWME1DPlotAdaptive`, `HME1DPlotClassical`, `HME1DPlotAdaptive` | Keep only `SWME1DPlotClassical` (does not import/subclass anything being deleted). |
| `main_HME_errorChecks.py`, `main_SWME_errorData.py` | entire files | Per user decision — drop, no replacement. |
| `config/ConfigHME1D/`, `config/Config Micro-Macro/` | entire dirs | Only consumed by files/models being deleted. |
| `recharge/source_terms.py` | ONLY the N=0/1/2 hardcoded functions and dispatchers (`compute_recharge_source_n0/n1/n2`, `compute_friction_matrix_n0/n1/n2`, `compute_recharge_source`, `compute_friction_matrix`, `compute_total_source`, `_evaluate_rainfall_and_infiltration`) | **Revised in Step 2**: the file itself is KEPT, not deleted — it now also holds the generic replacements (`compute_recharge_mass_source`, `compute_mixing_friction`, `compute_total_friction`), added in Step 2 alongside the old code for regression testing (`tests/test_recharge_regression.py`). Delete only the superseded old functions/dispatchers once Step 3 wires `RechargeSWME1D` to the generic ones. Do NOT delete this file wholesale — doing so would delete the new code too. |
| `symbolic_math/` | entire directory | Per user decision — its math is absorbed into `moment_sw/coefficients.py`; user has their own copy elsewhere, no in-repo wrapper needed. |

**Keep unchanged:** `recharge/laws.py`, `recharge/context.py` (already order-independent).
**Fix opportunistically, not blocking:** `recharge/initial_conditions.py` has small
per-order-capped hardcoding (effectively capped at N=2 for one profile) — trivially
loop-generalizable (`for i in range(1, order+1): initial_values[i+1] = coeff_i * pulse(x)`),
low priority since it's test-IC construction, not core physics.

**Mandatory validation step before deleting any hardcoded per-order block in `pde.py`**:
write a parametrized regression test that, for N=0..6, calls both the *old* hardcoded
`compute_system_matrix`/`compute_source_term`/`_compute_source_matrix_inverse` and the
*new* generic tensor-contraction versions on identical random `(h, u_m, alpha)` samples and
asserts numerical equality to float64 precision. Keep old code reachable (e.g. a `git tag`
taken before deletion, or a short-lived `_legacy.py` module) only until this suite passes
for all six orders, then delete for real. Do not delete the hardcoded blocks and write the
regression test after the fact — write the test first, against the still-present old code,
then swap the implementation.

---

## 4. Target package layout (`uv`)

**Superseded by the naming/layout decision recorded above — this is the current target.**
Post-deletion the codebase is roughly 4–6k lines — an order of magnitude below where
splitting `pde.py`/`simulation.py` further would pay for itself, but per the user's
explicit preference, `recharge` is a **sibling top-level package**, not nested inside the
core solver package (named `swme`, formerly planned as `moment_sw`). `matlab/` is deleted
outright (decision #4 above). `processing/` stays outside the installable package
(confirmed no code dependency on solver internals).

```
recharge-paper/                        # repo root is the uv project root
├── pyproject.toml
├── README.md
├── RESTRUCTURE_PLAN.md                # this file
├── mkdocs.yml                         # NEW — Step 9
├── docs/                              # NEW — Step 9
├── src/
│   ├── swme/                          # core solver (formerly planned as moment_sw)
│   │   ├── __init__.py
│   │   ├── coefficients.py            # §1, absorbs symbolic_math/symbo.py math
│   │   ├── source_terms.py            # §1, base Navier-slip friction ONLY (not recharge)
│   │   ├── mesh.py                    # + bed_elevation — §2.1
│   │   ├── pde.py                     # SWME1D only, genericized — §1/§2
│   │   ├── spatialDiscretization.py   # + augmented-path topography — §2.2
│   │   ├── timeIntegration.py         # unchanged
│   │   ├── simulation.py              # Simulation (ABC) + ClassicalSimulation1D only
│   │   ├── plotting.py                # Plotting (ABC) + SWME1DPlotClassical only
│   │   └── cli.py                     # replaces main.py's ad hoc script
│   └── recharge/                      # sibling package, NOT nested inside swme/
│       ├── __init__.py
│       ├── context.py
│       ├── laws.py
│       ├── source_terms.py            # rainfall/infiltration mass source + mixing friction
│       ├── initial_conditions.py
│       └── recharge_pde.py            # imports `from swme.pde import SWME1D` etc.
├── config/                            # wiped per §3/Step 4, YAML going forward
├── processing/                        # unchanged
└── tests/                             # regression suite (§1/§3) + lake-at-rest, wet-dry, mkdocs build sanity
```

`pyproject.toml` skeleton (updated for the sibling layout + YAML):
```toml
[project]
name = "moment-sw"
version = "0.1.0"
description = "1D finite-volume solver for Shallow Water Moment Equations (SWME/HSWME) with a rainfall-runoff extension"
requires-python = ">=3.11"
dependencies = [
    "numpy>=2.0",
    "scipy>=1.14",
    "matplotlib>=3.9",
    "pandas>=2.2",
    "sympy>=1.13",       # runtime dep: coefficients.py needs it for A/B/C at first use per N
    "pyyaml>=6.0",        # NEW — config format decision, see above
]

[project.scripts]
moment-sw = "swme.cli:main"

[dependency-groups]
dev = ["pytest>=8.0", "pytest-cov"]
docs = ["mkdocs>=1.6", "mkdocs-material>=9.5", "mkdocstrings[python]>=0.26"]  # NEW — Step 9

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/swme", "src/recharge"]
```
(The distribution/project name stays `moment-sw` and the installed console-script command
stays `moment-sw` for continuity — only the *importable* package names change. Revisit if
that's confusing once `swme`/`recharge` exist as separate top-level packages; trivial to
rename later either way.)

Install/run via `uv sync` then `uv run moment-sw ...` (or `uv run pytest`). Delete
`swme/requirements.txt`-equivalent and `Makefile` once `uv` is confirmed working
(`Makefile`'s only rule, `purge` — deleting `Data-processing/Results/Recharge/` — can move
to a small `uv run` script or a `[tool.hatch]` hook, low priority).

`main.py`'s current ~430-line script (config parsing + large if/elif dispatch tables, most
branches referencing deleted classes) is **rewritten as `cli.py`**, not just trimmed: a
YAML-driven (not `configparser`-driven, per the config-format decision above) construction
of exactly `SWME1D | RechargeSWME1D` × the PVM-family schemes (PRICE/LF/Roe/Osher) ×
`{ExplicitEuler, ImplicitEuler, Exact}` × `ClassicalSimulation1D`. Strip every branch
referencing deleted models/methods (`mpi`, `micro_macro`, `*Adaptive`) while rewriting.

---

## 5. Execution checklist

Work through this in order; each step should be a separately reviewable/testable unit.
Update the checkboxes as work lands. Do not skip the "write test against old code first"
steps — they're what makes the hardcoded-block deletions safe.

- [x] **Step 0 — scaffolding**: create `pyproject.toml`, `src/moment_sw/` layout, move
      existing files into it unchanged (pure move, no logic changes yet), get `uv sync` +
      a trivial existing config running end-to-end through the new layout before touching
      any physics/numerics code. This isolates "packaging works" from "physics is correct".
      **DONE.** `pyproject.toml` + `uv.lock` at repo root, `moment_sw/` moved to
      `src/moment_sw/` via `git mv` (history preserved), root `.venv/` removed in favor of
      `uv`-managed one. Two things had to be fixed beyond a pure move — neither is a
      physics/numerics change, both are required for the package to import at all:
        - Every intra-package module used flat, flat-namespace-style imports
          (`import pde`, `from recharge.context import SourceContext`, etc.), which only
          worked because the old flat `moment_sw/` directory was on `sys.path[0]` when run
          as `python3 main.py` from inside it. These are now relative imports
          (`from . import pde`, `from .context import SourceContext`, `from ..pde import
          SWME1D`). Fixed in: `main.py`, `plotting.py`, `simulation.py`,
          `recharge/recharge_pde.py`, `recharge/initial_conditions.py`. Added
          `src/moment_sw/__init__.py` (was missing entirely).
        - `mpmath` is imported in `spatialDiscretization.py` (present in the old
          `requirements.txt` but omitted from the first `pyproject.toml` draft) — added to
          `dependencies`. (It's also a transitive dep of `sympy`, so this was latent either
          way, but declare it explicitly since it's imported directly.)
        - `main.py`'s `config.read('config/config.txt')` was CWD-relative; changed to
          resolve relative to the package file itself
          (`Path(__file__).resolve().parent / 'config' / 'config.txt'`) so it works
          regardless of where `uv run` is invoked from.
        - `config/config.txt`'s `[numerical_method_information]` `order`/`start_order`
          keys were commented out (all three "Run 1/2/3" options disabled) — this is a
          pre-existing pattern of manually uncommenting one block per run, but with all
          three commented, `method = classical` crashes on a missing config key. Uncommented
          "Run 2" (`order = 1`) as the active default so the config is actually runnable;
          leave the other two commented as before.
      **Verified**: `uv sync` succeeds; `uv run python -c "import moment_sw.pde; ..."`
      (all core + recharge submodules) succeeds; a full run of the existing
      `RechargeSWME1D`/N=1/constant-exfiltration config via
      `MPLBACKEND=Agg uv run python -m moment_sw.main` completes t_end=0.4 without errors
      and writes the expected CSVs to `Data-processing/Results/Recharge/` (sane values,
      e.g. `h≈1.06, u_m≈0.75, a1≈0.042` at the final snapshot). `MPLBACKEND=Agg` is only
      needed because the VM this was run on is headless (`plt.show()` would otherwise
      block/error) — not a code change, just how to invoke it in this environment.
      `main_HME_errorChecks.py`/`main_SWME_errorData.py` were **not** import-fixed (still
      using flat imports) since they're deleted wholesale in Step 4 — not worth fixing
      dead-code-walking files.
- [x] **Step 1 — coefficients engine**: write `moment_sw/coefficients.py` (ported from
      `symbolic_math/symbo.py` + closed forms for r/s/E/F), with unit tests validating the
      closed forms against sympy integration for N=0..8.
      **DONE.** `src/moment_sw/coefficients.py` implements:
        - `Coefficients` frozen dataclass (`N, A, B, C, E, F, r, s, phi_at_1, phi_at_0`,
          all float64 numpy arrays, indices 0..N with index 0 = depth-averaged mode).
        - `get_coefficients(N)`, `@lru_cache`-memoized: `A` (Wigner-3j closed form, no
          basis needed), `B` (needs the `J_j` antiderivative table), `C` (basis-derivative
          inner product) computed via ported-not-imported sympy machinery
          (`_build_shifted_legendre_basis`, `_compute_A/_B/_C/_JP`, exact `Rational` →
          `float()`); `E`, `F`, `r`, `s` computed directly from the closed forms (no sympy
          at runtime).
        - `eval_phi(N, z)`: float64 Bonnet recurrence for `phi_i(z)`, for post-processing
          (vertical velocity profile reconstruction) only — never on the hot path.
        - Private sympy reference implementations `_compute_r_s_sympy` /
          `_compute_E_F_sympy` are kept (unused at runtime) specifically so the closed
          forms can be tested against them — see below.
      **Bug caught by the test suite and fixed**: the initial closed-form implementation
      set `r_0 = s_0 = phi_0(1) = phi_0(0)`-style values all from the same array, but
      `r_i`/`s_i` and `phi_i(1)`/`phi_i(0)` are genuinely different quantities at `i=0`.
      `phi_0(z) = 1` (constant), so `dphi_0/dz ≡ 0`, hence `r_0 = s_0 = 0` exactly, while
      `phi_0(1) = phi_0(0) = 1`. The closed forms `r_i=(-1)^i`, `s_i=1` (thesis Appendix
      B.1) only hold for `i>=1`; verified against direct sympy evaluation. Fixed in
      `get_coefficients`, decoupled `phi_at_1`/`phi_at_0` from `r`/`s` as independent
      arrays. Index 0 is not a physical moment variable either way, but the arrays must be
      numerically correct at every index since `pde.py`/`source_terms.py` will slice them
      generically in Step 3, not special-case index 0 by hand.
      **Verified**: `tests/test_coefficients.py`, 35 tests, all passing
      (`uv run pytest`) — closed-form `r/s/E/F` vs. sympy for N=0..8, `A` sanity checks
      (known value `A_000=1`, Wigner-3j parity zero `A_111=0`, symmetry in the last two
      indices), shape checks, `lru_cache` identity, negative-N rejection, `eval_phi` vs.
      exact basis evaluation. Additionally cross-checked by hand against the N=1 formulas
      already hand-verified in this plan's §0/§1 (`A_111=0`, `B_111=0`, `C_11=4` →
      `3*C_11=12` matching the old hardcoded `12*nu/h` friction term, `E_11=F_11=1/3`,
      `E_10=-1`, `F_10=1`) — all match exactly.
- [x] **Step 1.5 — package rename & config format migration** *(inserted after Steps 0-1
      landed under the old names; kept as "1.5" rather than renumbering every subsequent
      step)*. Implements decisions #4-6 in "User decisions locked in".
      **DONE.** `git mv src/moment_sw src/swme`, `git mv src/swme/recharge src/recharge`
      (un-nested to a sibling package). Import fixes: `recharge/recharge_pde.py`'s
      `from ..pde import SWME1D` → `from swme.pde import SWME1D` (cross-package, since
      `recharge` is no longer nested inside `swme`); `recharge/*.py`'s intra-package
      imports (`.context`, `.source_terms`, `.recharge_pde`, `.laws`) were already
      relative-within-`recharge` and needed no change; `swme/main.py`'s
      `from .recharge.initial_conditions import ...` / `from .recharge.laws import ...`
      → `from recharge.initial_conditions import ...` / `from recharge.laws import ...`
      (absolute, sibling package, not a `swme`-relative submodule anymore); `swme/*.py`'s
      existing `from . import X` imports (main.py, plotting.py, simulation.py) were
      unaffected — still relative *within* `swme/`. `tests/test_coefficients.py`'s
      `from moment_sw import coefficients` → `from swme import coefficients`.
      `pyproject.toml`: added `pyyaml>=6.0` to `dependencies`, added a `docs` dependency
      group (`mkdocs`, `mkdocs-material`, `mkdocstrings[python]` — added now per the
      original note so the file doesn't need touching twice before Step 9), script entry
      point → `swme.cli:main` (still unresolved until Step 7's `cli.py` exists, harmless),
      `[tool.hatch.build.targets.wheel] packages = ["src/swme", "src/recharge"]`.
      `matlab/` deleted in its entirety (`git rm -r matlab/`, decision #4). `config/`
      wiped: deleted `ConfigHME1D/`, `Config Micro-Macro/`, `ConfigSWME1D/`,
      `Config_Cyril-honoursProject/`; renamed the active `config.txt` → `config.ini`
      (same `configparser`-readable INI content, just an honest extension — chose to
      **keep it working via `configparser` for now** rather than force the YAML loader
      swap into this step, since `main.py` is fully rewritten as `cli.py` in Step 7
      anyway and doing the YAML rewrite twice would be wasted effort); added
      `config/example.yaml` as a **stub, not yet consumed by any code** — a
      forward-looking sketch of the intended YAML schema (mirrors `config.ini`'s current
      active RechargeSWME1D/N=1/constant-exfiltration case) so the target structure is
      visible ahead of the real Step 7 rewrite. `main.py`'s `config.read(...)` call and
      one comment updated from `config.txt` → `config.ini` accordingly.
      **Verified**: `uv sync` succeeds (pulls in `pyyaml`); every `swme.*` and
      `recharge.*` submodule imports cleanly as sibling packages; `uv run pytest` — all
      35 tests still pass; a full end-to-end run of the same RechargeSWME1D/N=1/constant-
      exfiltration config via `MPLBACKEND=Agg uv run python -m swme.main` reproduces the
      **exact same** `total mass = 242.14363128962492` as the pre-rename Step 0 run
      (byte-identical numerical result — confirms the move/rename changed nothing about
      behavior) and writes the same CSVs to `Data-processing/Results/Recharge/`.
- [x] **Step 2 — regression tests against old hardcoded code** (§3's mandatory step):
      parametrized N=0..6 equality tests for system matrix, friction source, and the
      matrix-inverse implicit source, comparing old hardcoded `pde.py` against not-yet-written
      generic implementations (write the generic implementation now, test immediately).
      **DONE**, and this step's own testing forced a real architectural correction plus
      caught two real bugs — see below.

      **Architecture correction (before any code was written)**: the original §1 draft put
      a single combined friction function in one new `moment_sw/source_terms.py`,
      parametrized by `f_R, f_I` (recharge-only concepts). The user caught that this
      violates the intended `swme`/`recharge` package boundary — `swme/` should hold only
      the base model's own physics (transport + its own Navier-slip friction, present with
      or without recharge), `recharge/` only what recharge actually *adds*. Corrected
      design (now reflected in §1's "Module consumption" above, superseding the original
      draft there):
        - `swme/source_terms.py` (NEW): `reconstruct_boundary_velocities` (shared u_s/u_b
          kinematic reconstruction — a fact about the moment representation, not friction
          physics, so legitimately shared), `compute_navier_slip_friction` (P_slip(U), the
          base model's own friction, f_R/f_I do not appear here at all),
          `compute_friction_operator_matrix` (S_slip(h), the linear operator behind the
          old `_compute_source_matrix_inverse`).
        - `recharge/source_terms.py` (existing file, new functions ADDED alongside the old
          N=0/1/2 code, not replacing it yet): `compute_recharge_mass_source` (S_{R,I}(U)),
          `compute_mixing_friction` (P_mix(U), the f_R/f_I-dependent piece only),
          `compute_total_friction` (= `swme`'s `compute_navier_slip_friction` +
          `compute_mixing_friction` — the direct generic analog of the old
          `compute_friction_matrix_n0/n1/n2`, which returned this same combined quantity).
        - `pde.py`: added `_compute_system_matrix_generic` as a new module-level function
          (imports `from . import coefficients`), staged next to the still-present
          `SWME1D.compute_system_matrix` hardcoded blocks — not yet wired into the class.

      **Bug #1 (legacy, in `pde.py`, order=6 system matrix)**: orders 0-5 matched the
      generic implementation to `atol=1e-10` on every entry, first try, for random states
      (`tests/test_pde_regression.py::test_system_matrix_matches_legacy`). Order=6 did not
      — not just the single expected `A[1][7] = (2*alpha5)/13.` copy-paste bug (should be
      `alpha6`, breaking the otherwise-universal pattern `A[1][i+1]=2*alpha_i/(2i+1)`) but
      **22 of the 64 entries**, reproducibly with the test's own seed. Ruled out a bug in
      the generic implementation via an independent, legacy-code-free consistency check
      (`test_system_matrix_order_reduction_consistency`): since `A_ijk`/`B_ijk` depend only
      on the basis functions up to the indices involved, not on the truncation order N,
      the generic order-N matrix with `alpha_N` set to 0 must exactly reproduce the
      order-(N-1) matrix on their shared block — verified for every order 1-6. Conclusion:
      the legacy order=6 block (its densest, most error-prone, hand-transcribed block) is
      simply unreliable; Step 3 deletes it outright rather than attempting to preserve its
      behavior. Fully documented (not hidden) in
      `test_order6_legacy_block_disagrees_at_multiple_entries`.
      **Bug #2 (own transcription error, not legacy)**: the recharge mass-source formula as
      first written (both in `recharge/source_terms.py` and in §1 above) had
      `- sum_j F[i,j]*alpha_j` for the infiltration term; the thesis formula (and the
      legacy `compute_recharge_source_n1`/`n2`, confirmed by hand-derivation for N=1) has
      `+ sum_j F[i,j]*alpha_j`. Caught immediately by
      `tests/test_recharge_regression.py::test_recharge_mass_source_matches_legacy` failing
      for N=1,2 (N=0 passed, since that entry doesn't involve E/F at all). Fixed in
      `recharge/source_terms.py` and corrected in §1 above, with a note left in place
      explaining the correction (do not silently re-introduce it).

      **Test files**: `tests/test_pde_regression.py` (116 tests: system matrix N=0-6 ×
      hyperbolic on/off, order-reduction consistency N=1-6, friction vector N=0-6 × 3
      viscosity/slip-length combos, friction operator matrix-inverse N=0-6 × 3
      viscosity/slip-length combos × 3 timesteps, friction-operator/friction-vector
      cross-consistency N=0-6) and `tests/test_recharge_regression.py` (34 tests: mass
      source N=0,1,2 × 3 param combos vs. legacy, total friction N=0,1,2 × 3 param combos
      vs. legacy, end-to-end `S-P` vs. legacy `compute_total_source`, generic functions
      execute and produce finite output for N=3-6 with no legacy code to compare against,
      `compute_total_friction` decomposes exactly into `swme` slip + `recharge` mixing).
      **Verified**: full suite `uv run pytest -q` — 185/185 passing. End-to-end solver run
      (`MPLBACKEND=Agg uv run python -m swme.main`) reproduces the exact same
      `total mass = 242.14363128962492` as before Step 2 — confirms these are pure
      additions with zero effect on the still-unchanged live execution path (Step 3 is
      what actually wires the generic engine in and will need this exact end-to-end check
      re-run afterward, since deleting the buggy order=6 block means an order=6 run's
      *numbers* will legitimately change — for the better).
- [ ] **Step 3 — swap in the generic engine** *(the generic implementations themselves are
      already written and regression-tested per Step 2 below — `swme.pde._compute_system_matrix_generic`,
      `swme.source_terms.{compute_navier_slip_friction,compute_friction_operator_matrix}`,
      `recharge.source_terms.{compute_recharge_mass_source,compute_mixing_friction,compute_total_friction}`
      all exist and pass regression tests against the legacy code. Step 3 is the mechanical
      wiring/deletion step, not further derivation)*: in `pde.py`, replace
      `SWME1D.compute_system_matrix`'s body with a call to (or inlining of)
      `_compute_system_matrix_generic` and delete the hardcoded `if order==N` blocks (note:
      do NOT try to preserve the order=6 block's behavior — it has confirmed bugs at 22/64
      entries, see Step 2 findings); delete `compute_system_matrix_diff` (already confirmed
      dead code, zero callers anywhere in the repo); replace `SWME1D.compute_source_term`'s
      non-implicit branch with `-swme.source_terms.compute_navier_slip_friction(...)`;
      replace `_compute_source_matrix_inverse`'s body with
      `np.linalg.inv(np.eye(n) - dt*swme.source_terms.compute_friction_operator_matrix(...))`
      and delete its ~1700 lines of hardcoded per-order blocks. In `recharge/recharge_pde.py`,
      replace `RechargeSWME1D.compute_source_term`'s call to `compute_total_source` with
      direct calls to `recharge.source_terms.compute_recharge_mass_source` and
      `compute_total_friction`, removing the `order not in (0,1,2)` restriction. In
      `recharge/source_terms.py`, delete the now-superseded
      `compute_recharge_source_n0/n1/n2`, `compute_friction_matrix_n0/n1/n2`,
      `compute_recharge_source`, `compute_friction_matrix`, `compute_total_source`, and
      `_evaluate_rainfall_and_infiltration` (keep the file itself — it now holds the generic
      functions, see the corrected §1 "Module consumption" above). Fix the
      `PDE.compute_source_term` ABC signature to include `delta_t`.
- [ ] **Step 4 — delete out-of-scope models**: `HermiteMomentEquations`,
      `VegetationSWME1D`, adaptive/`Micro_macro` simulation classes, matching `plotting.py`
      classes, `main_HME_errorChecks.py`, `main_SWME_errorData.py`, `symbolic_math/`.
      (`recharge/source_terms.py` is NOT deleted — only its superseded N=0/1/2 functions are
      removed, in Step 3 above, alongside the `RechargeSWME1D` wiring; see the corrected
      deletion-plan table in §3.)
      **Also, per user directive**: wipe `config/` down to nothing (or one minimal
      canonical example) — delete `config/ConfigHME1D/`, `config/Config Micro-Macro/`,
      `config/ConfigSWME1D/`, `config/Config_Cyril-honoursProject/`, and the current
      `config/config.txt` itself, so the config directory starts clean for the new
      format (see the config-format decision recorded below) rather than accumulating
      stale examples for deleted models.
      **Also (pending final confirmation, see chat)**: delete `matlab/` in its entirety —
      nothing in it is needed; see the write-up in the conversation for the full
      reasoning. If confirmed, add it to this step's deletion list; if the user wants it
      kept for provenance/citation reasons, mark it explicitly "kept, intentionally
      untouched" here instead and remove this TODO.
- [ ] **Step 5 — topography + well-balancing**: `mesh.py` bed elevation field, augmented
      `(U,Z)` path in `spatialDiscretization.py`, wiring in `simulation.py`. Write the
      lake-at-rest regression test as part of this step, not after.
- [ ] **Step 6 — wet-dry treatment**: `eps_div`/`h_dry`/`h_wet` params, desingularized
      primitive extraction, moment ramp, drying-timestep limiter, demote the hard-crash
      `RuntimeError` to a debug assertion. Write a wet-dry regression test (e.g. a
      partial dam-break onto initially dry bed) as part of this step.
- [ ] **Step 7 — CLI rewrite**: `cli.py` replacing `main.py`, trimmed config dispatch.
- [ ] **Step 8 — cleanup**: delete `requirements.txt`/`Makefile` once `uv` workflow is
      confirmed; update `README.md` (repo root and package-level) to describe the new
      layout, arbitrary-N support, topography/wet-dry usage, and `uv` quick-start.
- [ ] **Step 9 — documentation site (mkdocs)**: added per user request, appended after
      the (previously) final step. Scope:
        - Add `mkdocs`, `mkdocs-material`, `mkdocstrings[python]` as a new `docs`
          dependency group in `pyproject.toml` (`uv sync --group docs` to install).
        - `mkdocs.yml` at repo root, `docs/` directory with at minimum: an index/landing
          page (repo purpose, thesis link/context), a "Quick start" page (mirrors the
          root `README.md`'s command list, kept in sync rather than duplicated where
          possible — consider having the README be the canonical source and the mkdocs
          page `include`/reference it, or accept light duplication if that's simpler to
          maintain), a "Model overview" page (SWME/HSWME/RechargeSWME, arbitrary-N
          coefficients engine, well-balanced + wet-dry treatment — largely adapted from
          this plan's §1/§2 once implemented), a "Configuration reference" page
          (documenting the config format decided below), and an API-reference section
          generated via `mkdocstrings` from docstrings in `src/` (whatever the final
          package name(s) turn out to be, see the naming decision below).
        - `uv run mkdocs serve` for local preview, `uv run mkdocs build` for a static
          `site/` (gitignore `site/`).
        - Do this step *after* Steps 0–8 land (package renamed if applicable, CLI
          finalized, physics complete) so the docs describe the actual final structure
          rather than needing a rewrite partway through.

## Open items intentionally left for implementation time (not blocking this plan)

- Exact default numerical values for `eps_div`/`h_dry`/`h_wet` — pick sensible defaults
  during Step 6, expose as config, don't hardcode silently.
- Whether `Makefile`'s `purge` rule moves into a `uv run` script or a hatch build hook —
  decide during Step 8, low stakes either way.
- Soft practical upper bound on supported N (~10–12 estimated, where the moment closure
  itself becomes physically questionable, independent of the coefficient engine's own
  performance) — worth documenting explicitly once arbitrary N is live, not enforcing as a
  hard cap.
