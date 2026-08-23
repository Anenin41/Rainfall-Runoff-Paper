# Configuration

A run is described by one YAML file. Three sections are required, `pde`, `grid` and
`numerics`. The rest are optional and take defaults.

Unknown keys are rejected with an error that names the key and lists the valid ones for
that section. This is deliberate: under the previous INI format a misspelled key fell
back to a default and the run continued with settings nobody chose.

Values are checked by type when they are read, so a quoted `"false"` is an error rather
than a truthy string that quietly turns a feature on.

## `pde`

Required.

| Key | Type | Default | Meaning |
|:--|:--|:--|:--|
| `type` | choice | required | `SWME1D`, `HSWME1D` or `RechargeSWME1D` |
| `initial_condition` | string | required | See [Initial conditions](#initial-conditions) |
| `viscosity` | number | required | Kinematic viscosity \(\nu\). Use `0.0` for inviscid |
| `slip_length` | number | required | Navier-slip length \(\lambda\) |
| `linear_source` | bool | `false` | Return friction as a matrix. Requires `ImplicitEuler` |
| `hyperbolic` | bool | from `type` | Use the HSWME closure. Implied by `type: HSWME1D` |

`type: HSWME1D` and `hyperbolic: true` mean the same thing. Giving both is fine as long
as they agree; setting `type: HSWME1D` with `hyperbolic: false` is a contradiction and
raises an error.

### Rainfall and infiltration

These apply only when `type: RechargeSWME1D`.

| Key | Type | Meaning |
|:--|:--|:--|
| `rainfall_rate` | number | Rainfall rate \(R\) |
| `infiltration_type` | choice | `horton` or `constant` |
| `mixing_friction_model` | choice | `admissible` (the only option) |
| `alpha_R` | number | Mixing friction coefficient for rainfall |
| `alpha_I` | number | Mixing friction coefficient for infiltration |

Horton infiltration, \(I(t) = f_c + (f_0 - f_c) e^{-kt}\):

| Key | Type | Meaning |
|:--|:--|:--|
| `horton_f0` | number | Initial rate \(f_0\) |
| `horton_fc` | number | Final rate \(f_c\) |
| `horton_k` | number | Decay constant \(k\) |

Constant infiltration:

| Key | Type | Default | Meaning |
|:--|:--|:--|:--|
| `constant_infiltration_rate` | number | required | The rate \(I_0\) |
| `constant_infiltration_eps` | number | `1e-14` | Guard against removing water from an empty cell |
| `constant_limit_by_rainfall` | bool | `false` | Cap the rate at the rainfall rate |
| `constant_limit_by_available_water` | bool | `true` | Never remove more water than is present |

## `grid`

Required.

| Key | Type | Meaning |
|:--|:--|:--|
| `x1` | number | Left edge of the domain |
| `x2` | number | Right edge |
| `resolution_x` | integer | Number of cells |

The grid is uniform.

## `numerics`

Required.

| Key | Type | Default | Meaning |
|:--|:--|:--|:--|
| `order` | integer | required | Moment order \(N\). `0` gives classical shallow water |
| `t_end` | number | required | Final time |
| `method` | choice | `classical` | Only `classical` exists |
| `fvm_type` | choice | `PVM` | Only `PVM` exists |
| `pvm` | choice | required | `Roe`, `Osher`, `LF` or `PRICE` |
| `time_integrator` | choice | required | `ExplicitEuler` or `ImplicitEuler`. `Exact` is accepted but unusable — see below |
| `boundary_condition` | choice | required | `PERIODIC` or `INFLOW_OUTFLOW` |

See [Numerical method](numerics.md#fluctuations-instead-of-fluxes) for how the schemes
differ and which of them are well balanced.

!!! danger "Do not select `time_integrator: Exact`"
    It is a placeholder for a source term whose exact update is known in closed form,
    and none of the shipped models provides one. Selecting it overwrites the state with
    the source term at every step instead of integrating anything: an inviscid case ends
    with \(h = 0\) everywhere and no error is raised. Use `ExplicitEuler`, or
    `ImplicitEuler` with `linear_source: true` for stiff friction. See
    [What to watch out for](limitations.md#the-exact-time-integrator-is-a-stub).

## `topography`

Optional. Leaving it out means a flat bed at zero, and the solver stays on its simpler
non-augmented path.

| Key | Type | Default | Meaning |
|:--|:--|:--|:--|
| `bed_profile` | string | `flat` | See below |
| `reference_water_level` | number | `1.0` | Still water level \(H\), used by lake-at-rest cases |
| `perturbation_amplitude` | number | `0.0` | Size of the initial disturbance |
| `perturbation_center` | number | `0.0` | Where it sits |
| `perturbation_width` | number | `1.0` | How wide it is |

Any other key in this section is passed to the bed profile as a parameter. The key set
is open for that reason, but a name the profile does not accept is still an error.

Available profiles, with the shape each one makes and the keys it accepts. Every
parameter has a default, so a profile named with no parameters at all is valid.

| Profile | \(Z(x)\) | Parameters (defaults) |
|:--|:--|:--|
| `flat` | \(e\) | `elevation` (0) |
| `linear_slope` | \(e + s\,(x - x_r)\) | `slope` (0), `x_ref` (0), `elevation` (0) |
| `gaussian_bump` | \(a \exp\!\left[-\left(\frac{x-c}{w}\right)^2\right]\) | `amplitude` (0.2), `center` (0.5), `width` (0.1) |
| `parabolic_bump` | \(\max\!\left(0,\; a\left[1 - \left(\frac{x-c}{w_h}\right)^2\right]\right)\) | `amplitude` (0.2), `center` (10), `half_width` (2) |
| `sinusoidal` | \(e + a \sin\!\left(\frac{2\pi (x - p)}{L}\right)\) | `amplitude` (0.1), `wavelength` (1), `phase` (0), `elevation` (0) |
| `step` | \(e + a\) for \(x \geq p\), else \(e\) | `amplitude` (0.1), `position` (0.5), `elevation` (0) |
| `tanh_step` | \(e + \frac{a}{2}\left[1 + \tanh\frac{x-p}{w}\right]\) | `amplitude` (0.1), `position` (0.5), `width` (0.05), `elevation` (0) |

Three notes on the less obvious ones:

- `parabolic_bump` takes a **half width**, not a width, and its defaults are sized for
  the standard \([0, 25]\) channel benchmark rather than for a unit domain. Scale them
  to the domain in use.
- `step` and `tanh_step` take `amplitude`, not `height`, and a negative amplitude makes
  a descending step. `elevation` shifts the whole bed, so `elevation: 0.1` with
  `amplitude: -0.1` runs from 0.1 down to 0.
- With `PERIODIC` boundaries and `sinusoidal`, pick a wavelength that divides the domain
  length. Otherwise the periodic ghost cells introduce a bed jump at the wrap.

Example:

```yaml
topography:
  bed_profile: gaussian_bump
  amplitude: 0.4
  center: 0.5
  width: 0.1
  reference_water_level: 1.0
```

## `wet_dry`

Optional. The defaults never activate above a depth of about \(10^{-3}\), so a config
that leaves this out behaves as though wet-dry handling did not exist.

| Key | Type | Default | Meaning |
|:--|:--|:--|:--|
| `eps_div` | number | `1e-14` | Division guard. Not a physical quantity |
| `h_dry` | number | `1e-4` | At or below this a cell is dry |
| `h_wet` | number | `1e-3` | At or above this a cell is ordinary wet flow |

They must satisfy `eps_div <= h_dry < h_wet`, and this is checked.

!!! tip "Choosing h_dry"
    Put it well below the smallest depth the problem needs to resolve, then check that
    \(\nu / h_{\text{dry}}^2\) is still a sane number. An inviscid problem has no lower
    limit and can afford a very small value. A viscous one cannot. See
    [What to watch out for](limitations.md#h_dry-is-a-modelling-decision).

## `postprocessing`

Optional.

| Key | Type | Default | Meaning |
|:--|:--|:--|:--|
| `output_dir` | string | `results/<config-name>` | Where results go. `--output-dir` overrides it |
| `store_history` | bool | `true` | Store the state at intervals |
| `history_stride` | integer | `1` | Store every Nth step |
| `store_hyperbolicity` | bool | `false` | Record the eigenvalues of \(A(U)\) in every cell |
| `hyperbolicity_stride` | integer | `1` | Record every Nth step |

`store_hyperbolicity` is expensive. It builds and eigendecomposes a matrix in every
cell at every recorded step, and it writes one row per cell per step. Use a stride, and
turn it on only when you are investigating hyperbolicity specifically.

## Initial conditions

Available for `SWME1D` and `HSWME1D`:

`constantHeight_noVelocity`, `constantHeight_constantVelocity`,
`damBreak_noVelocity`, `damBreak_constantVelocity`, `symmetric_damBreak`,
`colliding_damBreak`, `linearDamBreak_noVelocity`, `linearHeight_noVelocity`,
`smooth_wave`, `smooth_constantVelocity`, `smooth_plus_damBreak`,
`smooth_wave_smallHeightGradient`, `damBreak_dryBed`, `damBreak_dryBed_partial`,
`lakeAtRest`, `perturbedLakeAtRest`.

`lakeAtRest` and `perturbedLakeAtRest` need a `topography` section. The first is the
C-property benchmark: still water over an uneven bed, which should stay still forever.

`damBreak_dryBed` releases water onto ground with \(h = 0\) exactly, not a thin film.

For `RechargeSWME1D`: `smooth_nested_profile_pulse_aggressive`,
`smooth_nested_profile_pulse_mild`, `horton_moment_order_pulse`.

All three are generic in the moment order `N` and *nested* in it: `h(x)` and `u_m(x)`
do not depend on `N`, and each further order adds one moment without changing any below
it, so runs at different `N` differ by the model and not by the problem. Moment `i` is
seeded at \((-1/2)^{i-1}\) times the first-moment amplitude — a ratio deliberately kept
clear of the non-hyperbolic wedge documented in `RESTRUCTURE_PLAN.md` §6, where what
matters is `alpha_2/alpha_1` rather than the size of the moments.

## A complete example

```yaml
# Dam break onto a dry bed.
pde:
  type: SWME1D
  initial_condition: damBreak_dryBed
  viscosity: 0.0
  slip_length: 1.0
  linear_source: false
  hyperbolic: false

grid:
  x1: -1.0
  x2: 1.0
  resolution_x: 200

numerics:
  order: 1
  t_end: 0.2
  method: classical
  fvm_type: PVM
  pvm: Roe
  time_integrator: ExplicitEuler
  boundary_condition: INFLOW_OUTFLOW

wet_dry:
  h_dry: 1.0e-10
  h_wet: 1.0e-9
  eps_div: 1.0e-14

postprocessing:
  store_history: true
  history_stride: 1
```

The unusually small thresholds here are on purpose. This case is inviscid, so there is
no friction term to be floored, and a vacuum front is exactly where a large
\(h_{\text{dry}}\) costs accuracy.
