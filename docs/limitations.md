# What to watch out for

Every item here was measured rather than assumed. Most are pinned by a test as well, so
they stay documented properties instead of being rediscovered later as mysteries. Where
an entry is reproducible but not pinned, it says so.

## Hyperbolicity depends on the ratio, not the size

SWME loses hyperbolicity from \(N = 2\) upwards. The natural guess is that this happens
when the moments get large, so keeping them small should be safe. That guess is wrong.

Hyperbolicity depends only on the scaled moments \(\alpha_i / \sqrt{g h}\). This was
checked directly: shifting \(u_m\) leaves the imaginary parts unchanged bit for bit,
and rescaling \(h \to c^2 h\) with \(\alpha \to c\, \alpha\) is exactly linear.

At \(N = 2\) the unstable set is a **narrow wedge in the ratio of the moments**,
confined to

\[
\left| \frac{\alpha_2}{\alpha_1} \right| \in [1.14,\ 1.40]
\]

Every direction outside that range stayed hyperbolic at every magnitude tested, out to
\(|\alpha| / \sqrt{g h} = 60\). Concretely, \(\alpha = (1.5,\ 1.8)\) is **not**
hyperbolic, while the strictly larger \(\alpha = (2.0,\ 3.0)\) is fine.

So "keep the moments small" is the wrong safety rule. The ratio is what matters.

If you need a guarantee rather than a check, use HSWME. It never lost hyperbolicity in
any state tested, at any order from 1 to 6, and it repaired every sampled state that
broke SWME.

!!! success "The reference results are clean, with room to spare"
    Every state visited by the reference simulations was replayed through both
    closures: 14417060 states across 20 runs, with no loss of hyperbolicity in either.
    That is not luck. The largest scaled moment reached anywhere is
    \(|\alpha_1| / \sqrt{g h} = 0.71\), and the initial conditions set
    \(\alpha_2 = -0.5\,\alpha_1\), a ratio of 0.5, which is outside the unstable wedge
    and therefore hyperbolic at any magnitude.

## The runtime warning measures the scheme, not the model

If a run prints a warning about complex spectra, read it carefully before concluding
anything about the model.

The always-on counter eigendecomposes the path-averaged interface matrix, not \(A(U)\).
Those are different questions. A dam break onto a dry bed trips the counter at 78 of
12462 interfaces at \(N = 0\), where the model is plain shallow water and cannot
possibly lose hyperbolicity. The count is identical at \(N = 1\), \(N = 2\) and for
HSWME.

Meanwhile \(A(U)\) itself was checked at all 62310 states those runs visited, and the
largest imaginary part was exactly zero.

To ask about the model rather than the scheme, set `store_hyperbolicity: true` and look
at the per-cell spectrum. The [run report](reports.md#reading-the-hyperbolicity-pages)
presents the two separately for this reason.

## h_dry is a modelling decision

At a wetting front the exact solution contains arbitrarily small depths. For a dam
break onto a dry bed, the depth vanishes quadratically at the front, so the leading edge
always sits below any fixed \(h_{\text{dry}}\) and is damped. The computed front comes
out too slow.

**Refining the mesh does not fix this**, because a finer grid only resolves more of the
region that is being truncated. Measured on the standard dry dam break at 800 cells,
against an exact front speed of 2.0:

| \(h_{\text{dry}}\) | Front speed | Error |
|:--|--:|--:|
| \(10^{-4}\) (default) | 1.758 | 12 % |
| \(10^{-8}\) | 1.934 | 3 % |
| \(10^{-12}\) | 1.984 | 0.8 % |

The trade-off runs the other way as soon as there is viscosity. \(h_{\text{dry}}\) also
floors the \(\nu / h^2\) friction term, so shrinking it is not free. At
\(h_{\text{dry}} = 10^{-10}\) that term reaches \(\nu \times 10^{20}\), which is a
finite number, so no check for infinities catches it, and it is meaningless.

> **Rule of thumb.** Put \(h_{\text{dry}}\) well below the smallest depth the problem
> needs to resolve, then check that \(\nu / h_{\text{dry}}^2\) is still a sane number.
> An inviscid problem has no lower limit. A viscous one does.

## Viscous drying needs the implicit source path

Navier-slip friction carries \(\nu / h^2\). Near a drying front this is stiff however
\(h_{\text{dry}}\) is chosen, since the floor only caps it at
\(\nu / h_{\text{dry}}^2\), still around \(10^6\) at the default.

Explicit integration goes unstable there. The symptom is misleading: the moment blows
up first and drags the depth negative, so the failure appears to come from the
positivity limiter, which is not at fault.

This was measured across viscosities from \(10^{-4}\) upwards and across
\(h_{\text{dry}}\) from \(10^{-4}\) to \(10^{-2}\), so it is genuine stiffness rather
than a threshold that needs tuning. It succeeds at every setting with
`linear_source: true` together with `ImplicitEuler`, which is what that path exists for.
`RechargeSWME1D` has no linear source form and rejects `linear_source: true`; there, use
`ImplicitEuler` with `linear_source: false`, which solves the source step by Newton
iteration.

The error message names this cause when it fires on a viscous explicit run.

## Runs overshoot their configured t_end

The time loop runs while \(t < t_{\text{end}}\) and takes no final partial step, so a
run finishes up to one timestep past the requested time. How far past depends on the
resolution.

This is **deliberately not fixed**. Correcting it would change every validated result,
and reproducing those results is the property this codebase is built to preserve.

It matters whenever a result is read at a specific time: comparing against an exact
solution, or comparing two resolutions with each other. In those cases, read the actual
time from the output rather than assuming it equals `t_end`.

## Shorelines over topography: what is and is not covered

Dry ground next to wet ground over a non-flat bed used to stop the run with a negative
depth: a lake with dry shores failed on its first step, and so did a dam break over a
bump or down a slope. The [shoreline treatment](numerics.md#shorelines-over-topography)
handles these. Everything below was measured with it, in the code's units (\(g = 1\)),
and is pinned in `tests/test_shoreline.py`, each case with a negative control that
fails when the treatment is switched off.

| Case | Result |
|:--|:--|
| Lake at rest in a parabolic bowl, dry shores, more than 200 steps | Depth changes by at most \(2.2 \times 10^{-16}\); no mass created by clamping. \(N = 0, 1, 2\), `Roe` and `Osher` |
| Sloshing in a parabolic bowl (Thacker's planar solution), one period | Mass drift below \(6 \times 10^{-16}\), no negative depth. Mean \(L^1\) error of the depth, relative to the volume: 2.6 % at 60 cells, 1.2 % at 120, 0.7 % at 240 (`Roe`, \(N = 0\)), so first order. With a uniform first moment \(0.2 h\): 3.4 % at \(N = 1\), 5.1 % at \(N = 2\) (60 cells; the exact solution is the \(N = 0\) one) |
| Dry dam break over a Gaussian bump (amplitude 0.2), 200 cells | Runs; mass change exactly 0. This is the case that used to fail at step 10 |
| Castro et al. Test 1, shock reflected by an emerged step | The step stays exactly dry and the inflow is fully accounted for. The reflected depth is 4.845 against an exact 5.131, 5.6 % low. With the correct Gauss-Legendre nodes it is 5.1305: the deficit is the [quadrature defect](#the-path-quadrature-is-not-gauss-legendre), not the treatment |
| Castro et al. Test 2, dam break over a plane inclined by \(\pm\pi/60\), \(\Delta x = 0.05\) | Runs both ways, mass conserved to round-off. Front speed: downhill 1.85 against 2.06 (10 % slow), flat bed 1.71 against 2.00 (14 % slow), uphill 1.60 against 1.94 (17 % slow). At \(\Delta x = 0.025\): downhill 12 %, flat 14 %, uphill 15 % slow. Relative to the flat front the exact slope effect is \(-3.0\) % uphill and \(+2.9\) % downhill; measured \(-6.4\) % and \(+7.8\) % at \(\Delta x = 0.05\), \(-4.7\) % and \(+5.7\) % at 0.025 |

What to keep in mind:

- **A front on a dry slope still lags.** Most of the lag is the flat-bed one: the
  first-order scheme at a vacuum front, read off at \(h = 10^{-3}\), which is there
  without any bed and barely shrinks with the mesh. The slope adds to it, uphill and
  downhill, by a few percent that do shrink with the mesh (see the table). Before the
  paper's Cases 2 and 3.2 were implemented (2026-11-19), an emerged dry cell was a wall
  until the water behind it was deeper than the step, and the uphill front was 24 %
  slow.
- **Three deliberate departures from the paper**, all argued and measured in
  [Numerical method](numerics.md#shorelines-over-topography): the wall of Case 3.1
  needs the bed to emerge; below a drop the supercritical state is used in every case,
  where the paper takes the subcritical one when the edge state is critical; flat
  wet/dry interfaces keep the ordinary fluctuation instead of the paper's Case 1.
- **The interface states are modelling choices for \(N \geq 1\).** Every depth (wall,
  climb, overfall) is the exact shallow-water one, computed from the depth-averaged
  velocity, and the moments per unit depth are carried into every interface state
  unchanged. For \(N = 0\) this is the scheme of the paper. For higher orders there is
  no reference to compare against.
- **Only interfaces with a dry cell are treated.** A thin layer that is still above
  \(h_{\text{dry}}\), perched on a bed step above deeper water, keeps the ordinary
  fluctuation. It then drains quickly and the positivity limiter cuts the timestep
  while it does. This costs time rather than correctness. Over one period of the
  sloshing run at 60 cells the smallest timestep is 180 times below the largest with
  `Roe` and 150 times with `Osher` at \(N = 0\), and 300 and 250 times with `Roe` at
  \(N = 1\) and \(N = 2\) (measured 2026-11-19; before Castro's Cases 2 and 3.2 were
  implemented the \(N = 2\) run fell to \(9 \times 10^{-6}\) against
  \(5 \times 10^{-2}\) for a few steps).
- **`h_dry` matters more here than on a flat bed.** A cell is handed to the treatment
  when it drops to \(h_{\text{dry}}\), so choose it well below the depths of interest.
- **`LF` and `PRICE` are not well balanced** with or without dry shores.

## The path quadrature is not Gauss-Legendre

*Measured in the implementation audit. Not fixed, because fixing it changes every
reference result.*

The fluctuation at an interface is the integral of the system matrix along the straight
path between the two cell states. The code evaluates it with five nodes and the
Gauss-Legendre weights, but the nodes are mis-transcribed: on \([-1, 1]\) they sit at
\(\pm 0.5232\), \(\pm 0.3109\) and 0 instead of \(\pm 0.9062\), \(\pm 0.5385\) and 0. The
rule integrates \(s^2\) over \([0, 1]\) to 0.2778 instead of 1/3, so it is exact for linear
integrands only.

What this does and does not affect:

- A jump between two states at rest is integrated exactly, because the matrix is then
  linear in \(h\). Well-balancing and mass conservation are unaffected.
- A jump with a velocity difference is not. For \((h, q) = (1, 0.5) \to (0.2, -0.3)\) at
  \(N = 0\) the momentum fluctuation sums to \(-0.628\) where the flux difference is
  \(-0.280\). Momentum is therefore not conserved exactly across strong shocks.
- For the Stoker dam break with depths 1 and 0.1, the plateau depth is 0.39370 at 800
  cells against an exact 0.39617, and the 0.6 % gap does not close under refinement. With
  the correct nodes the same run gives 0.39600. For the milder 2 to 1 dam break the two
  agree to four digits.

Smooth solutions are affected at higher order only, since the jumps between neighbouring
cells shrink with the mesh.

## Cell centres and cell width disagree

*Measured in the implementation audit. Not fixed, for the same reason.*

The mesh places its \(n\) cell centres with `linspace(x1, x2, n)`, a spacing of
\((x_2 - x_1)/(n - 1)\), while the finite-volume update uses the cell width
\((x_2 - x_1)/n\). Positions (initial conditions, bed sampling, the `x` column of the
output) and the update therefore disagree by a factor \(n/(n-1)\): 1.005 at 200 cells.
Wave speeds and bed slopes read from the output carry that factor. It vanishes as
\(1/n\).

## The Exact time integrator is a stub

*Reproducible, not yet pinned by a test. `tests/test_cli.py` checks only that the config
loader builds the right object, not what it then does.*

`time_integrator: Exact` is accepted by the config loader and does something actively
wrong.

It is written for a source term that can report its own exactly-integrated update, and
calls the source routine as though it returned one. No model in this package does: they
return the source term itself. So the state is *replaced* by the source at every step
rather than advanced by it. An inviscid case, whose source is zero, finishes with
\(h = 0\) in every cell — a completed run, no warning, no error, and a result that is
entirely meaningless.

Use `ExplicitEuler`, or `ImplicitEuler` with `linear_source: true` where the friction is
stiff. Nothing shipped selects `Exact`, and nothing should until a source term provides
the closed-form update it expects.

## Well balancing depends on the scheme

Only `Roe` and `Osher` preserve a lake at rest. `LF` and `PRICE` have a viscosity with
a nonzero constant term, so they leave a residual over an uneven bed no matter how the
bed slope is discretised.

Selecting `LF` or `PRICE` together with topography produces a warning at startup. The
combination is allowed, since it is sometimes useful for a robustness comparison, but
it is not well balanced and results from it should not be read as though it were.

## Performance

The solver calls the system matrix routine once per interface, per quadrature point,
per timestep, on arrays of size \(N + 2\). At small \(N\) the per-call overhead of
numpy dominates the actual arithmetic.

Evaluating the whole grid in one batch would remove that overhead, and it is the right
answer to the performance question. It is a substantial change to the time loop and has
not been made. Worth knowing before starting large production runs.
