# What to watch out for

Every item here was measured rather than assumed, and each is pinned by a test so it
stays a documented property instead of being rediscovered later as a mystery.

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
