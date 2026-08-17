# Numerical method

## Why a standard solver will not do

The moment system contains a term \(Q(U)\,\partial_x U\) that is not the derivative of
any flux. For a genuinely conservative system, a shock has a single correct jump
relation fixed by the equations. Here it does not: the jump depends on the path taken
through state space between the left and right values.

The method used is **path-conservative**, in the sense of Castro and Pares. A path
between the two states must be chosen explicitly, and the scheme is built around it.
This solver uses the straight line

\[
\psi(s) = (1 - s)\, U_L + s\, U_R, \qquad s \in [0, 1]
\]

The integral along the path is evaluated with five-point Gauss-Legendre quadrature.

## Fluctuations instead of fluxes

Each cell interface produces two fluctuations, \(D^-\) and \(D^+\), which are the
amounts sent left and right. They are built from the path-averaged matrix

\[
\bar{A} = \int_0^1 A\bigl(\psi(s)\bigr)\, \mathrm{d}s
\approx \sum_{k} w_k\, A\bigl(\psi(s_k)\bigr)
\]

together with a viscosity matrix that stabilises the scheme:

\[
D^{\pm} = \tfrac{1}{2}\left(\bar{A} \pm Q_{\text{visc}}\right)\left(U_R - U_L\right)
\]

The four available schemes differ only in how \(Q_{\text{visc}}\) is built.

| Scheme | Viscosity | Cost | Well balanced |
|:--|:--|:--|:--|
| `Roe` | \(\lvert \bar{A} \rvert\), from an eigendecomposition | Higher | Yes |
| `Osher` | Built per quadrature node | Highest | Yes |
| `LF` | \(\frac{\Delta x}{\Delta t} I\), Lax-Friedrichs | Lowest | No |
| `PRICE` | A polynomial in \(\bar{A}\) | Low | No |

`Roe` is the default choice for most cases. It resolves waves sharply and is well
balanced. `LF` is cheap and very robust but smears features.

The well balanced column is explained in the next section, and it matters more than it
might appear.

## Time integration

| Setting | Behaviour |
|:--|:--|
| `ExplicitEuler` | Forward Euler on the source term. The normal choice |
| `ImplicitEuler` | Backward Euler. Needed for stiff friction |
| `Exact` | No source integration, for source-free cases |

The timestep is chosen from a CFL condition using the largest wave speed on the grid.

When `linear_source: true`, the friction term is returned as a matrix rather than a
vector, and only `ImplicitEuler` knows how to apply it. Pairing it with any other
integrator is a silent wrong answer, so the config loader rejects the combination at
startup instead of letting it run.

## Bottom topography

A sloping bed adds a source term \(g h\, \partial_x Z\) to the momentum equation, where
\(Z(x)\) is the bed elevation. Discretising this as an ordinary cell-centred source
does not work, for a reason worth spelling out.

### The lake at rest problem

Consider still water over an uneven bed. The free surface \(h + Z\) is flat, the
velocity is zero, and nothing should ever move. But \(h\) itself is *not* constant,
because the bed is not flat. So the flux difference across a cell is nonzero, and the
bed slope source is nonzero. The correct answer requires the two to cancel exactly.

Discretise them separately and they cancel only approximately. The leftover drives a
spurious current, and small waves that a real simulation is meant to resolve get lost
underneath it. A scheme that keeps still water still is called **well balanced**, and
the property is known as the C-property.

### The augmented system

The fix is to stop treating the bed as a source and make it part of the state. The
state is extended with \(Z\), which does not change in time:

\[
\tilde{U} = \begin{pmatrix} U \\ Z \end{pmatrix},
\qquad
\tilde{A} = \begin{pmatrix} A(U) & b \\ 0 & 0 \end{pmatrix},
\qquad
b_1 = g h
\]

Now the bed slope travels through the same path integral as everything else, and at
rest the two contributions cancel by construction rather than by luck.

!!! warning "The sign of that entry is load bearing"
    The augmented entry is \(+gh\), not \(-gh\). With the wrong sign the two
    contributions do not cancel, they add: the residual becomes \(2 g h\, \Delta h\)
    instead of zero, which is as badly balanced as it is possible to be. A test pins
    the sign by flipping it and checking that the C-property fails.

### Well balancing needs the right viscosity

Making the bed part of the state is necessary but not sufficient. The fluctuation also
contains the viscosity matrix, and it must annihilate the jump as well. That happens
exactly when the viscosity polynomial satisfies \(P(0) = 0\).

`Roe` and `Osher` satisfy it. `LF` and `PRICE` do not: their viscosity has a nonzero
constant term, so they leave a residual at a lake at rest no matter how the bed is
discretised. Selecting one of them together with topography produces a warning at
startup rather than a quietly wrong result.

### Available bed profiles

`flat`, `gaussian_bump`, `parabolic_bump`, `linear_slope`, `sinusoidal`, `step`,
`tanh_step`.

Each takes its own parameters, given as extra keys in the `topography` section. An
unknown parameter name is an error, so a misspelling does not silently do nothing.

## Wet and dry cells

Dividing by \(h\) is unavoidable: the state stores \(h u_m\) and \(h\alpha_i\), while
every closure needs \(u_m\) and \(\alpha_i\). As \(h\) approaches zero those divisions
blow up, and the friction term divides by \(h\) twice, so it blows up sooner.

The rule for extracting primitives lives in one place, `swme.wetdry`, so the transport
matrix, the wave speeds, the friction and the recharge sources cannot disagree about
what a nearly dry cell means.

### Three regimes

| Depth | Treatment |
|:--|:--|
| \(h \geq h_{\text{wet}}\) | Ordinary wet flow. Plain \(q / h\), no regularisation of any kind |
| \(h_{\text{dry}} \leq h < h_{\text{wet}}\) | Velocity still \(q/h\); moments ramped linearly down to zero |
| \(h < h_{\text{dry}}\) | Velocity driven smoothly to zero; moments already zero |

Defaults are \(h_{\text{dry}} = 10^{-4}\) and \(h_{\text{wet}} = 10^{-3}\), with a
third threshold `eps_div` at \(10^{-14}\) guarding against dividing by exactly zero.
These are absolute depths, not ratios, and they suit problems where \(h\) is of order
one. A problem where \(h\) is everywhere \(10^{-3}\) would be entirely "dry" under
these defaults, so scale them to the case.

Below \(h_{\text{wet}}\) the velocity uses a desingularised form:

\[
u_m = \frac{2 h q}{h^2 + \max(h,\, h_{\text{dry}})^2}
\]

This is identically \(q/h\) whenever \(h \geq h_{\text{dry}}\), because the denominator
is exactly \(2h^2\) there, and it decays smoothly to zero below. Using the physical
threshold rather than the machine epsilon in the denominator matters: a hard cut at
\(h_{\text{dry}}\) would put a jump of size \(q / h_{\text{dry}}\) right at the wetting
front, which is the worst possible place for a discontinuity.

The moments are ramped to zero across the transition band because a vanishing film
cannot support a vertical velocity profile. It relaxes to plug flow instead.

Above \(h_{\text{wet}}\) the arithmetic is exactly what it was before any wet-dry
handling existed, so a simulation that never approaches drying reproduces earlier
results bit for bit.

### Keeping the depth positive

Even with careful primitives, an explicit update can overshoot and produce a negative
depth. A limiter reduces the timestep when a cell is heading that way. Cells already at
or below \(h_{\text{dry}}\) are excluded from the calculation, because a dry cell with
a small spurious outflow would otherwise drive the timestep to zero. Any mass created
by clamping is tracked and reported, so it can be checked rather than assumed small.

!!! note "The threshold is a modelling choice"
    \(h_{\text{dry}}\) is not a formality to be set and forgotten. It has a measurable
    effect on results at a wetting front, and the trade-off runs in both directions.
    See [What to watch out for](limitations.md#h_dry-is-a-modelling-decision).

## Boundary conditions

| Setting | Behaviour |
|:--|:--|
| `PERIODIC` | The domain wraps. What leaves one end enters the other |
| `INFLOW_OUTFLOW` | Zero gradient extrapolation at both ends |

When topography is present, the bed is given ghost cells filled the same way as the
state. If the two disagreed, the interfaces at the domain edges would see an
inconsistent pair of state and bed, and the well balancing would fail exactly there.
