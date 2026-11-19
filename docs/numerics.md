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

The integral along the path is evaluated with a five-point quadrature rule. It was
meant to be Gauss-Legendre, but its nodes are mis-transcribed, which makes it exact for
linear integrands only; see
[What to watch out for](limitations.md#the-path-quadrature-is-not-gauss-legendre).

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
| `Exact` | A placeholder for a source with a known closed-form update. **Not usable** — see the warning below |

The timestep is chosen from a CFL condition using the largest wave speed on the grid.

!!! danger "`Exact` does not mean \"skip the source\""
    It applies the source routine as though that routine returned the updated state,
    which none of the shipped models does — they return the source term itself. The
    state is therefore replaced by the source at every step. An inviscid run, where the
    source is zero, ends with \(h = 0\) in every cell, and nothing raises. It is kept
    only because a future source term could provide such an update.

When `linear_source: true`, the friction term is returned as a matrix rather than a
vector, and only `ImplicitEuler` knows how to apply it. Pairing it with any other
integrator is a silent wrong answer, so the config loader rejects the combination at
startup instead of letting it run.

`RechargeSWME1D` has no matrix form at all: its source is returned as a vector, and the
linear branch of `ImplicitEuler` would compute `S(U) @ U`, a scalar written into every
component. Both the config loader and the `RechargeSWME1D` constructor reject
`linear_source: true`. For stiff recharge runs, use `ImplicitEuler` with
`linear_source: false`, which solves the source step by Newton iteration.

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
unknown parameter name is an error, so a misspelling does not silently do nothing. The
formula and the parameter names for each profile are listed under
[Configuration](configuration.md#topography).

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

### Shorelines over topography

The limiter above governs cells that hold water. It cannot help a dry cell, and over a
non-flat bed the ordinary fluctuation does ask a dry cell for water: at a shoreline the
bank lies above the free surface, the straight path between the two cells is no longer
a lake at rest, and the numerical viscosity moves mass out of the dry cell. A lake with
dry shores then fails on its very first step.

The remedy is the modified Roe scheme of Castro, Gonzalez-Vida and Pares, *Numerical
treatment of wet/dry fronts in shallow flows with a modified Roe scheme*, M3AS 16(6)
(2006) 897-931, Section 5, written in fluctuation form and extended to the moment
variables. It is implemented in `swme.wetdry.shoreline_fluctuation`. Equation numbers
below are the paper's. The paper writes the bed as a depth \(H\) below a reference
level, so \(Z = -H\) here: its Case 2 (\(H_L < H_R\), dry cell on the right) is a bed
that *drops* towards the dry cell, its Case 3 (\(H_L > H_R\)) a bed that *rises*.

**Where it acts.** At an interface between two cells with different bed elevations,
at least one of them dry, \(h \leq h_{\text{dry}}\). Interfaces between two wet cells,
and interfaces where the bed is locally flat, keep the ordinary fluctuation. A run
without such an interface is unchanged bit for bit.

**The idea.** At such an interface the linear Riemann problem of the Roe scheme is
replaced by the nonlinear one, solved exactly for shallow water. Its solution gives two
interface states, \(W^-\) on the left bed and \(W^+\) on the right bed, joined by a
stationary contact at the step. Eq. (5.1) makes the numerical fluxes the physical
fluxes of those states, \(F^- = F(W^-)\) and \(F^+ = F(W^+)\). In fluctuation form,
with \(J(A \to B)\) the jump from state \(A\) to state \(B\),

\[
D^-_{i+1/2} = J(U_i \to W^-), \qquad D^+_{i+1/2} = J(W^+ \to U_{i+1}).
\]

In the mass and momentum rows \(J(A \to B) = F(B) - F(A)\) with the model's own
momentum flux,

\[
F_{\text{mom}}(U) = \frac{q^2}{h} + \frac{g h^2}{2} + \sum_i \frac{(h\alpha_i)^2}{(2i+1)\,h},
\]

whose gradient is the momentum row of \(A(U)\). For HSWME only \(\alpha_1\) enters,
because the system matrix is assembled with \(\alpha_2, \dots, \alpha_N = 0\). The
moment rows have no flux. There \(J\) is the path integral of \(A(U)\) from \(A\) to
\(B\), evaluated with the scheme's own quadrature. The contact itself carries the bed
force. Its fluctuation is zero by the generalised Rankine-Hugoniot condition (2.26) at
zero speed, so it appears in neither \(D^-\) nor \(D^+\).

**Two one-sided problems.** Let \(L\) be the cell on the lower bed, \(H\) the one on the
higher bed, \(s = Z_H - Z_L > 0\) the step. The interface problem is split into two of
the paper's problems, each for the water of one cell with the other cell taken to be
empty (`wetdry.one_sided_problem`):

- the water of \(L\) facing a bed that rises by \(s\) towards an empty cell (Case 3);
- the water of \(H\) facing a bed that drops by \(s\) towards an empty cell (Case 2).

When one cell is exactly empty, as in the paper, one of the two problems is trivial and
the scheme is Castro's. When the dry cell holds a film below \(h_{\text{dry}}\), the two
are superposed. For the problem of cell \(X\) with interface states \(W_{\text{near}}\)
on its own side and \(W_{\text{far}}\) on the other, \(X\) receives
\(J(U_X \to W_{\text{near}})\) and the other cell \(J(W_{\text{far}} \to 0)\), with the
jumps ordered left to right. The mass rows are then rewritten in flux form,
\(D^-_0 = F - q_{\text{left}}\) and \(D^+_0 = q_{\text{right}} - F\). Here \(F\) is
the sum of the two problems' discharges and \(q\) are the stored discharges, so the mass
that leaves one cell enters the other exactly.

**The cases.** For the water of one cell: \(u\) is the depth-averaged velocity towards
the empty neighbour, \(c = \sqrt{gh}\), \(E = h + u^2/(2g)\) the specific energy and
\(h_c = (q^2/g)^{1/3}\) the critical depth.

| Bed towards the empty cell | Condition | Case | Interface states |
|:--|:--|:--|:--|
| rises by \(s\) | \(u \geq c\) and \(s \leq E - \tfrac32 h_c\) | 3.2, climb | \(W_{\text{near}} = U\); \(W_{\text{far}}\): same \(q\), supercritical root of \(E(h) = E - s\) (5.29) |
| rises by \(s\) | \(s \geq h\), not the row above | 3.1, wall | \(W_{\text{near}} = (h_w, 0)\), \(h_w\) from (5.23)-(5.27); \(W_{\text{far}} = 0\) |
| rises by \(s\) | every other case (\(s < h\)) | 3.3 | ordinary fluctuation for the whole interface |
| drops by \(s\) | \(u \leq -2c\) | 2(a) | \(W_{\text{near}} = W_{\text{far}} = 0\) (5.10) |
| drops by \(s\) | \(u > -2c\) | 2(b), 2(c) | \(W_{\text{near}} = W_0\) of (5.8); \(W_{\text{far}}\): discharge of \(W_0\), supercritical root of \(E(h) = E(W_0) + s\) (5.11)-(5.14) |

with the wall depth

\[
h_w =
\begin{cases}
0, & u \leq -2c \quad \text{(a dry zone opens, 5.23)} \\[4pt]
\dfrac{(u/2 + c)^2}{g}, & -2c < u \leq 0 \quad \text{(rarefaction, 5.25)} \\[8pt]
\text{greatest root of } x^3 - h x^2 - \Big(h^2 + \dfrac{2 h u^2}{g}\Big) x + h^3, & u > 0 \quad \text{(reflected shock, 5.27)}
\end{cases}
\]

the state at the edge of the rarefaction into the dry bed, eq. (5.8),

\[
W_0 =
\begin{cases}
(0, 0), & u \leq -2c \\[4pt]
\Big(\dfrac{(u + 2c)^2}{9g},\ \dfrac{(u + 2c)^3}{27g}\Big), & -2c < u < c \quad \text{(critical: } u_0 = c_0\text{)} \\[8pt]
(h, q), & u \geq c
\end{cases}
\]

and the depths on a step from the steady contact (2.30) with \(q\) unchanged,
\(h + q^2/(2gh^2) = E\). Equations (5.15) and (5.18) are this relation multiplied by
\(h^2\). It has a supercritical root \(h < h_c\) and a subcritical one \(h > h_c\) when
\(E > \tfrac32 h_c\). Both are found by a monotone Newton iteration
(`wetdry.bernoulli_depth`).

**Deviations from the paper, deliberate.**

1. *The wall needs the bed to emerge.* The paper's second condition for Case 3.1, flow
   towards the step without the energy to climb it (5.20), does not repeat the
   emergence condition of the first. But its Fig. 8(b) draws the bed emerged, and the
   case is introduced as "the step acts as an obstacle". Read literally, it would also
   wall off a dry cell whose bed lies *below* the free surface whenever the discharge is
   too large for a steady flow over the step, although the water standing above that
   bed spreads onto it. Measured on Thacker's sloshing, the literal reading raises the
   mean depth error from 0.0118 / 0.0058 / 0.0037 to 0.0278 / 0.0111 / 0.0088 at
   60 / 120 / 240 cells. So the code requires \(s \geq h\) for the wall. An emerged step
   can be climbable only at \(\text{Fr} \geq 3^{3/4}\) (from \(u^2/(2g) \geq \tfrac32 h_c\)),
   so with this reading Case 3.3 is exactly the set of submerged steps.
2. *Below a drop, the supercritical root.* The paper takes it when the cell is
   supercritical, case (b), but the subcritical root (5.17)-(5.18) when \(W_0\) is
   critical, case (c). The code takes the supercritical root in both. With the
   subcritical root the state below the step has \(u - c < 0\), so the rarefaction that
   must join it to the dry bed would run back through the stationary step: the wave
   structure the state belongs to does not exist (the paper's Remark 5.1 notes the
   resonance). Its depth also grows like the drop while its discharge stays that of
   \(W_0\), which hands the empty cell a momentum flux of order \(g s^2/2\) with almost
   no water. The supercritical state is the free overfall: the water accelerates down
   the step. Measured on the shipped lake case, the subcritical root raises the largest
   velocity from 1.00 to 1.66 and the step count from 941 to 982. The lake level, the
   overtopping time and the spilled volume change in the fourth digit.
3. *Flat wet/dry interfaces (Case 1)* keep the ordinary Roe or Osher fluctuation, as
   they do on a flat-bed run, instead of the exact flux of \(W_0\). Measured on the
   flat dam break of Castro's Test 2 (dx = 0.05), the exact flux would move the front
   speed from 1.711 to 1.732 (exact 2.000). That is too small a gain to justify
   changing every flat-bed wet/dry result.
4. *Case 3.3 can drain a dry cell.* The paper notes this and suggests a smaller CFL
   number. Here, if the ordinary fluctuation would take more than half of a dry cell's
   water in one step (`wetdry.drains_a_dry_cell`, any amount for an empty cell), the
   lower cell gets the wall instead.
5. *Films.* The paper's dry cell holds no water. Here it may hold up to
   \(h_{\text{dry}}\); that film enters through its own one-sided problem, superposed as
   described above.

**The moments.** The paper treats \(N = 0\). For \(N \geq 1\) the moments per unit depth
\(\alpha_i\) are carried unchanged into every interface state, and every depth above is
the shallow-water one, computed from the depth-averaged velocity: the moments' effect on
the wave speeds and on the energy is not included. These are modelling choices. Carrying
\(\alpha_i\) is what keeps a uniformly sheared layer at rest in the mean, which is a
steady solution of the equations, steady next to a wall. Setting the moments to zero at
the wall instead left \(u = 0.5\) in the cell at the wall in Castro's Test 1 at
\(N = 1\) (measured when the wall was first built, 2026-11-17). Nothing in the
construction is specific to an order \(N\).

**Dry cells keep no residue.** A cell that drains can be left with momentum, or
moments, and almost no water to carry them. Both problems are posed on the regularised
state (`wetdry.regularised_state`), and when the treatment is on, every cell at or
below \(h_{\text{dry}}\) is reset after each step to the state its desingularised
primitives describe. The depth is untouched; the momentum is scaled by
\(2h^2/(h^2 + h_{\text{dry}}^2)\), which is 1 at \(h_{\text{dry}}\) and 0 for an empty
cell; the moments are removed. Without this, the next ordinary interface reads the
residue as a discharge and drains the cell.

**What is guaranteed.**

- A lake at rest with dry shores is preserved exactly: water at rest below an emerged
  step meets a wall whose state is the cell itself, and the empty cell above poses no
  problem, so the fluctuations are exact zeros (the extended C-property, Section 5.4 of
  the paper). Measured over more than 200 steps, at \(N = 0, 1, 2\) with `Roe` and
  `Osher`, the depth changes by at most \(2.2 \times 10^{-16}\) and no mass is created by
  clamping.
- Mass is conserved to round-off: one flux per interface, in flux form.
- A dry cell only gains water through a treated interface, and the cell that loses
  water loses at most half of it there. With the solver's CFL number 0.5,
  \(\Delta t \leq \tfrac12 \Delta x / (|u| + c)\) for every cell. The overfall flux
  satisfies \(q_0 \leq \tfrac12 h(|u| + c)\) for \(-2c < u < c\) (equality at
  \(u = c\)). The climbing flux is \(hu < h(|u| + c)\). Wet cells are protected by the
  positivity limiter in any case.

Because it is written in terms of the scheme's own path integrals, the treatment works
with `Osher` as well as `Roe`. How accurate it is, and what it does not cover, is
measured in
[What to watch out for](limitations.md#shorelines-over-topography-what-is-and-is-not-covered).
It can be switched off with `wet_dry.shoreline_treatment: false`, which exists for
comparison only.

## Boundary conditions

| Setting | Behaviour |
|:--|:--|
| `PERIODIC` | The domain wraps. What leaves one end enters the other |
| `INFLOW_OUTFLOW` | Zero gradient extrapolation at both ends |

When topography is present, the bed is given ghost cells filled the same way as the
state. If the two disagreed, the interfaces at the domain edges would see an
inconsistent pair of state and bed, and the well balancing would fail exactly there.
