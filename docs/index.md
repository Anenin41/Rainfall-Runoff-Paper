# Shallow Water Moment Equations

A 1D finite-volume solver for the Shallow Water Moment Equations (SWME), their
hyperbolic variant (HSWME), and an extension that adds rainfall and infiltration.

## The problem it solves

Classical shallow water theory assumes the horizontal velocity is the same at every
depth. Water is treated as a sliding block: one depth \(h\), one velocity \(u\).
That assumption is often wrong. Real flows are slower near the bed because of
friction and faster near the surface, and a model that cannot see the difference
cannot predict things that depend on it, such as how a wave front steepens or how
sediment moves.

Solving the full 3D equations instead is expensive. Shallow water moment equations
sit between the two. They keep the cheap 1D grid, but let the velocity vary with
depth by writing it as a small series:

\[
u(x, z, t) = u_m(x, t) + \sum_{i=1}^{N} \alpha_i(x, t)\, \phi_i(z)
\]

Here \(z\) runs from 0 at the bed to 1 at the surface, \(u_m\) is the depth-averaged
velocity, and the \(\alpha_i\) are coefficients that describe the shape of the
profile. The basis functions \(\phi_i\) are fixed and known. Setting \(N = 0\)
recovers the classical shallow water equations exactly.

The number \(N\) is yours to choose. Larger \(N\) means a richer velocity profile and
more equations to solve.

## What is in this package

| Package | Contents |
|:--|:--|
| `swme` | The base model, the numerical scheme, topography, wet-dry handling, the command line interface, and the report generator |
| `recharge` | The rainfall-runoff extension: rain falling on the surface, water infiltrating into the ground, and the friction that mixing introduces |

Three models can be selected from a config file:

- **SWME1D**, the moment model as written above.
- **HSWME1D**, the same model with a modification that guarantees the equations stay
  well posed. See [The model](model.md#the-hyperbolic-variant).
- **RechargeSWME1D**, SWME1D plus rainfall and infiltration.

## Where to go next

- [Quick start](quickstart.md) installs the package, runs a case, and points at the
  three demonstration runs that exercise most of the solver in about two minutes.
- [The model](model.md) sets out the equations and where each term comes from.
- [Numerical method](numerics.md) covers the finite-volume scheme, bed topography,
  and how dry ground is handled.
- [Configuration](configuration.md) lists every key a config file accepts.
- [Run reports](reports.md) explains the PDF report generator.
- [What to watch out for](limitations.md) collects measured limits of the solver,
  including the two combinations of settings that do not work. Read this before
  trusting a number.
- [API reference](api.md) is generated from the source docstrings.

## A note on this documentation

The statements here about accuracy and behaviour were measured, not assumed. Where a
number appears (a front speed, an error, a count of cells), it came from running the
code. Where something is a known weakness it is written down as one, in
[What to watch out for](limitations.md), rather than left out.
