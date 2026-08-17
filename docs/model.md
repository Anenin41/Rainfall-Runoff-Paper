# The model

## From shallow water to moments

The shallow water equations describe a thin layer of water by its depth \(h(x,t)\) and
a single velocity \(u(x,t)\). The velocity is the same at every depth, which is the
approximation that makes them cheap and also the one that limits them.

Moment models relax it. The velocity is allowed to vary through the depth, and that
variation is written as a finite series:

\[
u(x, z, t) = u_m(x, t) + \sum_{i=1}^{N} \alpha_i(x, t)\, \phi_i(z),
\qquad z \in [0, 1]
\]

The scaled coordinate \(z\) is 0 at the bed and 1 at the free surface. The first term
\(u_m\) is the depth-averaged velocity. Each \(\alpha_i\) says how much of the shape
\(\phi_i\) the profile contains.

### The basis

The \(\phi_i\) are shifted Legendre polynomials,

\[
\phi_i(z) = P_i(1 - 2z)
\]

where \(P_i\) is the ordinary Legendre polynomial of degree \(i\). The first few are

\[
\phi_0 = 1, \qquad
\phi_1(z) = 1 - 2z, \qquad
\phi_2(z) = 1 - 6z + 6z^2
\]

Two properties make this basis a good choice.

**They are orthogonal**, with

\[
\int_0^1 \phi_i(z)\, \phi_j(z)\, \mathrm{d}z = \frac{\delta_{ij}}{2i + 1}
\]

so each coefficient can be recovered from the profile independently of the others, and
truncating the series is a projection rather than a guess.

**\(\phi_i\) has zero mean for \(i \geq 1\)**, so \(u_m\) really is the depth average
whatever the other coefficients do. Adding a moment refines the shape of the profile
without changing the mean flow.

Because the basis is fixed and known, the coefficients that appear in the equations
below can be computed once for a given \(N\) and reused. That is what
`swme.coefficients` does, for any \(N\), with no upper limit built in.

## The equations

The conserved state is

\[
U = \bigl(h,\; h u_m,\; h\alpha_1,\; \ldots,\; h\alpha_N\bigr)^{\mathsf T}
\]

and the system has the form

\[
\partial_t U + \partial_x F(U) + Q(U)\, \partial_x U = S(U)
\]

The unusual part is \(Q(U)\,\partial_x U\), a term that cannot be written as the
derivative of a flux. Systems like this are called **non-conservative**, and they need
a numerical method built for them. See [Numerical method](numerics.md).

In practice the code works with the combined system matrix

\[
A(U) = \frac{\partial F}{\partial U} + Q(U)
\]

which for moment order \(N\) is an \((N+2) \times (N+2)\) matrix. Its entries are

\[
\begin{aligned}
A_{0,1} &= 1 \\[2pt]
A_{1,0} &= g h - u_m^2 - \sum_{i=1}^{N} \frac{\alpha_i^2}{2i + 1} \\[2pt]
A_{1,1} &= 2 u_m, \qquad
A_{1,i+1} = \frac{2\alpha_i}{2i + 1} \\[2pt]
A_{i+1,0} &= -2 u_m \alpha_i - \sum_{j,k} A_{ijk}\, \alpha_j \alpha_k \\[2pt]
A_{i+1,1} &= 2\alpha_i \\[2pt]
A_{i+1,l+1} &= u_m\, \delta_{il} + \sum_{k} \bigl(2A_{ilk} + B_{ilk}\bigr) \alpha_k
\end{aligned}
\]

The tensors \(A_{ijk}\) and \(B_{ijk}\) are integrals of products of basis functions.
They depend only on the basis, not on the flow, so they are computed once per moment
order and cached. \(A_{ijk}\) has a closed form through Wigner 3j symbols;
\(B_{ijk}\) comes from a table of antiderivatives.

At \(N = 0\) the matrix reduces to the classical shallow water system, as it should.

## Friction

Friction enters through a Navier-slip condition at the bed, which relates the bed shear
to the bed velocity through a slip length \(\lambda\). Writing \(u_s\) for the surface
velocity and \(u_b\) for the bed velocity,

\[
u_s = u_m + \sum_{i=1}^{N} \alpha_i\, \phi_i(1),
\qquad
u_b = u_m + \sum_{i=1}^{N} \alpha_i\, \phi_i(0)
\]

the friction term is

\[
\begin{aligned}
P_0 &= 0 \\
P_1 &= \frac{\nu}{\lambda}\, u_b \\
P_{i+2} &= (2i+1)\left[
  \frac{\nu}{\lambda}\, \phi_i(0)\, u_b
  + \frac{\nu}{h^2} \sum_j C_{ij}\, h\alpha_j
\right]
\end{aligned}
\]

with \(\nu\) the viscosity. The source term in the equations is \(-P(U)\).

Note the \(\nu / h^2\). As the water gets thin this grows quickly, and it is the reason
viscous flow near a drying front is numerically difficult. See
[What to watch out for](limitations.md#viscous-drying-needs-the-implicit-source-path).

## The hyperbolic variant

A system of this kind is **hyperbolic** when the matrix \(A(U)\) has real eigenvalues
and a full set of eigenvectors. This is not a technicality. The eigenvalues are the
speeds at which information travels. If some of them are complex, the initial value
problem is ill posed: small perturbations grow without bound at every wavelength, and
a finer grid makes the situation worse rather than better.

SWME loses hyperbolicity for \(N \geq 2\). Measured over uniformly sampled states with
\(h \in [0.2, 4]\) and \(|\alpha| \leq 3\):

| \(N\) | SWME states that are not hyperbolic | HSWME |
|--:|--:|--:|
| 1 | 0.00 % | 0.00 % |
| 2 | 3.07 % | 0.00 % |
| 3 | 11.24 % | 0.00 % |
| 4 | 21.06 % | 0.00 % |
| 5 | 34.59 % | 0.00 % |
| 6 | 48.03 % | 0.00 % |

**HSWME** fixes this. It sets \(\alpha_2, \ldots, \alpha_N\) to zero inside the
transport matrix only, leaving them untouched everywhere else. The eigenvalues then no
longer depend on those moments at all, while the outer wave speeds keep their correct
form

\[
\lambda_{\pm} = u_m \pm \sqrt{g h + \alpha_1^2}
\]

Across every state tested at \(N = 1\) through \(6\), HSWME never lost hyperbolicity,
and it repaired every sampled state that broke SWME.

Select it with `type: HSWME1D`, or equivalently `hyperbolic: true`.

!!! note "N = 0 and N = 1 are always safe"
    At those orders SWME and HSWME are the same model, and both are unconditionally
    hyperbolic. The \(N = 1\) spectrum is exactly
    \(\{u_m,\; u_m \pm \sqrt{g h + \alpha_1^2}\}\). No \(N \leq 1\) run can lose
    hyperbolicity, whatever the forcing does.

There is a surprise in *where* SWME fails, described in
[What to watch out for](limitations.md#hyperbolicity-depends-on-the-ratio-not-the-size).
Keeping the moments small is not the right safety rule.

## The rainfall-runoff extension

`RechargeSWME1D` adds two effects to the surface.

**Rainfall** at rate \(R\) adds water from above. **Infiltration** at rate \(I\)
removes water into the ground. Both change the mass balance, and because the water
entering and leaving does so at a particular height, both change the momentum and the
moments too:

\[
\begin{aligned}
S_0 &= R - I \\
S_1 &= R\, u_s - I\, u_b \\
S_{i+2} &= (2i+1)\, R \left[\phi_i(1)\, u_s - u_m r_i - \sum_j E_{ij} \alpha_j\right]
       + (2i+1)\, I \left[-\phi_i(0)\, u_b + u_m s_i + \sum_j F_{ij} \alpha_j\right]
\end{aligned}
\]

Rain arrives at the surface, so it carries the surface velocity \(u_s\). Infiltrating
water leaves through the bed, so it carries the bed velocity \(u_b\).

### Infiltration models

Two are available.

**Horton infiltration** models soil that saturates over time. The rate starts at
\(f_0\), decays towards \(f_c\), and the decay constant is \(k\):

\[
I(t) = f_c + (f_0 - f_c)\, e^{-k t}
\]

**Constant infiltration** uses a fixed rate. It can optionally be limited by the
rainfall rate or by the water actually available, so it cannot remove water that is
not there.

### Mixing friction

Water joining or leaving the layer has to be brought to the layer's velocity, and that
exchange of momentum acts like an extra friction. The `admissible` mixing model
represents it with two coefficients, `alpha_R` for rainfall and `alpha_I` for
infiltration.

## Choosing the moment order

There is no hard upper limit on \(N\) in the code, and the coefficient engine is
generic. Two practical bounds are worth knowing.

Above roughly \(N = 10\) to \(12\), the moment closure itself becomes physically
questionable, independent of anything numerical. That is a modelling limit rather than
a software one.

Cost grows with \(N\), because the system matrix is \((N+2) \times (N+2)\) and is built
at every interface, at every quadrature point, at every timestep.

For most work, \(N = 1\) or \(N = 2\) is the useful range: enough to represent a
velocity profile that classical shallow water cannot, without the expense or the
hyperbolicity trouble of going higher.
