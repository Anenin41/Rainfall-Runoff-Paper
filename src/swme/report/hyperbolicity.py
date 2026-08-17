"""The hyperbolicity pages.

**Phase 2 status: the overview page exists and reports the scheme-level
counters honestly; the model-level panels and the space-time maps land in Phase
3.** The page is present from the start so that page ordering in the assembled
document is final, and so a reader is never left wondering whether hyperbolicity
was considered.

The care taken here is not decoration. The always-on counters measure the
**path-averaged interface matrix**, `sum_k w_k A(psi(s_k))`, not `A(U)` at any
state. `A` is nonlinear in `U`, so that average is not `A(anything)` and need
not be hyperbolic even when every matrix being averaged is. A dam break onto a
dry bed trips it at 78 of 12 462 interfaces at **N = 0** - plain shallow water,
unconditionally hyperbolic, with no moments to destabilise - and the count is
identical at N=1, N=2 and HSWME, which is what gives it away.

So a single "hyperbolicity" number on a report would recreate exactly the
false-positive failure mode that Step 5.5's `ComplexWarning` and Step 6's
interface counter each had, and that Step 8.5's own defect D4 had again. Hence
two clearly separated tiers, and text rather than a bare number.
"""

from __future__ import annotations

from .data import RunData
from .style import ReportStyle

INTERFACE_CAVEAT = (
    "These counters eigendecompose the path-averaged interface matrix, not "
    "A(U) at any state. That average is not A(anything), and it leaves the "
    "real axis at a strong enough jump even for models that are hyperbolic at "
    "every point - a wet-dry front does so at N = 0, i.e. for plain shallow "
    "water. A non-zero count here is therefore not evidence that the model "
    "lost hyperbolicity."
)

ENABLE_HINT = (
    "Set postprocessing.store_hyperbolicity: true (and hyperbolicity_stride) "
    "in the config and re-run to capture the cell-by-cell spectrum of A(U)."
)


def scheme_counter_lines(run: RunData) -> list[str]:
    """The tier-1 text panel: counters, with what they do and do not mean.

    Never a bare number, and never "0 of 0" rendered as reassurance.
    """
    counters = run.scheme_counters
    if counters is None or counters.scheme is None:
        return [
            "SCHEME-LEVEL COUNTERS",
            "",
            "  Scheme not recorded, so the interface-matrix counters cannot be",
            "  interpreted and are not shown. Roe records one path-averaged",
            "  matrix per interface, Osher five weight-scaled single-node",
            "  matrices, and LF and PRICE never eigendecompose at all - the",
            "  same integer means a different thing in each case.",
        ]

    lines = ["SCHEME-LEVEL COUNTERS", "", f"  scheme                {counters.scheme}"]

    if not counters.eigendecomposes:
        lines += [
            "",
            f"  {counters.scheme} applies a scalar or polynomial viscosity and never",
            "  eigendecomposes the interface matrix. It records no spectra at",
            "  all, so its counters are structurally zero and carry no",
            "  information about this run. This is an absence of measurement,",
            "  not a clean result.",
        ]
        return lines

    lines += [
        f"  spectra recorded      {counters.spectra_examined}",
        f"  left the real axis    {counters.nonhyperbolic_count}",
        f"  largest |Im(lambda)|  {counters.max_abs_imaginary_eigenvalue:.3e}",
        f"  tolerance             {counters.tolerance:.1e}",
    ]
    if counters.spectra_examined == 0:
        lines += ["", "  No spectra were recorded, so nothing was measured."]
        return lines

    if counters.scheme == "Osher":
        lines += [
            "",
            "  Osher records five spectra per interface, one per quadrature",
            "  node, and each is a weight-scaled single-node matrix rather",
            "  than the path average. Its counts and magnitudes are therefore",
            "  not comparable with Roe's: the weights (0.118 to 0.284) scale",
            "  the imaginary parts down, making the fixed tolerance up to 8.4x",
            "  stricter in the units of A.",
        ]
    lines += ["", *_wrap(INTERFACE_CAVEAT, 68)]
    return lines


def _wrap(text: str, width: int) -> list[str]:
    import textwrap
    return ["  " + line for line in textwrap.wrap(text, width)]


def page_hyperbolicity(run: RunData, style: ReportStyle):
    """Tier 1 (scheme counters) and tier 2 (model spectrum)."""
    figure, axes = style.new_page(2, 1, height_ratios=[1.0, 1.0])
    style.page_header(figure, "Hyperbolicity",
                      "scheme-level counters and the model-level spectrum")

    style.text_block(axes[0, 0], scheme_counter_lines(run))

    hyper = run.hyperbolicity
    if hyper is None or not hyper.has_cells:
        style.placeholder(
            axes[1, 0], "Model-level spectrum not captured", ENABLE_HINT)
    else:
        style.placeholder(
            axes[1, 0], "Model-level spectrum: rendering lands in Phase 3",
            f"{len(hyper.cells)} per-cell records were loaded from this run and "
            "will be drawn once the hyperbolicity panels are implemented.")
    return figure
