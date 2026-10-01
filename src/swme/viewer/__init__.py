"""Interactive browser for run output (`moment-sw-view`).

    uv run moment-sw-view results/

A local Dash app over the same data layer the PDF report uses
(`swme.report.data.RunData`), so the two can never disagree about what a run
contains. The report is the archival, page-stable record; the viewer is for
exploring - a time cursor shared by every view, hover values, zoom, a chosen
`x` for the vertical profile, and runs overlaid against each other.

Nothing here imports dash or plotly at package import: `catalog` and `store`
are pure pandas/numpy, and the rendering modules are only imported once the
CLI starts the app. Both are installed with the package, so this is about
start-up time, not availability - `moment-sw` itself never pays for importing
a web framework it does not use.
"""
