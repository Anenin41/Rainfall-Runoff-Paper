"""Suite-wide test configuration.

The repo's first `conftest.py`, added with the report subpackage (Step 8.5).

Its one job is to force a headless matplotlib backend before anything imports
pyplot. `swme.report` itself never touches pyplot - it builds bare `Figure`
objects, so there is no global figure registry and no backend involved - but
`swme.plotting` does, and a test that reaches it on a machine with no display
would otherwise fail for reasons that have nothing to do with the code.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
