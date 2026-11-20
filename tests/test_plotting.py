"""`moment-sw --plot`: the interactive summary figure (`swme.plotting`)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from swme import mesh as mesh_module
from swme.pde import SWME1D
from swme.plotting import SWME1DPlotClassical


class _Finished:
    """The two attributes of a finished simulation the plot reads."""

    def __init__(self, order):
        self.order = order


@pytest.mark.parametrize('order', [0, 3, 6, 7, 10])
def test_the_summary_figure_has_one_panel_per_moment(order, monkeypatch):
    """The moment panels used to go into a fixed 3 x 3 grid, so N >= 7 asked
    matplotlib for panel 10 of 9 and the plot raised ValueError."""
    monkeypatch.setattr(plt, 'show', lambda: None)
    mesh = mesh_module.UniformRectangularMesh1D([0.0, 1.0], 8)
    data = np.column_stack([mesh.cell_center_positions, np.ones(8),
                            np.zeros((8, order + 1))])
    SWME1DPlotClassical(SWME1D('x', 0.0, 1.0, False, False), mesh,
                        _Finished(order)).plot(data)
    figure = plt.gcf()
    try:
        assert len(figure.axes) == 3 + order
        rows = figure.axes[0].get_subplotspec().get_gridspec().get_geometry()[0]
        assert rows == max(3, -(-(3 + order) // 3))
    finally:
        plt.close(figure)
