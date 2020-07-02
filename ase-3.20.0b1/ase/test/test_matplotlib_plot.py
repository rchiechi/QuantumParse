import os
import numpy as np
import pytest

from ase.lattice.cubic import FaceCenteredCubic
from ase.utils.plotting import SimplePlottingAxes
from ase.visualize.plot import plot_atoms


def test_matplotlib_plot(plt):
    slab = FaceCenteredCubic('Au', size=(2, 2, 2))

    fig, ax = plt.subplots()
    plot_atoms(slab, ax, radii=0.5, rotation=('10x,10y,10z'),
               show_unit_cell=0)

    assert len(ax.patches) == len(slab)
    print(ax)


class TestPlotManager:
    filename = 'plot.png'

    @classmethod
    def teardown_class(cls):
        if os.path.isfile(cls.filename):
            os.remove(cls.filename)

    @pytest.fixture
    def xy_data(self):
        return ([1, 2], [3, 4])

    def test_plot_manager_error(self, plt):
        # Boot up a figure to help the oldlibs tests manage without graphics
        fig = plt.figure()
        try:
            with pytest.raises(AssertionError):
                with SimplePlottingAxes(ax=None, show=False,
                                        filename=None) as _:
                    raise AssertionError()
        finally:
            plt.close(fig=fig)

    def test_plot_manager_no_file(self, plt, xy_data):
        x, y = xy_data

        # Boot up a figure to help the oldlibs tests manage without graphics
        fig = plt.figure()

        try:
            with SimplePlottingAxes(ax=None, show=False, filename=None) as ax:
                ax.plot(x, y)

            assert np.allclose(ax.lines[0].get_xydata().transpose(), xy_data)
            assert not os.path.isfile(self.filename)
        finally:
            plt.close(fig=fig)

    def test_plot_manager_axis_file(self, figure, xy_data):
        x, y = xy_data
        ax = figure.add_subplot()
        with SimplePlottingAxes(ax=ax, show=False,
                                filename=self.filename) as return_ax:
            assert return_ax is ax
            ax.plot(x, y)

        assert os.path.isfile(self.filename)
