import numpy as np


from Signal_source.Measurement_schemes import UniformMeasurement, LogMeasurement, RandomizedUniformMeasurement

from PySide6.QtWidgets import QWidget, QVBoxLayout
from analysis_tools.effective_rank import get_effective_rank
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class MatrixCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100, plots = (1,2)):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.subplots(plots[0], plots[1])  # TWO subplot axes
        super().__init__(self.fig)

class MatrixWindow(QWidget):
    def __init__(self, scheme_start, scheme_end, num_points, freq, measure_scheme, *args, **kwargs):
        super().__init__(*args, **kwargs)

        layout = QVBoxLayout(self)

        if measure_scheme == UniformMeasurement:
            measure_scheme = UniformMeasurement(start_position=scheme_start,
                                                end_position=scheme_end,
                                                num_points=num_points)

        self.plot_matrix = MatrixCanvas(parent=self, width=5, height=4, dpi=100, plots=(2, 1))
        self.effective_rank_plot = MatrixCanvas(parent=self, width=5, height=4, dpi=100, plots=(1, 2))


        effective_rank, matrix = get_effective_rank(freq=freq, measurement_scheme=measure_scheme)

        def P(A): 
            return np.abs(A)
        
        # Unpack axes
        ax_m1, ax_m2 = self.plot_matrix.axes
        ax_e1, ax_e2 = self.effective_rank_plot.axes

        # First matrix windows
        ax_m1.imshow(P(matrix))
        ax_m2.imshow(P(effective_rank.R))

        # Effective rank matrices
        ax_e1.imshow(P(effective_rank.X))
        ax_e2.imshow(P(effective_rank.Y))

        # Turn off ticks
        for ax in [ax_m1, ax_m2, ax_e1, ax_e2]:
            ax.set_xticks([])
            ax.set_yticks([])

        toolbar = NavigationToolbar(self.plot_matrix, self)
        layout.addWidget(toolbar)        
        layout.addWidget(self.plot_matrix)
        
        toolbar = NavigationToolbar(self.effective_rank_plot, self)
        layout.addWidget(toolbar)
        layout.addWidget(self.effective_rank_plot)
