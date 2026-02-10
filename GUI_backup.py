import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QLineEdit,
    QHBoxLayout, QPushButton, QComboBox, QTextEdit
)

from PySide6.QtGui import (
    QDoubleValidator, QIntValidator, QFont, QRegularExpressionValidator,
    QFont, QColor, QTextCursor, QPalette, QKeyEvent, 
)

from PySide6.QtCore import Qt, QRegularExpression, QObject, Signal
from SSH_terminal import SSHTerminal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sys
from Signal_source.Measurement_schemes import UniformMeasurement, LogMeasurement, RandomizedUniformMeasurement
import re
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from analysis_tools.effective_rank import get_effective_rank

def clear_layout(layout):
    """Recursively delete all widgets and child layouts from a layout."""
    if layout is None:
        return

    while layout.count():
        item = layout.takeAt(0)

        # Case 1: The item is a widget
        widget = item.widget()
        if widget is not None:
            widget.setParent(None)
            continue

        # Case 2: The item is a nested layout
        child_layout = item.layout()
        if child_layout is not None:
            clear_layout(child_layout)  # recursive call
            # After clearing, delete the child layout itself
            child_layout.setParent(None)

def get_pairs(self):
    pattern = r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)"
    matches = re.findall(pattern, self.text())
    return [(int(x), int(y)) for x, y in matches]



class MatrixInput(QLineEdit):
    def __init__(self):
        super().__init__()
        self.setPlaceholderText("Enter pairs like (1, 2), (3, 4)...")
        
        # Regular expression to allow tuples like (1, 2), (3, 4)
        regex = QRegularExpression(r"^\s*(\(\s*-?\d+\s*,\s*-?\d+\s*\)\s*,?\s*)*$")
        validator = QRegularExpressionValidator(regex)
        self.setValidator(validator)

        self.textChanged.connect(self.auto_add_brackets)

    def auto_add_brackets(self, text):
        text = text.strip()
        # If the last completed tuple is finished (ends with ')'), append ", ("
        if text and text[-1] == ')':
            # Avoid duplicating commas/brackets if already there
            if not text.endswith("), ("):
                self.blockSignals(True)  # Prevent recursion
                self.setText(text + ", (")
                self.setCursorPosition(len(self.text()))
                self.blockSignals(False)

class MplCanvas(FigureCanvas):
    """A QWidget-compatible Matplotlib canvas."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PySide6 + Matplotlib Example")

        # Create the Matplotlib canvas
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)

        # Measurement Scheme input
        Double_validator = QDoubleValidator(-100, 100.0, 2)
        Double_validator.setNotation(QDoubleValidator.StandardNotation)

        Int_Validator = QIntValidator(0, 9999999)

        float_validator = QDoubleValidator(0, 100.0, 2)
        float_validator.setNotation(QDoubleValidator.StandardNotation)

        freq_full_layout = QHBoxLayout()

        big_freq_label = QLabel('Frequency Range', alignment=Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(20)
        font.setBold(True) 
        big_freq_label.setFont(font)

        self.start_freq_input = QLineEdit(parent=self)
        self.start_freq_input.setValidator(Double_validator)
        self.start_freq_text =  QLabel('Start Position')

        start_freq_layout = QVBoxLayout()
        start_freq_layout.addWidget(self.start_freq_text)
        start_freq_layout.addWidget(self.start_freq_input)

        self.end_freq_input = QLineEdit(parent=self)
        self.end_freq_input.setValidator(Double_validator)
        self.end_freq_text =  QLabel('End Position')

        end_freq_layout = QVBoxLayout()
        end_freq_layout.addWidget(self.end_freq_text)
        end_freq_layout.addWidget(self.end_freq_input)

        self.num_freqs_input = QLineEdit(parent=self)
        self.num_freqs_input.setValidator(Int_Validator)
        self.num_freqs_text =  QLabel('Num Points')

        num_freqs_layout = QVBoxLayout()
        num_freqs_layout.addWidget(self.num_freqs_text)
        num_freqs_layout.addWidget(self.num_freqs_input)

        freq_full_layout.addLayout(start_freq_layout)
        freq_full_layout.addLayout(end_freq_layout)
        freq_full_layout.addLayout(num_freqs_layout)

        big_measure_scheme_label = QLabel('Measure Scheme', alignment=Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(20)
        font.setBold(True) 
        big_measure_scheme_label.setFont(font)

        self.start_position_input = QLineEdit(parent=self)
        self.start_position_input.setValidator(Double_validator)
        self.start_position_text =  QLabel('Start Position')

        self.end_position_input = QLineEdit(parent=self)
        self.end_position_input.setValidator(Double_validator)
        self.end_position_text =  QLabel('End Position')

        self.num_points_input = QLineEdit(parent=self)
        self.num_points_input.setValidator(Int_Validator)
        self.num_points_text =  QLabel('Num Points')

        self.measure_scheme_button = QPushButton('Submit Measure_scheme')
        self.measure_scheme_button.clicked.connect(self.submit_scheme)


        self.scheme_selector = QComboBox()
        self.scheme_selector.addItems(['UniformMeasurement', 'RandomizedUniformMeasurement', 'LogMeasurement'])
        self.selector_references = [UniformMeasurement, RandomizedUniformMeasurement, LogMeasurement]

        plots_layout = QVBoxLayout()
        self.fig1, self.ax1 = plt.subplots(ncols=1, nrows=2, figsize=(3,20))
        self.canvas1 = FigureCanvas(self.fig1)
        plots_layout.addWidget(self.canvas1, stretch=2.5)


        self.fig2, self.ax2 = plt.subplots(ncols=2, nrows=1, figsize=(40,20))
        self.canvas2 = FigureCanvas(self.fig2)
        plots_layout.addWidget(self.canvas2, stretch=2.5)

        plots_layout.addWidget(SSHTerminal(
            hostname='maxwell',
            username='miloszwegorski',
            password='Wegorszczak123'
        ), 3)

        #Measure Scheme Display
        self.layout_window = QHBoxLayout()
        
        self.start_pos = QVBoxLayout()
        self.start_pos.addWidget(self.start_position_text, alignment = Qt.AlignmentFlag.AlignBottom)
        self.start_pos.addWidget(self.start_position_input)

        self.end_pos = QVBoxLayout()
        self.end_pos.addWidget(self.end_position_text, alignment = Qt.AlignmentFlag.AlignBottom)
        self.end_pos.addWidget(self.end_position_input)
        
        num_points = QVBoxLayout()
        num_points.addWidget(self.num_points_text, alignment = Qt.AlignmentFlag.AlignBottom)
        num_points.addWidget(self.num_points_input)


        button = QVBoxLayout()
        button.addWidget(self.measure_scheme_button)

        # Simulation Settings
        self.Simulation_settings_layout = QHBoxLayout()

        big_Simulation_label = QLabel('Simulation Settings', alignment=Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(20)
        font.setBold(True) 
        big_Simulation_label.setFont(font)


        # Num Simulations
        self.num_sims_layout = QVBoxLayout()
        self.num_sims_layout.addWidget(QLabel('Num Simulations', alignment = Qt.AlignmentFlag.AlignBottom))
        self.num_sims_input = QLineEdit(parent=self)
        self.num_sims_input.setValidator(Int_Validator)
        self.num_sims_layout.addWidget(self.num_sims_input)

        #tolerance
        self.tolerance_layout = QVBoxLayout()
        self.tolerance_layout.addWidget(QLabel('Tolerance               ', alignment = Qt.AlignmentFlag.AlignBottom))
        self.tolerance_input = QLineEdit(parent=self)
        self.tolerance_input.setValidator(float_validator)
        self.tolerance_layout.addWidget(self.tolerance_input)

        self.name_layout = QVBoxLayout()
        self.name_layout.addWidget(QLabel('Simulation Name', alignment = Qt.AlignmentFlag.AlignBottom))
        self.name_input = QLineEdit(parent=self)
        self.name_layout.addWidget(self.name_input)


        big_Simulation_label = QLabel('Simulation Settings', alignment=Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(20)
        font.setBold(True) 
        big_Simulation_label.setFont(font)

        
        # Signal Settings
        Big_signal_label = QLabel('Signal Parameters', alignment=Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(20)
        font.setBold(True) 
        Big_signal_label.setFont(font)

        self.Signal_Parameters = QVBoxLayout()

        self.signal_select = QComboBox()

        self.signal_select.addItems(['', 'Simple Signal', 'Full Signal model'])

        self.signal_select.currentTextChanged.connect(self.on_select_signal)

        self.Signal_Parameters.addWidget(self.signal_select)

        self.Signal_input = QHBoxLayout()

        self.Signal_Parameters.addLayout(self.Signal_input)
    

        self.Left = QVBoxLayout()

        self.input_measure_scheme = QHBoxLayout()



        #measure scheme selector
        self.Simulation_settings_layout.addLayout(self.name_layout)
        self.Simulation_settings_layout.addLayout(self.num_sims_layout)
        self.Simulation_settings_layout.addLayout(self.tolerance_layout)

        self.input_measure_scheme.addLayout(self.start_pos)
        self.input_measure_scheme.addLayout(self.end_pos)
        self.input_measure_scheme.addLayout(num_points)

        #measure scheme buttons and input
        self.Left.addWidget(big_freq_label)
        self.Left.addLayout(freq_full_layout)

        self.Left.addWidget(big_measure_scheme_label)
        self.Left.addWidget(self.scheme_selector, alignment = Qt.AlignmentFlag.AlignBottom)
        self.Left.addLayout(self.input_measure_scheme)

        self.Left.addLayout(button)
        
        self.Left.addWidget(big_Simulation_label)
        
        self.Left.addLayout(self.Simulation_settings_layout)

        self.Left.addWidget(Big_signal_label)

        self.Left.addLayout(self.Signal_Parameters)

        self.layout_window.addLayout(self.Left, 3)
        self.layout_window.addLayout(plots_layout, 7)

        self.Left.addLayout(self.num_sims_layout)
        self.Left.addLayout(self.tolerance_layout)

        self.submit_button = QPushButton('Create .ini file')
        self.submit_button.clicked.connect(self.submit_ini)

        self.Left.addWidget(self.submit_button)

        self.Left.addStretch()

        # Embed canvas into layout
        
        layout = QVBoxLayout()
        layout.addLayout(self.layout_window)

        # Central widget
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
    
    def submit_ini(self):
        print('submit ini')
    def on_select_signal(self):
        if self.signal_select.currentText() == '':
            
            clear_layout(self.Signal_input)
        elif self.signal_select.currentText() == 'Simple Signal':
            clear_layout(self.Signal_input)

            # Amplitude
            self.Amp_Atten_layout = QVBoxLayout()
            self.Amplitude_Label = QLabel('Amplitude')
            self.Amplitude_input = QLineEdit()

            self.Amp_Atten_layout.addWidget(self.Amplitude_Label)
            self.Amp_Atten_layout.addWidget(self.Amplitude_input)

            #Attenuation
            self.Attenuation_Label = QLabel('Attenuation')
            self.Attenuation_input = QLineEdit()

            self.Amp_Atten_layout.addWidget(self.Attenuation_Label)
            self.Amp_Atten_layout.addWidget(self.Attenuation_input)

            #gamma
            self.gamma_source_dist_layout = QVBoxLayout()
            self.gamma_Label = QLabel('Gamma')
            self.gamma_input = QLineEdit()

            self.gamma_source_dist_layout.addWidget(self.gamma_Label)
            self.gamma_source_dist_layout.addWidget(self.gamma_input)

            #Source_dist
            self.Source_dist_Label = QLabel('Source dist')
            self.Source_dist_input = QLineEdit()

            self.gamma_source_dist_layout.addWidget(self.Source_dist_Label)
            self.gamma_source_dist_layout.addWidget(self.Source_dist_input)

            self.Simple_param_input = QHBoxLayout()
            self.Simple_param_input.addLayout(self.Amp_Atten_layout)
            self.Simple_param_input.addLayout(self.gamma_source_dist_layout)

            self.Simple_full_param_input = QVBoxLayout()

            self.param_matrix_label = QLabel('Parameter Matrix')
            self.param_matrix_input = MatrixInput()

            self.Simple_full_param_input.addLayout(self.Simple_param_input)
            self.Simple_full_param_input.addWidget(self.param_matrix_label)
            self.Simple_full_param_input.addWidget(self.param_matrix_input)

            self.Signal_input.addLayout(self.Simple_full_param_input)

            self.Signal_input.addStretch()


        elif self.signal_select.currentText() == 'Full Signal model':
            clear_layout(self.Signal_input)

            self.Simple_full_param_input = QVBoxLayout()

            self.param_matrix_label = QLabel('Parameter Matrix')
            self.param_matrix_input = MatrixInput()

            self.param_amps_label = QLabel('Parameter amplitudes')
            self.param_amps_input = MatrixInput()

            self.Simple_full_param_input.addWidget(self.param_matrix_label)
            self.Simple_full_param_input.addWidget(self.param_matrix_input)

            self.Simple_full_param_input.addWidget(self.param_amps_label)
            self.Simple_full_param_input.addWidget(self.param_amps_input)

            self.Signal_input.addLayout(self.Simple_full_param_input)

            self.Signal_input.addStretch()


    def submit_scheme(self):
        self.start_pos = float(self.start_position_input.text())
        self.end_pos = float(self.end_position_input.text())
        num_points = int(self.num_points_input.text())

        scheme = self.selector_references[self.scheme_selector.currentIndex()](
            start_position = self.start_pos,
            end_position = self.end_pos,
            num_points = num_points
        )

        effective_rank, matrix = get_effective_rank(measurement_scheme=scheme)

        def P(A): return np.abs(A)
        self.ax1[0].imshow(P(matrix))
        self.ax1[1].imshow(P(effective_rank.R))
        self.ax2[0].imshow(P(effective_rank.X))
        self.ax2[1].imshow(P(effective_rank.Y))
        for axs in self.ax1:
            axs.set_xticks([])
            axs.set_yticks([])
        for axs in self.ax2:
            axs.set_xticks([])
            axs.set_yticks([])
        self.canvas1.draw()
        self.canvas2.draw()




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())