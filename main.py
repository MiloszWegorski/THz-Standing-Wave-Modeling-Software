# File: main.py
import sys
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import QFile, QIODevice

from Signal_source.Measurement_schemes import UniformMeasurement, LogMeasurement, RandomizedUniformMeasurement

from analysis_tools.effective_rank import get_effective_rank
from matrix_window import MatrixWindow

from Measurement_tools.Libre_measurements import libre_vna_measurement


class MainWindow(QWidget):

    def __init__(self):
        super().__init__()

        # Load the .ui file
        ui_file_name = "Main_Input.ui"
        ui_file = QFile(ui_file_name)

        if not ui_file.open(QIODevice.ReadOnly):
            raise RuntimeError(f"Cannot open {ui_file_name}: {ui_file.errorString()}")

        loader = QUiLoader()
        self.ui = loader.load(ui_file)
        ui_file.close()

        if not self.ui:
            raise RuntimeError(loader.errorString())

        # Keep reference for matrix window so it doesn't get GC'd
        self.matrix_window = None

        # Connect button
        self.ui.RankMatrix.clicked.connect(self.create_matrix)
        self.ui.Libre_measure.clicked.connect(self.measure_libre_vna)

        # Show loaded UI
        self.ui.show()

    def create_matrix(self):
        # Validate fields -----------------------------------
        try:
            scheme_start = float(self.ui.Start_pos.text())
        except ValueError:
            print("Start pos not a float value")
            return

        try:
            scheme_end = float(self.ui.End_pos.text())
        except ValueError:
            print("End pos not a float")
            return

        try:
            num_points = int(self.ui.Num_points.text())
        except ValueError:
            print("Num points not an integer")
            return

        try:
            freq = float(self.ui.Freq_mat.text())
        except ValueError:
            print("Freq for matrix is not a float")
            return
        
        schemes = [UniformMeasurement, LogMeasurement, RandomizedUniformMeasurement]

        scheme_index = self.ui.Scheme_select.currentIndex()

        print(scheme_index)

        if scheme_index == -1:
            print('Please Select Measure_scheme')
            return
        
        measure_scheme = schemes[scheme_index]

        # Create matrix window -----------------------------
        if self.matrix_window is None:
            self.matrix_window = MatrixWindow(scheme_start=scheme_start,
                                              scheme_end=scheme_end,
                                              num_points=num_points,
                                              measure_scheme = measure_scheme,
                                              freq=freq)

        self.matrix_window.show()
        self.matrix_window.raise_()
        self.matrix_window.activateWindow()

    def measure_libre_vna(self):
        try:
            scheme_start = float(self.ui.Start_pos.text())
        except ValueError:
            print("Start pos not a float value")
            return

        try:
            scheme_end = float(self.ui.End_pos.text())
        except ValueError:
            print("End pos not a float")
            return

        try:
            num_points = int(self.ui.Num_points.text())
        except ValueError:
            print("Num points not an integer")
            return

        try:
            start_freq = float(self.ui.Start_freq.text())
        except ValueError:
            print("Freq for matrix is not a float")
            return

        try:
            end_freq = float(self.ui.End_freq.text())
        except ValueError:
            print("Freq for matrix is not a float")
            return
        
        try:
            filename = str(self.ui.file_name.text())
        except ValueError:
            print("Freq for matrix is not a float")
            return



        schemes = [UniformMeasurement, LogMeasurement, RandomizedUniformMeasurement]

        scheme_index = self.ui.Scheme_select.currentIndex()

        scheme = schemes[scheme_index](start_position = scheme_start, 
                                        end_position = scheme_end,
                                        num_points = num_points)

        

        libre_vna_measurement(measurement_scheme=scheme, start_freq=start_freq, end_freq=end_freq, filename=filename)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    main_window = MainWindow()

    sys.exit(app.exec())
