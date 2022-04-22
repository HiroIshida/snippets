import sys
from qtpy.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QSlider
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QApplication



class Window(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("v example")

        layout = QVBoxLayout()

        label = QLabel()
        label.setText("h")
        layout.addWidget(label)
        layout.addWidget(self._create_slider())

        label = QLabel()
        label.setText("s")
        layout.addWidget(label)
        layout.addWidget(self._create_slider())

        #layout.addWidget()
        self.setLayout(layout)

    def _create_slider(self):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 150)
        slider.setValue(50)
        slider.valueChanged.connect(self.onNewValue)
        return slider

    def onNewValue(self, value):
        return


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
