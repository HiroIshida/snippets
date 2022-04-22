import sys
from qtpy.QtCore import Qt
from qtpy import QtWidgets


class MyWindow(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()
        slider = QtWidgets.QSlider(Qt.Horizontal)
        slider.setRange(0, 150)
        slider.setValue(50)
        slider.valueChanged.connect(self.onNewValue)
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(slider)
        hbox.addSpacing(15)
        self.setLayout(hbox)

    def onNewValue(self, value):
        print(value)

app = QtWidgets.QApplication(sys.argv)
win = MyWindow()
win.show()
sys.exit(app.exec_())
