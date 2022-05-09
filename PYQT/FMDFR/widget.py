# This Python file uses the following encoding: utf-8
import os
from pathlib import Path
import sys

from PySide2.QtWidgets import QApplication, QWidget, QDialog, QPushButton
from PySide2.QtCore import QFile
from PySide2.QtUiTools import QUiLoader

from PyQt5.uic import loadUi
from PyQt5 import QtWidgets

class Widget(QDialog):
    def __init__(self):
        super(Widget, self).__init__()
        self.load_ui()
        self.b1=QPushButton("FR",self)
        self.b2=QPushButton("FMD",self)
        self.b1.clicked.connect(self.load_ui_FR)

    def load_ui(self):
        loader = QUiLoader()
        path = os.fspath(Path(__file__).resolve().parent / "form.ui")
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        loader.load(ui_file, self)
        ui_file.close()

    def load_ui_FR(self):
        FR=Face_Recognition()
        widget.addWidget(FR)
        widget.setCurrentIndex(widget.currentIndex()+1)

class Face_Recognition(QDialog):
    def __init__(self):
        super(Face_Recognition, self).__init__()
        loadUi("FR.ui",self)

    # def load_ui(self):
    #     loader = QUiLoader()
    #     path = os.fspath(Path(__file__).resolve().parent / "form.ui")
    #     ui_file = QFile(path)
    #     ui_file.open(QFile.ReadOnly)
    #     loader.load(ui_file, self)
    #     ui_file.close()

if __name__ == "__main__":
    app = QApplication([])
    widget = Widget()
    widget.show()
    sys.exit(app.exec_())
