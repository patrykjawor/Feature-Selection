from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QMainWindow, QPushButton, QLabel
from PyQt6.QtCore import Qt, QSize, QThread, QObject, pyqtSignal
from Old.test_genethic_algo import run_genethic_algo
import pyqtgraph as pg
import sys

class Worker(QObject):
    finished = pyqtSignal()

    def run(self):
        run_genethic_algo(30,20,30)
        self.finished.emit()
        
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Feature Selection Demo")
        self.setFixedSize(QSize(400, 300))

        layout = QVBoxLayout()

        self.button = QPushButton("CLICK ME!")
        self.button.setCheckable(True)
        self.button.clicked.connect(self.button_action)

        layout.addWidget(self.button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
    
    def button_action(self):
        self.button.setEnabled(False)
        self.thread = QThread()
        self.worker = Worker()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.finished.connect(self.enable_button)
        self.thread.start()
    
    def enable_button(self):
        self.button.setEnabled(True)
        
app = QApplication([])

window = MainWindow()
window.show()

app.exec()
