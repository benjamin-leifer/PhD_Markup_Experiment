import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import time
from PySide6 import QtWidgets

start_time = time.perf_counter()
app = QtWidgets.QApplication([])
window = QtWidgets.QWidget()
layout = QtWidgets.QVBoxLayout(window)
button = QtWidgets.QPushButton('Click')
label = QtWidgets.QLabel('Not clicked')
layout.addWidget(QtWidgets.QLabel('Battery Test Data Analysis'))
layout.addWidget(button)
layout.addWidget(label)
load_time = time.perf_counter() - start_time


def handle_click():
    label.setText('Clicked')

start_interact = time.perf_counter()
button.clicked.connect(handle_click)
button.click()
interaction_latency = time.perf_counter() - start_interact

print(f'PySide load time: {load_time:.4f} s')
print(f'PySide interaction latency: {interaction_latency:.4f} s')
