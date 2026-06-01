from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QObject, QTimer, QCoreApplication
from PyQt6.QtGui import QImage, QPixmap, QFont, QTextCursor, QAction, QVector3D
from PyQt6.QtWidgets import QApplication
import cv2
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import sys

class T(QThread):
    def run(self):
        print("Thread running")
        from ultralytics import YOLO
        y = YOLO('yolov8n.pt')
        print("YOLO loaded")

app = QApplication(sys.argv)
t = T()
t.start()
t.wait()
