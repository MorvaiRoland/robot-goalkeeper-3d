from PyQt6.QtCore import QCoreApplication, QThread
import sys

class T(QThread):
    def run(self):
        print("Thread running")
        from ultralytics import YOLO
        y = YOLO('yolov8n.pt')
        print("YOLO loaded")

app = QCoreApplication(sys.argv)
t = T()
t.start()
t.wait()
