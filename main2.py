import sys
import cv2
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from ui.design import Ui_Form

class HistogramCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(3.6, 1.2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

    def plot_histogram(self, channel_data, color):
        self.ax.clear()
        hist = cv2.calcHist([channel_data], [0], None, [256], [0, 256])
        self.ax.plot(hist, color=color)
        self.ax.set_xlim([0, 256])
        self.ax.set_yticks([])
        self.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.image = None

        self.ui.btn_load.clicked.connect(self.load_image)
        self.ui.btn_hr.clicked.connect(self.plot_red)
        self.ui.btn_hg.clicked.connect(self.plot_green)
        self.ui.btn_hb.clicked.connect(self.plot_blue)

        self.canvas_r = HistogramCanvas(self.ui.frame_r)
        self.ui.verticalLayout_2.addWidget(self.canvas_r)

        self.canvas_g = HistogramCanvas(self.ui.frame_g)
        self.ui.verticalLayout_3.addWidget(self.canvas_g)

        self.canvas_b = HistogramCanvas(self.ui.frame_b)
        self.ui.verticalLayout_4.addWidget(self.canvas_b)

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.bmp *.jpg *.png)")
        if fname:
            self.image = cv2.imread(fname)
            if self.image is not None:
                qimg = self.convert_cv_qt(self.image)
                self.ui.label_preview.setPixmap(qimg)

                # Reset histogram canvas
                # self.canvas_r.ax.clear()
                # self.canvas_r.draw()
                # self.canvas_g.ax.clear()
                # self.canvas_g.draw()
                # self.canvas_b.ax.clear()
                # self.canvas_b.draw()

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(qimg).scaled(180, 120)

    def plot_red(self):
        if self.image is not None:
            r_channel = self.image[:, :, 2]
            self.canvas_r.plot_histogram(r_channel, 'red')
            print_histogram_info(r_channel, "Red")

    def plot_green(self):
        if self.image is not None:
            g_channel = self.image[:, :, 1]
            self.canvas_g.plot_histogram(g_channel, 'green')
            print_histogram_info(g_channel, "Green")

    def plot_blue(self):
        if self.image is not None:
            b_channel = self.image[:, :, 0]
            self.canvas_b.plot_histogram(b_channel, 'blue')
            print_histogram_info(b_channel, "Blue")

def print_histogram_info(channel, label):
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256]).flatten()
    peak = int(np.argmax(hist))
    peak_val = int(hist[peak])
    mean_val = np.sum(hist * np.arange(256)) / np.sum(hist)
    total = int(np.sum(hist))
    print(f"[INFO] Histogram {label}:")
    print(f" - Peak at intensity {peak}: {peak_val} pixels")
    print(f" - Mean intensity: {mean_val:.2f}")
    print(f" - Total pixels: {total}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())