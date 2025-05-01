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

    def plot_histogram_manual_line(self, channel_data, color):
        """Menampilkan histogram channel dalam bentuk garis manual"""
        self.ax.clear()
        hist = np.zeros(256, dtype=np.uint32)
        height, width = channel_data.shape

        for y in range(height):
            for x in range(width):
                intensity = channel_data[y, x]
                hist[intensity] += 1

        hmax = np.max(hist) if np.max(hist) > 0 else 1
        hist = hist / hmax  # Normalisasi

        x_vals = np.arange(256)
        self.ax.plot(x_vals, hist, color=color)
        self.ax.set_xlim([0, 256])
        self.ax.set_ylim([0, 1])
        self.draw()

        print_histogram_info(hist * hmax, color, "Manual-Line")


    def plot_histogram_manual_filled(self, channel_data, color):
        """Menampilkan histogram channel dalam bentuk area terisi"""
        self.ax.clear()
        hist = np.zeros(256, dtype=np.uint32)
        height, width = channel_data.shape

        for y in range(height):
            for x in range(width):
                intensity = channel_data[y, x]
                hist[intensity] += 1

        hmax = np.max(hist) if np.max(hist) > 0 else 1
        hist = hist / hmax  # Normalisasi

        x_vals = np.arange(256)
        self.ax.fill_between(x_vals, 0, hist, color=color, alpha=0.7)
        self.ax.set_xlim([0, 256])
        self.ax.set_ylim([0, 1])
        self.draw()

        print_histogram_info(hist * hmax, color, "Manual-Filled")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.image = None
        self.current_mode = "Manual"

        self.ui.btn_load.clicked.connect(self.load_image)
        self.ui.btn_hr.clicked.connect(self.plot_red)
        self.ui.btn_hg.clicked.connect(self.plot_green)
        self.ui.btn_hb.clicked.connect(self.plot_blue)
        self.ui.btn_toggle_mode.clicked.connect(self.toggle_mode)

        self.canvas_r = HistogramCanvas(self.ui.frame_r)
        self.ui.verticalLayout_2.addWidget(self.canvas_r)

        self.canvas_g = HistogramCanvas(self.ui.frame_g)
        self.ui.verticalLayout_3.addWidget(self.canvas_g)

        self.canvas_b = HistogramCanvas(self.ui.frame_b)
        self.ui.verticalLayout_4.addWidget(self.canvas_b)

    def toggle_mode(self):
        if self.current_mode == "Manual":
            self.current_mode = "OpenCV"
            self.ui.btn_toggle_mode.setText("Line")
        else:
            self.current_mode = "Manual"
            self.ui.btn_toggle_mode.setText("Filled")

        if self.image is not None:
            self.plot_red()
            self.plot_green()
            self.plot_blue()

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.bmp *.jpg *.png)")
        if fname:
            self.image = cv2.imread(fname)
            if self.image is not None:
                qimg = self.convert_cv_qt(self.image)
                self.ui.label_preview.setPixmap(qimg)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(qimg).scaled(180, 120)

    def plot_red(self):
        if self.image is not None:
            r_channel = self.image[:, :, 2]
            if self.current_mode == "Manual":
                self.canvas_r.plot_histogram_manual_line(r_channel, 'red')
            else:
                self.canvas_r.plot_histogram_manual_filled(r_channel, 'red')


    def plot_green(self):
        if self.image is not None:
            g_channel = self.image[:, :, 1]
            if self.current_mode == "Manual":
                self.canvas_g.plot_histogram_manual_line(g_channel, 'green')
            else:
                self.canvas_g.plot_histogram_manual_filled(g_channel, 'green')


    def plot_blue(self):
        if self.image is not None:
            b_channel = self.image[:, :, 0]
            if self.current_mode == "Manual":
                self.canvas_b.plot_histogram_manual_line(b_channel, 'blue')
            else:
                self.canvas_b.plot_histogram_manual_filled(b_channel, 'blue')

def print_histogram_info(hist, label, mode):
    peak = int(np.argmax(hist))
    peak_val = int(hist[peak])
    mean_val = np.sum(hist * np.arange(256)) / np.sum(hist)
    total = int(np.sum(hist))
    print(f"[{mode}] Histogram {label.capitalize()}:")
    print(f" - Peak at intensity {peak}: {peak_val} pixels")
    print(f" - Mean intensity: {mean_val:.2f}")
    print(f" - Total pixels: {total}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())