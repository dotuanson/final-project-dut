import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import os
import cv2
import time
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from MODNet.src.models.modnet import MODNet

torch_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

pretrained_ckpt = './MODNet/pretrained/modnet_photographic_portrait_matting.ckpt'
modnet = MODNet(backbone_pretrained=False)
modnet = nn.DataParallel(modnet)

GPU = True if torch.cuda.device_count() > 0 else False
if GPU:
    print('Use GPU...')
    modnet = modnet.cuda()
    modnet.load_state_dict(torch.load(pretrained_ckpt))
else:
    print('Use CPU...')
    modnet.load_state_dict(torch.load(pretrained_ckpt, map_location=torch.device('cpu')))
modnet.eval()


class VideoThread(QtCore.QThread):
    change_pixmap_signal_source = QtCore.pyqtSignal(np.ndarray)
    change_pixmap_signal_target = QtCore.pyqtSignal(np.ndarray)

    def __init__(self):
        super(VideoThread, self).__init__()
        self._run_flag = True
        self.is_on = False
        self.background = np.full((512, 672, 3), 255.0)

    def run(self):
        # capture from source
        cap = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                frame_np = cv_img.copy()
                frame_np = cv2.resize(frame_np, (910, 512), cv2.INTER_AREA)
                frame_np = frame_np[:, 120:792, :]
                org = frame_np.copy()
                frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
                frame_PIL = Image.fromarray(frame_np)
                frame_tensor = torch_transforms(frame_PIL)
                frame_tensor = frame_tensor[None, :, :, :]
                if GPU:
                    frame_tensor = frame_tensor.cuda()

                start = time.time()
                with torch.no_grad():
                    _, _, matte_tensor = modnet(frame_tensor, True)
                end = time.time()

                matte_tensor = matte_tensor.repeat(1, 3, 1, 1)
                matte_np = matte_tensor[0].data.cpu().numpy().transpose(1, 2, 0)

                self.background = cv2.resize(self.background, (672, 512), cv2.INTER_AREA)
                fg_np = matte_np * frame_np + (1 - matte_np) * self.background

                target_img = np.uint8(fg_np)
                target_img = cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR)

                delta = str(int((end - start) * 1000)) + " ms"
                cv2.putText(target_img, delta, (8, 40), font, 1, (100, 255, 0), 2, cv2.LINE_AA)
                if self.is_on:
                    self.change_pixmap_signal_target.emit(target_img)
                else:
                    self.change_pixmap_signal_target.emit(org)
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()
        self._run_flag = True


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1019, 861)
        font = QtGui.QFont()
        font.setPointSize(12)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.start = QtWidgets.QPushButton(self.centralwidget)
        self.start.setGeometry(QtCore.QRect(920, 130, 81, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.start.setFont(font)
        self.start.setObjectName("start")
        self.target = QtWidgets.QLabel(self.centralwidget)
        self.target.setGeometry(QtCore.QRect(120, 100, 768, 512))
        self.target.setText("")
        self.target.setScaledContents(True)
        self.target.setObjectName("target")

        self.logo_ftech = QtWidgets.QLabel(self.centralwidget)
        self.logo_ftech.setGeometry(QtCore.QRect(47, 650, 371, 141))
        self.logo_ftech.setText("")
        self.logo_ftech.setPixmap(QtGui.QPixmap("./Pyqt/Images/ftech.png"))
        self.logo_ftech.setScaledContents(True)
        self.logo_ftech.setObjectName("logo_ftech")

        self.logo_dtvt = QtWidgets.QLabel(self.centralwidget)
        self.logo_dtvt.setGeometry(QtCore.QRect(480, 650, 161, 141))
        self.logo_dtvt.setText("")
        self.logo_dtvt.setPixmap(QtGui.QPixmap("./Pyqt/Images/dtvt.jpeg"))
        self.logo_dtvt.setScaledContents(True)
        self.logo_dtvt.setObjectName("logo_dtvt")

        self.svtt_1 = QtWidgets.QLabel(self.centralwidget)
        self.svtt_1.setGeometry(QtCore.QRect(697, 720, 221, 31))
        self.svtt_1.setObjectName("svtt_1")

        self.svtt_2 = QtWidgets.QLabel(self.centralwidget)
        self.svtt_2.setGeometry(QtCore.QRect(749, 750, 221, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.svtt_2.setFont(font)
        self.svtt_2.setObjectName("svtt_2")

        self.gvhd_1 = QtWidgets.QLabel(self.centralwidget)
        self.gvhd_1.setGeometry(QtCore.QRect(697, 650, 241, 31))
        self.gvhd_1.setObjectName("gvhd_1")

        self.gvhd_2 = QtWidgets.QLabel(self.centralwidget)
        self.gvhd_2.setGeometry(QtCore.QRect(750, 680, 171, 41))
        self.gvhd_2.setObjectName("gvhd_2")

        self.detai = QtWidgets.QLabel(self.centralwidget)
        self.detai.setGeometry(QtCore.QRect(17, 10, 971, 81))
        font = QtGui.QFont()
        font.setFamily("Sans")
        font.setPointSize(28)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.detai.setFont(font)
        self.detai.setObjectName("detai")

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1019, 24))
        self.menubar.setObjectName("menubar")

        self.menuOpen = QtWidgets.QMenu(self.menubar)
        self.menuOpen.setObjectName("menuOpen")
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.actionFile = QtWidgets.QAction(MainWindow)
        self.actionFile.setObjectName("actionFile")
        self.menuOpen.addAction(self.actionFile)
        self.menubar.addAction(self.menuOpen.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # # Thêm vô ở đây!!!!
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal_target.connect(self.update_image_target)
        # start the thread
        self.thread.start()
        self.start.clicked.connect(self.clicked_start)
        self.actionFile.triggered.connect(lambda: self.open_link())

    def update_image_target(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.target.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(w, h, QtCore.Qt.KeepAspectRatio)
        return QtGui.QPixmap.fromImage(p)

    def clicked_start(self):
        if self.thread.is_on:
            self.thread.is_on = False
        else:
            self.thread.is_on = True

    def open_link(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(None, ("Selecciona los medios"), os.getcwd(),
                                                             ("Video Files (*.jpg *.png *.jpeg)"))
        self.thread.background = cv2.imread(file_name)
        self.thread.background = cv2.cvtColor(self.thread.background, cv2.COLOR_RGB2BGR)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Đồ án tốt nghiệp"))
        self.start.setText(_translate("MainWindow", "ON/OFF"))
        self.svtt_1.setText(_translate("MainWindow", "SVTT :  Đỗ Tuấn Sơn - 17DT3"))
        self.svtt_2.setText(_translate("MainWindow", "Nguyễn Phước Toàn - 17DT2"))
        self.gvhd_1.setText(_translate("MainWindow", "GVHD : TS. Phan Trần Đăng Khoa"))
        self.gvhd_2.setText(_translate("MainWindow", "ThS. Doãn Đạt Phước"))
        self.detai.setText(_translate("MainWindow", "Tách chủ thể ra khỏi phông nền thời gian thực"))
        self.menuOpen.setTitle(_translate("MainWindow", "Open"))
        self.actionFile.setText(_translate("MainWindow", "File"))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
