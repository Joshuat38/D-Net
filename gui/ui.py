# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Depth_UI.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1684, 697)
        MainWindow.setStyleSheet("QWidget#centralwidget\n"
                                 "{\n"
                                 "    \n"
                                 "    \n"
                                 "    background-color: rgba(85, 85, 127, 50);\n"
                                 "}")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("QPushButton\n"
                                         "{\n"
                                         "    color: rgb(255, 255, 255);\n"
                                         "    background-color: rgb(0, 0, 127);\n"
                                         "    border: 3px solid rgb(0, 0, 0);\n"
                                         "    border-radius: 10px;\n"
                                         "    border-style: outset;\n"
                                         "}\n"
                                         "QPushButton:hover\n"
                                         "{\n"
                                         "    color: rgb(255, 255, 255);\n"
                                         "    background-color: rgb(85, 170, 255);\n"
                                         "}\n"
                                         "QPushButton:pressed\n"
                                         "{\n"
                                         "    color: rgb(255, 255, 255);\n"
                                         "    \n"
                                         "    background-color: rgb(170, 0, 127);\n"
                                         "}")
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.stackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
        self.stackedWidget.setObjectName("stackedWidget")
        self.home_page = QtWidgets.QWidget()
        self.home_page.setObjectName("home_page")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.home_page)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.home_label = QtWidgets.QLabel(self.home_page)
        self.home_label.setText("")
        self.home_label.setPixmap(QtGui.QPixmap("media/Depth_Home_Image.PNG"))
        self.home_label.setScaledContents(True)
        self.home_label.setObjectName("home_label")
        self.horizontalLayout_3.addWidget(self.home_label)
        self.stackedWidget.addWidget(self.home_page)
        self.depth_page = QtWidgets.QWidget()
        self.depth_page.setObjectName("depth_page")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.depth_page)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.depth_label = QtWidgets.QLabel(self.depth_page)
        self.depth_label.setText("")
        self.depth_label.setScaledContents(True)
        self.depth_label.setObjectName("depth_label")
        self.verticalLayout_2.addWidget(self.depth_label)
        self.stackedWidget.addWidget(self.depth_page)
        self.contours_page = QtWidgets.QWidget()
        self.contours_page.setObjectName("contours_page")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.contours_page)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setSpacing(0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.contours_label = QtWidgets.QLabel(self.contours_page)
        self.contours_label.setText("")
        self.contours_label.setScaledContents(True)
        self.contours_label.setObjectName("contours_label")
        self.verticalLayout_4.addWidget(self.contours_label)
        self.stackedWidget.addWidget(self.contours_page)
        self.settings_page = QtWidgets.QWidget()
        self.settings_page.setObjectName("settings_page")
        self.gridLayout = QtWidgets.QGridLayout(self.settings_page)
        self.gridLayout.setObjectName("gridLayout")
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 2, 2, 1, 1)
        self.change_contours_label = QtWidgets.QLabel(self.settings_page)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.change_contours_label.setFont(font)
        self.change_contours_label.setScaledContents(True)
        self.change_contours_label.setObjectName("change_contours_label")
        self.gridLayout.addWidget(self.change_contours_label, 3, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem1, 0, 2, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem2, 3, 3, 1, 1)
        self.change_contours = QtWidgets.QSpinBox(self.settings_page)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.change_contours.setFont(font)
        self.change_contours.setMinimum(1)
        self.change_contours.setMaximum(99)
        self.change_contours.setObjectName("change_contours")
        self.gridLayout.addWidget(self.change_contours, 3, 2, 1, 1)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.nyu_btn = QtWidgets.QRadioButton(self.settings_page)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.nyu_btn.setFont(font)
        self.nyu_btn.setChecked(True)
        self.nyu_btn.setObjectName("nyu_btn")
        self.buttonGroup = QtWidgets.QButtonGroup(MainWindow)
        self.buttonGroup.setObjectName("buttonGroup")
        self.buttonGroup.addButton(self.nyu_btn)
        self.verticalLayout_5.addWidget(self.nyu_btn)
        
        self.nyu_light_weight_btn = QtWidgets.QRadioButton(self.settings_page)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.nyu_light_weight_btn.setFont(font)
        self.nyu_light_weight_btn.setObjectName("nyu_light_weight_btn")
        self.buttonGroup.addButton(self.nyu_light_weight_btn)
        self.verticalLayout_5.addWidget(self.nyu_light_weight_btn)
        
        self.kitti_btn = QtWidgets.QRadioButton(self.settings_page)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.kitti_btn.setFont(font)
        self.kitti_btn.setObjectName("kitti_btn")
        self.buttonGroup.addButton(self.kitti_btn)
        self.verticalLayout_5.addWidget(self.kitti_btn)
        
        self.kitti_light_weight_btn = QtWidgets.QRadioButton(self.settings_page)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.kitti_light_weight_btn.setFont(font)
        self.kitti_light_weight_btn.setObjectName("kitti_light_weight_btn")
        self.buttonGroup.addButton(self.kitti_light_weight_btn)
        self.verticalLayout_5.addWidget(self.kitti_light_weight_btn)
        
        self.gridLayout.addLayout(self.verticalLayout_5, 1, 2, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem3, 3, 1, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem4, 4, 2, 1, 1)
        self.dataset_label = QtWidgets.QLabel(self.settings_page)
        self.dataset_label.setObjectName("dataset_label")
        self.gridLayout.addWidget(self.dataset_label, 1, 0, 1, 1)
        self.visual_label = QtWidgets.QLabel(self.settings_page)
        self.visual_label.setText("")
        self.visual_label.setPixmap(QtGui.QPixmap("media/NYU_Mode.PNG"))
        self.visual_label.setScaledContents(True)
        self.visual_label.setObjectName("visual_label")
        self.gridLayout.addWidget(self.visual_label, 1, 3, 1, 1)
        self.stackedWidget.addWidget(self.settings_page)
        self.verticalLayout_3.addWidget(self.stackedWidget)
        self.verticalLayout.addLayout(self.verticalLayout_3)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout.setSpacing(11)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.home_btn = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.home_btn.setFont(font)
        self.home_btn.setObjectName("home_btn")
        self.horizontalLayout.addWidget(self.home_btn)
        self.settings_btn = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.settings_btn.setFont(font)
        self.settings_btn.setObjectName("settings_btn")
        self.horizontalLayout.addWidget(self.settings_btn)
        self.depth_btn = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.depth_btn.setFont(font)
        self.depth_btn.setObjectName("depth_btn")
        self.horizontalLayout.addWidget(self.depth_btn)
        self.contour_btn = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.contour_btn.setFont(font)
        self.contour_btn.setObjectName("contour_btn")
        self.horizontalLayout.addWidget(self.contour_btn)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.verticalLayout.setStretch(0, 10)
        self.verticalLayout.setStretch(1, 1)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(3)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.change_contours_label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt;\">Select Contour Count:</span></p></body></html>"))
        self.nyu_btn.setText(_translate("MainWindow", "NYU Performance (Max 10m)"))
        self.nyu_light_weight_btn.setText(_translate("MainWindow", "NYU Lightweight (Max 10m)"))
        self.kitti_btn.setText(_translate("MainWindow", "KITTI Performance (Max 80m)"))
        self.kitti_light_weight_btn.setText(_translate("MainWindow", "KITTI Lightweight (Max 80m)"))
        self.dataset_label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt;\">Select Depth Dataset:</span></p></body></html>"))
        self.home_btn.setText(_translate("MainWindow", "Home"))
        self.depth_btn.setText(_translate("MainWindow", "Depth"))
        self.contour_btn.setText(_translate("MainWindow", "Contours"))
        self.settings_btn.setText(_translate("MainWindow", "Settings"))

