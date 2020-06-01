import os
import shutil
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QCoreApplication, Qt, QDir
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        #------------------------------------------------------
        # Main Window
        #==============
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1800, 1000)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        #------- Initialization --------------------------
        self.directory = None
        self.image     = None
        self.mask      = None
        #------------------------------------------------------
        # Main Font 1
        #==========================
        Font1 = QtGui.QFont()
        Font1.setFamily("Rockwell")
        Font1.setPointSize(10)
        Font1.setBold(True)
        Font1.setWeight(75)
        #------------------------------------------------------
        # Images
        #=========================
        # Image title
        self.LB_ImageTitle = QtWidgets.QLabel(self.centralwidget)
        self.LB_ImageTitle.setGeometry(QtCore.QRect(160, 30, 120, 30))
        self.LB_ImageTitle.setFont(Font1)
        self.LB_ImageTitle.setToolTip("Image")
        self.LB_ImageTitle.setObjectName("LB_ImageTitle")
        # Image 
        self.LB_Original_Image = QtWidgets.QLabel(self.centralwidget)
        self.LB_Original_Image.setGeometry(QtCore.QRect(70, 80, 381, 341))
        self.LB_Original_Image.setObjectName("LB_Original_Image")
        # Mask title
        self.LB_MaskTitle = QtWidgets.QLabel(self.centralwidget)
        self.LB_MaskTitle.setGeometry(QtCore.QRect(670, 30, 120, 30))
        self.LB_MaskTitle.setFont(Font1)
        self.LB_MaskTitle.setObjectName("LB_MaskTitle")
        # Mask
        self.LB_Mask_GT = QtWidgets.QLabel(self.centralwidget)
        self.LB_Mask_GT.setGeometry(QtCore.QRect(480, 80, 381, 341))
        self.LB_Mask_GT.setObjectName("LB_Mask_GT")
        # Prediction title
        self.LB_PredictionTitle = QtWidgets.QLabel(self.centralwidget)
        self.LB_PredictionTitle.setGeometry(QtCore.QRect(160, 440, 120, 30))
        self.LB_PredictionTitle.setFont(Font1)
        self.LB_PredictionTitle.setObjectName("LB_PredictionTitle")
        # Prediction
        self.LB_PredictionImage = QtWidgets.QLabel(self.centralwidget)
        self.LB_PredictionImage.setGeometry(QtCore.QRect(70, 460, 381, 341))
        self.LB_PredictionImage.setObjectName("LB_PredictionImage")
        #-------------------------------------------------------------------
        # Slider
        #=============
        self.Slider = QtWidgets.QSlider(self.centralwidget)
        self.Slider.setGeometry(QtCore.QRect(470, 770, 401, 22))
        self.Slider.setOrientation(QtCore.Qt.Horizontal)
        self.Slider.setObjectName("Slider")
        #-------------------------------------------------------------------
        # LCD
        #=============
        self.LCD = QtWidgets.QLCDNumber(self.centralwidget)
        self.LCD.setGeometry(QtCore.QRect(650, 750, 70, 25))
        self.LCD.setDigitCount(2)
        self.LCD.setObjectName("LCD")
        #-------------------------------------------------------------------
        # Push_Buttons
        #=============
        self.Frame_PB = QtWidgets.QFrame(self.centralwidget)
        self.Frame_PB.setGeometry(QtCore.QRect(1500, 40, 301, 400))
        self.Frame_PB.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Frame_PB.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Frame_PB.setFont(Font1)
        self.Frame_PB.setObjectName("Frame_PB")
        #-------------------------------------------------------------------
        # Load 
        self.PB_Load_and_Process = QtWidgets.QPushButton(self.Frame_PB)
        self.PB_Load_and_Process.setGeometry(QtCore.QRect(0, 0, 190, 50))
        self.PB_Load_and_Process.setObjectName("PB_Load_and_Process")
        #-------------------------------------------------------------------
        # Save
        self.PB_Save_Results = QtWidgets.QPushButton(self.Frame_PB)
        self.PB_Save_Results.setGeometry(QtCore.QRect(0, 70, 190, 50))
        self.PB_Save_Results.setObjectName("PB_Save_Results")
        #-------------------------------------------------------------------
        # Reset
        self.PB_Reset = QtWidgets.QPushButton(self.Frame_PB)
        self.PB_Reset.setGeometry(QtCore.QRect(0, 140, 190, 50))
        self.PB_Reset.setObjectName("PB_Reset")
        #-------------------------------------------------------------------
        # Plot
        self.PB_Plots = QtWidgets.QPushButton(self.Frame_PB)
        self.PB_Plots.setGeometry(QtCore.QRect(0, 210, 190, 50))
        self.PB_Plots.setObjectName("PB_Plots")
        #-------------------------------------------------------------------
        # Metric_Labels
        #==============
        self.Frame_LB = QtWidgets.QFrame(self.centralwidget)
        self.Frame_LB.setGeometry(QtCore.QRect(1500, 400, 301, 451))
        self.Frame_LB.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Frame_LB.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Frame_LB.setFont(Font1)
        self.Frame_LB.setObjectName("Frame_LB")
        #-------------------------------------------------------------------
        # Metric1Title
        self.LB_Metric1Title = QtWidgets.QLabel(self.Frame_LB)
        self.LB_Metric1Title.setGeometry(QtCore.QRect(0, 0, 200, 20))
        self.LB_Metric1Title.setObjectName("LB_Metric1Title")
        #-------------------------------------------------------------------
        # Metric1Value
        self.LB_Metric1Value = QtWidgets.QLabel(self.Frame_LB)
        self.LB_Metric1Value.setGeometry(QtCore.QRect(0, 60, 200, 20))
        self.LB_Metric1Value.setObjectName("LB_Metric1Value")
        #-------------------------------------------------------------------
        # Metric2Title
        self.LB_Metric2Title = QtWidgets.QLabel(self.Frame_LB)
        self.LB_Metric2Title.setGeometry(QtCore.QRect(0, 140, 200, 20))
        self.LB_Metric2Title.setObjectName("LB_Metric2Title")
        #-------------------------------------------------------------------
        # Metric2Value
        self.LB_Metric2Value = QtWidgets.QLabel(self.Frame_LB)
        self.LB_Metric2Value.setGeometry(QtCore.QRect(0, 200, 200, 20))
        self.LB_Metric2Value.setObjectName("LB_Metric2Value")
        MainWindow.setCentralWidget(self.centralwidget)
        #-------------------------------------------------------------------
        # Menu bar
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1407, 18))
        self.menubar.setObjectName("menubar")
        #-------------------------------------------------------------------
        # Menu
        self.menuMenu = QtWidgets.QMenu(self.menubar)
        self.menuMenu.setObjectName("menuMenu")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        #-------------------------------------------------------------------
        # Status bar
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        #-------------------------------------------------------------------
        _translate = QtCore.QCoreApplication.translate
        #-------------------------------------------------------------------
        # Open
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionOpen.setStatusTip(_translate("MainWindow", "Click to load Nifti Image"))
        self.actionOpen.setShortcut(_translate("MainWindow", "Ctrl + O"))
        #-------------------------------------------------------------------
        # Exit
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionExit.setStatusTip(_translate("MainWindow", "Press Alt + F4 to Exit"))
        self.actionExit.setShortcut(_translate("MainWindow", "Alt + F4"))
        #-------------------------------------------------------------------
        # Documentation
        self.actionDocumentation = QtWidgets.QAction(MainWindow)
        self.actionDocumentation.setObjectName("actionDocumentation")
        self.actionDocumentation.setStatusTip(_translate("MainWindow", "Click to see Documentation"))
        self.actionDocumentation.setShortcut(_translate("MainWindow", "F1"))
        #-------------------------------------------------------------------
        # Help
        self.actionHelp = QtWidgets.QAction(MainWindow)
        self.actionHelp.setObjectName("actionHelp")
        self.actionHelp.setStatusTip(_translate("MainWindow", "Click for Help"))
        #-------------------------------------------------------------------
        # Menu Action
        self.menuMenu.addAction(self.actionOpen)
        self.menuMenu.addAction(self.actionExit)
        #-------------------------------------------------------------------
        # Help Action
        self.menuHelp.addAction(self.actionDocumentation)
        self.menuHelp.addAction(self.actionHelp)
        #-------------------------------------------------------------------
        # Menu bar action
        self.menubar.addAction(self.menuMenu.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())
        #------------------------------------------------------
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        #===================================================================
        #======================== Signals ==================================
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        #
        self.actionOpen.triggered.connect(self.setExistingFile)
        self.actionDocumentation.triggered.connect(self.openUrl)
        self.PB_Load_and_Process.clicked.connect(self.setExistingFile)
        self.actionExit.triggered.connect(self.closeEvent)
        self.PB_Save_Results.clicked.connect(self.saveSeg)
        self.PB_Reset.clicked.connect(self.reset)
        self.Slider.sliderReleased.connect(self.sliderMovement)
        #==================================================================
        #==================================================================
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        #------------------------------------------------------
        # Labels
        #=============
        # Images
        self.LB_ImageTitle.setText(_translate("MainWindow", "image"))
        self.LB_MaskTitle.setText(_translate("MainWindow", "mask(GT)"))
        self.LB_Original_Image.setToolTip(_translate("MainWindow", "This is the Original Image"))
        self.LB_Original_Image.setStatusTip(_translate("MainWindow", "Original Imaage"))
        self.LB_Original_Image.setText(_translate("MainWindow", "Img"))
        self.LB_Mask_GT.setText(_translate("MainWindow", "Lbl"))
        self.LB_PredictionImage.setText(_translate("MainWindow", "pred"))
        self.LB_PredictionTitle.setText(_translate("MainWindow", "Prediction"))
        # Metrics
        self.LB_Metric1Title.setText(_translate("MainWindow", "Hausdorff Distance"))
        self.LB_Metric1Value.setText(_translate("MainWindow", "0"))
        self.LB_Metric2Title.setText(_translate("MainWindow", "Dice Metric"))
        self.LB_Metric2Value.setText(_translate("MainWindow", "0 %"))
        #------------------------------------------------------
        # Push Buttons
        #==============
        self.PB_Load_and_Process.setText(_translate("MainWindow", "Load and Process"))
        self.PB_Load_and_Process.setStatusTip(_translate("MainWindow", "Click to Load Nifti Image"))

        self.PB_Save_Results.setText(_translate("MainWindow", "Save Results"))
        self.PB_Save_Results.setText(_translate("MainWindow", "Save Results"))
        self.PB_Save_Results.setStatusTip(_translate("MainWindow", "Click to Save Results"))

        self.PB_Reset.setText(_translate("MainWindow", "Reset"))
        self.PB_Reset.setStatusTip(_translate("MainWindow", "Click to Reset"))

        self.Slider.setStatusTip(_translate("MainWindow", "Slide to change Slice Number"))

        self.PB_Plots.setText(_translate("MainWindow", "Show Plots"))
        self.PB_Plots.setStatusTip(_translate("MainWindow", "Click to see the Plots"))
        #------------------------------------------------------
        # Menus
        #=============
        self.menuMenu.setTitle(_translate("MainWindow", "Menu"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        #-------------------------------------------------------
        self.actionOpen.setText(_translate("MainWindow", "open"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        #-------------------------------------------------------
        self.actionDocumentation.setText(_translate("MainWindow", "Documentation"))
        self.actionHelp.setText(_translate("MainWindow", "Help"))
        #-------------------------------------------------------
        self.LCD.setStatusTip(_translate("MainWindow", "Slice Number"))
    #=============================================================================================
    def about(self):
        Q = QWidget()
        message = QMessageBox.information(Q,"About","Developed by VIBOT2019 ", QMessageBox.Ok)
    #--------------------------------------------------------------------------------------------
    def openUrl(self):
        url = QtCore.QUrl('Documentation.pdf')
        if not QDesktopServices.openUrl(url):
            pass
    #---------------------------------------------------------------------------------------------
    def reset(self):
        # if os.path.exists("temporary"):
        #     shutil.rmtree("temporary", ignore_errors=True) 
        if(self.directory is not None):
            self.directory = None
    #---------------------------------------------------------------------------------------------
    def closeEvent(self, event):
        """Generate 'question' dialog on clicking 'X' button in title bar.

        Reimplement the closeEvent() event handler to include a 'Question'
        dialog with options on how to proceed - Save, Close, Cancel buttons
        """
        Q = QWidget()
        reply = QMessageBox.question(Q,"Message","Are you sure you want to quit? Any unsaved work will be lost.",
            QMessageBox.Save | QMessageBox.Close | QMessageBox.Cancel,  QMessageBox.Save)

        if reply == QMessageBox.Close:
            app.quit()
        elif reply == QMessageBox.Save:
            self.saveSeg()
        else:
            pass

    #---------------------------------------------------------------------------------------------
    def close(self):
        QCoreApplication.instance().quit
    #---------------------------------------------------------------------------------------------
    def saveSeg(self):
        if(self.directory is not None):
            path = QFileDialog.getExistingDirectory(None,'Select file')
            # copytree('temporary/',path)
    #---------------------------------------------------------------------------------------------
    def setExistingFile(self):
        self.dialog = QFileDialog()
        self.directory = QFileDialog.getExistingDirectory(None, "Open File")
        if(self.directory != ''):
            # Load and Process Data
            pass
    #---------------------------------------------------------------------------------------------
    def sliderMovement(self):
        # What to d o if the slider moved
        self.LCD.display(self.Slider.value())
#======================================================
#                    Main Function
#======================================================
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    screenShape = QtWidgets.QDesktopWidget().screenGeometry()
    MainWindow.resize(screenShape.width(), screenShape.height())
    MainWindow.show()
    sys.exit(app.exec_())
