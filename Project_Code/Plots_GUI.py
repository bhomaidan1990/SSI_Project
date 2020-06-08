import os
import sys

from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QDialog

class PlotsWindow(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1600, 800)

        #------- Initialization --------------------------
        self.BA_directory  = 'BlandAltman.png'
        self.ROC_directory = 'ROC.png'
        self.BA  = None
        self.ROC = None
        #------------------------------------------------------
        # Main Fonts
        #==========================
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        # Font 12
        font12 = QtGui.QFont()
        font12.setFamily("Rockwell")
        font12.setPointSize(12)
        #---------------------------------------------------------------------       
        #-------------------------------------------------------------------
        # BA Title
        self.LB_BlandAltmanTitle = QtWidgets.QLabel(Dialog)
        self.LB_BlandAltmanTitle.setGeometry(QtCore.QRect(300, 20, 150, 20))
        self.LB_BlandAltmanTitle.setObjectName("LB_BlandAltmanTitle")
        self.LB_BlandAltmanTitle.setFont(font)
        # ROC Title
        self.LB_ROCTitle = QtWidgets.QLabel(Dialog)
        self.LB_ROCTitle.setGeometry(QtCore.QRect(1100, 20, 150, 20))
        self.LB_ROCTitle.setObjectName("LB_ROCTitle")
        self.LB_ROCTitle.setFont(font)
        # BA
        self.LB_BlandAltman = QtWidgets.QLabel(Dialog)
        self.LB_BlandAltman.setGeometry(QtCore.QRect(50, 60, 700, 650))
        self.LB_BlandAltman.setObjectName("LB_BlandAltman")
        # ROC
        self.LB_ROC = QtWidgets.QLabel(Dialog)
        self.LB_ROC.setGeometry(QtCore.QRect(850, 60, 700, 650))
        self.LB_ROC.setObjectName("LB_ROC")
        
        # MainWindow.setCentralWidget(Dialog)
        
        self.retranslateUi(Dialog)
        # QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Plots"))
        self.LB_BlandAltmanTitle.setText(_translate("Dialog", "Bland Altman"))
        self.LB_ROCTitle.setText(_translate("Dialog", "ROC"))
        self.LB_BlandAltman.setText(_translate("Dialog", ""))
        self.LB_ROC.setText(_translate("Dialog", ""))
        #-------------------------------------------------------------------------------
        if (os.path.isfile(str(self.BA_directory)) and os.path.isfile(str(self.ROC_directory))):

            qmap1 = QtGui.QPixmap(self.BA_directory)
            self.LB_BlandAltman.setPixmap(qmap1)
            self.LB_BlandAltman.setScaledContents(True)

            qmap2 = QtGui.QPixmap(self.ROC_directory)
            self.LB_ROC.setPixmap(qmap2)
            self.LB_ROC.setScaledContents(True)

            if(os.path.isfile("BlandAltman.png")):
                os.remove("BlandAltman.png")
            if(os.path.isfile("ROC.png")):
                os.remove("ROC.png")