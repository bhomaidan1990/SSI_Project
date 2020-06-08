#============================
#==  @Author: Belal Hmedan ==
#============================
# import neccessary libraries
import os
import sys
import shutil
import numpy as np
from skimage.io import imread, imsave
#---------------------------------------------------
# KerasPredict(image2D,modelPath)
from Predict import KerasPredict  
# eval(image2D, Backbone, modelPath) 
from Predict import crop2D_mask, eval 
from evaluate import *
from Plots_GUI import PlotsWindow
#---------------------------------------------------
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QCoreApplication, Qt, QDir
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        #------------------------------------------------------
        # Main Window
        #==============
        MainWindow.resize(1800, 1000)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        #------- Initialization --------------------------
        self.directory = None
        self.image     = None
        self.mask      = None
        self.segment   = None
        self.metrics   = {}
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
        #------------------------------------------------------
        # Image Title
        self.LB_ImageTitle = QtWidgets.QLabel(self.centralwidget)
        self.LB_ImageTitle.setGeometry(QtCore.QRect(250, 10, 150, 40))
        self.LB_ImageTitle.setFont(font12)
        self.LB_ImageTitle.setObjectName("LB_ImageTitle")
        # Mask Title
        self.LB_MaskTitle = QtWidgets.QLabel(self.centralwidget)
        self.LB_MaskTitle.setGeometry(QtCore.QRect(900, 10, 150, 40))
        self.LB_MaskTitle.setFont(font12)
        self.LB_MaskTitle.setObjectName("LB_MaskTitle")
        # Prediction Title
        self.LB_PredictionTitle = QtWidgets.QLabel(self.centralwidget)
        self.LB_PredictionTitle.setGeometry(QtCore.QRect(200, 470, 170, 40))
        self.LB_PredictionTitle.setFont(font12)
        self.LB_PredictionTitle.setObjectName("LB_PredictionTitle")
        # Image
        self.LB_Image = QtWidgets.QLabel(self.centralwidget)
        self.LB_Image.setGeometry(QtCore.QRect(100, 70, 400, 400))
        self.LB_Image.setObjectName("LB_Image")
        # Mask
        self.LB_Mask = QtWidgets.QLabel(self.centralwidget)
        self.LB_Mask.setGeometry(QtCore.QRect(800, 70, 400, 400))
        self.LB_Mask.setObjectName("LB_Mask")
        # Prediction
        self.LB_Prediction = QtWidgets.QLabel(self.centralwidget)
        self.LB_Prediction.setGeometry(QtCore.QRect(100, 520, 400, 400))
        self.LB_Prediction.setObjectName("LB_Prediction")
        #---------------------------------------------------------------------
        # -------------------- PB Frame --------------------------------------
        self.PB_Frame = QtWidgets.QFrame(self.centralwidget)
        self.PB_Frame.setGeometry(QtCore.QRect(1500, 10, 200, 340))
        self.PB_Frame.setFont(font12)
        self.PB_Frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.PB_Frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.PB_Frame.setObjectName("PB_Frame")
        #=====================================================================
        self.PB_LoadImage = QtWidgets.QPushButton(self.PB_Frame)
        self.PB_LoadImage.setGeometry(QtCore.QRect(0, 10, 150, 40))
        self.PB_LoadImage.setObjectName("PB_LoadImage")
        
        self.PB_LoadMask = QtWidgets.QPushButton(self.PB_Frame)
        self.PB_LoadMask.setGeometry(QtCore.QRect(0, 50,  150, 40))
        self.PB_LoadMask.setObjectName("PB_LoadMask")
        
        self.CB_Platform = QtWidgets.QComboBox(self.PB_Frame)
        self.CB_Platform.setGeometry(QtCore.QRect(0, 120,  150, 40))
        self.CB_Platform.setObjectName("CB_Platform")
        self.CB_Platform.addItem("")
        self.CB_Platform.addItem("")

        self.PB_Predict = QtWidgets.QPushButton(self.PB_Frame)
        self.PB_Predict.setGeometry(QtCore.QRect(0, 180,  150, 40))
        self.PB_Predict.setObjectName("PB_Predict")
        
        self.PB_Plots = QtWidgets.QPushButton(self.PB_Frame)
        self.PB_Plots.setGeometry(QtCore.QRect(0, 220,  150, 40))
        self.PB_Plots.setObjectName("PB_Plots")
        
        self.PB_SaveResults = QtWidgets.QPushButton(self.PB_Frame)
        self.PB_SaveResults.setGeometry(QtCore.QRect(0, 260,  150, 40))
        self.PB_SaveResults.setObjectName("PB_SaveResults")
        
        self.PB_Reset = QtWidgets.QPushButton(self.PB_Frame)
        self.PB_Reset.setGeometry(QtCore.QRect(0, 300,  150, 40))
        self.PB_Reset.setObjectName("PB_Reset")
        #---------------------------------------------------------------------
        # -------------------- LB Frame --------------------------------------        
        # LB Frame
        self.LB_Frame = QtWidgets.QFrame(self.centralwidget)
        self.LB_Frame.setGeometry(QtCore.QRect(1500, 400, 240, 450))
        self.LB_Frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.LB_Frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.LB_Frame.setFont(font12)
        self.LB_Frame.setObjectName("frame")
        #=====================================================================        
        # LB_Hausdorff       
        self.LB_Hausdorff = QtWidgets.QLabel(self.LB_Frame)
        self.LB_Hausdorff.setGeometry(QtCore.QRect(0, 0, 220, 20))
        self.LB_Hausdorff.setObjectName("LB_Hausdorff")
        # LB_HausdorffValue
        self.LB_HausdorffValue = QtWidgets.QLabel(self.LB_Frame)
        self.LB_HausdorffValue.setGeometry(QtCore.QRect(0, 30, 220, 20))
        self.LB_HausdorffValue.setObjectName("LB_HausdorffValue")
        # LB_Dice
        self.LB_Dice = QtWidgets.QLabel(self.LB_Frame)
        self.LB_Dice.setGeometry(QtCore.QRect(0, 80, 220, 20))
        self.LB_Dice.setObjectName("LB_Dice")
        # LB_DiceValue
        self.LB_DiceValue = QtWidgets.QLabel(self.LB_Frame)
        self.LB_DiceValue.setGeometry(QtCore.QRect(0, 110, 220, 20))
        self.LB_DiceValue.setObjectName("LB_DiceValue")
        # LB_Jaccard
        self.LB_Jaccard = QtWidgets.QLabel(self.LB_Frame)
        self.LB_Jaccard.setGeometry(QtCore.QRect(0, 160, 220, 25))
        self.LB_Jaccard.setObjectName("LB_Jaccard")
        # LB_JaccardVlaue
        self.LB_JaccardValue = QtWidgets.QLabel(self.LB_Frame)
        self.LB_JaccardValue.setGeometry(QtCore.QRect(0, 190, 220, 20))
        self.LB_JaccardValue.setObjectName("LB_JaccardValue")
        # LB_P_
        self.LB_P_ = QtWidgets.QLabel(self.LB_Frame)
        self.LB_P_.setGeometry(QtCore.QRect(0, 240, 220, 20))
        self.LB_P_.setObjectName("LB_P_")
        # LB_P_Value
        self.LB_P_Value = QtWidgets.QLabel(self.LB_Frame)
        self.LB_P_Value.setGeometry(QtCore.QRect(0, 270, 220, 20))
        self.LB_P_Value.setObjectName("LB_P_Value")
        # LB_Pearson
        self.LB_Pearson = QtWidgets.QLabel(self.LB_Frame)
        self.LB_Pearson.setGeometry(QtCore.QRect(0, 320, 220, 51))
        self.LB_Pearson.setObjectName("LB_Pearson")
        # LB_PearsonValue
        self.LB_PearsonValue = QtWidgets.QLabel(self.LB_Frame)
        self.LB_PearsonValue.setGeometry(QtCore.QRect(0, 380, 220, 20))
        self.LB_PearsonValue.setObjectName("LB_PearsonValue")
        #---------------------------------------------------------------
        #-------------- Menu -------------------------------------------
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1449, 21))
        self.menubar.setObjectName("menubar")
        self.menuMenu = QtWidgets.QMenu(self.menubar)
        self.menuMenu.setObjectName("menuMenu")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        #==============================================================
        # Exit
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setFont(font)
        self.actionExit.setObjectName("actionExit")
        # Documentation
        self.actionDocumentation = QtWidgets.QAction(MainWindow)
        self.actionDocumentation.setFont(font)
        self.actionDocumentation.setObjectName("actionDocumentation")
        #-------------------------------------------------------------
        self.menuMenu.addAction(self.actionExit)
        self.menuHelp.addAction(self.actionDocumentation)
        self.menubar.addAction(self.menuMenu.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())
        #-------------------------------------------------------------
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        #===================================================================
        #======================== Signals ==================================
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        #
        self.actionDocumentation.triggered.connect(self.openUrl)
        self.actionExit.triggered.connect(self.closeEvent)
        self.PB_LoadImage.clicked.connect(self.loadImage)
        self.PB_LoadMask.clicked.connect(self.loadMask)
        self.PB_Predict.clicked.connect(self.SegmentProcess)
        self.PB_Plots.clicked.connect(self.plotCurves)
        self.PB_SaveResults.clicked.connect(self.saveSeg)
        self.PB_Reset.clicked.connect(self.reset)
        #==================================================================
        #==================================================================
    def retranslateUi(self, MainWindow):
        #----------------------------------------------------------------------
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "SSI Project"))
        #----------------------------------------------------------------------
        self.LB_ImageTitle.setToolTip(_translate("MainWindow", "Image tilte"))
        self.LB_ImageTitle.setStatusTip(_translate("MainWindow", "Image title"))
        self.LB_ImageTitle.setWhatsThis(_translate("MainWindow", "Image title"))
        self.LB_ImageTitle.setText(_translate("MainWindow", "Image"))
        #--------------------------------------------------------------------------------------------------
        self.LB_MaskTitle.setToolTip(_translate("MainWindow", "Mask title"))
        self.LB_MaskTitle.setStatusTip(_translate("MainWindow", "Mask title"))
        self.LB_MaskTitle.setWhatsThis(_translate("MainWindow", "Mask title"))
        self.LB_MaskTitle.setText(_translate("MainWindow", "Ground Truth"))
        #--------------------------------------------------------------------------------------------------
        self.LB_PredictionTitle.setToolTip(_translate("MainWindow", "segmentation title"))
        self.LB_PredictionTitle.setStatusTip(_translate("MainWindow", "Prediction title"))
        self.LB_PredictionTitle.setWhatsThis(_translate("MainWindow", "Prediction title"))
        self.LB_PredictionTitle.setText(_translate("MainWindow", "Segmentation"))
        #--------------------------------------------------------------------------------------------------
        self.LB_Image.setToolTip(_translate("MainWindow", "Image"))
        self.LB_Image.setStatusTip(_translate("MainWindow", "Image"))
        self.LB_Image.setWhatsThis(_translate("MainWindow", "Image"))
        self.LB_Image.setText(_translate("MainWindow", ""))
        #--------------------------------------------------------------------------------------------------
        self.LB_Mask.setToolTip(_translate("MainWindow", "Ground Truth Mask"))
        self.LB_Mask.setStatusTip(_translate("MainWindow", "Ground Truth Mask"))
        self.LB_Mask.setWhatsThis(_translate("MainWindow", "Ground Truth Mask"))
        self.LB_Mask.setText(_translate("MainWindow", ""))
        #--------------------------------------------------------------------------------------------------
        self.LB_Prediction.setToolTip(_translate("MainWindow", "Prediction"))
        self.LB_Prediction.setStatusTip(_translate("MainWindow", "Prediction"))
        self.LB_Prediction.setWhatsThis(_translate("MainWindow", "Prediction"))
        self.LB_Prediction.setText(_translate("MainWindow", ""))
        #--------------------------------------------------------------------------------------------------
        self.PB_LoadImage.setToolTip(_translate("MainWindow", "Press to load the image in PNG format"))
        self.PB_LoadImage.setStatusTip(_translate("MainWindow", "Press to load the image in PNG format"))
        self.PB_LoadImage.setWhatsThis(_translate("MainWindow", "Image loading Push Button"))
        self.PB_LoadImage.setText(_translate("MainWindow", "Load Image"))
        #--------------------------------------------------------------------------------------------------
        self.PB_LoadMask.setToolTip(_translate("MainWindow", "Press to load the mask in PNG format"))
        self.PB_LoadMask.setStatusTip(_translate("MainWindow", "Press to load the mask in PNG format"))
        self.PB_LoadMask.setWhatsThis(_translate("MainWindow", "Mask Loading Push button"))
        self.PB_LoadMask.setText(_translate("MainWindow", "Load Mask"))
        #--------------------------------------------------------------------------------------------------
        self.PB_Predict.setToolTip(_translate("MainWindow", "Press to do the segmentaion"))
        self.PB_Predict.setStatusTip(_translate("MainWindow", "Press to do the segmentaion"))
        self.PB_Predict.setWhatsThis(_translate("MainWindow", "Processing Push Button"))
        self.PB_Predict.setText(_translate("MainWindow", "Segment"))
        #--------------------------------------------------------------------------------------------------
        self.PB_Plots.setToolTip(_translate("MainWindow", "Press to Draw Statistical Plots"))
        self.PB_Plots.setStatusTip(_translate("MainWindow", "Press to Draw Statistical Plots"))
        self.PB_Plots.setWhatsThis(_translate("MainWindow", "Push button to show figures"))
        self.PB_Plots.setText(_translate("MainWindow", "Show Plots"))
        #--------------------------------------------------------------------------------------------------
        self.PB_SaveResults.setToolTip(_translate("MainWindow", "Press to save results"))
        self.PB_SaveResults.setStatusTip(_translate("MainWindow", "Press to save the results"))
        self.PB_SaveResults.setWhatsThis(_translate("MainWindow", "Puch button to save the results"))
        self.PB_SaveResults.setText(_translate("MainWindow", "Save Results"))
        #--------------------------------------------------------------------------------------------------
        self.PB_Reset.setToolTip(_translate("MainWindow", "Press to Reset"))
        self.PB_Reset.setStatusTip(_translate("MainWindow", "Press to Reset"))
        self.PB_Reset.setWhatsThis(_translate("MainWindow", "Push Button to Reset"))
        self.PB_Reset.setText(_translate("MainWindow", "Reset"))
        #--------------------------------------------------------------------------------------------------
        self.CB_Platform.setToolTip(_translate("MainWindow", "Select Which platform to use"))
        self.CB_Platform.setStatusTip(_translate("MainWindow", "Select Keras or PyTorch to work with"))
        self.CB_Platform.setWhatsThis(_translate("MainWindow", "Platform selection"))
        self.CB_Platform.setItemText(0, _translate("MainWindow", "Keras"))
        self.CB_Platform.setItemText(1, _translate("MainWindow", "PyTorch"))
        #--------------------------------------------------------------------------------------------------
        self.LB_Hausdorff.setToolTip(_translate("MainWindow", "Hausdorff Distance"))
        self.LB_Hausdorff.setStatusTip(_translate("MainWindow", "Hausdorff Distance"))
        self.LB_Hausdorff.setWhatsThis(_translate("MainWindow", "Hausdorff Distance"))
        self.LB_Hausdorff.setText(_translate("MainWindow", "Hausdorff distance"))
        #--------------------------------------------------------------------------------------------------
        self.LB_HausdorffValue.setText(_translate("MainWindow", "0"))
        #--------------------------------------------------------------------------------------------------
        self.LB_Dice.setToolTip(_translate("MainWindow", "Dice Metric"))
        self.LB_Dice.setStatusTip(_translate("MainWindow", "Dice Metric"))
        self.LB_Dice.setWhatsThis(_translate("MainWindow", "Dice Metric"))
        self.LB_Dice.setText(_translate("MainWindow", "Dice Metric"))
        #--------------------------------------------------------------------------------------------------
        self.LB_DiceValue.setText(_translate("MainWindow", "0"))
        #--------------------------------------------------------------------------------------------------
        self.LB_Jaccard.setToolTip(_translate("MainWindow", "Jaccard Metric"))
        self.LB_Jaccard.setStatusTip(_translate("MainWindow", "Jaccard Metric"))
        self.LB_Jaccard.setWhatsThis(_translate("MainWindow", "Jaccard Metric"))
        self.LB_Jaccard.setText(_translate("MainWindow", "Jaccard Metric"))
        #--------------------------------------------------------------------------------------------------
        self.LB_JaccardValue.setText(_translate("MainWindow", "0"))
        #--------------------------------------------------------------------------------------------------        
        self.LB_P_.setToolTip(_translate("MainWindow", "P value"))
        self.LB_P_.setStatusTip(_translate("MainWindow", "P value"))
        self.LB_P_.setWhatsThis(_translate("MainWindow", "P value"))
        self.LB_P_.setText(_translate("MainWindow", "P-Value"))
        #--------------------------------------------------------------------------------------------------        
        self.LB_P_Value.setText(_translate("MainWindow", "0"))
        #--------------------------------------------------------------------------------------------------        
        self.LB_Pearson.setToolTip(_translate("MainWindow", "Pearson Correlation Coefficient"))
        self.LB_Pearson.setStatusTip(_translate("MainWindow", "Pearson Correlation Coefficient"))
        self.LB_Pearson.setWhatsThis(_translate("MainWindow", "Pearson Correlation Coefficient"))
        self.LB_Pearson.setText(_translate("MainWindow", "Pearson correlation\n coefficient"))
        #--------------------------------------------------------------------------------------------------
        self.LB_PearsonValue.setText(_translate("MainWindow", "0"))
        #--------------------------------------------------------------------------------------------------
        self.menuMenu.setTitle(_translate("MainWindow", "Menu"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        #--------------------------------------------------------------------------------------------------
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.actionExit.setStatusTip(_translate("MainWindow", "Press to Exit"))
        self.actionExit.setWhatsThis(_translate("MainWindow", "Exit"))
        self.actionExit.setShortcut(_translate("MainWindow", "Alt+F4"))
        #--------------------------------------------------------------------------------------------------
        self.actionDocumentation.setText(_translate("MainWindow", "Documentation"))
        self.actionDocumentation.setStatusTip(_translate("MainWindow", "Press to load the documentation"))
        self.actionDocumentation.setWhatsThis(_translate("MainWindow", "Help and Documentation"))
        self.actionDocumentation.setShortcut(_translate("MainWindow", "F1"))
        #--------------------------------------------------------------------------------------------------
    #====================
    # Auxiliary Functions
    #====================
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
            self.image     = None
            self.mask      = None
            self.segment   = None
            self.metrics   = {}
            #-----------------------------------
            mask_arr = np.zeros((128,128))
            qimage = QImage(mask_arr, mask_arr.shape[0],mask_arr.shape[1],QtGui.QImage.Format_RGB32)
            pix = QPixmap(qimage)
            mask = pix.createMaskFromColor(QColor(255, 255, 255), Qt.MaskOutColor)
            #-----------------------------------
            self.LB_Image.setPixmap(mask)
            self.LB_Image.setScaledContents(True)            
            self.LB_Mask.setPixmap(mask)
            self.LB_Mask.setScaledContents(True) 
            self.LB_Prediction.setPixmap(mask)
            self.LB_Prediction.setScaledContents(True)
            #-----------------------------------
            # Hausdorff
            self.LB_HausdorffValue.setText("0")
            # Dice
            self.LB_DiceValue.setText("0")
            # Jaccard
            self.LB_JaccardValue.setText("0")
            # P 
            self.LB_P_Value.setText("0")
            # Pearson Corellation Coefficient
            self.LB_PearsonValue.setText("0")
            # Reset Figures 
            if(os.path.isfile("segmentPyTorch.png")):
                os.remove("segmentPyTorch.png")
            if(os.path.isfile("BlandAltman.png")):
                os.remove("BlandAltman.png")
            if(os.path.isfile("ROC.png")):
                os.remove("ROC.png")
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
            if(os.path.isfile("segmentPyTorch.png")):
                os.remove("segmentPyTorch.png")
            if(os.path.isfile("BlandAltman.png")):
                os.remove("BlandAltman.png")
            if(os.path.isfile("ROC.png")):
                os.remove("ROC.png")

            app.quit()

        elif reply == QMessageBox.Save:
            if(self.segment is not None):
                self.saveSeg()

            if(os.path.isfile("segmentPyTorch.png")):
                os.remove("segmentPyTorch.png")
            if(os.path.isfile("BlandAltman.png")):
                os.remove("BlandAltman.png")
            if(os.path.isfile("ROC.png")):
                os.remove("ROC.png")

            app.quit()
        else:
            QCoreApplication.instance().quit
    #---------------------------------------------------------------------------------------------
    def saveSeg(self):
        if(self.directory is not None):
            path = QFileDialog.getExistingDirectory(None,'Select file')
            imsave(os.path.join(path,'segmentaion.png'), (255*self.segment[...,1:4]).astype(np.uint8))
    #---------------------------------------------------------------------------------------------
    def loadImage(self):
        self.dialog = QFileDialog()
        self.directory, _ = QFileDialog.getOpenFileName(None, "Select File", filter='*.png')
        if os.path.isfile(str(self.directory)):
           self.image = imread(self.directory)
           qimg = QImage(self.image,self.image.shape[0],self.image.shape[1],QImage.Format_Grayscale8)
           qmap = QPixmap(qimg)
           self.LB_Image.setPixmap(qmap)
           self.LB_Image.setScaledContents(True)
    #---------------------------------------------------------------------------------------------
    def loadMask(self):
        self.dialog = QFileDialog()
        self.directory, _ = QFileDialog.getOpenFileName(None, "Select File", filter='*.png')
        if os.path.isfile(str(self.directory)):
            self.mask = imread(self.directory)
            mask = (255*self.mask[...,1:4]).astype(np.uint8)
            qimg = QImage(mask,mask.shape[0],mask.shape[1],QImage.Format_RGB888) 
            qmap = QPixmap(qimg)
            self.LB_Mask.setPixmap(qmap)
            self.LB_Mask.setScaledContents(True)   
    #---------------------------------------------------------------------------------------------
    def getMetrics(self):
        # Hausdorff
        self.metrics['Hausdorff'] = hd(self.segment, self.mask)
        self.LB_HausdorffValue.setText(str(round(self.metrics['Hausdorff'],3))+" Pixels")
        # Dice
        self.metrics['Dice']      = 100*dc(self.segment, self.mask)
        self.LB_DiceValue.setText(str(round(self.metrics['Dice'],3))+" %")
        # Jaccard
        self.metrics['Jaccard']   = 100*jc(self.segment, self.mask)
        self.LB_JaccardValue.setText(str(round(self.metrics['Jaccard'],3))+" %")
        # P 
        self.metrics['P_Value']   = volume_change_correlation(self.segment, self.mask)[1]
        self.LB_P_Value.setText(str(round(self.metrics['P_Value'],3)))
        # Pearson Corellation Coefficient
        self.metrics['Pearson']   = volume_change_correlation(self.segment, self.mask)[0]
        self.LB_PearsonValue.setText(str(round(self.metrics['Pearson'],3)))
    #---------------------------------------------------------------------------------------------
    def SegmentProcess(self):
        # Get the Name of the selected platform
        platform = self.CB_Platform.currentText()
        # Keras
        if(platform == 'Keras'):
            # Check Image
            if(self.image is not None and self.image.shape[0]==256):
                # Check Mask
                if(self.mask is not None and self.mask.shape[0]==256):
                    self.segment = KerasPredict(self.image)
                    self.getMetrics()
                    # Show Segmentation
                    segment = (255*self.segment[...,1:4]).astype(np.uint8)
                    qimg = QImage(segment,segment.shape[0],segment.shape[1],QImage.Format_RGB888) 
                    qmap = QPixmap(qimg)
                    self.LB_Prediction.setPixmap(qmap)
                    self.LB_Prediction.setScaledContents(True) 

                else:
                    Q = QWidget()
                    message = QMessageBox.warning(Q, "About","Please Load the Mask also to calculate Metrics!", QMessageBox.Ok)

            else:
                Q = QWidget()
                message = QMessageBox.warning(Q, "About","Please Load the Image to Segment!", QMessageBox.Ok)

        # PyTorch
        elif(platform == 'PyTorch'):
            # Check Image
            if(self.image is not None): 
                # Check Mask
                if(self.mask is not None):
                    # Deng Prediction Interface function to be called here
                    #-----------------------------------------------------
                    self.image   = crop2D_mask(self.image, (224,224))
                    self.mask    = crop2D_mask(self.mask,  (224,224))
                    self.segment = eval(self.image)
                    #-----------------------------------------------------
                    self.getMetrics()
                    # Show Segmentation
                    segment = (255*self.segment[...,1:4]).astype(np.uint8)
                    imsave('segmentPyTorch.png', segment)
                    # qimg = QImage(segment, segment.shape[0], segment.shape[1], QImage.Format_RGB888) 
                    qmap = QPixmap('segmentPyTorch.png')
                    self.LB_Prediction.setPixmap(qmap)
                    self.LB_Prediction.setScaledContents(True)

                    if(os.path.isfile("segmentPyTorch.png")):
                        os.remove("segmentPyTorch.png")
                else:
                    Q = QWidget()
                    message = QMessageBox.warning(Q, "About","Please Load the Mask also to calculate Metrics!", QMessageBox.Ok)

            else:
                Q = QWidget()
                message = QMessageBox.warning(Q, "About","Please Load the Image to Segment!", QMessageBox.Ok)   
    #---------------------------------------------------------------------------------------------
    def plotCurves(self):
        if((self.segment is not None) and (self.mask is not None)):

            if(os.path.isfile("BlandAltman.png")):
                os.remove("BlandAltman.png")
            if(os.path.isfile("ROC.png")):
                os.remove("ROC.png")

            bland_altman_plot(self.mask, self.segment)
            pred = roc_preprocess(self.segment)
            draw_roc_curve(self.mask, pred)
            if (os.path.isfile(str('BlandAltman.png')) and os.path.isfile(str('ROC.png'))):
                Dialog = QtWidgets.QDialog()
                ui = PlotsWindow()
                ui.setupUi(Dialog)
                Dialog.show() 
                Dialog.exec_()
        else:
            Q = QWidget()
            message = QMessageBox.warning(Q, "About","Please Do Segmentation first!", QMessageBox.Ok)  
#==========================================================================================================
#======================================================
#                    Main Function
#======================================================
if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    screenShape = QtWidgets.QDesktopWidget().screenGeometry()
    MainWindow.resize(screenShape.width(), screenShape.height())
    MainWindow.show()
    sys.exit(app.exec_())
