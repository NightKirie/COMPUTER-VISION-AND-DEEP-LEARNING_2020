# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import Q1 as q1
import Q2 as q2
import Q3 as q3
import Q4 as q4
import Q5 as q5


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(800, 515)
        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setGeometry(QtCore.QRect(20, 10, 361, 81))
        self.groupBox.setObjectName("groupBox")
        self.btn_1_1 = QtWidgets.QPushButton(self.groupBox)
        self.btn_1_1.setGeometry(QtCore.QRect(50, 30, 251, 23))
        self.btn_1_1.setObjectName("btn_1_1")
        self.btn_1_1.clicked.connect(q1.Background_Subtraction)
        self.groupBox_2 = QtWidgets.QGroupBox(Form)
        self.groupBox_2.setGeometry(QtCore.QRect(20, 130, 361, 121))
        self.groupBox_2.setObjectName("groupBox_2")
        self.btn_2_1 = QtWidgets.QPushButton(self.groupBox_2)
        self.btn_2_1.setGeometry(QtCore.QRect(50, 30, 251, 23))
        self.btn_2_1.setObjectName("btn_2_1")
        self.btn_2_1.clicked.connect(q2.Preprocessing)
        self.btn_2_2 = QtWidgets.QPushButton(self.groupBox_2)
        self.btn_2_2.setGeometry(QtCore.QRect(50, 70, 251, 23))
        self.btn_2_2.setObjectName("btn_2_2")
        self.btn_2_2.clicked.connect(q2.Video_Tracking)
        self.groupBox_3 = QtWidgets.QGroupBox(Form)
        self.groupBox_3.setGeometry(QtCore.QRect(20, 270, 361, 81))
        self.groupBox_3.setObjectName("groupBox_3")
        self.btn_3_1 = QtWidgets.QPushButton(self.groupBox_3)
        self.btn_3_1.setGeometry(QtCore.QRect(50, 30, 251, 23))
        self.btn_3_1.setObjectName("btn_3_1")
        self.btn_3_1.clicked.connect(q3.Perspective_Transform)
        self.groupBox_4 = QtWidgets.QGroupBox(Form)
        self.groupBox_4.setGeometry(QtCore.QRect(20, 380, 361, 111))
        self.groupBox_4.setObjectName("groupBox_4")
        self.btn_4_1 = QtWidgets.QPushButton(self.groupBox_4)
        self.btn_4_1.setGeometry(QtCore.QRect(50, 30, 251, 23))
        self.btn_4_1.setObjectName("btn_4_1")
        self.btn_4_1.clicked.connect(q4.Image_Reconstruction)
        self.btn_4_2 = QtWidgets.QPushButton(self.groupBox_4)
        self.btn_4_2.setGeometry(QtCore.QRect(50, 70, 251, 23))
        self.btn_4_2.setObjectName("btn_4_2")
        self.btn_4_2.clicked.connect(q4.Compile_Error)
        self.groupBox_5 = QtWidgets.QGroupBox(Form)
        self.groupBox_5.setGeometry(QtCore.QRect(400, 10, 361, 450))
        self.groupBox_5.setObjectName("groupBox_5")
        self.btn_5_1 = QtWidgets.QPushButton(self.groupBox_5)
        self.btn_5_1.setGeometry(QtCore.QRect(50, 30, 251, 23))
        self.btn_5_1.setObjectName("btn_5_1")
        self.btn_5_1.clicked.connect(q5.Show_Train_Result)
        self.btn_5_2 = QtWidgets.QPushButton(self.groupBox_5)
        self.btn_5_2.setGeometry(QtCore.QRect(50, 130, 251, 23))
        self.btn_5_2.setObjectName("btn_5_2")
        self.btn_5_2.clicked.connect(q5.Show_Tensorboard)
        self.btn_5_3 = QtWidgets.QPushButton(self.groupBox_5)
        self.btn_5_3.setGeometry(QtCore.QRect(50, 230, 251, 23))
        self.btn_5_3.setObjectName("btn_5_3")
        self.btn_5_3.clicked.connect(q5.Classify_Random_Picture)
        self.btn_5_4 = QtWidgets.QPushButton(self.groupBox_5)
        self.btn_5_4.setGeometry(QtCore.QRect(50, 330, 251, 23))
        self.btn_5_4.setObjectName("btn_5_4")
        self.btn_5_4.clicked.connect(q5.Show_Acc_Diff)


        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.groupBox.setTitle(_translate("Form", "1. Background Subtraction"))
        self.btn_1_1.setText(_translate("Form", "1.1 Background Subtraction"))
        self.groupBox_2.setTitle(_translate("Form", "2. Optical Flow"))
        self.btn_2_1.setText(_translate("Form", "2.1 Preprocessing"))
        self.btn_2_2.setText(_translate("Form", "2.2 Video Tracking"))
        self.groupBox_3.setTitle(_translate("Form", "3. Perspective Transform"))
        self.btn_3_1.setText(_translate("Form", "3.1 Perspective Transform"))
        self.groupBox_4.setTitle(_translate("Form", "4. PCA"))
        self.btn_4_1.setText(_translate("Form", "4.1 Image Reconstruction"))
        self.btn_4_2.setText(_translate("Form", "4.2 Compute the Reconstruction Error"))
        self.groupBox_5.setTitle(_translate("Form", "5. Dogs and Cats classification Using ResNet50 "))
        self.btn_5_1.setText(_translate("Form", "5.1 Show Train Result"))
        self.btn_5_2.setText(_translate("Form", "5.2 Show TensorBoard"))
        self.btn_5_3.setText(_translate("Form", "5.3 Classify random picture"))
        self.btn_5_4.setText(_translate("Form", "5.4 Show Accuracy Comparison between models"))

