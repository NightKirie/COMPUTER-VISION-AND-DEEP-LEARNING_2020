from PyQt5 import QtCore, QtGui, QtWidgets
from q5 import *

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1059, 659)
        self.groupBox_6 = QtWidgets.QGroupBox(Form)
        self.groupBox_6.setGeometry(QtCore.QRect(40, 340, 931, 251))
        self.groupBox_6.setObjectName("groupBox_6")
        self.show_train_images_btn = QtWidgets.QPushButton(self.groupBox_6)
        self.show_train_images_btn.setGeometry(QtCore.QRect(90, 50, 141, 31))
        self.show_train_images_btn.setObjectName("show_train_images_btn")
        self.show_train_images_btn.clicked.connect(show_train_image)
        self.show_hyperparameters_btn = QtWidgets.QPushButton(self.groupBox_6)
        self.show_hyperparameters_btn.setGeometry(QtCore.QRect(90, 150, 141, 31))
        self.show_hyperparameters_btn.setObjectName("show_hyperparameters_btn")
        self.show_hyperparameters_btn.clicked.connect(show_hyperparameters)
        self.show_model_structure_btn = QtWidgets.QPushButton(self.groupBox_6)
        self.show_model_structure_btn.setGeometry(QtCore.QRect(360, 50, 141, 31))
        self.show_model_structure_btn.setObjectName("show_model_structure_btn")
        self.show_model_structure_btn.clicked.connect(show_model_structure)
        self.show_accuracy_btn = QtWidgets.QPushButton(self.groupBox_6)
        self.show_accuracy_btn.setGeometry(QtCore.QRect(360, 150, 141, 31))
        self.show_accuracy_btn.setObjectName("show_accuracy_btn")
        self.show_accuracy_btn.clicked.connect(show_accuracy)
        self.spinBox = QtWidgets.QSpinBox(self.groupBox_6)
        self.spinBox.setGeometry(QtCore.QRect(670, 60, 121, 22))
        self.spinBox.setMaximum(9999)
        self.spinBox.setObjectName("spinBox")
        self.test_btn = QtWidgets.QPushButton(self.groupBox_6)
        self.test_btn.setGeometry(QtCore.QRect(660, 150, 141, 31))
        self.test_btn.setObjectName("test_btn")
        self.test_btn.clicked.connect(lambda: test(self.spinBox.value()))

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.groupBox_6.setTitle(_translate("Form", "5. Training Cifar10 Classifire Using VGG16"))
        self.show_train_images_btn.setText(_translate("Form", "1. Show Train Images"))
        self.show_hyperparameters_btn.setText(_translate("Form", "2. Show Hyperparameters"))
        self.show_model_structure_btn.setText(_translate("Form", "3. Show Model Structure"))
        self.show_accuracy_btn.setText(_translate("Form", "4. Show Accuracy"))
        self.test_btn.setText(_translate("Form", "5. Test"))

