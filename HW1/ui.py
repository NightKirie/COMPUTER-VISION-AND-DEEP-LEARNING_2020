from PyQt5 import QtCore, QtGui, QtWidgets
from q1 import *
from q2 import *
from q3 import *
from q4 import *
from q5 import *

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1059, 659)
        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setGeometry(QtCore.QRect(30, 30, 581, 271))
        self.groupBox.setObjectName("groupBox")
        self.find_corners_btn = QtWidgets.QPushButton(self.groupBox)
        self.find_corners_btn.setGeometry(QtCore.QRect(80, 40, 131, 31))
        self.find_corners_btn.setObjectName("find_corners_btn")
        self.find_corners_btn.clicked.connect(findCorners)
        self.find_intrinsic_btn = QtWidgets.QPushButton(self.groupBox)
        self.find_intrinsic_btn.setGeometry(QtCore.QRect(80, 90, 131, 31))
        self.find_intrinsic_btn.setObjectName("find_intrinsic_btn")
        self.find_intrinsic_btn.clicked.connect(findInstrinsic)
        self.find_distortion_btn = QtWidgets.QPushButton(self.groupBox)
        self.find_distortion_btn.setGeometry(QtCore.QRect(80, 140, 131, 31))
        self.find_distortion_btn.setObjectName("find_distortion_btn")
        self.find_distortion_btn.clicked.connect(findDistorsion)
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_2.setGeometry(QtCore.QRect(270, 40, 251, 131))
        self.groupBox_2.setObjectName("groupBox_2")
        self.label = QtWidgets.QLabel(self.groupBox_2)
        self.label.setGeometry(QtCore.QRect(20, 20, 81, 31))
        self.label.setObjectName("label")
        self.find_extrinsic_combobox = QtWidgets.QComboBox(self.groupBox_2)
        self.find_extrinsic_combobox.setGeometry(QtCore.QRect(20, 50, 111, 22))
        self.find_extrinsic_combobox.setObjectName("find_extrinsic_combobox")
        self.find_extrinsic_combobox.addItem("")
        self.find_extrinsic_combobox.addItem("")
        self.find_extrinsic_combobox.addItem("")
        self.find_extrinsic_combobox.addItem("")
        self.find_extrinsic_combobox.addItem("")
        self.find_extrinsic_combobox.addItem("")
        self.find_extrinsic_combobox.addItem("")
        self.find_extrinsic_combobox.addItem("")
        self.find_extrinsic_combobox.addItem("")
        self.find_extrinsic_combobox.addItem("")
        self.find_extrinsic_combobox.addItem("")
        self.find_extrinsic_combobox.addItem("")
        self.find_extrinsic_combobox.addItem("")
        self.find_extrinsic_combobox.addItem("")
        self.find_extrinsic_combobox.addItem("")
        self.find_extrinsic_btn = QtWidgets.QPushButton(self.groupBox_2)
        self.find_extrinsic_btn.setGeometry(QtCore.QRect(20, 80, 131, 31))
        self.find_extrinsic_btn.setObjectName("find_extrinsic_btn")
        self.find_extrinsic_btn.clicked.connect(lambda: findExtrinsic(self.find_extrinsic_combobox.currentText()))
        self.groupBox_3 = QtWidgets.QGroupBox(Form)
        self.groupBox_3.setGeometry(QtCore.QRect(650, 30, 321, 81))
        self.groupBox_3.setObjectName("groupBox_3")
        self.augmented_reality_btn = QtWidgets.QPushButton(self.groupBox_3)
        self.augmented_reality_btn.setGeometry(QtCore.QRect(80, 30, 151, 31))
        self.augmented_reality_btn.setObjectName("augmented_reality_btn")
        self.augmented_reality_btn.clicked.connect(augmentedReality)
        self.groupBox_4 = QtWidgets.QGroupBox(Form)
        self.groupBox_4.setGeometry(QtCore.QRect(650, 120, 321, 81))
        self.groupBox_4.setObjectName("groupBox_4")
        self.stereo_disparity_map_btn = QtWidgets.QPushButton(self.groupBox_4)
        self.stereo_disparity_map_btn.setGeometry(QtCore.QRect(80, 30, 151, 31))
        self.stereo_disparity_map_btn.setObjectName("stereo_disparity_map_btn")
        self.stereo_disparity_map_btn.clicked.connect(disparityMap)
        self.groupBox_5 = QtWidgets.QGroupBox(Form)
        self.groupBox_5.setGeometry(QtCore.QRect(650, 220, 321, 120))
        self.groupBox_5.setObjectName("groupBox_5")
        self.keypoint_btn = QtWidgets.QPushButton(self.groupBox_5)
        self.keypoint_btn.setGeometry(QtCore.QRect(80, 30, 151, 31))
        self.keypoint_btn.setObjectName("keypoint_btn")
        self.keypoint_btn.clicked.connect(createKeyPoint)
        self.matched_keypoint_btn = QtWidgets.QPushButton(self.groupBox_5)
        self.matched_keypoint_btn.setGeometry(QtCore.QRect(80, 70, 151, 31))
        self.matched_keypoint_btn.setObjectName("keypoint_btn")
        self.matched_keypoint_btn.clicked.connect(matchedKeyPoint)
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
        self.groupBox.setTitle(_translate("Form", "1. Calibration"))
        self.find_corners_btn.setText(_translate("Form", "1.1 Find Corners"))
        self.find_intrinsic_btn.setText(_translate("Form", "1.2 Find Intrinsic"))
        self.find_distortion_btn.setText(_translate("Form", "1.4 Find Distortion"))
        self.groupBox_2.setTitle(_translate("Form", "1.3 Find Extrinsic"))
        self.label.setText(_translate("Form", "Select Image"))
        self.find_extrinsic_combobox.setItemText(0, _translate("Form", "1"))
        self.find_extrinsic_combobox.setItemText(1, _translate("Form", "2"))
        self.find_extrinsic_combobox.setItemText(2, _translate("Form", "3"))
        self.find_extrinsic_combobox.setItemText(3, _translate("Form", "4"))
        self.find_extrinsic_combobox.setItemText(4, _translate("Form", "5"))
        self.find_extrinsic_combobox.setItemText(5, _translate("Form", "6"))
        self.find_extrinsic_combobox.setItemText(6, _translate("Form", "7"))
        self.find_extrinsic_combobox.setItemText(7, _translate("Form", "8"))
        self.find_extrinsic_combobox.setItemText(8, _translate("Form", "9"))
        self.find_extrinsic_combobox.setItemText(9, _translate("Form", "10"))
        self.find_extrinsic_combobox.setItemText(10, _translate("Form", "11"))
        self.find_extrinsic_combobox.setItemText(11, _translate("Form", "12"))
        self.find_extrinsic_combobox.setItemText(12, _translate("Form", "13"))
        self.find_extrinsic_combobox.setItemText(13, _translate("Form", "14"))
        self.find_extrinsic_combobox.setItemText(14, _translate("Form", "15"))
        self.find_extrinsic_btn.setText(_translate("Form", "1.3 Find Extrinsic"))
        self.groupBox_3.setTitle(_translate("Form", "2. Augmented Reality"))
        self.augmented_reality_btn.setText(_translate("Form", "2.1 Augmented Reality"))
        self.groupBox_4.setTitle(_translate("Form", "3. Stereo Disparity Map"))
        self.stereo_disparity_map_btn.setText(_translate("Form", "3.1 Stereo Disparity Map"))
        self.groupBox_5.setTitle(_translate("Form", "4. SIFT"))
        self.keypoint_btn.setText(_translate("Form", "4.1 Keypoints"))
        self.matched_keypoint_btn.setText(_translate("Form", "4.2 Matched Keypoints"))
        self.groupBox_6.setTitle(_translate("Form", "5. Training Cifar10 Classifire Using VGG16"))
        self.show_train_images_btn.setText(_translate("Form", "1. Show Train Images"))
        self.show_hyperparameters_btn.setText(_translate("Form", "2. Show Hyperparameters"))
        self.show_model_structure_btn.setText(_translate("Form", "3. Show Model Structure"))
        self.show_accuracy_btn.setText(_translate("Form", "4. Show Accuracy"))
        self.test_btn.setText(_translate("Form", "5. Test"))

