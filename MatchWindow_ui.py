# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MatchWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(580, 382)
        self.HomeComboBox = QtWidgets.QComboBox(Dialog)
        self.HomeComboBox.setGeometry(QtCore.QRect(160, 20, 391, 31))
        self.HomeComboBox.setObjectName("HomeComboBox")
        self.AwayComboBox = QtWidgets.QComboBox(Dialog)
        self.AwayComboBox.setGeometry(QtCore.QRect(160, 90, 391, 31))
        self.AwayComboBox.setObjectName("AwayComboBox")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(40, 10, 111, 41))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(90, 90, 56, 21))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(70, 60, 191, 16))
        self.label_3.setObjectName("label_3")
        self.HRedComboBox = QtWidgets.QComboBox(Dialog)
        self.HRedComboBox.setGeometry(QtCore.QRect(260, 60, 51, 22))
        self.HRedComboBox.setObjectName("HRedComboBox")
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(320, 60, 171, 16))
        self.label_4.setObjectName("label_4")
        self.HInjurComboBox = QtWidgets.QComboBox(Dialog)
        self.HInjurComboBox.setGeometry(QtCore.QRect(500, 60, 51, 22))
        self.HInjurComboBox.setObjectName("HInjurComboBox")
        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_5.setGeometry(QtCore.QRect(320, 130, 171, 16))
        self.label_5.setObjectName("label_5")
        self.AInjurComboBox = QtWidgets.QComboBox(Dialog)
        self.AInjurComboBox.setGeometry(QtCore.QRect(500, 130, 51, 22))
        self.AInjurComboBox.setObjectName("AInjurComboBox")
        self.label_6 = QtWidgets.QLabel(Dialog)
        self.label_6.setGeometry(QtCore.QRect(70, 130, 191, 16))
        self.label_6.setObjectName("label_6")
        self.ARedComboBox = QtWidgets.QComboBox(Dialog)
        self.ARedComboBox.setGeometry(QtCore.QRect(260, 130, 51, 22))
        self.ARedComboBox.setObjectName("ARedComboBox")
        self.SimulateButton = QtWidgets.QPushButton(Dialog)
        self.SimulateButton.setGeometry(QtCore.QRect(40, 170, 511, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.SimulateButton.setFont(font)
        self.SimulateButton.setObjectName("SimulateButton")
        self.SaveButton = QtWidgets.QPushButton(Dialog)
        self.SaveButton.setGeometry(QtCore.QRect(40, 310, 511, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.SaveButton.setFont(font)
        self.SaveButton.setObjectName("SaveButton")
        self.tableWidget = QtWidgets.QTableWidget(Dialog)
        self.tableWidget.setGeometry(QtCore.QRect(40, 220, 511, 81))
        font = QtGui.QFont()
        font.setPointSize(7)
        self.tableWidget.setFont(font)
        self.tableWidget.setTabKeyNavigation(True)
        self.tableWidget.setCornerButtonEnabled(False)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        self.backButton = QtWidgets.QPushButton(Dialog)
        self.backButton.setGeometry(QtCore.QRect(40, 340, 511, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.backButton.setFont(font)
        self.backButton.setObjectName("backButton")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Symulacja Meczu"))
        self.label.setText(_translate("Dialog", "Gospodarz:"))
        self.label_2.setText(_translate("Dialog", "Gość:"))
        self.label_3.setText(_translate("Dialog", "Ilość graczy z czerwoną kartką:"))
        self.label_4.setText(_translate("Dialog", "Ilość kontuzjowanych graczy:"))
        self.label_5.setText(_translate("Dialog", "Ilość kontuzjowanych graczy:"))
        self.label_6.setText(_translate("Dialog", "Ilość graczy z czerwoną kartką:"))
        self.SimulateButton.setText(_translate("Dialog", "Symulacja wyników meczu"))
        self.SaveButton.setText(_translate("Dialog", "Zapisz wynik"))
        self.backButton.setText(_translate("Dialog", "Powrót"))