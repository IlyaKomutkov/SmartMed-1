# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'startingWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class StartingWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(600, 480)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(70, 30, 463, 41))
        self.label.setObjectName("label")
        self.pushButtonDone = QtWidgets.QPushButton(self.centralwidget)
        self.pushButtonDone.setGeometry(QtCore.QRect(450, 420, 121, 32))
        self.pushButtonDone.setObjectName("pushButtonDone")
        self.pushButtonStat = QtWidgets.QPushButton(self.centralwidget)
        self.pushButtonStat.setGeometry(QtCore.QRect(60, 100, 201, 32))
        self.pushButtonStat.setObjectName("pushButtonStat")
        self.pushButtonBioeq = QtWidgets.QPushButton(self.centralwidget)
        self.pushButtonBioeq.setGeometry(QtCore.QRect(60, 220, 201, 32))
        self.pushButtonBioeq.setObjectName("pushButtonBioeq")
        self.pushButtonPred = QtWidgets.QPushButton(self.centralwidget)
        self.pushButtonPred.setGeometry(QtCore.QRect(60, 330, 201, 32))
        self.pushButtonPred.setObjectName("pushButtonPred")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(340, 310, 381, 91))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(340, 180, 381, 111))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(340, 80, 381, 81))
        self.label_4.setObjectName("label_4")
        self.label.raise_()
        self.pushButtonDone.raise_()
        self.pushButtonBioeq.raise_()
        self.pushButtonPred.raise_()
        self.pushButtonStat.raise_()
        self.label_2.raise_()
        self.label_3.raise_()
        self.label_4.raise_()
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:18pt;\">Выберите способ анализа</span></p><p><span style=\" font-size:18pt;\"><br/></span></p></body></html>"))
        self.pushButtonDone.setText(_translate("MainWindow", "Завершить"))
        self.pushButtonStat.setText(_translate("MainWindow", "Описательный анализ"))
        self.pushButtonBioeq.setText(_translate("MainWindow", "Биоэквивалентность"))
        self.pushButtonPred.setText(_translate("MainWindow", "Предсказательные модели"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p align=\"justify\">Построение статистических </p><p align=\"justify\">и предсказательных моделей, </p><p align=\"justify\">ROC-анализ</p></body></html>"))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p align=\"justify\">Исследование идентичности</p><p align=\"justify\">свойств биодоступности </p><p align=\"justify\">у исходного препарата </p><p align=\"justify\">и дженерика</p></body></html>"))
        self.label_4.setText(_translate("MainWindow", "<html><head/><body><p align=\"justify\">Получение обобщенной </p><p align=\"justify\">информации о данных, </p><p align=\"justify\">визуальный анализ</p><p align=\"justify\"><br/></p></body></html>"))
