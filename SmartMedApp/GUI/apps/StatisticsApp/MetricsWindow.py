# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MetricsWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class MetricsWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(600, 480)
        MainWindow.setToolTipDuration(5)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButtonNext = QtWidgets.QPushButton(self.centralwidget)
        self.pushButtonNext.setGeometry(QtCore.QRect(460, 420, 113, 32))
        self.pushButtonNext.setToolTipDuration(10)
        self.pushButtonNext.setObjectName("pushButtonNext")
        self.pushButtonBack = QtWidgets.QPushButton(self.centralwidget)
        self.pushButtonBack.setGeometry(QtCore.QRect(330, 420, 113, 32))
        self.pushButtonBack.setToolTipDuration(10)
        self.pushButtonBack.setObjectName("pushButtonBack")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(70, 30, 291, 31))
        self.label.setObjectName("label")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(120, 90, 341, 281))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.checkBoxCount = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.checkBoxCount.setObjectName("checkBoxCount")
        self.verticalLayout.addWidget(self.checkBoxCount)
        self.checkBoxMean = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.checkBoxMean.setObjectName("checkBoxMean")
        self.verticalLayout.addWidget(self.checkBoxMean)
        self.checkBoxStd = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.checkBoxStd.setObjectName("checkBoxStd")
        self.verticalLayout.addWidget(self.checkBoxStd)
        self.checkBoxMax = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.checkBoxMax.setObjectName("checkBoxMax")
        self.verticalLayout.addWidget(self.checkBoxMax)
        self.checkBoxMin = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.checkBoxMin.setObjectName("checkBoxMin")
        self.verticalLayout.addWidget(self.checkBoxMin)
        self.checkBoxQuants = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.checkBoxQuants.setObjectName("checkBoxQuants")
        self.verticalLayout.addWidget(self.checkBoxQuants)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButtonNext.setText(_translate("MainWindow", "Вперед"))
        self.pushButtonBack.setText(_translate("MainWindow", "Назад"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:18pt;\">Выбор статистических метрик</span></p></body></html>"))
        self.checkBoxCount.setText(_translate("MainWindow", "  Объем выборки"))
        self.checkBoxMean.setText(_translate("MainWindow", "  Среднее по столбцу"))
        self.checkBoxStd.setToolTip(_translate("MainWindow", "<html><head/><body><p>Показывает разброс значений в столбце</p></body></html>"))
        self.checkBoxStd.setText(_translate("MainWindow", "  Стандартное отклонение по столбцу"))
        self.checkBoxMax.setText(_translate("MainWindow", "  Максимальное значение в столбце  "))
        self.checkBoxMin.setText(_translate("MainWindow", "  Минимальное значение в столбце"))
        self.checkBoxQuants.setToolTip(_translate("MainWindow", "<html><head/><body><p>По этим значениям можно оценить распределение выборки</p><p><br/></p></body></html>"))
        self.checkBoxQuants.setText(_translate("MainWindow", "  Квантили 25, 50, 75"))
