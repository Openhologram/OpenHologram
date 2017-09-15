/********************************************************************************
** Form generated from reading UI file 'hologramdepthmap.ui'
**
** Created by: Qt User Interface Compiler version 5.6.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_HOLOGRAMDEPTHMAP_H
#define UI_HOLOGRAMDEPTHMAP_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_HologramDepthmapClass
{
public:
    QAction *actionTest;
    QWidget *centralWidget;
    QPushButton *pbGen;
    QPushButton *pbRecon;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *HologramDepthmapClass)
    {
        if (HologramDepthmapClass->objectName().isEmpty())
            HologramDepthmapClass->setObjectName(QStringLiteral("HologramDepthmapClass"));
        HologramDepthmapClass->resize(600, 400);
        actionTest = new QAction(HologramDepthmapClass);
        actionTest->setObjectName(QStringLiteral("actionTest"));
        centralWidget = new QWidget(HologramDepthmapClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        pbGen = new QPushButton(centralWidget);
        pbGen->setObjectName(QStringLiteral("pbGen"));
        pbGen->setGeometry(QRect(180, 30, 100, 100));
        QIcon icon;
        icon.addFile(QStringLiteral(":/HologramDepthmap/Resources/hologram_generation.png"), QSize(), QIcon::Normal, QIcon::Off);
        pbGen->setIcon(icon);
        pbGen->setIconSize(QSize(100, 100));
        pbRecon = new QPushButton(centralWidget);
        pbRecon->setObjectName(QStringLiteral("pbRecon"));
        pbRecon->setGeometry(QRect(310, 30, 100, 100));
        QIcon icon1;
        icon1.addFile(QStringLiteral(":/HologramDepthmap/Resources/hologram_reconstruction.png"), QSize(), QIcon::Normal, QIcon::Off);
        pbRecon->setIcon(icon1);
        pbRecon->setIconSize(QSize(100, 100));
        HologramDepthmapClass->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(HologramDepthmapClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 600, 21));
        HologramDepthmapClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(HologramDepthmapClass);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        HologramDepthmapClass->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(HologramDepthmapClass);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        HologramDepthmapClass->setStatusBar(statusBar);

        retranslateUi(HologramDepthmapClass);

        QMetaObject::connectSlotsByName(HologramDepthmapClass);
    } // setupUi

    void retranslateUi(QMainWindow *HologramDepthmapClass)
    {
        HologramDepthmapClass->setWindowTitle(QApplication::translate("HologramDepthmapClass", "Hologram Generator", 0));
        actionTest->setText(QApplication::translate("HologramDepthmapClass", "test", 0));
#ifndef QT_NO_TOOLTIP
        pbGen->setToolTip(QApplication::translate("HologramDepthmapClass", "Generate Hologram", 0));
#endif // QT_NO_TOOLTIP
        pbGen->setText(QString());
#ifndef QT_NO_TOOLTIP
        pbRecon->setToolTip(QApplication::translate("HologramDepthmapClass", "Reconstruct Image", 0));
#endif // QT_NO_TOOLTIP
        pbRecon->setText(QString());
    } // retranslateUi

};

namespace Ui {
    class HologramDepthmapClass: public Ui_HologramDepthmapClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_HOLOGRAMDEPTHMAP_H
