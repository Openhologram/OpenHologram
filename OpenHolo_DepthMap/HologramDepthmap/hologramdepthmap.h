#ifndef HOLOGRAMDEPTHMAP_H
#define HOLOGRAMDEPTHMAP_H

#include <QtWidgets/QMainWindow>
#include "ui_hologramdepthmap.h"
#include "Hologram/HologramGenerator.h"
#include "graphics/sys.h"

/**
* @brief Test class for executing the sample program, which shows how to use a hologram library.
* @details The sample program has a main window form and the user can choose the execution type - CPU and GPU.
*/
class HologramDepthmap : public QMainWindow
{
	Q_OBJECT

public:
	HologramDepthmap(QWidget *parent = 0);
	~HologramDepthmap();

private slots:

	void GenHologram();
	void ReconImage();


private:
	Ui::HologramDepthmapClass ui;

	HologramGenerator* hologram_;

};

#endif // HOLOGRAMDEPTHMAP_H
