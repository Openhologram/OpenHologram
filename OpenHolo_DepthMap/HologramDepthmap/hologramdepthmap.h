#ifndef HOLOGRAMDEPTHMAP_H
#define HOLOGRAMDEPTHMAP_H

#include <QtWidgets/QMainWindow>
#include "ui_hologramdepthmap.h"
#include "graphics/sys.h"
#include "Hologram/HologramGenerator_CPU.h"

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
