#include "hologramdepthmap.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	HologramDepthmap w;
	w.show();
	return a.exec();
}
