/* OpenHolo Beta v1.0 - Generation Module - PointCloud
Example for Point Cloud based Fringe Pattern Generation
*/

#include <iostream>
using namespace std;

// Point Cloud Based CGH Library
#include "../HologramPointCloud/HologramPointCloud.h"

// Define I/O Data File Path
#define INPUT_3D_MODEL	"TestPointCloud.dat" // 3D Point Cloud Data Base
#define INPUT_CONFIG	"TestSpec.config" // Parameters Config Specification
#define OUTPUT_BMP		"Result_FringePattern" // Fringe Pattern Image Output Bitmap File Name


int main(int argc, char **argv)
{
	cout << "OpenHolo Library : Generation Hologram - Point Cloud Example" << endl;

	openholo::HologramPointCloud* Hologram = new openholo::HologramPointCloud();

	Hologram->setMode(MODE_CPU); //Select CPU or GPU Processing

	if (Hologram->loadPointCloud(INPUT_3D_MODEL) == -1)
	{
		cerr << "Failed to load Point Cloud Data File(*.dat)" << endl;
		return -1;
	}

	if (!Hologram->readConfig(INPUT_CONFIG))
	{
		cerr << "Failed to load Config Specification Data File(*.config)" << endl;
		return -1;
	}

	double time = Hologram->generateHologram();
	cout << "Implement Time : " << time << " sec" << endl;

	Hologram->saveFileBmp(OUTPUT_BMP);

	return 0;
}