#ifndef __glMulti_Gpu_h
#define __glMulti_Gpu_h

#include "graphics/sys.h"
#include <vector>

namespace graphics {

class glMultiGpu{


public:

	struct RenderContext {
		HDC affdc_;
		HGLRC affrc_;
		RenderContext(int gpuIndx = 0);
		void createGPUContext(int gpuIndex);
	};

	glMultiGpu(int num_gpu = 2);

	void makeCurrent(int gpuIndex);

private:

	int num_gpu_;
	int pf_;

	std::vector<RenderContext*> contexts_;
	

};
}
#endif