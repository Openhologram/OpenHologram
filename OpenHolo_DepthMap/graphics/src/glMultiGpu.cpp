#include "graphics/glMultiGpu.h"
#include "GL/wglew.h"
namespace graphics{

#define MAX_GPU 10

PIXELFORMATDESCRIPTOR pfd_ =
{
	sizeof(PIXELFORMATDESCRIPTOR),
	1,
	PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,    //Flags
	PFD_TYPE_RGBA,            //The kind of framebuffer. RGBA or palette.
	24,                        //Colordepth of the framebuffer.
	0, 0, 0, 0, 0, 0,
	0,
	0,
	0,
	0, 0, 0, 0,
	24,                        //Number of bits for the depthbuffer
	8,                        //Number of bits for the stencilbuffer
	0,                        //Number of Aux buffers in the framebuffer.
	PFD_MAIN_PLANE,
	0,
	0, 0, 0
};

void glMultiGpu::RenderContext::createGPUContext(int gpuIndex)
{
	HGPUNV hGPU[MAX_GPU];
	HGPUNV GpuMask[MAX_GPU];

	UINT displayDeviceIdx;
	GPU_DEVICE gpuDevice;
	bool bPrimary;
	// Get a list of the first MAX_GPU GPUs in the system
	if ((gpuIndex < MAX_GPU) && wglEnumGpusNV(gpuIndex, &hGPU[gpuIndex])) {

		printf("Device# %d:\n", gpuIndex);

		// Now get the detailed information about this device:
		// how many displays it's attached to
		displayDeviceIdx = 0;
		if (wglEnumGpuDevicesNV(hGPU[gpuIndex], displayDeviceIdx, &gpuDevice))
		{

			bPrimary |= (gpuDevice.Flags & DISPLAY_DEVICE_PRIMARY_DEVICE) != 0;
			LOG(" Display# %d:\n", displayDeviceIdx);
			LOG("  Name: %s\n", gpuDevice.DeviceName);
			LOG("  String: %s\n", gpuDevice.DeviceString);
			if (gpuDevice.Flags & DISPLAY_DEVICE_ATTACHED_TO_DESKTOP)
			{
				LOG("  Attached to the desktop: LEFT=%d, RIGHT=%d, TOP=%d, BOTTOM=%d\n",
					gpuDevice.rcVirtualScreen.left, gpuDevice.rcVirtualScreen.right, gpuDevice.rcVirtualScreen.top, gpuDevice.rcVirtualScreen.bottom);
			}
			else
			{
				LOG("  Not attached to the desktop\n");
			}

			// See if it's the primary GPU
			if (gpuDevice.Flags & DISPLAY_DEVICE_PRIMARY_DEVICE)
			{
				LOG("  This is the PRIMARY Display Device\n");
			}


		}

		///=======================   CREATE a CONTEXT HERE 
		GpuMask[0] = hGPU[gpuIndex];
		GpuMask[1] = NULL;
		affdc_ = wglCreateAffinityDCNV(GpuMask);

		if (!affdc_)
		{
			LOG("wglCreateAffinityDCNV failed");
		}
	}
}

glMultiGpu::RenderContext::RenderContext(int GpuIndex)
{
	createGPUContext(GpuIndex);
}

glMultiGpu::glMultiGpu(int num_gpu)
	: num_gpu_(num_gpu)
{
	contexts_.resize(num_gpu_);
	for (int i = 0; i < num_gpu; i++) {
		RenderContext* rc = new RenderContext(i);
		pf_ = ChoosePixelFormat(rc->affdc_, &pfd_);
		SetPixelFormat(rc->affdc_, pf_, &pfd_);
		DescribePixelFormat(rc->affdc_, pf_, sizeof(PIXELFORMATDESCRIPTOR),
			&pfd_);
		rc->affrc_ = wglCreateContext(rc->affdc_);
		contexts_[i] = rc;
	}
}
void glMultiGpu::makeCurrent(int gpuIndex)
{
	wglMakeCurrent(contexts_[gpuIndex]->affdc_, contexts_[gpuIndex]->affrc_);
}

}