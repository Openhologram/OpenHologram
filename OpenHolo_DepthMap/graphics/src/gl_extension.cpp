
#include "graphics/gl_extension.h"

namespace graphics {
#ifdef _WIN32

bool kGlExtensionInitialized = false;

void glExtensionInitialize()
{
	if (kGlExtensionInitialized) return;

	kGlExtensionInitialized = true;
	glewInit();
}

#endif

};