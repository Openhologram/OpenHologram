#include "graphics/log.h"

#include "graphics/sys.h"
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>

namespace graphics {

void out_log(char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
}

void fatal(char *fmt, ...)
{
    va_list ap;

    printf("fatal:");

    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);

    printf("\n");
    fflush(stdout);

    exit(1);
}
}; // namespace graphics