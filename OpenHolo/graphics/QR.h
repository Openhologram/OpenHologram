#ifndef __QR_h
#define __QR_h

#include "graphics/sys.h"
#include "graphics/real.h"
#include "graphics/matrix.h"
#include "graphics/vector.h"

namespace graphics {
/*|
 *  QR_factorize :
 *
 *	Do QR factorization  QA = R (Trapezoidal form)
 *  Q : A.n1 x A.n1 (Square Matrix)
 *  R : A.n1 x A.n2
 */
void QR_factorize
(matrix& A, matrix& Q, matrix& R);

}; // namespace
#endif
