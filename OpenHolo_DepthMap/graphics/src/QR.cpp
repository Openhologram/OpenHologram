#include "graphics/QR.h"

namespace graphics {

/*|
 *  QR_factorize :
 *
 *	Do QR factorization  QA = R (Trapezoidal form)
 *  Q : A.n1 x A.n1 (Square Matrix)
 *  R : A.n1 x A.n2
 */
void QR_factorize
(matrix& A, matrix& Q, matrix& R)
{
    /*|
     * vv  : selected column from A or d
     * uu  : gamma * uu is householder transformation
     *       when it is subtracted from vv
     * Qvv : householder-transformed vv
     */
    vector<real> uu(A.n1, A.n1), vv(A.n1, A.n1), Qvv(A.n1, A.n1);

    /*|
     * copy
     */
    R = A;

    matrix iden(A.n1, A.n1);
    matrix temp(A.n1, A.n1);
    iden.identity(iden.n1);

    Q.resize(A.n1, A.n1);
    Q.identity(Q.n1);


    for (int j = 0 ; j < A.n2 ; ++j) {
	/*|
	 * for each column, create uu vector, 
	 * so Qvv = vv - uu to be [v0,..,vk-1,s,0,..,0]^T
	 */
	vv = R.sel_col(j);
	int i, k, l;
	for (i = 0 ; i < j ;++i)
	    uu[i] = 0.0;

	real s = 0;
	
	for (k = j ; k < uu.size() ;++k)
	    s += vv[k]*vv[k];

	s = sqrt(s);

	if ( s * vv[j] > 0.0 ) s = -s;

	uu[j] = vv[j] - s;
	for (k = j+1 ; k < uu.size() ;++k)
	    uu[k] = vv[k];

	vector<real> tmp;
	real uu_uu = sum(tmp = uu*uu);
	real b = 2.0/uu_uu;

	for (k = 0; k < temp.n1 ;++k)
	    for (l = 0; l < temp.n2 ; l++) {
		real keep;
		keep=iden(k, l) - (b * uu[k] * uu[l]);
		temp(k,l) = (fabs(keep)<epsilon)? 0.0 : keep;
	    }
	/*|
	 * matrix multiplication
	 */


	Q = temp ^ Q;


	for (l = 0 ; l < A.n2 ; l++) {

	    /*|
	     * for each column , apply householder 
	     * transformation
	     */
	    vv = R.sel_col(l);

	    real gamma = b * sum(tmp = uu * vv);
	    Qvv = vv - (gamma * uu);
	    R.rep_col(l, Qvv);

	}
    }

    Q = Q.transpose();
}

}; // namespace graphics