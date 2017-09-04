#ifndef __qr_decomp_h
#define __qr_decomp_h


#include "graphics/mtl/mtl.h"
#include "graphics/mtl/matrix.h"
#include "graphics/mtl/mtl2lapack2.h"
#include "graphics/matrix.h"



namespace graphics {
//
// Description:
//  QR decomposition of matrix A (m x n, m >= n)
//
class qr_decomp {

public:

    qr_decomp(): m_Q(), m_R(), m_tau(), m(0), n(0) {}
    
    // Description:
    //	Setup computation environment: 
    //	If explicit_qr, compute Q and R matrix explicitly.
    bool set_matrix (const matrix& A, bool explicit_qr = false)
    {
	m = A.n1;
	n = A.n2;

	m_Q.resize(m, n);

	for (int i = 0 ; i < m ;++i){ 
	    for (int j = 0 ; j < n ; ++j)
		m_Q(i, j) = A(i, j);
	}
	
	m_explicit_qr = explicit_qr;

	bool result = factorize(explicit_qr);
	return result;
    }


    template <class T>
    bool solve(const vector<T>& b_, vector<T>& x_)
    {
	if (b_.size() == 0) return false;

        lapack_mat b(m, b_[0].n);
	lapack_mat c(n, b_[0].n);

	int d;
        
	// transform the vector form, b_, to matrix form b.
        for (d = 0; d < b_[0].n; d++) {
	    for (int i = 0 ; i < b_.size() ;++i)
		b(i, d) = b_[i][d];
        }
        
	// solve R*X = Q^t * b, X is overwritten to C
	if (!solve(b, c)) return false;
        
	x_.resize(n);
	for (d = 0; d < b_[0].n ; d++) {
	    for (int i = 0 ; i < x_.size() ;++i)
		x_[i][d] = c(i, d);
	}
        
        return true;
    }

    bool solve(const vector<real>& b_, vector<real>& x_, bool)
    {
	if (b_.size() == 0) return false;

        lapack_mat b(m, 1);
	lapack_mat c(n, 1);

	int d;
        
	// transform the vector form, b_, to matrix form b.
        for (d = 0; d < 1; d++) {
	    for (int i = 0 ; i < b_.size() ;++i)
		b(i, d) = b_[i];
        }
        
	// solve R*X = Q^t * b, X is overwritten to C
	if (!solve(b, c)) return false;
        
	x_.resize(n);
	for (d = 0; d < 1 ; d++) {
	    for (int i = 0 ; i < x_.size() ;++i)
		x_[i] = c(i, d);
	}
        
        return true;
    }

    bool get_Q(matrix& ret) {

	ret.resize(m, m);

	for (int i = 0 ; i < m ;++i){
	    for (int j = 0 ; j < m ; ++j) {
		ret(i, j) = m_Q(i, j);
	    }
	}
	return true;
    }

    bool get_R(matrix& ret) {
	ret.resize(n, n);

	for (int i = 0 ; i < n ;++i){
	    for (int j = 0 ; j < n  ; ++j) {
		ret(i,j) = m_R(i, j);
	    }
	}
	return true;
    }

private:

    bool    factorize(bool explicit_qr)
    {
	m_tau.resize(min(m, n));

	// Calls LAPACK QR factorization
	int info = mtl2lapack::geqrf(m_Q, m_tau);

	if (info != 0) return false;
	
	// m_A = Q1 * R, where R is n x n and Q1 m x n
	// R is upper triangular
	collect_R();

	if (explicit_qr) {
	    // now m_Q converts to Q1
	    m_Q.resize(m, m);	// lapack matrix is column major, so resize will not
				// hurt the content.
	    mtl2lapack::orgqr(m_Q, m_tau);
	}
	return true;
    }

    void    collect_R()
    {
	m_R.resize(n, n);

	for (int i = 0 ; i < n ;++i){
	    for (int j = i ; j < n ; ++j) {
		m_R(i, j) = m_Q(i, j);
	    }
	}
    }

    bool QTxB(lapack_mat& b, lapack_mat& c)
    {
	if (m_explicit_qr) {
	    lapack_mat::transpose_type V = mtl::trans(m_Q);

	    mtl::mult(V.sub_matrix(0, n, 0, m), b, c);
	}
	else 
	{
	    mtl2lapack::ormqr(m_Q, b, m_tau);
	    mtl::copy(b.sub_matrix(0, c.nrows(), 0, c.ncols()), c);
	}

	return true;
    }


    // Ax = b;
    bool solve(lapack_mat& b, lapack_mat& c)
    {
        int nrhs = b.ncols(),
            info = 0;

        // transform Q(:,0:n)'*B = C
	QTxB(b, c);
        
        int lda = m_R.minor(),
            ldb = c.minor();

        // consider only K-part of the triangular matrix, i.e:
        // solve R(0:n,0:n)*x(0:n) = C
        mtl_lapack_dispatch2::getrtrs('U', // upper triangle 
                                      'N', // A.x=b, no transpose
                                      'N', // no unit triangle
                                      n,   // consider R(0:n,0:n) only!
                                      nrhs,
                                      m_R.data(), 
                                      lda,
                                      c.data(),
                                      ldb,
                                      info);

	if (info == 0) return true;
        return false;
    }

private:

    // A = m_Q * m_R

    lapack_mat	    m_Q;
    lapack_mat	    m_R;
    
    lapack_vector   m_tau;

    int	    m, n;

    // For efficiency reason, we mostly don't need explicit Q matrix.
    bool    m_explicit_qr;
};

};  // namespace
#endif
