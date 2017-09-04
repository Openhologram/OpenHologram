
#include "graphics/fmatrix.h"
#include "graphics/epsilon.h"

/*!
 */
namespace graphics {

real hypot(real x,real y)
{
    real t;
    x = fabs(x);
    y = fabs(y);
    t = min(x,y);
    x = max(x,y);
    t = t/x;
    return x*sqrt(1+t*t);
}
	

fmatrix operator + (const fmatrix& a, const fmatrix& b)
{
    fmatrix c(a.n1, a.n2);
    for(int i = 0; i < a.n;++i) {
	c[i] = a.v[i] + b.v[i];
    }
    return c;
}


fmatrix operator + (real a, const fmatrix& b)
{
    fmatrix c(b.n1, b.n2);
    for(int i = 0; i < b.n;++i) {
	c[i] = a + b.v[i];
    }
    return c;
}


fmatrix operator + (const fmatrix& a, real b)
{
    fmatrix c(a.n1, a.n2);
    for(int i = 0; i < a.n;++i) {
	c[i] = a.v[i] + b;
    }
    return c;
}




fmatrix operator - (const fmatrix& a, const fmatrix& b)
{
    fmatrix c(a.n1, a.n2);
    for(int i = 0; i < a.n;++i) {
	c[i] = a.v[i] - b.v[i];
    }
    return c;
}


fmatrix operator - (real a, const fmatrix& b)
{
    fmatrix c(b.n1, b.n2);
    for(int i = 0; i < b.n;++i) {
	c[i] = a - b.v[i];
    }
    return c;
}

fmatrix operator - (const fmatrix& a, real b)
{
    fmatrix c(a.n1, a.n2);
    for(int i = 0; i < a.n;++i) {
	c[i] = a.v[i] - b;
    }
    return c;
}




fmatrix operator * (const fmatrix& a, const fmatrix& b)
{
    fmatrix c(a.n1, a.n2);
    for(int i = 0; i < a.n;++i) {
	c[i] = a.v[i] * b.v[i];
    }
    return c;
}


fmatrix operator * (real a, const fmatrix& b)
{
    fmatrix c(b.n1, b.n2);
    for(int i = 0; i < b.n;++i) {
	c[i] = a * b.v[i];
    }
    return c;
}


fmatrix operator * (const fmatrix& a, real b)
{
    fmatrix c(a.n1, a.n2);
    for(int i = 0; i < a.n;++i) {
	c[i] = a.v[i] * b;
    }
    return c;
}




fmatrix operator / (const fmatrix& a, const fmatrix& b)
{
    fmatrix c(a.n1, a.n2);
    for(int i = 0; i < a.n;++i) {
	c[i] = a.v[i] / b.v[i];
    }
    return c;
}

fmatrix operator / (real a, const fmatrix& b)
{
    fmatrix c(b.n1, b.n2);
    for(int i = 0; i < b.n;++i) {
	c[i] = a / b.v[i];
    }
    return c;
}


fmatrix operator / (const fmatrix& a, real b)
{
    fmatrix c(a.n1, a.n2);
    for(int i = 0; i < a.n;++i) {
	c[i] = a.v[i] / b;
    }
    return c;
}



//| cumulative : component-wise operations



fmatrix operator += (fmatrix& a, const fmatrix& b)
{
    return a = (a + b);
}


fmatrix operator += (fmatrix& a, real b)
{
    return a = (a + b);
}




fmatrix operator -= (fmatrix& a, const fmatrix& b)
{
    return a = (a - b);
}


fmatrix operator -= (fmatrix& a, real b)
{
    return a = (a - b);
}




fmatrix operator *= (fmatrix& a, const fmatrix& b)
{
    return a = (a * b);
}


fmatrix operator *= (fmatrix& a, real b)
{
    return a = (a * b);
}




fmatrix operator /= (fmatrix& a, const fmatrix& b)
{
    return a = (a / b);
}


fmatrix operator /= (fmatrix& a, real b)
{
    return a = (a / b);
}

real sum(const std::vector<real>& a)
{
    real s = 0;
    for(int i = 0; i < a.size();++i) {
		s += a[i];
    }
    return s;
}
real sum(const fmatrix& a)
{
    real s = 0;
    for(int i = 0; i < a.n ;++i) {
		s += a[i];
    }
    return s;
}
//|
//| matrix multiplication
//|

fmatrix operator ^ (const fmatrix& a, const fmatrix& b)
{
    if(a.n2 != b.n1) 
    {
		LOG("(%d X %d) * (%d X %d)\n",a.n1, a.n2, b.n1, b.n2); 
		fatal("matrix multiplication: matrix mismatch !\n");
    }

    fmatrix c(a.n1, b.n2);
    std::vector<real> d, e;

    for (int i = 0 ; i < a.n1 ;++i) {
	for(int j = 0 ; j < b.n2 ; ++j) {
	    c(i,j) = sum(a.sel_row(i) * b.sel_col(j));
	}
    }

    return c;
}


fmatrix operator ^ (const fmatrix& a, const std::vector<real>& b)
{
    fmatrix c(b);
    return a^c;
}


fmatrix operator ^ (const std::vector<real>& a, const fmatrix& b)

{
    fmatrix c(a);
    fmatrix d = c.transpose();
    return d^b; 
}
/*!
  \brief assignment operator

  Assignment operator

  \param a  the LU matrix to copy
  \return a reference to itself

  \author Philippe Lavoie 
  \date 22 October 1997
*/
flumatrix& flumatrix::operator=(const flumatrix& a){
  resize(a.n1,a.n2) ;
	for(int i=0;i<fmatrix::n1;++i)
    for(int j=0;j<fmatrix::n2;++j)
      (*this)(i,j) = a(i,j) ;
  pivot_ = a.pivot_ ;
  return *this ;
}

/*!
  \brief Generetes a LU matrix from a matrix a

  Generetes a LU matrix from a matrix a.

  \param a  the matrix to decompose
  \return a reference to itself

  \author Philippe Lavoie 
  \date 22 October 1997
*/

flumatrix& flumatrix::decompose(const fmatrix& a)
{
  int i, j, k, l, kp1, nm1;
  real t, q;
  real den, ten;
  
  int N = a.n1;

  resize(N, N) ;

  if (a.n1 != a.n2) {
    fatal( "The a matrix is not squared!") ;
  }

  //	lu = a;	 must do it by copying or LUFACT will be recursively called !
  for (i = 0 ; i < N ; ++i)
    for(j = 0 ; j < N ; ++j)
      (*this)(i,j) = a(i,j) ;

  errval = 0;
  nm1 = N - 1;
  for (k = 0; k < N;++k)
    pivot_[k] = k ;

  sign = 1 ;
  
  if (nm1 >= 1)	/* non-trivial problem */
    {
      for (k = 0; k < nm1;++k)
	{
	  kp1 = k + 1;
	  // partial pivoting ROW exchanges
	  // -- search over column 

	  ten = absolute(a(k,k));
	  l = k;
	  for (i = kp1; i < N;++i)
	    {
	      den = absolute(a(i,k));
	      if ( den > ten )
		{
		  ten = den;
		  l = i;
		}
	    }
	  pivot_[k] = l;

	  if ( (*this)(l,k) != 0.0 )
	    {			// nonsingular pivot found 
	      if (l != k ){	// interchange needed 
		for (i = k; i < N;++i)
		  {
		    t = (*this)(l,i) ;
		    (*this)(l,i) = (*this)(k,i) ;
		    (*this)(k,i) = t ; 
		  }
		sign = -sign ;
	      }
	      q =  (*this)(k,k);	/* scale row */
	      for (i = kp1; i < N;++i)
		{
		  t = - (*this)(i,k)/q;
		  (*this)(i,k) = t;
		  for (j = kp1; j < N; ++j)
		    (*this)(i,j) += t * (*this)(k,j);
		}
	    }
	  else		/* pivot singular */
	    errval = k;
	}
      
    }
  
  pivot_[nm1] = nm1;
  if ((*this)(nm1,nm1) == 0.0)
    errval = nm1;  
  return *this;
}


/*!
  \brief Finds the determinant of a LU matrix

  Finds the determinant of a LU matrix.

  \return The value of the determinant

  \author Philippe Lavoie 
  \date 22 October 1997
*/

real flumatrix::determinant(){
  real det = (*this)(0,0) ;
  for(int i=1;i<fmatrix::n1;++i)
    det *= (*this)(i,i) ;
  return det * (real)sign ;
}

/*!
  \brief Finds the inverse of the LU matrix

  Finds the inverse of the LU matrix.

  \param inv  the inverse of the matrix

  \author Philippe Lavoie 
  \date 22 October 1997
*/

void flumatrix::inverseIn(fmatrix& inv) const
{
  real ten;
  int i, j, k, l, kb, kp1, nm1, coln;

  if ( fmatrix::n1 != fmatrix::n2 )
    {
      fatal("matrix inverse, not square: ");
    }

  int N = coln = fmatrix::n1;


  inv = *this ;
  
  nm1 = N - 1;

  // inverse U 
  for (k = 0; k < N;++k)
    {
      inv(k,k) = ten = 1.0 / inv(k,k) ;
      ten = -ten;
      for (i = 0; i < k;++i)
	inv(i,k) *= ten;
      kp1 = k + 1;
      if (nm1 >= kp1)
	{
	  for (j = kp1; j < N; ++j)
	    {
	      ten = inv(k,j) ;
	      inv(k,j) = 0.0 ;
	      for (i = 0; i < kp1;++i)
		inv(i,j) += ten * inv(i,k) ;	      
	    }
	}
    }

  // INV(U) * INV(L)   
  if (nm1 >= 1)
    {
      std::vector<real> work(N) ;
      //error.memory( work.memory() );

      for (kb = 0; kb < nm1; kb++)
	{
	  k = nm1 - kb - 1;
	  kp1 = k + 1;
	  for (i = kp1; i < N;++i)
	    {
	      work[i] = inv(i,k) ;
	      inv(i,k) = 0.0;
	    }
	  for (j = kp1; j < N; ++j)
	    {
	      ten = work[j];
	      for (i = 0; i < N;++i)
		inv(i,k) += ten * inv(i,j) ;
	    }
	  l = pivot[k];
	  if (l != k)
	    for (i = 0; i < N;++i)
	      {
		ten = inv(i,k) ;
		inv(i,k) = inv(i,l);
		inv(i,l) = ten;
	      }
	}
      
    } 
}

/*!
  \brief Finds the inverse of the LU matrix

  Finds the inverse of the LU matrix.

  \return the inverse of the matrix

  \author Philippe Lavoie 
  \date 22 October 1997
*/

fmatrix flumatrix::inverse() const
{
  if ( fmatrix::n1 != fmatrix::n2 )
    {
      fatal("no square\n");
    }

  fmatrix inverse ;
  inverseIn(inverse) ;

  return inverse;
}



inline real sqr(real a){
  return a*a ;
}

/*
 *------------------------------------------------------------------------
 *				Bidiagonalization
 */

 /*
 *			Left Householder Transformations
 *
 * Zero out an entire subdiagonal of the i-th column of A and compute the
 * modified A[i,i] by multiplication (UP' * A) with a matrix UP'
 *   (1)  UP' = E - UPi * UPi' / beta
 *
 * where a column-vector UPi is as follows
 *   (2)  UPi = [ (i-1) zeros, A[i,i] + Norm, vector A[i+1:M,i] ]
 * where beta = UPi' * A[,i] and Norm is the norm of a vector A[i:M,i]
 * (sub-diag part of the i-th column of A). Note we assign the Norm the
 * same sign as that of A[i,i].
 * By construction, (1) does not affect the first (i-1) rows of A. Since
 * A[*,1:i-1] is bidiagonal (the result of the i-1 previous steps of
 * the bidiag algorithm), transform (1) doesn't affect these i-1 columns
 * either as one can easily verify.
 * The i-th column of A is transformed as
 *   (3)  UP' * A[*,i] = A[*,i] - UPi
 * (since UPi'*A[*,i]/beta = 1 by construction of UPi and beta)
 * This means effectively zeroing out A[i+1:M,i] (the entire subdiagonal
 * of the i-th column of A) and replacing A[i,i] with the -Norm. Thus
 * modified A[i,i] is returned by the present function.
 * The other (i+1:N) columns of A are transformed as
 *    (4)  UP' * A[,j] = A[,j] - UPi * ( UPi' * A[,j] / beta )
 * Note, due to (2), only elements of rows i+1:M actually  participate
 * in above transforms; the upper i-1 rows of A are not affected.
 * As was mentioned earlier,
 * (5)  beta = UPi' * A[,i] = (A[i,i] + Norm)*A[i,i] + A[i+1:M,i]*A[i+1:M,i]
 *	= ||A[i:M,i]||^2 + Norm*A[i,i] = Norm^2 + Norm*A[i,i]
 * (note the sign of the Norm is the same as A[i,i])
 * For extra precision, vector UPi (and so is Norm and beta) are scaled,
 * which would not affect (4) as easy to see.
 *
 * To satisfy the definition
 *   (6)  .SIG = U' A V
 * the result of consecutive transformations (1) over matrix A is accumulated
 * in matrix U' (which is initialized to be a unit matrix). At each step,
 * U' is left-multiplied by UP' = UP (UP is symmetric by construction,
 * see (1)). That is, U is right-multiplied by UP, that is, rows of U are
 * transformed similarly to columns of A, see eq. (4). We also keep in mind
 * that multiplication by UP at the i-th step does not affect the first i-1
 * columns of U.
 * Note that the vector UPi doesn't have to be allocated explicitly: its
 * first i-1 components are zeros (which we can always imply in computations),
 * and the rest of the components (but the UPi[i]) are the same as those
 * of A[i:M,i], the subdiagonal of A[,i]. This column, A[,i] is affected only
 * trivially as explained above, that is, we don't need to carry this
 * transformation explicitly (only A[i,i] is going to be non-trivially
 * affected, that is, replaced by -Norm, but we will use sig[i] to store
 * the result).
 *
 */

 
real fsvdmatrix::left_householder(fmatrix& A, const int i)
 {
   int j,k;
   real scale = 0;			// Compute the scaling factor
   for(k=i; k<M;++k)
     scale += absolute(A(k,i));
   if( scale == 0 )			// If A[,i] is a null vector, no
     return 0;				// transform is required
   real Norm_sqr = 0;			// Scale UPi (that is, A[,i])
   for(k=i; k<M;++k)			// and compute its norm, Norm^2
     Norm_sqr += sqr(A(k,i) /= scale);
   real new_Aii = sqrt(Norm_sqr);	// new_Aii = -Norm, Norm has the
   if( A(i,i) > 0 ) new_Aii = -new_Aii;	// same sign as Aii (that is, UPi[i])
   const real beta = - A(i,i)*new_Aii + Norm_sqr;
   A(i,i) -= new_Aii;			// UPi[i] = A[i,i] - (-Norm)
   
   for(j=i+1; j<N; ++j)		// Transform i+1:N columns of A
   {
     real factor = 0;
     for(k=i; k<M;++k)
       factor += A(k,i) * A(k,j);	
     factor /= beta;
     for(k=i; k<M;++k)
       A(k,j) -= A(k,i) * factor;
   }

   for(j=0; j<M; ++j)			// Accumulate the transform in U
   {
     real factor = 0;
     for(k=i; k<M;++k)
       factor += A(k,i) * U_(j,k);	// Compute  U[j,] * UPi
     factor /= beta;
     for(k=i; k<M;++k)
        U_(j,k) -= A(k,i) * factor;
   }
   return new_Aii * scale;		// Scale new Aii back (our new Sig[i])
 }
 
/*
 *			Right Householder Transformations
 *
 * Zero out i+2:N elements of a row A[i,] of matrix A by right
 * multiplication (A * VP) with a matrix VP
 *   (1)  VP = E - VPi * VPi' / beta
 *
 * where a vector-column .VPi is as follows
 *   (2)  VPi = [ i zeros, A[i,i+1] + Norm, vector A[i,i+2:N] ]
 * where beta = A[i,] * VPi and Norm is the norm of a vector A[i,i+1:N]
 * (right-diag part of the i-th row of A). Note we assign the Norm the
 * same sign as that of A[i,i+1].
 * By construction, (1) does not affect the first i columns of A. Since
 * A[1:i-1,] is bidiagonal (the result of the previous steps of
 * the bidiag algorithm), transform (1) doesn't affect these i-1 rows
 * either as one can easily verify.
 * The i-th row of A is transformed as
 *  (3)  A[i,*] * VP = A[i,*] - VPi'
 * (since A[i,*]*VPi/beta = 1 by construction of VPi and beta)
 * This means effectively zeroing out A[i,i+2:N] (the entire right super-
 * diagonal of the i-th row of A, but ONE superdiag element) and replacing
 * A[i,i+1] with - Norm. Thus modified A[i,i+1] is returned as the result of
 * the present function.
 * The other (i+1:M) rows of A are transformed as
 *    (4)  A[j,] * VP = A[j,] - VPi' * ( A[j,] * VPi / beta )
 * Note, due to (2), only elements of columns i+1:N actually  participate
 * in above transforms; the left i columns of A are not affected.
 * As was mentioned earlier,
 * (5)  beta = A[i,] * VPi = (A[i,i+1] + Norm)*A[i,i+1]
 *			   + A[i,i+2:N]*A[i,i+2:N]
 *	= ||A[i,i+1:N]||^2 + Norm*A[i,i+1] = Norm^2 + Norm*A[i,i+1]
 * (note the sign of the Norm is the same as A[i,i+1])
 * For extra precision, vector VPi (and so is Norm and beta) are scaled,
 * which would not affect (4) as easy to see.
 *
 * The result of consecutive transformations (1) over matrix A is accumulated
 * in matrix V (which is initialized to be a unit matrix). At each step,
 * V is right-multiplied by VP. That is, rows of V are transformed similarly
 * to rows of A, see eq. (4). We also keep in mind that multiplication by
 * VP at the i-th step does not affect the first i rows of V.
 * Note that vector VPi doesn't have to be allocated explicitly: its
 * first i components are zeros (which we can always imply in computations),
 * and the rest of the components (but the VPi[i+1]) are the same as those
 * of A[i,i+1:N], the superdiagonal of A[i,]. This row, A[i,] is affected
 * only trivially as explained above, that is, we don't need to carry this
 * transformation explicitly (only A[i,i+1] is going to be non-trivially
 * affected, that is, replaced by -Norm, but we will use super_diag[i+1] to
 * store the result).
 *
 */
 

real fsvdmatrix::right_householder(fmatrix& A, const int i)
{
   int j,k;
   real scale = 0;				// Compute the scaling factor
   for(k=i+1; k<N;++k)
     scale += absolute(A(i,k));
   if( scale == 0 )			// If A[i,] is a null vector, no
     return 0;				// transform is required

   real Norm_sqr = 0;			// Scale VPi (that is, A[i,])
   for(k=i+1; k<N;++k)			// and compute its norm, Norm^2
     Norm_sqr += sqr(A(i,k) /= scale);
   real new_Aii1 = sqrt(Norm_sqr);	// new_Aii1 = -Norm, Norm has the
   if( A(i,i+1) > 0 )			// same sign as
     new_Aii1 = -new_Aii1; 		// Aii1 (that is, VPi[i+1])
   const real beta = - A(i,i+1)*new_Aii1 + Norm_sqr;
   A(i,i+1) -= new_Aii1;		// VPi[i+1] = A[i,i+1] - (-Norm)
   
   for(j=i+1; j<M; ++j)			// Transform i+1:M rows of A
   {
     real factor = 0;
     for(k=i+1; k<N;++k)
       factor += A(i,k) * A(j,k);	// Compute A[j,] * VPi
     factor /= beta;
     for(k=i+1; k<N;++k)
       A(j,k) -= A(i,k) * factor;
   }

   for(j=0; j<N; ++j)			// Accumulate the transform in V
   {
     real factor = 0;
     for(k=i+1; k<N;++k)
       factor += A(i,k) * V_(j,k);	// Compute  V[j,] * VPi
     factor /= beta;
     for(k=i+1; k<N;++k)
       V_(j,k) -= A(i,k) * factor;
   }
   return new_Aii1 * scale;		// Scale new Aii1 back
}

/*
 *------------------------------------------------------------------------
 *			  Bidiagonalization
 * This nethod turns matrix A into a bidiagonal one. Its N diagonal elements
 * are stored in a vector sig, while N-1 superdiagonal elements are stored
 * in a vector super_diag(2:N) (with super_diag(1) being always 0).
 * Matrices U and V store the record of orthogonal Householder
 * reflections that were used to convert A to this form. The method
 * returns the norm of the resulting bidiagonal matrix, that is, the
 * maximal column sum.
 */


real fsvdmatrix::bidiagonalize(std::vector<real>& super_diag, const fmatrix& _A)
{
  real norm_acc = 0;
  super_diag[0] = 0;			// No superdiag elem above A(1,1)
  fmatrix A = _A;			// A being transformed
  for(int i=0; i<N;++i)
  {
    const real& diagi = sig_[i] = left_householder(A,i);
    if( i < N-1 )
      super_diag[i+1] = right_householder(A,i);
    real t2 = absolute(diagi)+absolute(super_diag[i]) ;
    norm_acc = max(norm_acc, t2);
  }
  return norm_acc;
}

/*
 *------------------------------------------------------------------------
 *		QR-diagonalization of a bidiagonal matrix
 *
 * After bidiagonalization we get a bidiagonal matrix J:
 *    (1)  J = U' * A * V
 * The present method turns J into a matrix JJ by applying a set of
 * orthogonal transforms
 *    (2)  JJ = S' * J * T
 * Orthogonal matrices S and T are chosen so that JJ were also a
 * bidiagonal matrix, but with superdiag elements smaller than those of J.
 * We repeat (2) until non-diag elements of JJ become smaller than EPS
 * and can be disregarded.
 * Matrices S and T are constructed as
 *    (3)  S = S1 * S2 * S3 ... Sn, and similarly T
 * where Sk and Tk are matrices of simple rotations
 *    (4)  Sk[i,j] = i==j ? 1 : 0 for all i>k or i<k-1
 *         Sk[k-1,k-1] = cos(Phk),  Sk[k-1,k] = -sin(Phk),
 *         SK[k,k-1] = sin(Phk),    Sk[k,k] = cos(Phk), k=2..N
 * gmatrix Tk is constructed similarly, only with angle Thk rather than Phk.
 *
 * Thus left multiplication of J by SK' can be spelled out as
 *    (5)  (Sk' * J)[i,j] = J[i,j] when i>k or i<k-1,
 *                  [k-1,j] = cos(Phk)*J[k-1,j] + sin(Phk)*J[k,j]
 *                  [k,j] =  -sin(Phk)*J[k-1,j] + cos(Phk)*J[k,j]
 * That is, k-1 and k rows of J are replaced by their linear combinations;
 * the rest of J is unaffected. Right multiplication of J by Tk similarly
 * changes only k-1 and k columns of J.
 * gmatrix T2 is chosen the way that T2'J'JT2 were a QR-transform with a
 * shift. Note that multiplying J by T2 gives rise to a J[2,1] element of
 * the product J (which is below the main diagonal). Angle Ph2 is then
 * chosen so that multiplication by S2' (which combines 1 and 2 rows of J)
 * gets rid of that elemnent. But this will create a [1,3] non-zero element.
 * T3 is made to make it disappear, but this leads to [3,2], etc.
 * In the end, Sn removes a [N,N-1] element of J and matrix S'JT becomes
 * bidiagonal again. However, because of a special choice
 * of T2 (QR-algorithm), its non-diag elements are smaller than those of J.
 *
 * All this process in more detail is described in
 *    J.H. Wilkinson, C. Reinsch. Linear algebra - Springer-Verlag, 1971
 *
 * If during transforms (1), JJ[N-1,N] turns 0, then JJ[N,N] is a singular
 * number (possibly with a wrong (that is, negative) sign). This is a
 * consequence of Frantsis' Theorem, see the book above. In that case, we can
 * eliminate the N-th row and column of JJ and carry out further transforms
 * with a smaller matrix. If any other superdiag element of JJ turns 0,
 * then JJ effectively falls into two independent matrices. We will process
 * them independently (the bottom one first).
 *
 * Since matrix J is a bidiagonal, it can be stored efficiently. As a matter
 * of fact, its N diagonal elements are in array Sig, and superdiag elements
 * are stored in array super_diag.
 */
 
				// Carry out U * S with a rotation matrix
				// S (which combines i-th and j-th columns
				// of U, i>j)

void fsvdmatrix::rotate(fmatrix& Mu, const int i, const int j,
		   const real cos_ph, const real sin_ph)
{
  for(int l=0; l<Mu.n1; l++)
  {
    real& Uil = Mu(l,i); real& Ujl = Mu(l,j);
    const real Ujl_was = Ujl;
    Ujl =  cos_ph*Ujl_was + sin_ph*Uil;
    Uil = -sin_ph*Ujl_was + cos_ph*Uil;
  }
}

/*
 * A diagonal element J[l-1,l-1] turns out 0 at the k-th step of the
 * algorithm. That means that one of the original matrix' singular numbers
 * shall be zero. In that case, we multiply J by specially selected
 * matrices S' of simple rotations to eliminate a superdiag element J[l-1,l].
 * After that, matrix J falls into two pieces, which can be dealt with
 * in a regular way (the bottom piece first).
 * 
 * These special S transformations are accumulated into matrix U: since J
 * is left-multiplied by S', U would be right-multiplied by S. Transform
 * formulas for doing these rotations are similar to (5) above. See the
 * book cited above for more details.
 */

void fsvdmatrix::rip_through(std::vector<real>& super_diag, const int k, const int l, const real eps)
{
  real cos_ph = 0, sin_ph = 1;	// Accumulate cos,sin of Ph
  				// The first step of the loop below
  				// (when i==l) would eliminate J[l-1,l],
  				// which is stored in super_diag(l)
  				// However, it gives rise to J[l-1,l+1]
  				// and J[l,l+2]
  				// The following steps eliminate these
  				// until they fall below
  				// significance
  for(int i=l; i<=k;++i)
  {
    const real f = sin_ph * super_diag[i];
    super_diag[i] *= cos_ph;
    if( absolute(f) <= eps )
      break;			// Current J[l-1,l] became unsignificant
    cos_ph = sig[i]; sin_ph = -f;	// unnormalized sin/cos
    const real norm = (sig_[i] = hypot(cos_ph,sin_ph)); // sqrt(sin^2+cos^2)
    cos_ph /= norm, sin_ph /= norm;	// Normalize sin/cos
    rotate(U_,i,l-1,cos_ph,sin_ph);
  }
}

			// We're about to proceed doing QR-transforms
			// on a (bidiag) matrix J[1:k,1:k]. It may happen
			// though that the matrix splits (or can be
			// split) into two independent pieces. This function
			// checks for splitting and returns the lowerbound
			// index l of the bottom piece, J[l:k,l:k]

int fsvdmatrix::get_submatrix_to_work_on(std::vector<real>& super_diag, const int k, const real eps)
{
  for(register int l=k; l>0; l--)
    if( absolute(super_diag[l]) <= eps )
      return l;				// The breaking point: zero J[l-1,l]
    else if( absolute(sig[l-1]) <= eps )	// Diagonal J[l,l] turns out 0
    {					// meaning J[l-1,l] _can_ be made
      rip_through(super_diag,k,l,eps);	// zero after some rotations
      return l;
    }
  return 0;			// Deal with J[1:k,1:k] as a whole
}

		// Diagonalize root module

void fsvdmatrix::diagonalize(std::vector<real>& super_diag, const real eps)
{
  for(register int k=N-1; k>=0; k--)	// QR-iterate upon J[l:k,l:k]
  {
    int l;
    //    while(l=get_submatrix_to_work_on(super_diag,k,eps),
    //	  absolute(super_diag[k]) > eps)	// until superdiag J[k-1,k] becomes 0
    // Phil
    while(absolute(super_diag[k]) > eps)// until superdiag J[k-1,k] becomes 0
    {
      l=get_submatrix_to_work_on(super_diag,k,eps) ;
      real shift;			// Compute a QR-shift from a bottom
      {					// corner minor of J[l:k,l:k] order 2
      	real Jk2k1 = super_diag[k-1],	// J[k-2,k-1]
      	     Jk1k  = super_diag[k],
      	     Jk1k1 = sig[k-1],		// J[k-1,k-1]
      	     Jkk   = sig[k],
      	     Jll   = sig[l];		// J[l,l]
      	shift = (Jk1k1-Jkk)*(Jk1k1+Jkk) + (Jk2k1-Jk1k)*(Jk2k1+Jk1k);
      	shift /= 2*Jk1k*Jk1k1;
      	shift += (shift < 0 ? -1 : 1) * sqrt(shift*shift+1);
      	shift = ( (Jll-Jkk)*(Jll+Jkk) + Jk1k*(Jk1k1/shift-Jk1k) )/Jll;
      }
      				// Carry on multiplications by T2, S2, T3...
      real cos_th = 1, sin_th = 1;
      real Ji1i1 = sig[l];	// J[i-1,i-1] at i=l+1...k
      for(register int i=l+1; i<=k;++i)
      {
      	real Ji1i = super_diag[i], Jii = sig[i];  // J[i-1,i] and J[i,i]
      	sin_th *= Ji1i; Ji1i *= cos_th; cos_th = shift;
      	real norm_f = (super_diag[i-1] = hypot(cos_th,sin_th));
      	cos_th /= norm_f, sin_th /= norm_f;
      					// Rotate J[i-1:i,i-1:i] by Ti
      	shift = cos_th*Ji1i1 + sin_th*Ji1i;	// new J[i-1,i-1]
      	Ji1i = -sin_th*Ji1i1 + cos_th*Ji1i;	// J[i-1,i] after rotation
      	const real Jii1 = Jii*sin_th;		// Emerged J[i,i-1]
      	Jii *= cos_th;				// new J[i,i]
        rotate(V_,i,i-1,cos_th,sin_th); // Accumulate T rotations in V
        
        real cos_ph = shift, sin_ph = Jii1;// Make Si to get rid of J[i,i-1]
        sig_[i-1] = (norm_f = hypot(cos_ph,sin_ph));	// New J[i-1,i-1]
        if( norm_f == 0 )		// If norm =0, rotation angle
          cos_ph = cos_th, sin_ph = sin_th; // can be anything now
        else
          cos_ph /= norm_f, sin_ph /= norm_f;
      					// Rotate J[i-1:i,i-1:i] by Si
        shift = cos_ph * Ji1i + sin_ph*Jii;	// New J[i-1,i]
        Ji1i1 = -sin_ph*Ji1i + cos_ph*Jii;	// New Jii, would carry over
        					// as J[i-1,i-1] for next i
        rotate(U_,i,i-1,cos_ph,sin_ph);  // Accumulate S rotations in U
        				// Jii1 disappears, sin_th would
        cos_th = cos_ph, sin_th = sin_ph; // carry over a (scaled) J[i-1,i+1]
        				// to eliminate on the next i, cos_th
        				// would carry over a scaled J[i,i+1]
      }
      super_diag[l] = 0;		// Supposed to be eliminated by now
      super_diag[k] = shift;
      sig_[k] = Ji1i1;
    }		// --- end-of-QR-iterations
    if( sig[k] < 0 )		// Correct the sign of the sing number
    {
      sig_[k] = -sig_[k];
      for(register int j=0; j<V_.n1; ++j)
        V_(j,k) = -V_(j,k);
    }
  }
} 

/*!
  \brief Performs the SVD decomposition of A

  Performs the SVD decomposition of A. It is not recommended
  to use this routine has errors from the decomposition 
  routine are discarded

  \param A  the matrix to decompose

  \warning The matrix A must have a number equal or higher of rows than 
           columns.

  \author Philippe Lavoie 
  \date 22 October 1997
*/


/*!
  \brief Performs the SVD decomposition of A

  Performs the SVD decomposition of A.

  \param A  the matrix to decompose
  \return 1 if the decomposition was succesfull, 0 otherwise.

  \warning The matrix A must have a number equal or higher of rows than 
               columns.

  \author Philippe Lavoie 
  \date 22 October 1997
*/

int fsvdmatrix::decompose(const fmatrix& A){
  M = A.n1 ;
  N = A.n2 ;

  U_.resize(M, M) ;
  V_.resize(N, N) ;
  sig_.resize(N) ;
  std::vector<real>& m_sigma = sig_;

  U_.set_diagonal_elements(1) ;
  V_.set_diagonal_elements(1) ;
  //
  std::vector<real> super_diag(N);
  const real bidiag_norm = bidiagonalize(super_diag,A);
  const real eps = zero_epsilon * bidiag_norm;	// Significance threshold
  diagonalize(super_diag, eps);
  sort();

  return 1 ;
}


/*!
  \brief Finds the minimal and maximal sigma values

  Finds the minimal and maximal sigma values.

  \param min_sig  the minimal sigma value
  \param max_sig  the maximal sigma value

  \warning The SVD matrix must be valid.

  \author Philippe Lavoie 
  \date 22 October 1997
*/

void fsvdmatrix::minMax(real& min_sig, real& max_sig) const
{
  max_sig = min_sig = sig[0];
  for(register int i=0; i<sig.size(); ++i)
    if( sig[i] > max_sig )
      max_sig = sig[i];
    else if( sig[i] < min_sig )
      min_sig = sig[i];
}

/*!
  \brief Finds the condition number

  Finds the condition number which corresponds to the maximal
  sigma value divided by its minimal value.

  \return The condition number

  \warning The SVD matrix must be valid.

  \author Philippe Lavoie 
  \date 22 October 1997
*/

real fsvdmatrix::q_cond_number(void) const
{
  real mins,maxs ;
  minMax(mins,maxs) ;
  return (maxs/mins);
}

/*!
  \brief Finds the (pseudo-)inverse of a SVD matrix

  Finds the (pseudo-)inverse of a SVD matrix.

  \param   A  the inverse of the SVD matrix
  \param tau  the minimal singular value accepted

  \warning The SVD matrix must be valid.

  \author Philippe Lavoie 
  \date 22 October 1997
*/

void fsvdmatrix::inverseIn(fmatrix& A, real tau) {
  fmatrix S ;

  real min_sig, max_sig ;
  minMax(min_sig,max_sig) ;

  if(tau==0)
    tau = V.n1*max_sig* epsilon ;

  if(min_sig<tau){
    fatal("\nSVD found some singular value null coefficients.\n") ;
  }

  S.resize(V.n2,U.n1);
  S = 0;
  for(int i=0;i<sig.size();++i)
    S(i,i) = (real)1/sig[i] ;

  //A = V*S*transpose(U) ; 
  A = U.transpose() ;
  A = (const fmatrix&)S^(const fmatrix&)A ; // transpose(U) ; 
  A = (const fmatrix&)V^(const fmatrix&)A ;
}


/*!
  \brief Finds the (pseudo-)inverse of a SVD matrix

  Finds the (pseudo-)inverse of a SVD matrix.

  \param tau  the minimal singular value accepted

  \warning The SVD matrix must be valid.

  \author Philippe Lavoie 
  \date 22 October 1997
*/

fmatrix fsvdmatrix::inverse(real tau) {
  fmatrix A ;

  inverseIn(A,tau) ;
  return A ;
}

/*!
  \brief Solves the linear system \a A \a X = \a B

   Solves the linear system \a A \a X = \a B. Given \a A and \a B it
   finds the value of \a X. \a A is the SVD matrix properly
   initialized and the only other input parameter is \a B.

  \param B  the right hand side of the equation
  \param X  the resulting matrix
  \param tau  the minimal singular value accepted

  \return 1 if the routine was able to solve the problem, 0 otherwise.

  \warning The SVD matrix must be valid and correspond to \a A.

  \author Philippe Lavoie 
  \date 22 October 1997
*/

int fsvdmatrix::solve(const fmatrix&B, fmatrix&X, real tau) {
  real min_sig, max_sig ;
  minMax(min_sig,max_sig) ;

  if(B.n1 != U.n1){
    fatal("SVDMatrix::solve:The matrix B doesn't have a proper size for this SVD matrix.") ;
  }

  X.resize(V.n1,B.n2) ;

  if(tau==0)
    tau = V.n1*max_sig*epsilon ;
  int stable = 1 ;

  std::vector<real> S(sig.size()) ;

  // doing one column from B at a time 
  int i,j,k ;
  for(j=0;j<B.n2;++j){
    for(i=0;i<V.n2;++i){
      real s = 0.0 ;
      if(sig[i]>tau){
	for(k=0;k<U.n2;++k)
	  s += U(k,i)*B(k,j);
	s /= sig[i] ;
      }
      else
	stable = 0 ;
      S[i] = s ;
    }
    for(i=0;i<V.n1;++i){
      X(i,j) = 0.0 ;
      for(k=0;k<V.n1;++k)
	X(i,j) += V(i,k)*S[k] ;
    }
  }
  return stable ;
}


void fsvdmatrix::sort()
{
	abnormal_ = false;

	fmatrix save_u(U_.n1, U_.n2);
	fmatrix save_v(V_.n1, V_.n2);
	std::vector<real>  save_sig(sig_.size());

	std::vector<bool> stat(sig_.size());

	int count = 0;
	int test_count = 0;

	for (int i = 0 ; i < sig_.size() ;++i)
		stat[i] = false;

	while(count < sig_.size()) {

		int where_ = -1;
		real max_val = -1.0;

		for (int i = 0 ; i < sig_.size() ;++i) {
			if (!stat[i] && sig_[i] > max_val) {
				max_val = sig_[i];
				where_ = i;
			}
		}

		if (where_ != -1) {
			stat[where_] = true;

			save_u.set_col(count, U_.sel_col(where_));
			save_v.set_col(count, V_.sel_col(where_));
			save_sig[count] = sig_[where_];
			count++;
		}
		test_count++;
		if (test_count == sig_.size() + 1) {
			abnormal_ = true;
			break;
		}
	}

	sig_ = save_sig;
	U_ = save_u;
	V_ = save_v;
}

/*!
  \brief Solves the linear system \a A \a X = \a B

  Solves the linear system \a A \a X = \a B. Given \a A and B it
  finds the value of \a X. The routine uses LU decomposition
  if the A matrix is square and it uses SVD decomposition
  otherwise.

  \param   A  the \a A matrix
  \param B  the right hand side of the equation
  \param X  the resulting matrix

  \return 1 if the SVD decomposition worked.

  \author Philippe Lavoie 
  \date 22 October 1997
*/

int solve(const fmatrix& A, 
	   const fmatrix& B, 
	   fmatrix& X)
{
  if(A.n1==A.n2){ 
    // use LU decomposition to solve the problem
    flumatrix lu(A) ;
    X = lu.inverse()^B ;
  }
  else{
    fsvdmatrix svd(A) ;
    return svd.solve(B,X) ;
  }
  return 1;
}

/*!
  \brief finds the inverse of a matrix

  Finds the inverse of a matrix. It uses LU decomposition if
  the matrix is square and it uses SVD decomposition otherwise.

  \param A  the matrix to inverse

  \return The inverse of A

  \warning The A matrix must have a number of rows greater or equal to 
               its number of columns.

  \author Philippe Lavoie 
  \date 22 October 1997
*/

fmatrix inverse(const fmatrix&A)
{
  fmatrix inv ;
  if(A.n1==A.n2){
    flumatrix lu(A) ;
    lu.inverseIn(inv) ;
  }
  else{
    fsvdmatrix svd(A) ;
    svd.inverseIn(inv) ;
  }
  return inv ;
}


};

