/*=============================================================================
        File: matrixMat.h
     Purpose:       
    Revision: $Id: matrixMat.h,v 1.2 2002/05/13 21:07:45 philosophil Exp $
  Created by: Philippe Lavoie          (22 Oct, 1997)
 Modified by: 

 Copyright notice:
          Copyright (C) 1996-1997 Philippe Lavoie
 
	  This library is free software; you can redistribute it and/or
	  modify it under the terms of the GNU Library General Public
	  License as published by the Free Software Foundation; either
	  version 2 of the License, or (at your option) any later version.
 
	  This library is distributed in the hope that it will be useful,
	  but WITHOUT ANY WARRANTY; without even the implied warranty of
	  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
	  Library General Public License for more details.
 
	  You should have received a copy of the GNU Library General Public
	  License along with this library; if not, write to the Free
	  Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

  I (Daehyun Kim) ported this code to my existing codes to avoid conflicts..
=============================================================================*/
#ifndef __matrix_math_h_
#define __matrix_math_h_

#include    "matrix.h"


namespace graphics {
/*!
\class LUMatrix matrixMat.h matrix/matrixMat.h
\brief a class for the LU decomposition of a matrix

\latexonly
This class was adapted from a class in the Kalman library
written by Skip Carter (\verb|skip@taygeta.oc.nps.navy.mil|)
available from 
\verb|ftp:://usc.edu/pub/C-numanal/kalman.tar.gz|
\endlatexonly
\htmlonly
This class was adapted from a class in the 
<A HREF="ftp:://usc.edu/pub/C-numanal/kalman.tar.gz">Kalman library</A>
written by Skip Carter 
(<A HREF="mailto:skip@taygeta.oc.nps.navy.mil">skip@taygeta.oc.nps.navy.mil</A>
\endhtmlonly

It performs the LU decomposition of a matrix. It can be
used to find it's inverse and to solve linear systems.

This class can only be used with floating point values
(float or double).

\author Skip Carter
\author Philippe Lavoie 
\date 22 October 1997
*/
template <class T>
class LUMatrix : public gmatrix<T> 
{
public:
    LUMatrix(int r, int c) : gmatrix<T>(r,c), pivot(pivot_) { pivot_.resize(r); }
    LUMatrix()  : gmatrix<T>(), pivot(pivot_) { }
    LUMatrix(const LUMatrix<T>& lu): gmatrix<T>(lu), pivot(pivot_) { pivot_ = lu.pivot_;}
    LUMatrix(const gmatrix<T>& a): gmatrix<T>(a), pivot(pivot_) { decompose(a) ; }

    void resize(const int r, const int c) { gmatrix<T>::resize(r,c) ; pivot_.resize(r) ; }
    LUMatrix& operator=(const LUMatrix<T>&);
    LUMatrix& decompose(const gmatrix<T> &a);
    T determinant() ;

    gmatrix<T> inverse() const;
    void inverseIn(gmatrix<T>&) const;

    const vector<int>& pivot ;

private:
    vector<int> pivot_ ;
    real  min_pivot_value;
    int	  rank;

protected:
    int errval ;
    int sign ;
};


/*!
\class SVDMatrix  matrixMat.h matrix/matrixMat.h
\brief A matrix for the SVD decomposition

\latexonly
This matrix was adapted from the Numerical Math Class Library
developed by Oleg Kiselyov and available from
\verb|http://www.lh.com/oleg/ftp/packages.html|.
\endlatexonly
\htmlonly
This matrix was adapted from the 
<A HREF="http://www.lh.com/oleg/ftp/packages.html">Numerical Math 
Class Library</A>
developed by Oleg Kiselyov.
\endhtmlonly

This class can only be used with floating point values
(float or double).

\latexonly
The following comments are from Oleg Kiselyov.

Singular Value Decomposition of a rectangular matrix
$ A = U  Sig  V'$
where matrices $U$ and $V$ are orthogonal and $Sig$ is a 
diagonal matrix.

The singular value decomposition is performed by constructing 
an SVD  object from an $M\times N$ matrix $A$ with $M \ge N$
(that is, at least as many rows as columns). Note, in case 
$M > N,$ matrix $Sig$ has to be a $M \times N$ diagonal
matrix. However, it has only $N$ diagonal elements, which 
we store in a vector sig.

{\bf Algorithm:}	Bidiagonalization with Householder 
reflections followed by a modification of a QR-algorithm. 
For more details, see G.E. Forsythe, M.A. Malcolm, C.B. Moler 
Computer methods for mathematical computations, Prentice-Hall, 
1977.  However, in the present implementation, matrices $U$ 
and $V$ are computed right away rather than delayed until 
after all Householder reflections.
 
This code is based for the most part on a Algol68 code I 
(Oleg Kiselyov) wrote ca. 1987.

Look at the source code for more information about the 
algorithm.
\endlatexonly

\htmlonly
<P>
The following comments are from Oleg Kiselyov.
</P>
<P>
Singular Value Decomposition of a rectangular matrix
<code>A=U Sig V'</code>
where matrices U and V are orthogonal and Sig is a 
diagonal matrix.
</p>
<P>
The singular value decomposition is performed by constructing 
an SVD  object from an M*N matrix A with M \ge N
(that is, at least as many rows as columns). Note, in case 
M &gt; N, matrix  Sig has to be a M*N diagonal
matrix. However, it has only N diagonal elements, which 
we store in a vector sig.
</p>
<P>
<strong>Algorithm:</strong> Bidiagonalization with Householder 
reflections followed by a modification of a QR-algorithm. 
For more details, see G.E. Forsythe, M.A. Malcolm, C.B. Moler 
Computer methods for mathematical computations, Prentice-Hall, 
1977.  However, in the present implementation, matrices  U
and V are computed right away rather than delayed until 
after all Householder reflections.
</p>
<p>
This code is based for the most part on a Algol68 code I 
(Oleg Kiselyov) wrote ca. 1987.
</P>
<p>
Look at the source code for more information about the 
algorithm.
</p>
\endhtmlonly

\author Oleg Kiselyov
\author Philippe Lavoie 
\date 22 Oct. 1997
*/
template <class T>
class SVDMatrix 
{
public:
    inline SVDMatrix():U(), V(), sig(sig_) { ; }
    
	
	inline SVDMatrix(const gmatrix<T>& A)
	: U(U_),V(V_),sig(sig_), abnormal_(false)
	{
		decompose(A) ;
	};
	
    const gmatrix<T>&	U ;
    const gmatrix<T>&	V ;
    const vector<T>&	sig ;

    int	    decompose(const gmatrix<T>& A) ;
    void    minMax(T& min_sig, T& max_sig) const;
    real    q_cond_number(void) const;	

    void    cut_off(const real min_sig);
    void    inverseIn(gmatrix<T>& inv, real tau=0) ;
    gmatrix<T> inverse(real tau=0) ;
    int	    solve(const gmatrix<T>& B, gmatrix<T>& X, real tau=0) ;


	void	sort();

//protected:
    int		M,  N;		    //!< Dimensions of the problem (M > N)
    gmatrix<T>	U_;	    //!< M * M orthogonal matrix \a U
    gmatrix<T>	V_;	    //!< N * N orthogonal matrix \a V
    vector<T>	sig_;	    //!< Vector of \a N unordered singular values

	bool		abnormal_;

    // Internal procedures used in SVD
    real    left_householder(gmatrix<T>& A, const int i);
    real    right_householder(gmatrix<T>& A, const int i);
    real    bidiagonalize(vector<T>& super_diag, const gmatrix<T>& _A);

    void    rotate(gmatrix<T>& U, const int i, const int j,
		const real cos_ph, const real sin_ph);
    void    rip_through(vector<T>& super_diag, const int k, const int l, const real eps);
    int	    get_submatrix_to_work_on(vector<T>& super_diag, const int k, const real eps);
    void    diagonalize(vector<T>& super_diag, const real eps);

};

template <class T> int solve(const gmatrix<T>& A, const gmatrix<T>& B, gmatrix<T>& C) ;
template <class T> gmatrix<T> inverse(const gmatrix<T>& A) ;

};  // namespace

#endif
