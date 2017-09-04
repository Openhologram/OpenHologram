#ifndef __fmatrix_h_
#define __fmatrix_h_

#include "graphics/real.h"
#include <vector>
#include "graphics/sys.h"
#include <math.h>
#include "graphics/misc.h"
#include "graphics/log.h"
#include "graphics/matrix3x3.h"
#include "graphics/matrix3x4.h"
#include "graphics/matrix4x4.h"



namespace graphics {

class fmatrix
{
public:
	
	std::vector<real>	v;
    int n, n1, n2;	
    int grain;
    int a_size() const { return ceil(n, grain); }
    

	void set_zero() {
		for (int i = 0 ; i < n;++i)
			v[i] = 0;
	}

    fmatrix() : n(0), n1(0), n2(0), grain(16), v()
    {
    }


    fmatrix(int n_) : n(n_), n1(1), n2(n_), grain(16), v()
    {
		if(n < 0) fatal("fmatrix: bad fmatrix size"); 
		if (a_size()) v.resize(a_size());
    }

    fmatrix(int n1_, int n2_) : n(n1_*n2_), n1(n1_), n2(n2_), grain(16), v()
    {
		if(n < 0) fatal("fmatrix: bad fmatrix size"); 
		if (a_size()) v.resize(a_size());
    }

    fmatrix(int n1_, int n2_, int _grain) : n(n1_*n2_), n1(n1_), n2(n2_), grain(_grain), v()
    {
		if(n < 0) fatal("fmatrix: bad fmatrix size"); 
		if (a_size()) v.resize(a_size());
    }

    fmatrix(const std::vector<real>& a) : n(a.size()), n1(a.size()), n2(1), grain(16), v()
    {
		if (a_size()) v.resize(a_size());
		for(int i = 0 ; i < n ;++i)
			v[i] = a[i];
    }

    fmatrix(const fmatrix& a) : n(a.n), n1(a.n1), n2(a.n2), grain(a.grain), v()
    {
		if (a_size()) v.resize(a_size());
		*this = a;
    }
    
    inline real* get_array() const { return const_cast<real*>(&(*(v.begin()))); }

    ~fmatrix()
    {
    }

    fmatrix& operator=(int a)
    {
		for(int i = 0; i < n;++i)
			v[i] = a;
		return *this;
    }

    fmatrix& operator=(const std::vector<real>& a)
    {
		resize(a.size(), 1);
		for(int i = 0 ; i < a.size() ;++i)
			v[i] = a[i];
		return *this;
    }

    fmatrix& operator=(const fmatrix& a)
    {
		resize(a.n1, a.n2);
		for(int i = 0; i < a.n;++i)
			v[i] = a.v[i];
		return *this;

    }
 
    real& operator [](int i)
    {
		return ( v[i%n]);
    }

    const real& operator [](int i) const
    {
		return ( v[i%n]);
    }
    real& operator ()(int i)
    {
		return ( v[i%n]);
    }

    const real& operator ()(int i) const
    {
		return ( v[i%n]);
    }

    inline int cal_index(int a, int b) const
    {
		return ((a* n2) + b);
    }

    real& operator ()(int a, int b)
    {
		return v[(a* n2) + b];
    }

    const real& operator ()(int a, int b) const
    {
		return v[(a* n2) + b];
    }

    operator std::vector<real> ()
    {
		if(n1 != 1 && n2 != 1) 
		{    
			LOG("fmatrix: cannot change %d X %d fmatrixrix to std::vector", n1, n2);
			fatal("fmatrix: error type casting");
		}

		if(n1 == 1)
		{
			std::vector<real> c(n2); 
			for(int i = 0 ; i < n2 ;++i)
			c[i] = v[i];
			return c;
		}

		std::vector<real> c(n1);
		for(int i = 0 ; i < n1 ;++i)
			c[i] = v[i];
		return c;
    }

    virtual void resize(int n_new)
    {
		if(n_new < 0) fatal("fmatrix: bad new fmatrix size");
		if(ceil(n,grain) >= ceil(n_new,grain))
			n = n_new;
		else {
			n = n_new;
			v.resize(a_size());
		}
    }

    virtual void resize(int n1_, int n2_)
    {
		n1 = n1_; n2 = n2_;
		int n_new = n1 * n2;
		if(n_new < 0) fatal("fmatrix: bad new fmatrix size");
		resize(n_new);
    }

    void add_row(const std::vector<real>& a)
    {
		if(a.size() != n2)
			fatal("fmatrix: bad row length");
		resize(n1+1, n2);
		for(int i = 0 ; i < a.size() ;++i)
			v[cal_index(n1-1, i)] = a[i];
    }

    void add_col(const std::vector<real>& a)
    {
		int i, j;
		if(a.size() != n1)
			fatal("fmatrix: bad col length");
		fmatrix temp = *this;
		resize(n1, n2+1);
		for(i = 0 ; i < n1 ;++i)
			for(j = 0 ; j < n2-1 ; ++j)
			v[cal_index(i, j)] = temp(i, j);
		for(i = 0 ; i < n1 ;++i)
			v[cal_index(i, (n2-1))] = a[i];
    }

    void ins_row(int index, const std::vector<real>& a)
    {
		int i, j;
		if(a.size() != n2)
			fatal("fmatrix: bad row length");
		fmatrix temp = *this;
		resize(n1+1, n2);
		for(i = 0 ; i < index ;++i)
			for(j = 0 ; j < n2 ; ++j)
			v[cal_index(i,j)] = temp(i,j);

		for(j = 0 ; j < n2 ; ++j)
			v[cal_index(index, j)] = a[j];

		for(i = index+1 ; i < n1 ;++i)
			for(j = 0 ; j < n2 ; ++j)
			v[cal_index(i, j)] = temp(i-1,j);
    }

    void ins_col(int index, const std::vector<real>& a)
    {
		int i, j;
		if(a.size() != n1)
			fatal("fmatrix: bad input col length");

		fmatrix temp = *this;

		resize(n1, n2+1);
		for(i = 0 ; i < n1;++i)
			for(j = 0 ; j < index ; ++j)
			v[cal_index(i,j)] = temp(i,j);

		for(i = 0 ; i < n1 ;++i)
			v[cal_index(i,index)] = a[i];

		for(i = 0 ; i < n1 ;++i)
			for(j = index+1 ; j < n2 ; ++j)
			v[cal_index(i,j)] = temp(i,j-1);
    }

    void del_row(int index)
    {
		for(int i = (index*n2); i < n - n2;++i)
			v[i] = v[i + n2];
		resize(n1-1, n2);
    }

    void del_col(int index)
    {
		int i, j;
		fmatrix temp = *this;
		resize(n1, n2-1);
		for(i = 0 ; i < n1;++i)
			for(j = 0 ; j < index ; ++j)
			v[cal_index(i,j)] = temp(i,j);

		for(i = 0 ; i < n1 ;++i)
			for(j = index+1 ; j < temp.n2 ; ++j)
			v[cal_index(i,j-1)] = temp(i,j);
    }

    std::vector<real> sel_col(int index) const
    {
		std::vector<real> temp(n1);
		for(int i = 0 ; i < n1 ;++i)
			temp[i] = v[cal_index(i, index)];
		return temp;
    }

    std::vector<real> sel_row(int index) const
    {
		std::vector<real> temp(n2);
		for(int i = 0 ; i < n2 ;++i)
			temp[i] = v[cal_index(index, i)];
		return temp;
    }

    void set_col(int index, const std::vector<real>& a)
    {
		int min_num = (a.size()< n1)?a.size():n1;
		for(int i = 0 ; i < min_num ;++i)
			v[cal_index(i, index)] = a[i];
    }
	void set_col(int index, real a)
    {
		int min_num = n1;
		for(int i = 0 ; i < min_num ;++i)
			v[cal_index(i, index)] = a;
    }
    void set_row(int index, const std::vector<real>& a)
    {
		int min_num = (a.size()<n2)?a.size():n2;
		for(int i = 0 ; i < min_num ;++i)
			v[cal_index(index, i)] = a[i];
    }

	void set_row(int index, const fmatrix& a)
    {
		int min_num = (a.n<n2)?a.n:n2;
		for(int i = 0 ; i < min_num ;++i)
			v[cal_index(index, i)] = a[i];
    }
    void set_row(int index, real a)
    {
		int min_num = n2;
		for(int i = 0 ; i < min_num ;++i)
			v[cal_index(index, i)] = a;
    }
	void set_block(int k1, int k2, const fmatrix& b)
	{
		int kk1 = (n1<k1 + b.n1)?n1:k1+b.n1;
		int kk2 = (n2<k2 + b.n2)?n2:k2+b.n2;

		for (int i = k1 ; i < kk1 ;++i) {
			for ( int j = k2 ; j < kk2 ; ++j) {
				(*this)(i, j) = b(i-k1, j-k2);
			}
		}
	}
	void set_block(int k1, int k2, const matrix3x3& b)
	{
		int kk1 = (n1<k1 + 3)?n1:k1+3;
		int kk2 = (n2<k2 + 3)?n2:k2+3;

		for (int i = k1 ; i < kk1 ;++i) {
			for ( int j = k2 ; j < kk2 ; ++j) {
				(*this)(i, j) = b(i-k1, j-k2);
			}
		}
	}
	void set_block(int k1, int k2, const matrix4x4& b)
	{
		int kk1 = (n1< k1 + 4)?n1:k1+4;
		int kk2 = (n2< k2 + 4)?n2:k2+4;

		for (int i = k1 ; i < kk1 ;++i) {
			for ( int j = k2 ; j < kk2 ; ++j) {
				(*this)(i, j) = b(i-k1, j-k2);
			}
		}
	}
	void set_block(int k1, int k2, const matrix3x4& b)
	{
		int kk1 = (n1< k1 + 3)?n1:k1+3;
		int kk2 = (n2< k2 + 4)?n2:k2+4;

		for (int i = k1 ; i < kk1 ;++i) {
			for ( int j = k2 ; j < kk2 ; ++j) {
				(*this)(i, j) = b(i-k1, j-k2);
			}
		}
	}

	fmatrix get_block(int k1, int k2, int kk1, int kk2) const
	{
		kk1 = (kk1<n1-k1)?kk1:n1-k1;
		kk2 = (kk2<n2-k2)?kk2:n2-k2;

		fmatrix ret(kk1, kk2);
		for (int i = 0 ; i < kk1 ;++i) 
			for (int j = 0 ; j < kk2 ; ++j)
				ret(i,j) = (*this)(i + k1, j + k2);
		return ret;
	}
	void get_block(int k1, int k2, matrix3x3& out) const
	{
		out.setZero();

		int kk1 = (3<n1-k1)?3:n1-k1;
		int kk2 = (3, n2-k2)?3:n2-k2;

		for (int i = 0 ; i < kk1 ;++i) 
			for (int j = 0 ; j < kk2 ; ++j)
				out(i,j) = (*this)(i + k1, j + k2);
	}
	void get_block(int k1, int k2, matrix3x4& out) const
	{
		out.setZero();

		int kk1 = (3< n1-k1)?3:n1-k1;
		int kk2 = (4< n2-k2)?4:n2-k2;

		for (int i = 0 ; i < kk1 ;++i) 
			for (int j = 0 ; j < kk2 ; ++j)
				out(i,j) = (*this)(i + k1, j + k2);
	}
	
	void get_block(int k1, int k2, matrix4x4& out) const
	{
		out.setZero();

		int kk1 = (4< n1-k1)?4:n1-k1;
		int kk2 = (4, n2-k2)?4:n2-k2;

		for (int i = 0 ; i < kk1 ;++i) 
			for (int j = 0 ; j < kk2 ; ++j)
				out(i,j) = (*this)(i + k1, j + k2);
	}

	fmatrix col(int index) const
	{
		fmatrix ret(n1, 1);
		for (int i = 0 ; i < n1 ;++i) 
			ret(i,0) = v[cal_index(i, index)];
		return ret;
	}

	std::vector<real> col_vector(int index) const
	{
		std::vector<real> ret(n1);
		for (int i = 0 ; i < n1 ;++i) 
			ret[i] = v[cal_index(i, index)];
		return ret;
	}
	fmatrix row(int index) const
	{
		fmatrix ret(1, n2);
		for (int i = 0 ; i < n2 ;++i) 
			ret(0,i) = v[cal_index(index, i)];
		return ret;
	}

    fmatrix transpose()  const
    {
		fmatrix mt(n2, n1);

		for (int i = 0; i < mt.n1;++i) {
			for (int j = 0; j < mt.n2; ++j) {
			mt(i,j) = v[cal_index(j,i)];
			}
		}
		return mt;
    }

    void set_diagonal_elements(const real& rhs) 
    {
		int i, iend;

		iend = n1;
		if ( iend > n2 )
			iend = n2;

		for (i = iend-1; i >=0; --i)
			(*this)(i,i) = rhs;

    }

	static fmatrix diagonal_matrix(const std::vector<real>& val)
	{
		fmatrix ret(val.size(), val.size());
		ret.set_zero();
		for (int i = 0 ; i < val.size() ;++i)
			ret(i,i) = val[i];
		return ret;
	}
	static fmatrix diagonal_matrix(const vec3& val)
	{
		fmatrix ret(3, 3);
		ret.set_zero();
		for (int i = 0 ; i < 3 ;++i)
			ret(i,i) = val[i];
		return ret;
	}
	real norm() const {
		real sum = 0;
		for (int i = 0 ; i < n ;++i)
			sum += (v[i] * v[i]);
		return sqrt(sum);
	}
	real squaredNorm() const {
		real sum = 0;
		for (int i = 0 ; i < n ;++i)
			sum += (v[i] * v[i]);
		return sum;
	}

	real rowwise_sum(int i) const {
		real sum = 0;
		for (int j = 0 ; j < n2; ++j) {
			sum += (*this)(i,j);
		}
		return sum;
	}

	real rowwise_squared_norm(int i) const {
		real sum = 0;
		for (int j = 0 ; j < n2; ++j) {
			sum += ((*this)(i,j) * (*this)(i,j));
		}
		return sqrt(sum);
	}
	real columnwise_sum(int i) const {
		real sum = 0;
		for (int j = 0 ; j < n1; ++j) {
			sum += (*this)(j,i);
		}
		return sum;
	}

	void make_identity() {
		if (n1 != n2) return;
		set_zero();
		for (int i = 0 ; i < n1; i++)
			(*this)(i,i) = 1.0;
	}
};


//|
//| binary : component-wise operations
//|



fmatrix operator + (const fmatrix& a, const fmatrix& b);

fmatrix operator + (real a, const fmatrix& b);


fmatrix operator + (const fmatrix& a, real b);




fmatrix operator - (const fmatrix& a, const fmatrix& b);

fmatrix operator - (real a, const fmatrix& b);

fmatrix operator - (const fmatrix& a, real b);




fmatrix operator * (const fmatrix& a, const fmatrix& b);


fmatrix operator * (real a, const fmatrix& b);


fmatrix operator * (const fmatrix& a, real b);




fmatrix operator / (const fmatrix& a, const fmatrix& b);

fmatrix operator / (real a, const fmatrix& b);


fmatrix operator / (const fmatrix& a, real b);



//| cumulative : component-wise operations



fmatrix operator += (fmatrix& a, const fmatrix& b);

fmatrix operator += (fmatrix& a, real b);




fmatrix operator -= (fmatrix& a, const fmatrix& b);

fmatrix operator -= (fmatrix& a, real b);




fmatrix operator *= (fmatrix& a, const fmatrix& b);

fmatrix operator *= (fmatrix& a, real b);




fmatrix operator /= (fmatrix& a, const fmatrix& b);

fmatrix operator /= (fmatrix& a, real b);

real sum(const std::vector<real>& a);
real sum(const fmatrix& a);
//|
//| matrix multiplication
//|

fmatrix operator ^ (const fmatrix& a, const fmatrix& b);


fmatrix operator ^ (const fmatrix& a, const std::vector<real>& b);


fmatrix operator ^ (const std::vector<real>& a, const fmatrix& b);

class flumatrix : public fmatrix 
{
public:
    flumatrix(int r, int c) : fmatrix(r,c), pivot(pivot_) { pivot_.resize(r); }
    flumatrix()  : fmatrix(), pivot(pivot_) { }
    flumatrix(const flumatrix& lu): fmatrix(lu), pivot(pivot_) { pivot_ = lu.pivot_;}
    flumatrix(const fmatrix& a): fmatrix(a), pivot(pivot_) { decompose(a) ; }

    void resize(const int r, const int c) { fmatrix::resize(r,c) ; pivot_.resize(r) ; }
    flumatrix& operator=(const flumatrix&);
    flumatrix& decompose(const fmatrix &a);
    real determinant() ;

    fmatrix inverse() const;
    void inverseIn(fmatrix&) const;

	void solve(const fmatrix& B,  fmatrix& X) 
	{
		X = inverse()^B;
	}

    const std::vector<int>& pivot ;

private:
    std::vector<int> pivot_ ;
    real  min_pivot_value;
    int	  rank;

protected:
    int errval ;
    int sign ;
};



class fsvdmatrix 
{
public:
    inline fsvdmatrix():U(U_), V(V_), sig(sig_) { ; }
    
	
	inline fsvdmatrix(const fmatrix& A)
	: U(U_),V(V_),sig(sig_), abnormal_(false)
	{
		if (A.n1 < A.n2) {
			fmatrix A_extended(A.n2, A.n2);
			A_extended.set_zero();
			for (int i = 0 ; i < A.n1 ;++i)
			for (int j = 0 ; j < A.n2 ; ++j)
				A_extended(i,j) = A(i,j);
			decompose(A_extended);
		}
		else decompose(A) ;
	};

	inline fsvdmatrix(const matrix3x3& A)
	: U(U_),V(V_),sig(sig_), abnormal_(false)
	{
		fmatrix B(3,3);
		B.set_block(0,0, A);
		decompose(B) ;
	};	

    const fmatrix&	U ;
    const fmatrix&	V ;
    const std::vector<real>&	sig ;

    int	    decompose(const fmatrix& A) ;
    void    minMax(real& min_sig, real& max_sig) const;
    real    q_cond_number(void) const;	

    void    cut_off(const real min_sig);
    void    inverseIn(fmatrix& inv, real tau=0) ;
    fmatrix inverse(real tau=0) ;
    int	    solve(const fmatrix& B, fmatrix& X, real tau=0) ;


	void	sort();

//protected:
    int		M,  N;		    //!< Dimensions of the problem (M > N)
    fmatrix	U_;	    //!< M * M orthogonal matrix \a U
    fmatrix	V_;	    //!< N * N orthogonal matrix \a V
	std::vector<real>	sig_;	    //!< Vector of \a N unordered singular values

	bool		abnormal_;

    // Internal procedures used in SVD
    real    left_householder(fmatrix& A, const int i);
    real    right_householder(fmatrix& A, const int i);
    real    bidiagonalize(std::vector<real>& super_diag, const fmatrix& _A);

    void    rotate(fmatrix& U, const int i, const int j,
		const real cos_ph, const real sin_ph);
    void    rip_through(std::vector<real>& super_diag, const int k, const int l, const real eps);
    int	    get_submatrix_to_work_on(std::vector<real>& super_diag, const int k, const real eps);
    void    diagonalize(std::vector<real>& super_diag, const real eps);

};

int solve(const fmatrix& A, const fmatrix& B, fmatrix& C) ;
fmatrix inverse(const fmatrix& A) ;
}
#endif