#ifndef __matrix_h
#define __matrix_h
//|
//| matrix.e
//| Dae Hyun Kim
//|

#include "graphics/sys.h"
#include "graphics/real.h"

#include <math.h>
#include "graphics/log.h"
#include "graphics/vec.h"
#include "graphics/vector.h"
#include "graphics/misc.h"
#include "graphics/ivec.h"
#include "graphics/epsilon.h"
#include "graphics/minmax.h"

namespace graphics {
 
//|
//| Normal 2dimensional matrix (n1 x n2)
//|
struct matrix
{

    vector<real> v;

    int n;
    int n1;					// row count
    int n2;					// column count
    int grain;          			// memory block size

    int a_size() const { return ceil(n, grain); }

    matrix() : n(0), n1(0), n2(0), grain(16), v()
    {
    }

    matrix(int n_) : n(n_), n1(1), n2(n_), grain(16), v()
    {
	if(n < 0) fatal("matrix : bad mat size"); 
	
	if (a_size()) v.resize(a_size());
    }

    matrix(int n1_, int n2_) : n(n1_*n2_), n1(n1_), n2(n2_), grain(16), v()
    {
	if(n < 0) fatal("matrix : bad mat size"); 
	if (a_size()) v.resize(a_size());
    }

    matrix(int n1_, int n2_, int _grain) : n(n1_*n2_), n1(n1_), n2(n2_), grain(_grain), v()
    {
	if(n < 0) fatal("matrix : bad mat size"); 
	if (a_size()) v.resize(a_size());
    }

    
    matrix(const vector<real>& a) : n(a.size()), n1(a.size()), n2(1), grain(16), v()
    {
	if (a_size()) v.resize(a_size());
	for(int i = 0 ; i < n ;++i)
	    v[i] = a[i];
    }

    
    matrix(const vec2& a) : n(a.n), n1(a.n), n2(1), grain(16), v()
    {
	if (a_size()) v.resize(a_size());	
	for(int i = 0 ; i < n ;++i)
	    v[i] = a.v[i];
    }
    
    matrix(const vec3& a) : n(a.n), n1(a.n), n2(1), grain(16), v()
    {
	if (a_size()) v.resize(a_size());
	for(int i = 0 ; i < n ;++i)
	    v[i] = a.v[i];
    }
    
    matrix(const vec4& a) : n(a.n), n1(a.n), n2(1), grain(16), v()
    {
	if (a_size()) v.resize(a_size());
	for(int i = 0 ; i < n ;++i)
	    v[i] = a.v[i];
    }
    

    matrix(const matrix& a) : n(a.n), n1(a.n1), n2(a.n2), grain(a.grain), v()
    {
	if (a_size()) v.resize(a_size());
	*this = a;
    }
    
    ~matrix()
    {
    }

    matrix& operator=(int a);

    
    matrix& operator=(const vec2& a);    
    matrix& operator=(const vec3& a);    
    matrix& operator=(const vec4& a);
    

    real* get_array() { return v.get_array(); }
    matrix& operator=(const vector<real>& a);
    matrix& operator = (const matrix& a);
 
    real& operator [](int i)
    {
	return (i>=0?v[i % n]:v[n - ((-i)%n)]);
    }

    const real& operator [](int i) const
    {
	return (i>=0?v[i % n]:v[n - ((-i)%n)]);
    }

    int cal_index(int a, int b) const
    {
	return (((a >= 0 ? a : n1 -((-a)%n1))%n1) * n2) + ((b >= 0 ? b : n2 -((-b)%n2))%n2);
    }

    real& operator ()(int a, int b)
    {
	return v[cal_index(a,b)];
    }

    const real& operator ()(int a, int b) const
    {
	return v[cal_index(a,b)];
    }

    operator vector<real> ();			    // type conversion operator

    void resize(int n_new);
    void resize(int n1_, int n2_);

    void add_row(const vector<real>& a);
    void add_col(const vector<real>& a);
    void ins_row(int index, const vector<real>& a); // insert a in the index'th row
    void ins_col(int index, const vector<real>& a); // insert a in the index'th column
    void del_row(int index);

    void del_col(int index);

    vector<real> sel_col(int index) const;	    // read indexth column
    vector<real> sel_row(int index) const;	    // read indexth row

    void rep_col(int index, const vector<real>& a); // replace indexth colum with a
    void rep_row(int index, const vector<real>& a); // replace indexth row with a

    bool swap_rows(int ith, int jth); // swap ith row and jth row
    bool swap_cols(int ith, int jth); // swap ith col and jth col

public:

    // Description:
    //	Just usual array function
    // Refernece:
    //	graphics::vector
    void add(real a);		    // attach a at the end of this matrix
    void ins(int index, real a);    // insert at (*this)[this.size()]
    void del(int index);	    // delete ith element, and resize the arrasy.
    
    
    bool is_affine() const;

    inline real det3x3(real m00,real m01,real m02,
			real m10,real m11,real m12,
			real m20,real m21,real m22) const
    {
	return m00*(m11*m22-m21*m12) -
	       m01*(m10*m22-m20*m12) +
	       m02*(m10*m21-m20*m11);
    }

    void mat_zero(int r, int c) 
    {

	int i, j;
	resize(r,c);
	for ( i = 0 ; i < r ;++i)
	    for ( j = 0 ; j < c ; j++ )
		v[cal_index(i,j)] = 0.0;
    }

    void mat_zero(int size) 
    {
	mat_zero(size,size);
    }

    int  rows() const
    { 
	return n1; 
    }

    int  cols() const
    { 
	return n2; 
    }

    bool is_null() const
    { 
	return n1==0 && n2==0; 
    }

    real determ() const;
    

    matrix adjoint(int i, int j) const;

    matrix inverse() const;
    matrix transpose() const;

    void identity(int dim = 4);

    void set_diagonal_elements(real rhs)
    {
	int iend = (n1 > n2) ? n2 : n1;
	for (int i = iend-1; i >=0; --i)
	    (*this)(i,i) = rhs;

    }

};

//|
//| binary : component-wise operations
//|


matrix operator + (const matrix& a, const matrix& b)
;

matrix operator + (real a, const matrix& b)
;

matrix operator + (const matrix& a, real b)
;



matrix operator - (const matrix& a, const matrix& b)
;

matrix operator - (real a, const matrix& b)
;

matrix operator - (const matrix& a, real b)
;



matrix operator * (const matrix& a, const matrix& b)
;

matrix operator * (real a, const matrix& b)
;

matrix operator * (const matrix& a, real b)
;



matrix operator / (const matrix& a, const matrix& b)
;

matrix operator / (real a, const matrix& b)
;

matrix operator / (const matrix& a, real b)
;



//| cumulative : component-wise operations


matrix operator += (matrix& a, const matrix& b)
;

matrix operator += (matrix& a, real b)
;



matrix operator -= (matrix& a, const matrix& b)
;

matrix operator -= (matrix& a, real b)
;



matrix operator *= (matrix& a, const matrix& b)
;

matrix operator *= (matrix& a, real b)
;



matrix operator /= (matrix& a, const matrix& b)
;

matrix operator /= (matrix& a, real b)
;



//| logical : component-wise operations


int operator == (const matrix& a, const matrix& b)
;

int operator == (real a, const matrix& b)
;

int operator == (const matrix& a, real b)
;



int operator < (const matrix& a, const matrix& b)
;

int operator < (real a, const matrix& b)
;

int operator < (const matrix& a, real b)
;



int operator <= (const matrix& a, const matrix& b)
;

int operator <= (real a, const matrix& b)
;

int operator <= (const matrix& a, real b)
;



int operator > (const matrix& a, const matrix& b)
;

int operator > (real a, const matrix& b)
;

int operator > (const matrix& a, real b)
;



int operator >= (const matrix& a, const matrix& b)
;

int operator >= (real a, const matrix& b)
;

int operator >= (const matrix& a, real b)
;



//|
//| matrix multiplication
//|
matrix operator ^ (const matrix& a, const matrix& b)
;

matrix operator ^ (const matrix& a, const vector<real>& b)
;

matrix operator ^ (const vector<real>& a, const matrix& b)
;

int operator != (const matrix& a, const matrix& b)
;


real sum(const matrix& a)
;

void print(matrix& a)
;


template <class T> 
class gmatrix
{
public:
	
	vector<T>	v;
    int n;
    int n1;		// when It saves nurbs control points, stride0 = n2 * T:n
    int n2;		//				       stride1 = T:n
    int grain;
    int a_size() const { return ceil(n, grain); }
    
    gmatrix() : n(0), n1(0), n2(0), grain(16), v()
    {
    }

    gmatrix(int n_) : n(n_), n1(1), n2(n_), grain(16), v()
    {
	if(n < 0) fatal("gmatrix: bad gmatrix size"); 
	if (a_size()) v.resize(a_size());
    }

    gmatrix(int n1_, int n2_) : n(n1_*n2_), n1(n1_), n2(n2_), grain(16), v()
    {
	if(n < 0) fatal("gmatrix: bad gmatrix size"); 
	if (a_size()) v.resize(a_size());
    }

    gmatrix(int n1_, int n2_, int _grain) : n(n1_*n2_), n1(n1_), n2(n2_), grain(_grain), v()
    {
	if(n < 0) fatal("gmatrix: bad gmatrix size"); 
	if (a_size()) v.resize(a_size());
    }

    gmatrix(const vector<T>& a) : n(a.size()), n1(a.size()), n2(1), grain(16), v()
    {
	if (a_size()) v.resize(a_size());
	for(int i = 0 ; i < n ;++i)
	    v[i] = a[i];
    }

    gmatrix(const gmatrix& a) : n(a.n), n1(a.n1), n2(a.n2), grain(a.grain), v()
    {
	if (a_size()) v.resize(a_size());
	*this = a;
    }

    
    inline T* get_array() const { return v.get_array(); }

    ~gmatrix()
    {
    }

    gmatrix& operator=(int a)
    {
	for(int i = 0; i < n;++i)
	    v[i] = a;
	return *this;
    }

    gmatrix& operator=(const vector<T>& a)
    {
	resize(a.size(), 1);
	for(int i = 0 ; i < a.size() ;++i)
	    v[i] = a[i];
	return *this;
    }

    gmatrix& operator=(const gmatrix& a)
    {
	resize(a.n1, a.n2);
	for(int i = 0; i < a.n;++i)
	    v[i] = a.v[i];
	return *this;

    }
 
    T& operator [](int i)
    {
	return ( i >= 0 ? v[i % n]:v[n -((-i)%n)]);
    }

    const T& operator [](int i) const
    {
	return ( i >= 0 ? v[i % n]:v[n -((-i)%n)]);
    }

    inline int cal_index(int a, int b) const
    {
	return (((a >= 0 ? a : n1 -((-a)%n1))%n1) * n2) + ((b >= 0 ? b : n2 -((-b)%n2))%n2);
    }

    T& operator ()(int a, int b)
    {
	return v[cal_index(a, b)];
    }

    const T& operator ()(int a, int b) const
    {
	return v[cal_index(a, b)];
    }

    operator vector<T> ()
    {
	if(n1 != 1 && n2 != 1) 
	{    
	    LOG("gmatrix: cannot change %d X %d gmatrixrix to vector", n1, n2);
	    fatal("gmatrix: error type casting");
	}

	if(n1 == 1)
	{
	    vector<T> c(n2); 
	    for(int i = 0 ; i < n2 ;++i)
		c[i] = v[i];
	    return c;
	}

	vector<T> c(n1);
	for(int i = 0 ; i < n1 ;++i)
	    c[i] = v[i];
	return c;
    }

    virtual void resize(int n_new)
    {
	if(n_new < 0) fatal("gmatrix: bad new gmatrix size");
	if(ceil(n,grain) >= ceil(n_new,grain))
	    n = n_new;
	else {
	    n = n_new;
	    v.resize(n);
	}
    }

    virtual void resize(int n1_, int n2_)
    {
	n1 = n1_; n2 = n2_;
	int n_new = n1 * n2;
	if(n_new < 0) fatal("gmatrix: bad new gmatrix size");
	resize(n_new);
    }

    void add_row(const vector<T>& a)
    {
	if(a.size() != n2)
	    fatal("gmatrix: bad row length");
	resize(n1+1, n2);
	for(int i = 0 ; i < a.size() ;++i)
	    v[cal_index(n1-1, i)] = a.v[i];
    }

    void add_col(const vector<T>& a)
    {
	int i, j;
	if(a.size() != n1)
	    fatal("gmatrix: bad col length");
	gmatrix temp = *this;
	resize(n1, n2+1);
	for(i = 0 ; i < n1 ;++i)
	    for(j = 0 ; j < n2-1 ; ++j)
		v[cal_index(i, j)] = temp(i, j);
	for(i = 0 ; i < n1 ;++i)
	    v[cal_index(i, (n2-1))] = a.v[i];
    }

    void ins_row(int index, const vector<T>& a)
    {
	int i, j;
	if(a.size() != n2)
	    fatal("gmatrix: bad row length");
	gmatrix temp = *this;
	resize(n1+1, n2);
	for(i = 0 ; i < index ;++i)
	    for(j = 0 ; j < n2 ; ++j)
		v[cal_index(i,j)] = temp(i,j);

	for(j = 0 ; j < n2 ; ++j)
	    v[cal_index(index, j)] = a.v[j];

	for(i = index+1 ; i < n1 ;++i)
	    for(j = 0 ; j < n2 ; ++j)
		v[cal_index(i, j)] = temp(i-1,j);
    }

    void ins_col(int index, const vector<T>& a)
    {
	int i, j;
	if(a.size() != n1)
	    fatal("gmatrix: bad input col length");

	gmatrix temp = *this;

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
	gmatrix temp = *this;
	resize(n1, n2-1);
	for(i = 0 ; i < n1;++i)
	    for(j = 0 ; j < index ; ++j)
		v[cal_index(i,j)] = temp(i,j);

	for(i = 0 ; i < n1 ;++i)
	    for(j = index+1 ; j < temp.n2 ; ++j)
		v[cal_index(i,j-1)] = temp(i,j);
    }

    vector<T> sel_col(int index) const
    {
	vector<T> temp(n1);
	for(int i = 0 ; i < n1 ;++i)
	    temp[i] = v[cal_index(i, index)];
	return temp;
    }

    vector<T> sel_row(int index) const
    {
	vector<T> temp(n2);
	for(int i = 0 ; i < n2 ;++i)
	    temp[i] = v[cal_index(index, i)];
	return temp;
    }

    void rep_col(int index, const vector<T>& a)
    {
	int min_num = min(a.size(), n1);
	for(int i = 0 ; i < min_num ;++i)
	    v[cal_index(i, index)] = a[i];
    }

    void rep_row(int index, const vector<T>& a)
    {
	int min_num = min(a.size(), n2);
	for(int i = 0 ; i < min_num ;++i)
	    v[cal_index(index, i)] = a[i];
    }

    void add(const T& a)
    {
	if(n1 == 1 || n2 == 1) {
	    if(n1 == 1) resize(1, n + 1);
	    else resize(n + 1, 1);
	    v[n - 1] = a;
	}
    }

    void ins(int index, const T& a)
    {
	if(n1 == 1 || n2 == 1) {
	    if(n1 == 1) resize(1, n + 1);
	    else resize(n+1, 1);
	    for(int i = n - 1; i > index; i--)
		v[i] = v[i - 1];
	    v[index] = a;
	}
    }

    void del(int index)
    {
	if(n1 == 1 || n2 == 1) {
	    for(int i = index; i < n - 1;++i)
		v[i] = v[i + 1];
	    if(n1 == 1) resize(1, n - 1);
	    else resize(n-1, 1);
	}
    }

    gmatrix transpose()  const
    {
	gmatrix mt(n2, n1);

	for (int i = 0; i < mt.n1;++i) {
	    for (int j = 0; j < mt.n2; ++j) {
		mt(i,j) = v[cal_index(j,i)];
	    }
	}
	return mt;
    }

    void set_diagonal_elements(const T& rhs) 
    {
	int i, iend;

	iend = n1;
	if ( iend > n2 )
	    iend = n2;

	for (i = iend-1; i >=0; --i)
	    (*this)(i,i) = rhs;

    }
};


//|
//| binary : component-wise operations
//|


template <class T> 
gmatrix<T> operator + (const gmatrix<T>& a, const gmatrix<T>& b)
{
    gmatrix<T> c(a.n1, a.n2);
    for(int i = 0; i < a.n;++i) {
	c[i] = a.v[i] + b.v[i];
    }
    return c;
}

template <class T> 
gmatrix<T> operator + (real a, const gmatrix<T>& b)
{
    gmatrix<T> c(b.n1, b.n2);
    for(int i = 0; i < b.n;++i) {
	c[i] = a + b.v[i];
    }
    return c;
}

template <class T> 
gmatrix<T> operator + (const gmatrix<T>& a, real b)
{
    gmatrix<T> c(a.n1, a.n2);
    for(int i = 0; i < a.n;++i) {
	c[i] = a.v[i] + b;
    }
    return c;
}



template <class T> 
gmatrix<T> operator - (const gmatrix<T>& a, const gmatrix<T>& b)
{
    gmatrix<T> c(a.n1, a.n2);
    for(int i = 0; i < a.n;++i) {
	c[i] = a.v[i] - b.v[i];
    }
    return c;
}

template <class T> 
gmatrix<T> operator - (real a, const gmatrix<T>& b)
{
    gmatrix<T> c(b.n1, b.n2);
    for(int i = 0; i < b.n;++i) {
	c[i] = a - b.v[i];
    }
    return c;
}

template <class T> 
gmatrix<T> operator - (const gmatrix<T>& a, real b)
{
    gmatrix<T> c(a.n1, a.n2);
    for(int i = 0; i < a.n;++i) {
	c[i] = a.v[i] - b;
    }
    return c;
}



template <class T> 
gmatrix<T> operator * (const gmatrix<T>& a, const gmatrix<T>& b)
{
    gmatrix<T> c(a.n1, a.n2);
    for(int i = 0; i < a.n;++i) {
	c[i] = a.v[i] * b.v[i];
    }
    return c;
}

template <class T> 
gmatrix<T> operator * (real a, const gmatrix<T>& b)
{
    gmatrix<T> c(b.n1, b.n2);
    for(int i = 0; i < b.n;++i) {
	c[i] = a * b.v[i];
    }
    return c;
}

template <class T> 
gmatrix<T> operator * (const gmatrix<T>& a, real b)
{
    gmatrix<T> c(a.n1, a.n2);
    for(int i = 0; i < a.n;++i) {
	c[i] = a.v[i] * b;
    }
    return c;
}



template <class T> 
gmatrix<T> operator / (const gmatrix<T>& a, const gmatrix<T>& b)
{
    gmatrix<T> c(a.n1, a.n2);
    for(int i = 0; i < a.n;++i) {
	c[i] = a.v[i] / b.v[i];
    }
    return c;
}

template <class T> 
gmatrix<T> operator / (real a, const gmatrix<T>& b)
{
    gmatrix<T> c(b.n1, b.n2);
    for(int i = 0; i < b.n;++i) {
	c[i] = a / b.v[i];
    }
    return c;
}

template <class T> 
gmatrix<T> operator / (const gmatrix<T>& a, real b)
{
    gmatrix<T> c(a.n1, a.n2);
    for(int i = 0; i < a.n;++i) {
	c[i] = a.v[i] / b;
    }
    return c;
}



//| cumulative : component-wise operations


template <class T> 
gmatrix<T> operator += (gmatrix<T>& a, const gmatrix<T>& b)
{
    return a = (a + b);
}

template <class T> 
gmatrix<T> operator += (gmatrix<T>& a, real b)
{
    return a = (a + b);
}



template <class T> 
gmatrix<T> operator -= (gmatrix<T>& a, const gmatrix<T>& b)
{
    return a = (a - b);
}

template <class T> 
gmatrix<T> operator -= (gmatrix<T>& a, real b)
{
    return a = (a - b);
}



template <class T> 
gmatrix<T> operator *= (gmatrix<T>& a, const gmatrix<T>& b)
{
    return a = (a * b);
}

template <class T> 
gmatrix<T> operator *= (gmatrix<T>& a, real b)
{
    return a = (a * b);
}



template <class T> 
gmatrix<T> operator /= (gmatrix<T>& a, const gmatrix<T>& b)
{
    return a = (a / b);
}

template <class T> 
gmatrix<T> operator /= (gmatrix<T>& a, real b)
{
    return a = (a / b);
}

//|
//| matrix multiplication
//|
template <class T>
gmatrix<T> operator ^ (const gmatrix<T>& a, const gmatrix<T>& b)
{
    if(a.n2 != b.n1) 
    {
	LOG("(%d X %d) * (%d X %d)\n",a.n1, a.n2, b.n1, b.n2); 
	fatal("matrix multiplication: matrix mismatch !\n");
    }

    gmatrix<T> c(a.n1, b.n2);
    vector<T> d, e;

    for (int i = 0 ; i < a.n1 ;++i) {
	for(int j = 0 ; j < b.n2 ; ++j) {
	    c(i,j) = sum(a.sel_row(i) * b.sel_col(j));
	}
    }

    return c;
}

template <class T>
gmatrix<T> operator ^ (const gmatrix<T>& a, const vector<T>& b)
{
    gmatrix<T> c(b);
    return a^c;
}

template <class T>
gmatrix<T> operator ^ (const vector<T>& a, const gmatrix<T>& b)

{
    gmatrix<T> c(a);
    gmatrix<T> d = c.transpose();
    return d^b; 
}

}; // graphics namespace
 
#endif
