#include "graphics/matrix.h"
//|
//| matrix.e
//| Dae Hyun Kim
//|

namespace graphics {

//|
//| Normal 2dimensional matrix (n1 x n2)
//|
matrix& matrix::operator=(int a)
{
    for(int i = 0; i < n;++i)
	v[i] = a;
    return *this;
}

matrix& matrix::operator=(const vec2& a)
{
    resize(a.n, 1);
    for(int i = 0 ; i < a.n ;++i)
	v[i] = a.v[i];
    return *this;
}

matrix& matrix::operator=(const vec3& a)
{
    resize(a.n, 1);
    for(int i = 0 ; i < a.n ;++i)
	v[i] = a.v[i];
    return *this;
}

matrix& matrix::operator=(const vec4& a)
{
    resize(a.n, 1);
    for(int i = 0 ; i < a.n ;++i)
	v[i] = a.v[i];
    return *this;
}

matrix& matrix::operator=(const vector<real>& a)
{
    resize(a.size(), 1);
    for(int i = 0 ; i < a.size() ;++i)
	v[i] = a[i];
    return *this;
}

matrix& matrix::operator = (const matrix& a)
{
    resize(a.n1, a.n2);
    for(int i = 0; i < a.n;++i)
	v[i] = a.v[i];
    return *this;

}

matrix::operator vector<real> ()
{
    if(n2 != 1) 
    {    
	LOG("matrix  : cannot change %d X %d matrix to vector", n1, n2);
	fatal("matrix: error when doing type casting");
    }
    vector<real> c(n1);
    for(int i = 0 ; i < n1 ;++i)
	c[i] = v[i];
    return c;
}

void matrix::resize(int n_new)
{
    if(n_new < 0) fatal("matrix : bad new mat size");
    if(ceil(n,grain) >= ceil(n_new,grain))
	n = n_new;
    else {
	n = n_new;
	v.resize(a_size());
    }
}

void matrix::resize(int n1_, int n2_)
{
    n1 = n1_; n2 = n2_;
    int n_new = n1 * n2;

    if(n_new < 0) fatal("matrix : bad new mat size");
    if(ceil(n,grain) >= ceil(n_new,grain))
	n = n_new;
    else {
	n = n_new;
	v.resize(a_size());
    }
}

void matrix::add_row(const vector<real>& a)
{
    if(a.size() == n2)
	n1++;
    else
	fatal("matrix: bad row size");
    resize(n + n2);
    for(int i = 0 ; i < a.size() ;++i)
	v[cal_index(n1-1, i)] = a[i];
}

void matrix::add_col(const vector<real>& a)
{
    if(a.size() != n1)
	fatal("matrix : bad column size");
    matrix temp = *this;
    n2++;
    resize(n+n1);
    int i;
    for(i = 0 ; i < n1 ;++i)
	for(int j = 0 ; j < n2-1 ; ++j)
	    v[cal_index(i, j)] = temp(i, j);
    for(i = 0 ; i < n1 ;++i)
	v[cal_index(i, (n2-1))] = a[i];
}

void matrix::ins_row(int index, const vector<real>& a)
{
    if(a.size() != n2)
	fatal("matrix: bad row size");
    matrix temp = *this;
    resize(n1+1, n2);
    int i, j;
    for(i = 0 ; i < index ;++i)
	for(j = 0 ; j < n2 ; ++j)
	    v[cal_index(i,j)] = temp(i,j);

    for(j = 0 ; j < n2 ; ++j)
	v[cal_index(index, j)] = a[j];

    for(i = index+1 ; i < n1 ;++i)
	for(j = 0 ; j < n2 ; ++j)
	    v[cal_index(i, j)] = temp(i-1,j);
}

void matrix::ins_col(int index, const vector<real>& a)
{
    if(a.size() != n1)
	fatal("matrix: bad input column size");
    matrix temp = *this;
    resize(n1, n2+1);
    int i, j;
    for(i = 0 ; i < n1;++i)
	for(j = 0 ; j < index ; ++j)
	    v[cal_index(i,j)] = temp(i,j);

    for(i = 0 ; i < n1 ;++i)
	v[cal_index(i,index)] = a[i];

    for(i = 0 ; i < n1 ;++i)
	for(j = index+1 ; j < n2 ; ++j)
	    v[cal_index(i,j)] = temp(i,j-1);
}

void matrix::del_row(int index)
{
    for(int i = (index*n2); i < n - n2;++i)
	v[i] = v[i + n2];
    resize(n1-1, n2);
}

void matrix::del_col(int index)
{
    matrix temp = *this;
    resize(n1, n2-1);
    int i, j;
    for(i = 0 ; i < n1;++i)
	for(j = 0 ; j < index ; ++j)
	    v[cal_index(i,j)] = temp(i,j);

    for(i = 0 ; i < n1 ;++i)
	for(j = index+1 ; j < temp.n2 ; ++j)
	    v[cal_index(i,j-1)] = temp(i,j);
}

vector<real> matrix::sel_col(int index) const
{
    vector<real> temp(n1);
    for(int i = 0 ; i < n1 ;++i)
	temp[i] = v[cal_index(i, index)];
    return temp;
}

vector<real> matrix::sel_row(int index) const
{
    vector<real> temp(n2);
    for(int i = 0 ; i < n2 ;++i)
	temp[i] = v[cal_index(index, i)];
    return temp;
}

void matrix::rep_col(int index, const vector<real>& a)
{
    int min_num = min(a.size(), n1);
    for(int i = 0 ; i < min_num ;++i)
	v[cal_index(i, index)] = a[i];
}

void matrix::rep_row(int index, const vector<real>& a)
{
    int min_num = min(a.size(), n2);
    for(int i = 0 ; i < min_num ;++i)
	v[cal_index(index, i)] = a[i];
}

void matrix::add(real a)
{
    if(n1 == 1 || n2 == 1) {
	if(n1 == 1) resize(1, n + 1);
	else resize(n + 1, 1);
	v[n - 1] = a;
    }
}

void matrix::ins(int index, real a)
{
    if(n1 == 1 || n2 == 1) {
	if(n1 == 1) resize(1, n + 1);
	else resize(n+1, 1);
	for(int i = n - 1; i > index; i--)
	    v[i] = v[i - 1];
	v[index] = a;
    }
}

void matrix::del(int index)
{
    if(n1 == 1 || n2 == 1) {
	for(int i = index; i < n - 1;++i)
	    v[i] = v[i + 1];
	if(n1 == 1) resize(1, n - 1);
	else resize(n-1, 1);
    }
}

bool matrix::is_affine() const
{
    if (n1 != n2) return false;
    int size = n1;
    for(int i = 0 ; i < size-1 ;++i)
	if(v[cal_index(i, size-1)] != .0)
	    return false;
    return (v[cal_index(size-1,size-1)] == 1.0);
}

real matrix::determ() const
{
    const matrix& a = *this;

    if (is_null()) {
	LOG("Attempt to compute determinant on Null matrix.");
	return 0.0;
    }
    if (n1 != n2) {
	LOG("Attempt to compute determinant on non-square matrix.");
	return 0.0;
    }

    switch (n1) {
    case 1: return a(0,0);
    case 2: return (a(0,0) * a(1,1) - a(1,0) * a(0,1));
    case 3: return det3x3(a(0,0), a(0,1), a(0,2),
			  a(1,0), a(1,1), a(1,2),
		  	  a(2,0), a(2,1), a(2,2));
    case 4:
	//| Check for an affine matrix
	if (is_affine()) {
	    return det3x3(a(0,0), a(0,1), a(0,2),
			  a(1,0), a(1,1), a(1,2),
			  a(2,0), a(2,1), a(2,2));
	} else {
	    //| Do full 4x4 calculation
	    real d00, d01, d02, d03;

	    d00 = det3x3(a(1,1), a(1,2), a(1,3),
			 a(2,1), a(2,2), a(2,3),
			 a(3,1), a(3,2), a(3,3));
	    d01 = det3x3(a(1,0), a(1,2), a(1,3),
			 a(2,0), a(2,2), a(2,3),
			 a(3,0), a(3,2), a(3,3));
	    d02 = det3x3(a(1,0), a(1,1), a(1,3),
			 a(2,0), a(2,1), a(2,3),
			 a(3,0), a(3,1), a(3,3));
	    d03 = det3x3(a(1,0), a(1,1), a(1,2),
			 a(2,0), a(2,1), a(2,2),
			 a(3,0), a(3,1), a(3,2));
	    return(a(0,0)*d00 - a(0,1)*d01 +
		   a(0,2)*d02 - a(0,3)*d03);
		   
	}
    }
	return 0;
}

matrix matrix::adjoint(int i, int j) const
{
    int row = 0, col, size = n1;
    matrix adj;

    adj.mat_zero(size-1);
    for(int i3 = 0; i3 < size; i3++) {
	if (i3 != i) {
	    col = 0;
	    for(int j3 = 0; j3 < size; j3++) {
		if(j3 != j) {
			adj(row,col++) = v[cal_index(i3,j3)];
		}
	    }
	    row++;
	}
    }
    return adj;
}

matrix matrix::inverse() const
{

    if (n1 != n2) {
	matrix null(0,0);
	LOG("Attempt to invert non-square matrix.");
	return null;
    }
    if (is_null()) {
	matrix null(0,0);
	LOG("Attempt to invert null matrix.");
	return null;
    }

    if (n1 > 4) {
	matrix null(0,0);
	LOG("Can't invert matrices larger than 4x4.");
	return null;
    }

    int sign, size = n1;
    real deta, detm = determ();
    matrix minv, temp;

    //if (apx_equal(fabs(detm), 0.0, zero_epsilon/10000.0)) {
	if (apx_equal(fabs(detm), 0.0, zero_epsilon)) {
	matrix null(0,0);
	//LOG("Attempt to invert a singular matrix.");
	return null;
    }

    minv.mat_zero(size);
    for(int i = 0; i < size;++i){
	sign = (i&1) ? ~0 : 0;
	for(int j = 0; j < size; ++j) {
	    temp = adjoint(i,j);
	    deta =  temp.determ();
	    minv(j,i) = sign ? -deta : deta;
	    sign = ~sign;
	}
    }
    return (1.0/detm) * minv;
}

matrix matrix::transpose() const
{
    matrix mt(n2, n1);

    for (int i = 0; i < mt.n1;++i){
	for (int j = 0; j < mt.n2; ++j) {
	    mt(i,j) = v[cal_index(j,i)];
	}
    }
    return mt;
}

void matrix::identity(int dim)
{
    resize(dim,dim);

    for (int i = 0 ; i < n1 ;++i){
	for (int j = 0 ; j < n2 ; ++j) {
	    v[cal_index(i,j)] = ( i == j ? 1.0 : 0.0);
	}
    }
    return;
}


bool
matrix::swap_rows(int ith, int jth)
{
    if (ith == jth) return true;
    vector<real> ith_a = sel_row(ith);
    vector<real> jth_a = sel_row(jth);

    rep_row(ith, jth_a);
    rep_row(jth, ith_a);
    return true;
}

bool
matrix::swap_cols(int ith, int jth)
{
    if (ith == jth) return true;
    vector<real> ith_a = sel_col(ith);
    vector<real> jth_a = sel_col(jth);

    rep_col(ith, jth_a);
    rep_col(jth, ith_a);
    return true;
}


//|
//| binary : component-wise operations
//|


matrix operator + (const matrix& a, const matrix& b)
{
    matrix c(a.n1, a.n2);
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] + b.v[i];
    }
    return c;
}

matrix operator + (real a, const matrix& b)
{
    matrix c(b.n1, b.n2);
    for(int i = 0; i < b.n;++i){
	c[i] = a + b.v[i];
    }
    return c;
}

matrix operator + (const matrix& a, real b)
{
    matrix c(a.n1, a.n2);
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] + b;
    }
    return c;
}



matrix operator - (const matrix& a, const matrix& b)
{
    matrix c(a.n1, a.n2);
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] - b.v[i];
    }
    return c;
}

matrix operator - (real a, const matrix& b)
{
    matrix c(b.n1, b.n2);
    for(int i = 0; i < b.n;++i){
	c[i] = a - b.v[i];
    }
    return c;
}

matrix operator - (const matrix& a, real b)
{
    matrix c(a.n1, a.n2);
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] - b;
    }
    return c;
}



matrix operator * (const matrix& a, const matrix& b)
{
    matrix c(a.n1, a.n2);
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] * b.v[i];
    }
    return c;
}

matrix operator * (real a, const matrix& b)
{
    matrix c(b.n1, b.n2);
    for(int i = 0; i < b.n;++i){
	c[i] = a * b.v[i];
    }
    return c;
}

matrix operator * (const matrix& a, real b)
{
    matrix c(a.n1, a.n2);
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] * b;
    }
    return c;
}



matrix operator / (const matrix& a, const matrix& b)
{
    matrix c(a.n1, a.n2);
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] / b.v[i];
    }
    return c;
}

matrix operator / (real a, const matrix& b)
{
    matrix c(b.n1, b.n2);
    for(int i = 0; i < b.n;++i){
	c[i] = a / b.v[i];
    }
    return c;
}

matrix operator / (const matrix& a, real b)
{
    matrix c(a.n1, a.n2);
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] / b;
    }
    return c;
}



//| cumulative : component-wise operations


matrix operator += (matrix& a, const matrix& b)
{
    a = (a + b);
    return a;
}

matrix operator += (matrix& a, real b)
{
    a = (a + b);
    return a;
}



matrix operator -= (matrix& a, const matrix& b)
{
    a = (a - b);
    return a;
}

matrix operator -= (matrix& a, real b)
{
    a = (a - b);
    return a;
}



matrix operator *= (matrix& a, const matrix& b)
{
    a = (a * b);
    return a;
}

matrix operator *= (matrix& a, real b)
{
    a = (a * b);
    return a;
}



matrix operator /= (matrix& a, const matrix& b)
{
    a = (a / b);
    return a;
}

matrix operator /= (matrix& a, real b)
{
    a = (a / b);
    return a;
}



//| logical : component-wise operations


int operator == (const matrix& a, const matrix& b)
{
    int c = 1;
    for(int i = 0; i < a.n;++i){
	c = c && a.v[i] == b.v[i];
    }
    return c;
}

int operator == (real a, const matrix& b)
{
    int c = 1;
    for(int i = 0; i < b.n;++i){
	c = c && a == b.v[i];
    }
    return c;
}

int operator == (const matrix& a, real b)
{
    int c = 1;
    for(int i = 0; i < a.n;++i){
	c = c && a.v[i] == b;
    }
    return c;
}



int operator < (const matrix& a, const matrix& b)
{
    int c = 1;
    for(int i = 0; i < a.n;++i){
	c = c && a.v[i] < b.v[i];
    }
    return c;
}

int operator < (real a, const matrix& b)
{
    int c = 1;
    for(int i = 0; i < b.n;++i){
	c = c && a < b.v[i];
    }
    return c;
}

int operator < (const matrix& a, real b)
{
    int c = 1;
    for(int i = 0; i < a.n;++i){
	c = c && a.v[i] < b;
    }
    return c;
}



int operator <= (const matrix& a, const matrix& b)
{
    int c = 1;
    for(int i = 0; i < a.n;++i){
	c = c && a.v[i] <= b.v[i];
    }
    return c;
}

int operator <= (real a, const matrix& b)
{
    int c = 1;
    for(int i = 0; i < b.n;++i){
	c = c && a <= b.v[i];
    }
    return c;
}

int operator <= (const matrix& a, real b)
{
    int c = 1;
    for(int i = 0; i < a.n;++i){
	c = c && a.v[i] <= b;
    }
    return c;
}



int operator > (const matrix& a, const matrix& b)
{
    int c = 1;
    for(int i = 0; i < a.n;++i){
	c = c && a.v[i] > b.v[i];
    }
    return c;
}

int operator > (real a, const matrix& b)
{
    int c = 1;
    for(int i = 0; i < b.n;++i){
	c = c && a > b.v[i];
    }
    return c;
}

int operator > (const matrix& a, real b)
{
    int c = 1;
    for(int i = 0; i < a.n;++i){
	c = c && a.v[i] > b;
    }
    return c;
}



int operator >= (const matrix& a, const matrix& b)
{
    int c = 1;
    for(int i = 0; i < a.n;++i){
	c = c && a.v[i] >= b.v[i];
    }
    return c;
}

int operator >= (real a, const matrix& b)
{
    int c = 1;
    for(int i = 0; i < b.n;++i){
	c = c && a >= b.v[i];
    }
    return c;
}

int operator >= (const matrix& a, real b)
{
    int c = 1;
    for(int i = 0; i < a.n;++i){
	c = c && a.v[i] >= b;
    }
    return c;
}



//|
//| matrix multiplication
//|
matrix operator ^ (const matrix& a, const matrix& b)
{
    if(a.n2 != b.n1) 
    {
	LOG("(%d X %d) * (%d X %d)\n",a.n1, a.n2, b.n1, b.n2); 
	fatal("matrix multiplication: matrix mismatch !\n");
    }

    matrix c(a.n1, b.n2);
    vector<real> d, e;

    for (int i = 0 ; i < a.n1 ;++i){
	for(int j = 0 ; j < b.n2 ; ++j) {
	    c(i,j) = sum(a.sel_row(i) * b.sel_col(j));
	}
    }

    return c;
}

matrix operator ^ (const matrix& a, const vector<real>& b)
{
    matrix c(b);
    return a^c;
}

matrix operator ^ (const vector<real>& a, const matrix& b)
{
    matrix c(a);
    matrix d = c.transpose();
    return d^b; 
}

int operator != (const matrix& a, const matrix& b)
{
    return !(a == b);
}


real sum(const matrix& a)
{
    real s;
    s = 0;
    for(int i = 0; i < a.n;++i){
        s += a.v[i];
    }
    return s;
}

void print(matrix& a)
{
    for (int i = 0 ; i < a.n1 ;++i){
	for (int j = 0 ; j < a.n2 ; ++j) 
	    LOG("%g  ", a(i,j));
	LOG("\n");
    }
}




 
}; // graphics namespace