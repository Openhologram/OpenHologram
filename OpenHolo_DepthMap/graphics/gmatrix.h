#ifndef __gmatrix_h
#define __gmatrix_h
//|
//| To make the elements of matrix general.
//| This can be used for NURBS evaluation.
//| Dae Hyun Kim
//|

#include "graphics/sys.h"
#include "graphics/real.h"
#include "graphics/log.h"

#include "graphics/vec.h"
#include "graphics/vector.h"
#include "graphics/matrix.h"
#include "graphics/misc.h"
#include "graphics/ivec.h"


namespace graphics {					
			
struct gmatrix3_real
{
    vector<real> v;
    int n;
     
    int n1;  
    int n2;  
    int n3; 
    
    int  grain;
    int  a_size() const { return ceil(n, grain); }

    void init() { v = 0; n = 0; grain = 16;}

    gmatrix3_real() : n(0), n1(0), n2(0), n3(0), grain(16), v()
    {
    }

    gmatrix3_real(int n_) : grain(16), v()
    {
	n1 = n_;
	
	n3 = 1;
	n3 = 1;
	if(n < 0) fatal("gmatrix3_real: bad gmatrix size"); 
	if (a_size()) v.resize(a_size());
    }

    gmatrix3_real(int n1_, int n2_, int n3_) : 
    n(n1_*n2_*n3_), n1(n1_), n2(n2_), n3(n3_), grain(16), v()
    {
	if(n < 0) fatal("gmatrix3_real: bad gmatrix size"); 
	if (a_size()) v.resize(a_size());
    }

    gmatrix3_real(const gmatrix3_real& a) : 
    n(a.n), n1(a.n1), n2(a.n2), n3(a.n3), grain(a.grain), v()
    {
	if (a_size()) v.resize(a_size());
	*this = a;
    }

    ~gmatrix3_real()
    {
    }

     gmatrix3_real& operator=(int a)
    ;

     gmatrix3_real& operator=(const gmatrix3_real& a)
    ;

    real& operator[](int i)
    {
	return v[i % n];
    }

     real& operator() (int n1_ ,int n2_ ,int n3_)
    ;

    const real& operator[](int i) const
    {
	return v[i % n];
    }

     const real& operator() (int n1_ ,int n2_ ,int n3_) const
    ;

     void resize(int n_new)
    ;

     void resize(int n1_, int n2_, int n3_)
    ;

    void sel_line( int n3_,  gmatrix<real>& a)
    {
	a.resize(n1, n2);
	for(int k = 0 ; k < n1 ;++k)
	    for(int l = 0 ; l < n2 ; l++)
		a(k, l) = (*this)(k,l  ,n3_ );
    }

    void rep_line( int n3_,  gmatrix<real>& a)
    {
	if(a.n1 == n1 && a.n2 == n2)
	for(int k = 0 ; k < n1 ;++k)
	    for(int l = 0 ; l < n2 ; l++)
		(*this)(k,l  ,n3_ ) = a(k,l);
    }

};

//| operators

gmatrix3_real operator + (const gmatrix3_real& a, const gmatrix3_real& b)
;

gmatrix3_real operator - (const gmatrix3_real& a, const gmatrix3_real& b)
;

gmatrix3_real operator * (const gmatrix3_real& a, const gmatrix3_real& b)
;

gmatrix3_real operator / (const gmatrix3_real& a, const gmatrix3_real& b)
;


//| cumulative

gmatrix3_real operator += (gmatrix3_real& a, const gmatrix3_real& b)
;

gmatrix3_real operator -= (gmatrix3_real& a, const gmatrix3_real& b)
;

gmatrix3_real operator *= (gmatrix3_real& a, const gmatrix3_real& b)
;

gmatrix3_real operator /= (gmatrix3_real& a, const gmatrix3_real& b)
;


			
struct gmatrix3_vec2
{
    //vec2* v;
    vector<vec2> v;
    int n;
     
    int n1;  
    int n2;  
    int n3; 
    
    int  grain;
    int  a_size() const { return ceil(n, grain); }

    //| General list protocol
    void init() { v = 0; n = 0; grain = 16;}

    gmatrix3_vec2() : n(0), n1(0), n2(0), n3(0), grain(16), v()
    {
	//v = 0;
    }

    gmatrix3_vec2(int n_) : grain(16), v()
    {
	n1 = n_;
	
	n3 = 1;
	n3 = 1;
	if(n < 0) fatal("gmatrix3_vec2: bad gmatrix size"); 
	if (a_size()) v.resize(a_size());
    }

    gmatrix3_vec2(int n1_, int n2_, int n3_) : 
    n(n1_*n2_*n3_), n1(n1_), n2(n2_), n3(n3_), grain(16), v()
    {
	if(n < 0) fatal("gmatrix3_vec2: bad gmatrix size"); 
	if (a_size()) v.resize(a_size());
    }

    gmatrix3_vec2(const gmatrix3_vec2& a) : 
    n(a.n), n1(a.n1), n2(a.n2), n3(a.n3), grain(a.grain), v()
    {
	if (a_size()) v.resize(a_size());
	*this = a;
    }

    ~gmatrix3_vec2()
    {
    }

     gmatrix3_vec2& operator=(int a)
    ;

     gmatrix3_vec2& operator=(const gmatrix3_vec2& a)
    ;

    vec2& operator[](int i)
    {
	return v[i % n];
    }

     vec2& operator() (int n1_ ,int n2_ ,int n3_)
    ;

    const vec2& operator[](int i) const
    {
	return v[i % n];
    }

     const vec2& operator() (int n1_ ,int n2_ ,int n3_) const
    ;

     void resize(int n_new)
    ;

     void resize(int n1_, int n2_, int n3_)
    ;

    void sel_line( int n3_,  gmatrix<vec2>& a)
    {
	a.resize(n1, n2);
	for(int k = 0 ; k < n1 ;++k)
	    for(int l = 0 ; l < n2 ; l++)
		a(k, l) = (*this)(k,l  ,n3_ );
    }

    void rep_line( int n3_,  gmatrix<vec2>& a)
    {
	if(a.n1 == n1 && a.n2 == n2)
	for(int k = 0 ; k < n1 ;++k)
	    for(int l = 0 ; l < n2 ; l++)
		(*this)(k,l  ,n3_ ) = a(k,l);
    }

};

//| operators

gmatrix3_vec2 operator + (const gmatrix3_vec2& a, const gmatrix3_vec2& b)
;

gmatrix3_vec2 operator - (const gmatrix3_vec2& a, const gmatrix3_vec2& b)
;

gmatrix3_vec2 operator * (const gmatrix3_vec2& a, const gmatrix3_vec2& b)
;

gmatrix3_vec2 operator / (const gmatrix3_vec2& a, const gmatrix3_vec2& b)
;


//| cumulative

gmatrix3_vec2 operator += (gmatrix3_vec2& a, const gmatrix3_vec2& b)
;

gmatrix3_vec2 operator -= (gmatrix3_vec2& a, const gmatrix3_vec2& b)
;

gmatrix3_vec2 operator *= (gmatrix3_vec2& a, const gmatrix3_vec2& b)
;

gmatrix3_vec2 operator /= (gmatrix3_vec2& a, const gmatrix3_vec2& b)
;


			
struct gmatrix3_vec3
{
    vector<vec3>    v;
    int n;
     
    int n1;  
    int n2;  
    int n3; 
    
    int  grain;
    int  a_size() const { return ceil(n, grain); }

    //| General list protocol
    void init() { v = 0; n = 0; grain = 16;}

    gmatrix3_vec3() : n(0), n1(0), n2(0), n3(0), grain(16), v()
    {
    }

    gmatrix3_vec3(int n_) : grain(16), v()
    {
	n1 = n_;
	
	n3 = 1;
	n3 = 1;
	if(n < 0) fatal("gmatrix3_vec3: bad gmatrix size"); 
	if (a_size()) v.resize(a_size());
    }

    gmatrix3_vec3(int n1_, int n2_, int n3_) : 
    n(n1_*n2_*n3_), n1(n1_), n2(n2_), n3(n3_), grain(16), v()
    {
	if(n < 0) fatal("gmatrix3_vec3: bad gmatrix size"); 
	if (a_size()) v.resize(a_size());
    }

    gmatrix3_vec3(const gmatrix3_vec3& a) : 
    n(a.n), n1(a.n1), n2(a.n2), n3(a.n3), grain(a.grain), v()
    {
	if (a_size()) v.resize(a_size());
	*this = a;
    }

    ~gmatrix3_vec3()
    {
    }

     gmatrix3_vec3& operator=(int a)
    ;

     gmatrix3_vec3& operator=(const gmatrix3_vec3& a)
    ;

    vec3& operator[](int i)
    {
	return v[i % n];
    }

     vec3& operator() (int n1_ ,int n2_ ,int n3_)
    ;

    const vec3& operator[](int i) const
    {
	return v[i % n];
    }

     const vec3& operator() (int n1_ ,int n2_ ,int n3_) const
    ;

     void resize(int n_new)
    ;

     void resize(int n1_, int n2_, int n3_)
    ;

    void sel_line( int n3_,  gmatrix<vec3>& a)
    {
	a.resize(n1, n2);
	for(int k = 0 ; k < n1 ;++k)
	    for(int l = 0 ; l < n2 ; l++)
		a(k, l) = (*this)(k,l  ,n3_ );
    }

    void rep_line( int n3_,  gmatrix<vec3>& a)
    {
	if(a.n1 == n1 && a.n2 == n2)
	for(int k = 0 ; k < n1 ;++k)
	    for(int l = 0 ; l < n2 ; l++)
		(*this)(k,l  ,n3_ ) = a(k,l);
    }

};

//| operators

gmatrix3_vec3 operator + (const gmatrix3_vec3& a, const gmatrix3_vec3& b)
;

gmatrix3_vec3 operator - (const gmatrix3_vec3& a, const gmatrix3_vec3& b)
;

gmatrix3_vec3 operator * (const gmatrix3_vec3& a, const gmatrix3_vec3& b)
;

gmatrix3_vec3 operator / (const gmatrix3_vec3& a, const gmatrix3_vec3& b)
;


//| cumulative

gmatrix3_vec3 operator += (gmatrix3_vec3& a, const gmatrix3_vec3& b)
;

gmatrix3_vec3 operator -= (gmatrix3_vec3& a, const gmatrix3_vec3& b)
;

gmatrix3_vec3 operator *= (gmatrix3_vec3& a, const gmatrix3_vec3& b)
;

gmatrix3_vec3 operator /= (gmatrix3_vec3& a, const gmatrix3_vec3& b)
;


			
struct gmatrix3_vec4
{
    vector<vec4> v;
    int n;
     
    int n1;  
    int n2;  
    int n3; 
    
    int  grain;
    int  a_size() const { return ceil(n, grain); }


    gmatrix3_vec4() : n(0), n1(0), n2(0), n3(0), grain(16), v()
    {
    }

    gmatrix3_vec4(int n_) : grain(16), v()
    {
	n1 = n_;
	
	n3 = 1;
	n3 = 1;
	if(n < 0) fatal("gmatrix3_vec4: bad gmatrix size"); 
	if (a_size()) v.resize(a_size());
    }

    gmatrix3_vec4(int n1_, int n2_, int n3_) : 
    n(n1_*n2_*n3_), n1(n1_), n2(n2_), n3(n3_), grain(16), v()
    {
	if(n < 0) fatal("gmatrix3_vec4: bad gmatrix size"); 
	if (a_size()) v.resize(a_size());
    }

    gmatrix3_vec4(const gmatrix3_vec4& a) : 
    n(a.n), n1(a.n1), n2(a.n2), n3(a.n3), grain(a.grain), v()
    {
	if (a_size()) v.resize(a_size());
	*this = a;
    }

    ~gmatrix3_vec4()
    {
    }

     gmatrix3_vec4& operator=(int a)
    ;

     gmatrix3_vec4& operator=(const gmatrix3_vec4& a)
    ;

    vec4& operator[](int i)
    {
	return v[i % n];
    }

     vec4& operator() (int n1_ ,int n2_ ,int n3_)
    ;

    const vec4& operator[](int i) const
    {
	return v[i % n];
    }

     const vec4& operator() (int n1_ ,int n2_ ,int n3_) const
    ;

     void resize(int n_new)
    ;

     void resize(int n1_, int n2_, int n3_)
    ;

    void sel_line( int n3_,  gmatrix<vec4>& a)
    {
	a.resize(n1, n2);
	for(int k = 0 ; k < n1 ;++k)
	    for(int l = 0 ; l < n2 ; l++)
		a(k, l) = (*this)(k,l  ,n3_ );
    }

    void rep_line( int n3_,  gmatrix<vec4>& a)
    {
	if(a.n1 == n1 && a.n2 == n2)
	for(int k = 0 ; k < n1 ;++k)
	    for(int l = 0 ; l < n2 ; l++)
		(*this)(k,l  ,n3_ ) = a(k,l);
    }

};

//| operators

gmatrix3_vec4 operator + (const gmatrix3_vec4& a, const gmatrix3_vec4& b)
;

gmatrix3_vec4 operator - (const gmatrix3_vec4& a, const gmatrix3_vec4& b)
;

gmatrix3_vec4 operator * (const gmatrix3_vec4& a, const gmatrix3_vec4& b)
;

gmatrix3_vec4 operator / (const gmatrix3_vec4& a, const gmatrix3_vec4& b)
;


//| cumulative

gmatrix3_vec4 operator += (gmatrix3_vec4& a, const gmatrix3_vec4& b)
;

gmatrix3_vec4 operator -= (gmatrix3_vec4& a, const gmatrix3_vec4& b)
;

gmatrix3_vec4 operator *= (gmatrix3_vec4& a, const gmatrix3_vec4& b)
;

gmatrix3_vec4 operator /= (gmatrix3_vec4& a, const gmatrix3_vec4& b)
;



					
			
struct gmatrix4_real
{
    vector<real> v;
    int n;
     
    int n1;  
    int n2;  
    int n3;  
    int n4; 
    
    int  grain;
    int  a_size() const { return ceil(n, grain); }

    //| General list protocol
    void init() { v = 0; n = 0; grain = 16;}

    gmatrix4_real() : n(0), n1(0), n2(0), n3(0), n4(0), grain(16), v()
    {
    }

    gmatrix4_real(int n_) : grain(16), v()
    {
	n1 = n_;
	
	n4 = 1;
	n4 = 1;
	n4 = 1;
	if(n < 0) fatal("gmatrix4_real: bad gmatrix size"); 
	if (a_size()) v.resize(a_size());
    }

    gmatrix4_real(int n1_, int n2_, int n3_, int n4_) : 
    n(n1_*n2_*n3_*n4_), n1(n1_), n2(n2_), n3(n3_), n4(n4_), grain(16), v()
    {
	if(n < 0) fatal("gmatrix4_real: bad gmatrix size"); 
	if (a_size()) v.resize(a_size());
    }

    gmatrix4_real(const gmatrix4_real& a) : 
    n(a.n), n1(a.n1), n2(a.n2), n3(a.n3), n4(a.n4), grain(a.grain), v()
    {
	if (a_size()) v.resize(a_size());
	*this = a;
    }

    ~gmatrix4_real()
    {
    }

     gmatrix4_real& operator=(int a)
    ;

     gmatrix4_real& operator=(const gmatrix4_real& a)
    ;

    real& operator[](int i)
    {
	return v[i % n];
    }

     real& operator() (int n1_ ,int n2_ ,int n3_ ,int n4_)
    ;

    const real& operator[](int i) const
    {
	return v[i % n];
    }

     const real& operator() (int n1_ ,int n2_ ,int n3_ ,int n4_) const
    ;

     void resize(int n_new)
    ;

     void resize(int n1_, int n2_, int n3_, int n4_)
    ;

    void sel_line( int n3_,  int n4_,  gmatrix<real>& a)
    {
	a.resize(n1, n2);
	for(int k = 0 ; k < n1 ;++k)
	    for(int l = 0 ; l < n2 ; l++)
		a(k, l) = (*this)(k,l  ,n3_  ,n4_ );
    }

    void rep_line( int n3_,  int n4_,  gmatrix<real>& a)
    {
	if(a.n1 == n1 && a.n2 == n2)
	for(int k = 0 ; k < n1 ;++k)
	    for(int l = 0 ; l < n2 ; l++)
		(*this)(k,l  ,n3_  ,n4_ ) = a(k,l);
    }

};

//| operators

gmatrix4_real operator + (const gmatrix4_real& a, const gmatrix4_real& b)
;

gmatrix4_real operator - (const gmatrix4_real& a, const gmatrix4_real& b)
;

gmatrix4_real operator * (const gmatrix4_real& a, const gmatrix4_real& b)
;

gmatrix4_real operator / (const gmatrix4_real& a, const gmatrix4_real& b)
;


//| cumulative

gmatrix4_real operator += (gmatrix4_real& a, const gmatrix4_real& b)
;

gmatrix4_real operator -= (gmatrix4_real& a, const gmatrix4_real& b)
;

gmatrix4_real operator *= (gmatrix4_real& a, const gmatrix4_real& b)
;

gmatrix4_real operator /= (gmatrix4_real& a, const gmatrix4_real& b)
;


			
struct gmatrix4_vec2
{

    vector<vec2> v;
    int n;
     
    int n1;  
    int n2;  
    int n3;  
    int n4; 
    
    int  grain;
    int  a_size() const { return ceil(n, grain); }

    //| General list protocol
    void init() { v = 0; n = 0; grain = 16;}

    gmatrix4_vec2() : n(0), n1(0), n2(0), n3(0), n4(0), grain(16), v()
    {
    }

    gmatrix4_vec2(int n_) : grain(16), v()
    {
	n1 = n_;
	
	n4 = 1;
	n4 = 1;
	n4 = 1;
	if(n < 0) fatal("gmatrix4_vec2: bad gmatrix size"); 
	if (a_size()) v.resize(a_size());
    }

    gmatrix4_vec2(int n1_, int n2_, int n3_, int n4_) : 
    n(n1_*n2_*n3_*n4_), n1(n1_), n2(n2_), n3(n3_), n4(n4_), grain(16), v()
    {
	if(n < 0) fatal("gmatrix4_vec2: bad gmatrix size"); 
	if (a_size()) v.resize(a_size());
    }

    gmatrix4_vec2(const gmatrix4_vec2& a) : 
    n(a.n), n1(a.n1), n2(a.n2), n3(a.n3), n4(a.n4), grain(a.grain), v()
    {
	if (a_size()) v.resize(a_size());
	*this = a;
    }

    ~gmatrix4_vec2()
    {
    }

     gmatrix4_vec2& operator=(int a)
    ;

     gmatrix4_vec2& operator=(const gmatrix4_vec2& a)
    ;

    vec2& operator[](int i)
    {
	return v[i % n];
    }

     vec2& operator() (int n1_ ,int n2_ ,int n3_ ,int n4_)
    ;

    const vec2& operator[](int i) const
    {
	return v[i % n];
    }

     const vec2& operator() (int n1_ ,int n2_ ,int n3_ ,int n4_) const
    ;

     void resize(int n_new)
    ;

     void resize(int n1_, int n2_, int n3_, int n4_)
    ;

    void sel_line( int n3_,  int n4_,  gmatrix<vec2>& a)
    {
	a.resize(n1, n2);
	for(int k = 0 ; k < n1 ;++k)
	    for(int l = 0 ; l < n2 ; l++)
		a(k, l) = (*this)(k,l  ,n3_  ,n4_ );
    }

    void rep_line( int n3_,  int n4_,  gmatrix<vec2>& a)
    {
	if(a.n1 == n1 && a.n2 == n2)
	for(int k = 0 ; k < n1 ;++k)
	    for(int l = 0 ; l < n2 ; l++)
		(*this)(k,l  ,n3_  ,n4_ ) = a(k,l);
    }

};

//| operators

gmatrix4_vec2 operator + (const gmatrix4_vec2& a, const gmatrix4_vec2& b)
;

gmatrix4_vec2 operator - (const gmatrix4_vec2& a, const gmatrix4_vec2& b)
;

gmatrix4_vec2 operator * (const gmatrix4_vec2& a, const gmatrix4_vec2& b)
;

gmatrix4_vec2 operator / (const gmatrix4_vec2& a, const gmatrix4_vec2& b)
;


//| cumulative

gmatrix4_vec2 operator += (gmatrix4_vec2& a, const gmatrix4_vec2& b)
;

gmatrix4_vec2 operator -= (gmatrix4_vec2& a, const gmatrix4_vec2& b)
;

gmatrix4_vec2 operator *= (gmatrix4_vec2& a, const gmatrix4_vec2& b)
;

gmatrix4_vec2 operator /= (gmatrix4_vec2& a, const gmatrix4_vec2& b)
;


			
struct gmatrix4_vec3
{
    vector<vec3> v;
    int n;
     
    int n1;  
    int n2;  
    int n3;  
    int n4; 
    
    int  grain;
    int  a_size() const { return ceil(n, grain); }

    //| General list protocol
    void init() { v = 0; n = 0; grain = 16;}

    gmatrix4_vec3() : n(0), n1(0), n2(0), n3(0), n4(0), grain(16), v()
    {
    }

    gmatrix4_vec3(int n_) : grain(16), v()
    {
	n1 = n_;
	
	n4 = 1;
	n4 = 1;
	n4 = 1;
	if(n < 0) fatal("gmatrix4_vec3: bad gmatrix size"); 
	if (a_size()) v.resize(a_size());
    }

    gmatrix4_vec3(int n1_, int n2_, int n3_, int n4_) : 
    n(n1_*n2_*n3_*n4_), n1(n1_), n2(n2_), n3(n3_), n4(n4_), grain(16), v()
    {
	if(n < 0) fatal("gmatrix4_vec3: bad gmatrix size"); 
	if (a_size()) v.resize(a_size());
    }

    gmatrix4_vec3(const gmatrix4_vec3& a) : 
    n(a.n), n1(a.n1), n2(a.n2), n3(a.n3), n4(a.n4), grain(a.grain), v()
    {
	if (a_size()) v.resize(a_size());
	*this = a;
    }

    ~gmatrix4_vec3()
    {
    }

     gmatrix4_vec3& operator=(int a)
    ;

     gmatrix4_vec3& operator=(const gmatrix4_vec3& a)
    ;

    vec3& operator[](int i)
    {
	return v[i % n];
    }

     vec3& operator() (int n1_ ,int n2_ ,int n3_ ,int n4_)
    ;

    const vec3& operator[](int i) const
    {
	return v[i % n];
    }

     const vec3& operator() (int n1_ ,int n2_ ,int n3_ ,int n4_) const
    ;

     void resize(int n_new)
    ;

     void resize(int n1_, int n2_, int n3_, int n4_)
    ;

    void sel_line( int n3_,  int n4_,  gmatrix<vec3>& a)
    {
	a.resize(n1, n2);
	for(int k = 0 ; k < n1 ;++k)
	    for(int l = 0 ; l < n2 ; l++)
		a(k, l) = (*this)(k,l  ,n3_  ,n4_ );
    }

    void rep_line( int n3_,  int n4_,  gmatrix<vec3>& a)
    {
	if(a.n1 == n1 && a.n2 == n2)
	for(int k = 0 ; k < n1 ;++k)
	    for(int l = 0 ; l < n2 ; l++)
		(*this)(k,l  ,n3_  ,n4_ ) = a(k,l);
    }

};

//| operators

gmatrix4_vec3 operator + (const gmatrix4_vec3& a, const gmatrix4_vec3& b)
;

gmatrix4_vec3 operator - (const gmatrix4_vec3& a, const gmatrix4_vec3& b)
;

gmatrix4_vec3 operator * (const gmatrix4_vec3& a, const gmatrix4_vec3& b)
;

gmatrix4_vec3 operator / (const gmatrix4_vec3& a, const gmatrix4_vec3& b)
;


//| cumulative

gmatrix4_vec3 operator += (gmatrix4_vec3& a, const gmatrix4_vec3& b)
;

gmatrix4_vec3 operator -= (gmatrix4_vec3& a, const gmatrix4_vec3& b)
;

gmatrix4_vec3 operator *= (gmatrix4_vec3& a, const gmatrix4_vec3& b)
;

gmatrix4_vec3 operator /= (gmatrix4_vec3& a, const gmatrix4_vec3& b)
;


			
struct gmatrix4_vec4
{
    vector<vec4> v;
    int n;
     
    int n1;  
    int n2;  
    int n3;  
    int n4; 
    
    int  grain;
    int  a_size() const { return ceil(n, grain); }

    //| General list protocol
    void init() { v = 0; n = 0; grain = 16;}

    gmatrix4_vec4() : n(0), n1(0), n2(0), n3(0), n4(0), grain(16), v()
    {
    }

    gmatrix4_vec4(int n_) : grain(16), v()
    {
	n1 = n_;
	
	n4 = 1;
	n4 = 1;
	n4 = 1;
	if(n < 0) fatal("gmatrix4_vec4: bad gmatrix size"); 
	if (a_size()) v.resize(a_size());
    }

    gmatrix4_vec4(int n1_, int n2_, int n3_, int n4_) : 
    n(n1_*n2_*n3_*n4_), n1(n1_), n2(n2_), n3(n3_), n4(n4_), grain(16), v()
    {
	if(n < 0) fatal("gmatrix4_vec4: bad gmatrix size"); 
	if (a_size()) v.resize(a_size());
    }

    gmatrix4_vec4(const gmatrix4_vec4& a) : 
    n(a.n), n1(a.n1), n2(a.n2), n3(a.n3), n4(a.n4), grain(a.grain), v()
    {
	if (a_size()) v.resize(a_size());
	*this = a;
    }

    ~gmatrix4_vec4()
    {
    }

     gmatrix4_vec4& operator=(int a)
    ;

     gmatrix4_vec4& operator=(const gmatrix4_vec4& a)
    ;

    vec4& operator[](int i)
    {
	return v[i % n];
    }

     vec4& operator() (int n1_ ,int n2_ ,int n3_ ,int n4_)
    ;

    const vec4& operator[](int i) const
    {
	return v[i % n];
    }

     const vec4& operator() (int n1_ ,int n2_ ,int n3_ ,int n4_) const
    ;

     void resize(int n_new)
    ;

     void resize(int n1_, int n2_, int n3_, int n4_)
    ;

    void sel_line( int n3_,  int n4_,  gmatrix<vec4>& a)
    {
	a.resize(n1, n2);
	for(int k = 0 ; k < n1 ;++k)
	    for(int l = 0 ; l < n2 ; l++)
		a(k, l) = (*this)(k,l  ,n3_  ,n4_ );
    }

    void rep_line( int n3_,  int n4_,  gmatrix<vec4>& a)
    {
	if(a.n1 == n1 && a.n2 == n2)
	for(int k = 0 ; k < n1 ;++k)
	    for(int l = 0 ; l < n2 ; l++)
		(*this)(k,l  ,n3_  ,n4_ ) = a(k,l);
    }

};

//| operators

gmatrix4_vec4 operator + (const gmatrix4_vec4& a, const gmatrix4_vec4& b)
;

gmatrix4_vec4 operator - (const gmatrix4_vec4& a, const gmatrix4_vec4& b)
;

gmatrix4_vec4 operator * (const gmatrix4_vec4& a, const gmatrix4_vec4& b)
;

gmatrix4_vec4 operator / (const gmatrix4_vec4& a, const gmatrix4_vec4& b)
;


//| cumulative

gmatrix4_vec4 operator += (gmatrix4_vec4& a, const gmatrix4_vec4& b)
;

gmatrix4_vec4 operator -= (gmatrix4_vec4& a, const gmatrix4_vec4& b)
;

gmatrix4_vec4 operator *= (gmatrix4_vec4& a, const gmatrix4_vec4& b)
;

gmatrix4_vec4 operator /= (gmatrix4_vec4& a, const gmatrix4_vec4& b)
;





//|
//| utility functions to calculate the index inside the class gmatrix
//|
inline int cal_index(int n1, int n2, int n3, int n1_, int n2_, int n3_)
;

inline int cal_index(int n1, int n2, int n3, int n4,
	      int n1_, int n2_, int n3_, int n4_)
;

inline int cal_index(int n1, int n2, int n3, int n4, int n5,
	      int n1_, int n2_, int n3_, int n4_, int n5_)
;

inline int cal_index(int n1, int n2, int n3, int n4, int n5, int n6,
	      int n1_, int n2_, int n3_, int n4_, int n5_, int n6_)
;


}; // graphics namespace
#endif
