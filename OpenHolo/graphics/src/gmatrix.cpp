#include "graphics/gmatrix.h"
//|
//| To make the elements of matrix general.
//| This can be used for NURBS evaluation.
//| Dae Hyun Kim
//|


namespace graphics {					
			
 gmatrix3_real& gmatrix3_real::operator=(int a)
    {
	for(int i = 0; i < n;++i)
	    v[i] = a;
	return *this;
    }

 gmatrix3_real& gmatrix3_real::operator=(const gmatrix3_real& a)
    {
	
	n1 = a.n1;
	
	n2 = a.n2;
	
	n3 = a.n3;
	
	resize(a.n);
	for(int i = 0; i < a.n;++i)
	    v[i] = a.v[i];
	return *this;
    }

 real& gmatrix3_real::operator() (int n1_ ,int n2_ ,int n3_)
    {
	return v[cal_index( n1, n2, n3, n1_, n2_, n3_)];	
    }

 const real& gmatrix3_real::operator() (int n1_ ,int n2_ ,int n3_) const
    {
	return v[cal_index( n1, n2, n3, n1_, n2_, n3_)];	
    }

 void gmatrix3_real::resize(int n_new)
    {
	if(n_new < 0) fatal("gmatrix3_real: bad new gmatrix size");
	if(ceil(n,grain) >= ceil(n_new,grain))
	    n = n_new;
	else {
	    int old_n = n;
	    n = n_new;
	    v.resize(a_size());
	}
    }

 void gmatrix3_real::resize(int n1_, int n2_, int n3_)
    {
	
	n1 = n1_;
	n2 = n2_;
	n3 = n3_;
	

	int n_new = n1_ *n2_ *n3_ ;
	if(n_new < 0) fatal("gmatrix3_real: bad new gmatrix size");
	if(ceil(n,grain) >= ceil(n_new,grain))
	    n = n_new;
	else {
	    int old_n = n;
	    n = n_new;
	    v.resize(a_size());
	}
    }



//| operators

gmatrix3_real operator + (const gmatrix3_real& a, const gmatrix3_real& b)
{
    
    if(a.n1 != b.n1) fatal("gmatrix3_real + gmatrix3_real: size dismatch");
    
    if(a.n2 != b.n2) fatal("gmatrix3_real + gmatrix3_real: size dismatch");
    
    if(a.n3 != b.n3) fatal("gmatrix3_real + gmatrix3_real: size dismatch");
    

    gmatrix3_real c(a.n1  ,a.n2  ,a.n3 );
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] + b.v[i];
    }
    return c;
}

gmatrix3_real operator - (const gmatrix3_real& a, const gmatrix3_real& b)
{
    
    if(a.n1 != b.n1) fatal("gmatrix3_real + gmatrix3_real: size dismatch");
    
    if(a.n2 != b.n2) fatal("gmatrix3_real + gmatrix3_real: size dismatch");
    
    if(a.n3 != b.n3) fatal("gmatrix3_real + gmatrix3_real: size dismatch");
    

    gmatrix3_real c(a.n1  ,a.n2  ,a.n3 );
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] - b.v[i];
    }
    return c;
}

gmatrix3_real operator * (const gmatrix3_real& a, const gmatrix3_real& b)
{
    
    if(a.n1 != b.n1) fatal("gmatrix3_real + gmatrix3_real: size dismatch");
    
    if(a.n2 != b.n2) fatal("gmatrix3_real + gmatrix3_real: size dismatch");
    
    if(a.n3 != b.n3) fatal("gmatrix3_real + gmatrix3_real: size dismatch");
    

    gmatrix3_real c(a.n1  ,a.n2  ,a.n3 );
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] * b.v[i];
    }
    return c;
}

gmatrix3_real operator / (const gmatrix3_real& a, const gmatrix3_real& b)
{
    
    if(a.n1 != b.n1) fatal("gmatrix3_real + gmatrix3_real: size dismatch");
    
    if(a.n2 != b.n2) fatal("gmatrix3_real + gmatrix3_real: size dismatch");
    
    if(a.n3 != b.n3) fatal("gmatrix3_real + gmatrix3_real: size dismatch");
    

    gmatrix3_real c(a.n1  ,a.n2  ,a.n3 );
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] / b.v[i];
    }
    return c;
}


//| cumulative

gmatrix3_real operator += (gmatrix3_real& a, const gmatrix3_real& b)
{
    return a = (a + b);
}

gmatrix3_real operator -= (gmatrix3_real& a, const gmatrix3_real& b)
{
    return a = (a - b);
}

gmatrix3_real operator *= (gmatrix3_real& a, const gmatrix3_real& b)
{
    return a = (a * b);
}

gmatrix3_real operator /= (gmatrix3_real& a, const gmatrix3_real& b)
{
    return a = (a / b);
}


			
 gmatrix3_vec2& gmatrix3_vec2::operator=(int a)
    {
	for(int i = 0; i < n;++i)
	    v[i] = a;
	return *this;
    }

 gmatrix3_vec2& gmatrix3_vec2::operator=(const gmatrix3_vec2& a)
    {
	
	n1 = a.n1;
	
	n2 = a.n2;
	
	n3 = a.n3;
	
	resize(a.n);
	for(int i = 0; i < a.n;++i)
	    v[i] = a.v[i];
	return *this;
    }

 vec2& gmatrix3_vec2::operator() (int n1_ ,int n2_ ,int n3_)
    {
	return v[cal_index( n1, n2, n3, n1_, n2_, n3_)];	
    }

 const vec2& gmatrix3_vec2::operator() (int n1_ ,int n2_ ,int n3_) const
    {
	return v[cal_index( n1, n2, n3, n1_, n2_, n3_)];	
    }

 void gmatrix3_vec2::resize(int n_new)
    {
	if(n_new < 0) fatal("gmatrix3_vec2: bad new gmatrix size");
	if(ceil(n,grain) >= ceil(n_new,grain))
	    n = n_new;
	else {
	    n = n_new;
	    v.resize(a_size());
	}
    }

 void gmatrix3_vec2::resize(int n1_, int n2_, int n3_)
    {
	
	n1 = n1_;
	
	n2 = n2_;
	
	n3 = n3_;
	

	int n_new = n1_ *n2_ *n3_ ;
	if(n_new < 0) fatal("gmatrix3_vec2: bad new gmatrix size");
	if(ceil(n,grain) >= ceil(n_new,grain))
	    n = n_new;
	else {
	    n = n_new;
	    v.resize(a_size());
	}
    }



//| operators

gmatrix3_vec2 operator + (const gmatrix3_vec2& a, const gmatrix3_vec2& b)
{
    
    if(a.n1 != b.n1) fatal("gmatrix3_vec2 + gmatrix3_vec2: size dismatch");
    
    if(a.n2 != b.n2) fatal("gmatrix3_vec2 + gmatrix3_vec2: size dismatch");
    
    if(a.n3 != b.n3) fatal("gmatrix3_vec2 + gmatrix3_vec2: size dismatch");
    

    gmatrix3_vec2 c(a.n1  ,a.n2  ,a.n3 );
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] + b.v[i];
    }
    return c;
}

gmatrix3_vec2 operator - (const gmatrix3_vec2& a, const gmatrix3_vec2& b)
{
    
    if(a.n1 != b.n1) fatal("gmatrix3_vec2 + gmatrix3_vec2: size dismatch");
    
    if(a.n2 != b.n2) fatal("gmatrix3_vec2 + gmatrix3_vec2: size dismatch");
    
    if(a.n3 != b.n3) fatal("gmatrix3_vec2 + gmatrix3_vec2: size dismatch");
    

    gmatrix3_vec2 c(a.n1  ,a.n2  ,a.n3 );
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] - b.v[i];
    }
    return c;
}

gmatrix3_vec2 operator * (const gmatrix3_vec2& a, const gmatrix3_vec2& b)
{
    
    if(a.n1 != b.n1) fatal("gmatrix3_vec2 + gmatrix3_vec2: size dismatch");
    
    if(a.n2 != b.n2) fatal("gmatrix3_vec2 + gmatrix3_vec2: size dismatch");
    
    if(a.n3 != b.n3) fatal("gmatrix3_vec2 + gmatrix3_vec2: size dismatch");
    

    gmatrix3_vec2 c(a.n1  ,a.n2  ,a.n3 );
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] * b.v[i];
    }
    return c;
}

gmatrix3_vec2 operator / (const gmatrix3_vec2& a, const gmatrix3_vec2& b)
{
    
    if(a.n1 != b.n1) fatal("gmatrix3_vec2 + gmatrix3_vec2: size dismatch");
    
    if(a.n2 != b.n2) fatal("gmatrix3_vec2 + gmatrix3_vec2: size dismatch");
    
    if(a.n3 != b.n3) fatal("gmatrix3_vec2 + gmatrix3_vec2: size dismatch");
    

    gmatrix3_vec2 c(a.n1  ,a.n2  ,a.n3 );
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] / b.v[i];
    }
    return c;
}


//| cumulative

gmatrix3_vec2 operator += (gmatrix3_vec2& a, const gmatrix3_vec2& b)
{
    return a = (a + b);
}

gmatrix3_vec2 operator -= (gmatrix3_vec2& a, const gmatrix3_vec2& b)
{
    return a = (a - b);
}

gmatrix3_vec2 operator *= (gmatrix3_vec2& a, const gmatrix3_vec2& b)
{
    return a = (a * b);
}

gmatrix3_vec2 operator /= (gmatrix3_vec2& a, const gmatrix3_vec2& b)
{
    return a = (a / b);
}


			
 gmatrix3_vec3& gmatrix3_vec3::operator=(int a)
    {
	for(int i = 0; i < n;++i)
	    v[i] = a;
	return *this;
    }

 gmatrix3_vec3& gmatrix3_vec3::operator=(const gmatrix3_vec3& a)
    {
	
	n1 = a.n1;
	
	n2 = a.n2;
	
	n3 = a.n3;
	
	resize(a.n);
	for(int i = 0; i < a.n;++i)
	    v[i] = a.v[i];
	return *this;
    }

 vec3& gmatrix3_vec3::operator() (int n1_ ,int n2_ ,int n3_)
    {
	return v[cal_index( n1, n2, n3, n1_, n2_, n3_)];	
    }

 const vec3& gmatrix3_vec3::operator() (int n1_ ,int n2_ ,int n3_) const
    {
	return v[cal_index( n1, n2, n3, n1_, n2_, n3_)];	
    }

 void gmatrix3_vec3::resize(int n_new)
    {
	if(n_new < 0) fatal("gmatrix3_vec3: bad new gmatrix size");
	if(ceil(n,grain) >= ceil(n_new,grain))
	    n = n_new;
	else {
	    n = n_new;
	    v.resize(a_size());
	}
    }

 void gmatrix3_vec3::resize(int n1_, int n2_, int n3_)
    {
	
	n1 = n1_;
	
	n2 = n2_;
	
	n3 = n3_;
	

	int n_new = n1_ *n2_ *n3_ ;
	if(n_new < 0) fatal("gmatrix3_vec3: bad new gmatrix size");
	if(ceil(n,grain) >= ceil(n_new,grain))
	    n = n_new;
	else {
	    n = n_new;
	    vec3 initial = 0;
	    v.resize(a_size());
	}
    }



//| operators

gmatrix3_vec3 operator + (const gmatrix3_vec3& a, const gmatrix3_vec3& b)
{
    
    if(a.n1 != b.n1) fatal("gmatrix3_vec3 + gmatrix3_vec3: size dismatch");
    
    if(a.n2 != b.n2) fatal("gmatrix3_vec3 + gmatrix3_vec3: size dismatch");
    
    if(a.n3 != b.n3) fatal("gmatrix3_vec3 + gmatrix3_vec3: size dismatch");
    

    gmatrix3_vec3 c(a.n1  ,a.n2  ,a.n3 );
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] + b.v[i];
    }
    return c;
}

gmatrix3_vec3 operator - (const gmatrix3_vec3& a, const gmatrix3_vec3& b)
{
    
    if(a.n1 != b.n1) fatal("gmatrix3_vec3 + gmatrix3_vec3: size dismatch");
    
    if(a.n2 != b.n2) fatal("gmatrix3_vec3 + gmatrix3_vec3: size dismatch");
    
    if(a.n3 != b.n3) fatal("gmatrix3_vec3 + gmatrix3_vec3: size dismatch");
    

    gmatrix3_vec3 c(a.n1  ,a.n2  ,a.n3 );
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] - b.v[i];
    }
    return c;
}

gmatrix3_vec3 operator * (const gmatrix3_vec3& a, const gmatrix3_vec3& b)
{
    
    if(a.n1 != b.n1) fatal("gmatrix3_vec3 + gmatrix3_vec3: size dismatch");
    
    if(a.n2 != b.n2) fatal("gmatrix3_vec3 + gmatrix3_vec3: size dismatch");
    
    if(a.n3 != b.n3) fatal("gmatrix3_vec3 + gmatrix3_vec3: size dismatch");
    

    gmatrix3_vec3 c(a.n1  ,a.n2  ,a.n3 );
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] * b.v[i];
    }
    return c;
}

gmatrix3_vec3 operator / (const gmatrix3_vec3& a, const gmatrix3_vec3& b)
{
    
    if(a.n1 != b.n1) fatal("gmatrix3_vec3 + gmatrix3_vec3: size dismatch");
    
    if(a.n2 != b.n2) fatal("gmatrix3_vec3 + gmatrix3_vec3: size dismatch");
    
    if(a.n3 != b.n3) fatal("gmatrix3_vec3 + gmatrix3_vec3: size dismatch");
    

    gmatrix3_vec3 c(a.n1  ,a.n2  ,a.n3 );
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] / b.v[i];
    }
    return c;
}


//| cumulative

gmatrix3_vec3 operator += (gmatrix3_vec3& a, const gmatrix3_vec3& b)
{
    return a = (a + b);
}

gmatrix3_vec3 operator -= (gmatrix3_vec3& a, const gmatrix3_vec3& b)
{
    return a = (a - b);
}

gmatrix3_vec3 operator *= (gmatrix3_vec3& a, const gmatrix3_vec3& b)
{
    return a = (a * b);
}

gmatrix3_vec3 operator /= (gmatrix3_vec3& a, const gmatrix3_vec3& b)
{
    return a = (a / b);
}


			
 gmatrix3_vec4& gmatrix3_vec4::operator=(int a)
    {
	for(int i = 0; i < n;++i)
	    v[i] = a;
	return *this;
    }

 gmatrix3_vec4& gmatrix3_vec4::operator=(const gmatrix3_vec4& a)
    {
	
	n1 = a.n1;
	
	n2 = a.n2;
	
	n3 = a.n3;
	
	resize(a.n);
	for(int i = 0; i < a.n;++i)
	    v[i] = a.v[i];
	return *this;
    }

 vec4& gmatrix3_vec4::operator() (int n1_ ,int n2_ ,int n3_)
    {
	return v[cal_index( n1, n2, n3, n1_, n2_, n3_)];	
    }

 const vec4& gmatrix3_vec4::operator() (int n1_ ,int n2_ ,int n3_) const
    {
	return v[cal_index( n1, n2, n3, n1_, n2_, n3_)];	
    }

 void gmatrix3_vec4::resize(int n_new)
    {
	if(n_new < 0) fatal("gmatrix3_vec4: bad new gmatrix size");
	if(ceil(n,grain) >= ceil(n_new,grain))
	    n = n_new;
	else {
	    n = n_new;
	    v.resize(a_size());
	}
    }

 void gmatrix3_vec4::resize(int n1_, int n2_, int n3_)
    {
	
	n1 = n1_;
	
	n2 = n2_;
	
	n3 = n3_;
	

	int n_new = n1_ *n2_ *n3_ ;
	if(n_new < 0) fatal("gmatrix3_vec4: bad new gmatrix size");
	if(ceil(n,grain) >= ceil(n_new,grain))
	    n = n_new;
	else {
	    n = n_new;
	    v.resize(a_size());
	}
    }



//| operators

gmatrix3_vec4 operator + (const gmatrix3_vec4& a, const gmatrix3_vec4& b)
{
    
    if(a.n1 != b.n1) fatal("gmatrix3_vec4 + gmatrix3_vec4: size dismatch");
    
    if(a.n2 != b.n2) fatal("gmatrix3_vec4 + gmatrix3_vec4: size dismatch");
    
    if(a.n3 != b.n3) fatal("gmatrix3_vec4 + gmatrix3_vec4: size dismatch");
    

    gmatrix3_vec4 c(a.n1  ,a.n2  ,a.n3 );
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] + b.v[i];
    }
    return c;
}

gmatrix3_vec4 operator - (const gmatrix3_vec4& a, const gmatrix3_vec4& b)
{
    
    if(a.n1 != b.n1) fatal("gmatrix3_vec4 + gmatrix3_vec4: size dismatch");
    
    if(a.n2 != b.n2) fatal("gmatrix3_vec4 + gmatrix3_vec4: size dismatch");
    
    if(a.n3 != b.n3) fatal("gmatrix3_vec4 + gmatrix3_vec4: size dismatch");
    

    gmatrix3_vec4 c(a.n1  ,a.n2  ,a.n3 );
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] - b.v[i];
    }
    return c;
}

gmatrix3_vec4 operator * (const gmatrix3_vec4& a, const gmatrix3_vec4& b)
{
    
    if(a.n1 != b.n1) fatal("gmatrix3_vec4 + gmatrix3_vec4: size dismatch");
    
    if(a.n2 != b.n2) fatal("gmatrix3_vec4 + gmatrix3_vec4: size dismatch");
    
    if(a.n3 != b.n3) fatal("gmatrix3_vec4 + gmatrix3_vec4: size dismatch");
    

    gmatrix3_vec4 c(a.n1  ,a.n2  ,a.n3 );
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] * b.v[i];
    }
    return c;
}

gmatrix3_vec4 operator / (const gmatrix3_vec4& a, const gmatrix3_vec4& b)
{
    
    if(a.n1 != b.n1) fatal("gmatrix3_vec4 + gmatrix3_vec4: size dismatch");
    
    if(a.n2 != b.n2) fatal("gmatrix3_vec4 + gmatrix3_vec4: size dismatch");
    
    if(a.n3 != b.n3) fatal("gmatrix3_vec4 + gmatrix3_vec4: size dismatch");
    

    gmatrix3_vec4 c(a.n1  ,a.n2  ,a.n3 );
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] / b.v[i];
    }
    return c;
}


//| cumulative

gmatrix3_vec4 operator += (gmatrix3_vec4& a, const gmatrix3_vec4& b)
{
    return a = (a + b);
}

gmatrix3_vec4 operator -= (gmatrix3_vec4& a, const gmatrix3_vec4& b)
{
    return a = (a - b);
}

gmatrix3_vec4 operator *= (gmatrix3_vec4& a, const gmatrix3_vec4& b)
{
    return a = (a * b);
}

gmatrix3_vec4 operator /= (gmatrix3_vec4& a, const gmatrix3_vec4& b)
{
    return a = (a / b);
}



					
			
 gmatrix4_real& gmatrix4_real::operator=(int a)
    {
	for(int i = 0; i < n;++i)	    v[i] = a;
	return *this;
    }

 gmatrix4_real& gmatrix4_real::operator=(const gmatrix4_real& a)
    {
	
	n1 = a.n1;
	
	n2 = a.n2;
	
	n3 = a.n3;
	
	n4 = a.n4;
	
	resize(a.n);
	for(int i = 0; i < a.n;++i)
	    v[i] = a.v[i];
	return *this;
    }

 real& gmatrix4_real::operator() (int n1_ ,int n2_ ,int n3_ ,int n4_)
    {
	return v[cal_index( n1, n2, n3, n4, n1_, n2_, n3_, n4_)];	
    }

 const real& gmatrix4_real::operator() (int n1_ ,int n2_ ,int n3_ ,int n4_) const
    {
	return v[cal_index( n1, n2, n3, n4, n1_, n2_, n3_, n4_)];	
    }

 void gmatrix4_real::resize(int n_new)
    {
	if(n_new < 0) fatal("gmatrix4_real: bad new gmatrix size");
	if(ceil(n,grain) >= ceil(n_new,grain))
	    n = n_new;
	else {
	    n = n_new;
	    v.resize(a_size());
	}
    }

 void gmatrix4_real::resize(int n1_, int n2_, int n3_, int n4_)
    {
	
	n1 = n1_;
	
	n2 = n2_;
	
	n3 = n3_;
	
	n4 = n4_;
	

	int n_new = n1_ *n2_ *n3_ *n4_ ;
	if(n_new < 0) fatal("gmatrix4_real: bad new gmatrix size");
	if(ceil(n,grain) >= ceil(n_new,grain))
	    n = n_new;
	else {
	    n = n_new;
	    v.resize(a_size());
	}
    }



//| operators

gmatrix4_real operator + (const gmatrix4_real& a, const gmatrix4_real& b)
{
    
    if(a.n1 != b.n1) fatal("gmatrix4_real + gmatrix4_real: size dismatch");
    
    if(a.n2 != b.n2) fatal("gmatrix4_real + gmatrix4_real: size dismatch");
    
    if(a.n3 != b.n3) fatal("gmatrix4_real + gmatrix4_real: size dismatch");
    
    if(a.n4 != b.n4) fatal("gmatrix4_real + gmatrix4_real: size dismatch");
    

    gmatrix4_real c(a.n1  ,a.n2  ,a.n3  ,a.n4 );
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] + b.v[i];
    }
    return c;
}

gmatrix4_real operator - (const gmatrix4_real& a, const gmatrix4_real& b)
{
    
    if(a.n1 != b.n1) fatal("gmatrix4_real + gmatrix4_real: size dismatch");
    
    if(a.n2 != b.n2) fatal("gmatrix4_real + gmatrix4_real: size dismatch");
    
    if(a.n3 != b.n3) fatal("gmatrix4_real + gmatrix4_real: size dismatch");
    
    if(a.n4 != b.n4) fatal("gmatrix4_real + gmatrix4_real: size dismatch");
    

    gmatrix4_real c(a.n1  ,a.n2  ,a.n3  ,a.n4 );
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] - b.v[i];
    }
    return c;
}

gmatrix4_real operator * (const gmatrix4_real& a, const gmatrix4_real& b)
{
    
    if(a.n1 != b.n1) fatal("gmatrix4_real + gmatrix4_real: size dismatch");
    
    if(a.n2 != b.n2) fatal("gmatrix4_real + gmatrix4_real: size dismatch");
    
    if(a.n3 != b.n3) fatal("gmatrix4_real + gmatrix4_real: size dismatch");
    
    if(a.n4 != b.n4) fatal("gmatrix4_real + gmatrix4_real: size dismatch");
    

    gmatrix4_real c(a.n1  ,a.n2  ,a.n3  ,a.n4 );
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] * b.v[i];
    }
    return c;
}

gmatrix4_real operator / (const gmatrix4_real& a, const gmatrix4_real& b)
{
    
    if(a.n1 != b.n1) fatal("gmatrix4_real + gmatrix4_real: size dismatch");
    
    if(a.n2 != b.n2) fatal("gmatrix4_real + gmatrix4_real: size dismatch");
    
    if(a.n3 != b.n3) fatal("gmatrix4_real + gmatrix4_real: size dismatch");
    
    if(a.n4 != b.n4) fatal("gmatrix4_real + gmatrix4_real: size dismatch");
    

    gmatrix4_real c(a.n1  ,a.n2  ,a.n3  ,a.n4 );
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] / b.v[i];
    }
    return c;
}


//| cumulative

gmatrix4_real operator += (gmatrix4_real& a, const gmatrix4_real& b)
{
    return a = (a + b);
}

gmatrix4_real operator -= (gmatrix4_real& a, const gmatrix4_real& b)
{
    return a = (a - b);
}

gmatrix4_real operator *= (gmatrix4_real& a, const gmatrix4_real& b)
{
    return a = (a * b);
}

gmatrix4_real operator /= (gmatrix4_real& a, const gmatrix4_real& b)
{
    return a = (a / b);
}


			
 gmatrix4_vec2& gmatrix4_vec2::operator=(int a)
    {
	for(int i = 0; i < n;++i)
	    v[i] = a;
	return *this;
    }

 gmatrix4_vec2& gmatrix4_vec2::operator=(const gmatrix4_vec2& a)
    {
	
	n1 = a.n1;
	
	n2 = a.n2;
	
	n3 = a.n3;
	
	n4 = a.n4;
	
	resize(a.n);
	for(int i = 0; i < a.n;++i)
	    v[i] = a.v[i];
	return *this;
    }

 vec2& gmatrix4_vec2::operator() (int n1_ ,int n2_ ,int n3_ ,int n4_)
    {
	return v[cal_index( n1, n2, n3, n4, n1_, n2_, n3_, n4_)];	
    }

 const vec2& gmatrix4_vec2::operator() (int n1_ ,int n2_ ,int n3_ ,int n4_) const
    {
	return v[cal_index( n1, n2, n3, n4, n1_, n2_, n3_, n4_)];	
    }

 void gmatrix4_vec2::resize(int n_new)
    {
	if(n_new < 0) fatal("gmatrix4_vec2: bad new gmatrix size");
	if(ceil(n,grain) >= ceil(n_new,grain))
	    n = n_new;
	else {
	    n = n_new;
	    v.resize(a_size());
	}
    }

 void gmatrix4_vec2::resize(int n1_, int n2_, int n3_, int n4_)
    {
	
	n1 = n1_;
	
	n2 = n2_;
	
	n3 = n3_;
	
	n4 = n4_;
	

	int n_new = n1_ *n2_ *n3_ *n4_ ;
	if(n_new < 0) fatal("gmatrix4_vec2: bad new gmatrix size");
	if(ceil(n,grain) >= ceil(n_new,grain))
	    n = n_new;
	else {
	    n = n_new;
	    v.resize(a_size());
	}
    }



//| operators

gmatrix4_vec2 operator + (const gmatrix4_vec2& a, const gmatrix4_vec2& b)
{
    
    if(a.n1 != b.n1) fatal("gmatrix4_vec2 + gmatrix4_vec2: size dismatch");
    
    if(a.n2 != b.n2) fatal("gmatrix4_vec2 + gmatrix4_vec2: size dismatch");
    
    if(a.n3 != b.n3) fatal("gmatrix4_vec2 + gmatrix4_vec2: size dismatch");
    
    if(a.n4 != b.n4) fatal("gmatrix4_vec2 + gmatrix4_vec2: size dismatch");
    

    gmatrix4_vec2 c(a.n1  ,a.n2  ,a.n3  ,a.n4 );
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] + b.v[i];
    }
    return c;
}

gmatrix4_vec2 operator - (const gmatrix4_vec2& a, const gmatrix4_vec2& b)
{
    
    if(a.n1 != b.n1) fatal("gmatrix4_vec2 + gmatrix4_vec2: size dismatch");
    
    if(a.n2 != b.n2) fatal("gmatrix4_vec2 + gmatrix4_vec2: size dismatch");
    
    if(a.n3 != b.n3) fatal("gmatrix4_vec2 + gmatrix4_vec2: size dismatch");
    
    if(a.n4 != b.n4) fatal("gmatrix4_vec2 + gmatrix4_vec2: size dismatch");
    

    gmatrix4_vec2 c(a.n1  ,a.n2  ,a.n3  ,a.n4 );
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] - b.v[i];
    }
    return c;
}

gmatrix4_vec2 operator * (const gmatrix4_vec2& a, const gmatrix4_vec2& b)
{
    
    if(a.n1 != b.n1) fatal("gmatrix4_vec2 + gmatrix4_vec2: size dismatch");
    
    if(a.n2 != b.n2) fatal("gmatrix4_vec2 + gmatrix4_vec2: size dismatch");
    
    if(a.n3 != b.n3) fatal("gmatrix4_vec2 + gmatrix4_vec2: size dismatch");
    
    if(a.n4 != b.n4) fatal("gmatrix4_vec2 + gmatrix4_vec2: size dismatch");
    

    gmatrix4_vec2 c(a.n1  ,a.n2  ,a.n3  ,a.n4 );
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] * b.v[i];
    }
    return c;
}

gmatrix4_vec2 operator / (const gmatrix4_vec2& a, const gmatrix4_vec2& b)
{
    
    if(a.n1 != b.n1) fatal("gmatrix4_vec2 + gmatrix4_vec2: size dismatch");
    
    if(a.n2 != b.n2) fatal("gmatrix4_vec2 + gmatrix4_vec2: size dismatch");
    
    if(a.n3 != b.n3) fatal("gmatrix4_vec2 + gmatrix4_vec2: size dismatch");
    
    if(a.n4 != b.n4) fatal("gmatrix4_vec2 + gmatrix4_vec2: size dismatch");
    

    gmatrix4_vec2 c(a.n1  ,a.n2  ,a.n3  ,a.n4 );
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] / b.v[i];
    }
    return c;
}


//| cumulative

gmatrix4_vec2 operator += (gmatrix4_vec2& a, const gmatrix4_vec2& b)
{
    return a = (a + b);
}

gmatrix4_vec2 operator -= (gmatrix4_vec2& a, const gmatrix4_vec2& b)
{
    return a = (a - b);
}

gmatrix4_vec2 operator *= (gmatrix4_vec2& a, const gmatrix4_vec2& b)
{
    return a = (a * b);
}

gmatrix4_vec2 operator /= (gmatrix4_vec2& a, const gmatrix4_vec2& b)
{
    return a = (a / b);
}


			
 gmatrix4_vec3& gmatrix4_vec3::operator=(int a)
    {
	for(int i = 0; i < n;++i)	    v[i] = a;
	return *this;
    }

 gmatrix4_vec3& gmatrix4_vec3::operator=(const gmatrix4_vec3& a)
    {
	
	n1 = a.n1;
	
	n2 = a.n2;
	
	n3 = a.n3;
	
	n4 = a.n4;
	
	resize(a.n);
	for(int i = 0; i < a.n;++i)	    v[i] = a.v[i];
	return *this;
    }

 vec3& gmatrix4_vec3::operator() (int n1_ ,int n2_ ,int n3_ ,int n4_)
    {
	return v[cal_index( n1, n2, n3, n4, n1_, n2_, n3_, n4_)];	
    }

 const vec3& gmatrix4_vec3::operator() (int n1_ ,int n2_ ,int n3_ ,int n4_) const
    {
	return v[cal_index( n1, n2, n3, n4, n1_, n2_, n3_, n4_)];	
    }

 void gmatrix4_vec3::resize(int n_new)
    {
	if(n_new < 0) fatal("gmatrix4_vec3: bad new gmatrix size");
	if(ceil(n,grain) >= ceil(n_new,grain))
	    n = n_new;
	else {
	    n = n_new;
	    v.resize(a_size());
	}
    }

 void gmatrix4_vec3::resize(int n1_, int n2_, int n3_, int n4_)
    {
	
	n1 = n1_;
	
	n2 = n2_;
	
	n3 = n3_;
	
	n4 = n4_;
	

	int n_new = n1_ *n2_ *n3_ *n4_ ;
	if(n_new < 0) fatal("gmatrix4_vec3: bad new gmatrix size");
	if(ceil(n,grain) >= ceil(n_new,grain))
	    n = n_new;
	else {
	    n = n_new;
	    v.resize(a_size());
	}
    }



//| operators

gmatrix4_vec3 operator + (const gmatrix4_vec3& a, const gmatrix4_vec3& b)
{
    
    if(a.n1 != b.n1) fatal("gmatrix4_vec3 + gmatrix4_vec3: size dismatch");
    
    if(a.n2 != b.n2) fatal("gmatrix4_vec3 + gmatrix4_vec3: size dismatch");
    
    if(a.n3 != b.n3) fatal("gmatrix4_vec3 + gmatrix4_vec3: size dismatch");
    
    if(a.n4 != b.n4) fatal("gmatrix4_vec3 + gmatrix4_vec3: size dismatch");
    

    gmatrix4_vec3 c(a.n1  ,a.n2  ,a.n3  ,a.n4 );
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] + b.v[i];
    }
    return c;
}

gmatrix4_vec3 operator - (const gmatrix4_vec3& a, const gmatrix4_vec3& b)
{
    
    if(a.n1 != b.n1) fatal("gmatrix4_vec3 + gmatrix4_vec3: size dismatch");
    
    if(a.n2 != b.n2) fatal("gmatrix4_vec3 + gmatrix4_vec3: size dismatch");
    
    if(a.n3 != b.n3) fatal("gmatrix4_vec3 + gmatrix4_vec3: size dismatch");
    
    if(a.n4 != b.n4) fatal("gmatrix4_vec3 + gmatrix4_vec3: size dismatch");
    

    gmatrix4_vec3 c(a.n1  ,a.n2  ,a.n3  ,a.n4 );
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] - b.v[i];
    }
    return c;
}

gmatrix4_vec3 operator * (const gmatrix4_vec3& a, const gmatrix4_vec3& b)
{
    
    if(a.n1 != b.n1) fatal("gmatrix4_vec3 + gmatrix4_vec3: size dismatch");
    
    if(a.n2 != b.n2) fatal("gmatrix4_vec3 + gmatrix4_vec3: size dismatch");
    
    if(a.n3 != b.n3) fatal("gmatrix4_vec3 + gmatrix4_vec3: size dismatch");
    
    if(a.n4 != b.n4) fatal("gmatrix4_vec3 + gmatrix4_vec3: size dismatch");
    

    gmatrix4_vec3 c(a.n1  ,a.n2  ,a.n3  ,a.n4 );
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] * b.v[i];
    }
    return c;
}

gmatrix4_vec3 operator / (const gmatrix4_vec3& a, const gmatrix4_vec3& b)
{
    
    if(a.n1 != b.n1) fatal("gmatrix4_vec3 + gmatrix4_vec3: size dismatch");
    
    if(a.n2 != b.n2) fatal("gmatrix4_vec3 + gmatrix4_vec3: size dismatch");
    
    if(a.n3 != b.n3) fatal("gmatrix4_vec3 + gmatrix4_vec3: size dismatch");
    
    if(a.n4 != b.n4) fatal("gmatrix4_vec3 + gmatrix4_vec3: size dismatch");
    

    gmatrix4_vec3 c(a.n1  ,a.n2  ,a.n3  ,a.n4 );
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] / b.v[i];
    }
    return c;
}


//| cumulative

gmatrix4_vec3 operator += (gmatrix4_vec3& a, const gmatrix4_vec3& b)
{
    return a = (a + b);
}

gmatrix4_vec3 operator -= (gmatrix4_vec3& a, const gmatrix4_vec3& b)
{
    return a = (a - b);
}

gmatrix4_vec3 operator *= (gmatrix4_vec3& a, const gmatrix4_vec3& b)
{
    return a = (a * b);
}

gmatrix4_vec3 operator /= (gmatrix4_vec3& a, const gmatrix4_vec3& b)
{
    return a = (a / b);
}


			
 gmatrix4_vec4& gmatrix4_vec4::operator=(int a)
    {
	for(int i = 0; i < n;++i)
	    v[i] = a;
	return *this;
    }

 gmatrix4_vec4& gmatrix4_vec4::operator=(const gmatrix4_vec4& a)
    {
	
	n1 = a.n1;
	
	n2 = a.n2;
	
	n3 = a.n3;
	
	n4 = a.n4;
	
	resize(a.n);
	for(int i = 0; i < a.n;++i)
	    v[i] = a.v[i];
	return *this;
    }

 vec4& gmatrix4_vec4::operator() (int n1_ ,int n2_ ,int n3_ ,int n4_)
    {
	return v[cal_index( n1, n2, n3, n4, n1_, n2_, n3_, n4_)];	
    }

 const vec4& gmatrix4_vec4::operator() (int n1_ ,int n2_ ,int n3_ ,int n4_) const
    {
	return v[cal_index( n1, n2, n3, n4, n1_, n2_, n3_, n4_)];	
    }

 void gmatrix4_vec4::resize(int n_new)
    {
	if(n_new < 0) fatal("gmatrix4_vec4: bad new gmatrix size");
	if(ceil(n,grain) >= ceil(n_new,grain))
	    n = n_new;
	else {
	    n = n_new;
	    v.resize(a_size());
	}
    }

 void gmatrix4_vec4::resize(int n1_, int n2_, int n3_, int n4_)
    {
	
	n1 = n1_;
	
	n2 = n2_;
	
	n3 = n3_;
	
	n4 = n4_;
	

	int n_new = n1_ *n2_ *n3_ *n4_ ;
	if(n_new < 0) fatal("gmatrix4_vec4: bad new gmatrix size");
	if(ceil(n,grain) >= ceil(n_new,grain))
	    n = n_new;
	else {
	    n = n_new;
	    v.resize(a_size());
	}
    }



//| operators

gmatrix4_vec4 operator + (const gmatrix4_vec4& a, const gmatrix4_vec4& b)
{
    
    if(a.n1 != b.n1) fatal("gmatrix4_vec4 + gmatrix4_vec4: size dismatch");
    
    if(a.n2 != b.n2) fatal("gmatrix4_vec4 + gmatrix4_vec4: size dismatch");
    
    if(a.n3 != b.n3) fatal("gmatrix4_vec4 + gmatrix4_vec4: size dismatch");
    
    if(a.n4 != b.n4) fatal("gmatrix4_vec4 + gmatrix4_vec4: size dismatch");
    

    gmatrix4_vec4 c(a.n1  ,a.n2  ,a.n3  ,a.n4 );
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] + b.v[i];
    }
    return c;
}

gmatrix4_vec4 operator - (const gmatrix4_vec4& a, const gmatrix4_vec4& b)
{
    
    if(a.n1 != b.n1) fatal("gmatrix4_vec4 + gmatrix4_vec4: size dismatch");
    
    if(a.n2 != b.n2) fatal("gmatrix4_vec4 + gmatrix4_vec4: size dismatch");
    
    if(a.n3 != b.n3) fatal("gmatrix4_vec4 + gmatrix4_vec4: size dismatch");
    
    if(a.n4 != b.n4) fatal("gmatrix4_vec4 + gmatrix4_vec4: size dismatch");
    

    gmatrix4_vec4 c(a.n1  ,a.n2  ,a.n3  ,a.n4 );
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] - b.v[i];
    }
    return c;
}

gmatrix4_vec4 operator * (const gmatrix4_vec4& a, const gmatrix4_vec4& b)
{
    
    if(a.n1 != b.n1) fatal("gmatrix4_vec4 + gmatrix4_vec4: size dismatch");
    
    if(a.n2 != b.n2) fatal("gmatrix4_vec4 + gmatrix4_vec4: size dismatch");
    
    if(a.n3 != b.n3) fatal("gmatrix4_vec4 + gmatrix4_vec4: size dismatch");
    
    if(a.n4 != b.n4) fatal("gmatrix4_vec4 + gmatrix4_vec4: size dismatch");
    

    gmatrix4_vec4 c(a.n1  ,a.n2  ,a.n3  ,a.n4 );
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] * b.v[i];
    }
    return c;
}

gmatrix4_vec4 operator / (const gmatrix4_vec4& a, const gmatrix4_vec4& b)
{
    
    if(a.n1 != b.n1) fatal("gmatrix4_vec4 + gmatrix4_vec4: size dismatch");
    
    if(a.n2 != b.n2) fatal("gmatrix4_vec4 + gmatrix4_vec4: size dismatch");
    
    if(a.n3 != b.n3) fatal("gmatrix4_vec4 + gmatrix4_vec4: size dismatch");
    
    if(a.n4 != b.n4) fatal("gmatrix4_vec4 + gmatrix4_vec4: size dismatch");
    

    gmatrix4_vec4 c(a.n1  ,a.n2  ,a.n3  ,a.n4 );
    for(int i = 0; i < a.n;++i){
	c[i] = a.v[i] / b.v[i];
    }
    return c;
}


//| cumulative

gmatrix4_vec4 operator += (gmatrix4_vec4& a, const gmatrix4_vec4& b)
{
    return a = (a + b);
}

gmatrix4_vec4 operator -= (gmatrix4_vec4& a, const gmatrix4_vec4& b)
{
    return a = (a - b);
}

gmatrix4_vec4 operator *= (gmatrix4_vec4& a, const gmatrix4_vec4& b)
{
    return a = (a * b);
}

gmatrix4_vec4 operator /= (gmatrix4_vec4& a, const gmatrix4_vec4& b)
{
    return a = (a / b);
}





//|
//| utility functions to calculate the index inside the class gmatrix
//|
inline int cal_index(int n1, int n2, int n3, int n1_, int n2_, int n3_)
{
    return (n1_%n1)*n2*n3 + (n2_%n2)*n3 + n3_%n3;
}

inline int cal_index(int n1, int n2, int n3, int n4,
	      int n1_, int n2_, int n3_, int n4_)
{
    return (n1_%n1)*n2*n3*n4 + (n2_%n2)*n3*n4 + (n3_%n3)*n4 + (n4_%n4);
}

inline int cal_index(int n1, int n2, int n3, int n4, int n5,
	      int n1_, int n2_, int n3_, int n4_, int n5_)
{
    return (n1_%n1)*n2*n3*n4*n5 + (n2_%n2)*n3*n4*n5 + (n3_%n3)*n4*n5 + (n4_%n4)*n5 + (n5_%n5);
}

inline int cal_index(int n1, int n2, int n3, int n4, int n5, int n6,
	      int n1_, int n2_, int n3_, int n4_, int n5_, int n6_)
{
    return (n1_%n1)*n2*n3*n4*n5*n6 + (n2_%n2)*n3*n4*n5*n6 + (n3_%n3)*n4*n5 + (n4_%n4)*n5*n6 + (n5_%n5)*n6 + (n6_%n6);

}


}; 