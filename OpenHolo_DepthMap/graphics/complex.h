
//-------------------------------------------------------------------------
// Complex Numbers
// Revision :   $Rev$
// Last changed : $Date$
//
// Author       : Seungtaik Oh
// Last Update  : 19 JUL 2011
//-------------------------------------------------------------------------
#ifndef COMPLEX_
#define COMPLEX_

#include <iostream>
#include <cmath>

const double PI = 3.141592653589793238462643383279502884197169399375105820974944592308;
const double TWO_PI = 2.0*PI;

class ComplexEuler;

class Complex
{
public:
    Complex() : a(0.0), b(0.0) {}
    Complex( double ta, double tb ) : a(ta), b(tb) {}
    Complex( const Complex& p )
    {
        a = p.a;
        b = p.b;
    }
    Complex( const ComplexEuler& p );

    double mag2() const  { return a*a+b*b; }
    double mag()  const  { return sqrt(a*a+b*b); }

    double arg() const
    {
        double r = mag();
        double theta = acos( a/r );

        if( sin(theta)-b/r < 10e-6 )    
            return theta;
        else 
            return 2.0*PI-theta;
    }

    void euler( double& r, double& theta )
    {
        r = mag();
        theta = arg();
    }

    Complex conj() const {    return Complex(a, -b); }

    // arithmetic
    const Complex& operator= (const Complex& p )
    {
        a = p.a;
        b = p.b;

        return *this;
    }

    const Complex& operator+= (const Complex& p )
    {
        a += p.a;
        b += p.b;

        return *this; 
    }

    const Complex& operator-= (const Complex& p)
    {
        a -= p.a;
        b -= p.b;

        return *this;
    }

    const Complex& operator*= (const double k)
    {
        a *= k;
        b *= k;

        return *this;
    }

    const Complex& operator*= (const Complex& p)
    {
        const double ta = a;
        const double tb = b;

        a = ta*p.a-tb*p.b;
        b = ta*p.b+tb*p.a;

        return *this;
    }

    const Complex& operator/= (const double k)
    {
        a /= k;
        b /= k;

        return *this;
    }

    friend const Complex operator+ ( const Complex& p, const Complex& q)
    {
        return Complex(p) += q;
    }

    friend const Complex operator- ( const Complex& p, const Complex& q )
    {
        return Complex(p) -= q;
    }

    friend const Complex operator* ( const double k, const Complex& p )
    {
        return Complex(p) *= k;
    }

    friend const Complex operator* ( const Complex& p, const double k )
    {
        return Complex(p) *= k;
    }

    friend const Complex operator* ( const Complex& p, const Complex& q )
    {
        return Complex( p.a*q.a-p.b*q.b, p.a*q.b+p.b*q.a );
    }

    friend const Complex operator/ ( const Complex& p, const Complex& q )
    {
        return Complex((1.0/q.mag2())*(p*q.conj()));
    }

    // stream
    friend std::ostream& operator << ( std::ostream& os, const Complex& p )
    {
        os << "(" << p.a << ", " << p.b << ")";
        return os;
    }

public:
    double a, b;
};

class ComplexEuler
{
public:
    ComplexEuler() : r_(0.0), theta_(0.0) {}
    ComplexEuler( double r, double theta ) : r_(r), theta_(theta) {}
    ComplexEuler( const Complex& p ) : r_(p.mag2()), theta_(p.arg()) {}
    ComplexEuler( const ComplexEuler& p ) : r_(p.r_), theta_(p.theta_) {}

    double real() const  { return r_*cos(theta_); }
    double imaginary() const { return r_*sin(theta_); }

    double mag2() const  { return r_*r_; }
    double mag()  const  { return r_; }
    double arg()  const  { return theta_; }

    ComplexEuler conj() const { return ComplexEuler(r_, -theta_); }


    // arithmetic
    const ComplexEuler& operator= (const ComplexEuler& p )
    {
        r_      = p.r_;
        theta_  = p.theta_;

        return *this;
    }

    const ComplexEuler& operator*= (const double k)
    {
        r_ *= k;
        if( r_ < 0.0 )
            theta_ += PI;

        return *this;
    }

    const ComplexEuler& operator*= (const ComplexEuler& p )
    {
        r_      *= p.r_;
        theta_  += p.theta_;

        return *this; 
    }

    const ComplexEuler& operator/= (const double k)
    {
        r_ /= k;
        if( r_ < 0.0 )
            theta_ += PI;

        return *this;
    }

    const ComplexEuler& operator/= (const ComplexEuler& p)
    {
        r_      /= p.r_;
        theta_  -= p.theta_;

        return *this;
    }

    friend const ComplexEuler operator* ( const double k, 
                                          const ComplexEuler& p )
    {
        return ComplexEuler(p) *= k;
    }

    friend const ComplexEuler operator* ( const ComplexEuler& p, 
                                          const double k )
    {
        return ComplexEuler(p) *= k;
    }

    friend const ComplexEuler operator* ( const ComplexEuler& p, 
                                          const ComplexEuler& q )
    {
        return ComplexEuler( p.r_*q.r_, p.theta_+q.theta_ );
    }

    friend const ComplexEuler operator/ ( const ComplexEuler& p, 
                                          const ComplexEuler& q )
    {
        return ComplexEuler( p.r_/q.r_, p.theta_-q.theta_ );
    }

    // stream
    friend std::ostream& operator << ( std::ostream& os, 
                                       const ComplexEuler& p )
    {
        os << p.r_ << "exp(i" << p.theta_ << ")";
        return os;
    }

public:
    double r_, theta_;
};

inline Complex::Complex(const ComplexEuler &p)
{
    a = p.real();
    b = p.imaginary();
}
#endif
