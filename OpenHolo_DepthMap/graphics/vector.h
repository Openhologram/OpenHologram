#ifndef __vector_h
#define __vector_h
// Description:
//  vector : general list for mathematical date
//  * Myung Joon Kim
//  * Dae Hyun Kim
// Note:
//   Now it is now, long after modification, STL, but
//   with familiar interfaces.



#include "graphics/sys.h"
#include "graphics/log.h"
#include "graphics/misc.h"
#include <vector>
#include <list>


namespace graphics {

	
template<class T> 
struct vector
{

private:
    std::vector<T> v;
    int grain;

    inline int  a_size() const { return ceil(v.size(), grain); }

public:

    int  last() const { return size() - 1; }
    T* get_array() const { return const_cast<T*>(&(*(v.begin()))); }
    operator T*()
    {
	return (&(*(v.begin())));
    }

    // constructor
    vector() : v(), grain(10)
    {
    }

    vector(int n_) : v(n_), grain(10)
    {
	if (a_size()) v.reserve(a_size());
    }

    vector(int n_, int g) : v(n_), grain(g)
    {
	if (a_size()) v.reserve(a_size());
    }

    vector(const vector& a) : v(a.size()), grain(a.grain)
    {
	*this = a;
    }

    //| destructor
    ~vector()
    {
    }

    vector& operator=(int a)
    {
	for(int i = 0; i < size();++i)
	    v[i] = a;
	return *this;
    }

    vector& operator=(const vector& a)
    {
	grain = a.grain;
	v.resize(a.size());

	for(int i = 0; i < a.size();++i)
	    v[i] = a.v[i];
	return *this;
    }

    vector reverse() const
    {
	vector b(size());
	for (int i = 0 ; i < size() ;++i)
	    b[size()-i-1] = v[i];
	return b;
    }

    inline T& operator[](int i)
    {
	if (i >= 0)
	    return v[i % size()];
	else return v[size() -((-i)%size())];
    }

    inline const T& operator[](int i) const
    {
	if (i >= 0)
	    return v[i % size()];
	else return v[size() -((-i)%size())];
    }

    inline void resize(int n_new)
    {
	T initial = 0;	
	v.resize(n_new, initial);
	v.reserve(a_size());
    }

    void add(const T a)
    {
	v.push_back(a);
    }

    void attach(const vector& a)
    {
	for(int i = 0 ; i < a.size() ;++i)
	    v.push_back(a[i]);
    }

    void ins(int index, const T& a)
    {
	v.resize(size() + 1);
	for(int i = size() - 1; i > index; i--)
	    v[i] = v[i - 1];
	v[index] = a;
    }

    bool add_set(const vector<T>& val) 
    {
	if (val.size() == 0) return false;
	bool ret = false;
	for (int i = 0; i <val.size();++i){
	    if (find(val[i]) == -1) {
		ret = true;
		add(val[i]);
	    }
	}

	return true;
    }

    bool add_set(const T& val) 
    {
	if (find(val) == -1) {
	    add(val);
	    return true;
	}
	return false;	// already exist!
    }

    // This has one of val inside?
    bool has(const vector<T>& val) const 
    {
	if (val.size() == 0) return false;

	for (int i = 0 ; i < val.size();++i){
	    if (find(val[i]) != -1) return true;
	}
	return false;
    }

    bool has(const T& val) const
    {
	if (find(val) != -1) return true;
	return false;
    }

    int find(int index, const T& a) const
    {
	for(int i = index ; i < size() ;++i)
	    if(v[i] == a) return i;
	return -1;
    }

    int find(const T& a) const
    {
	for(int i = 0 ; i < size() ;++i)
	    if(v[i] == a) return i;
	return -1;
    }

    int find_reverse(const T& a) const
    {
	for(int i = size()-1 ; i >= 0 ; i--)
	    if(v[i] == a) return i;
	return -1;
    }

    void del(int index)
    {
	for(int i = index; i < size() - 1;++i)
	    v[i] = v[i + 1];
	resize(size() - 1);
    }

    void del(int from, int to)
    {
	for(int i = 0 ; i <= (to - from) ;++i)
	    del(from);
    }

    inline int	size()  const
    {
	return v.size();
    }

    inline int grain_size() const
    {
	return grain;
    }

};

//|
//| for a pointer type as a template argument, 
//| following operations are not applicable.
//|
//| binary : componentwise operation


template<class T> 
vector<T> list_to_vector(std::list<T>& input) 
{

    typename std::list<T>::iterator i;
	typename std::list<T>::iterator end;
	i = input.begin();
	end = input.end();
    vector<T> ret;

    for ( ; i != end ;++i){
	ret.add(*i);
    }
    return ret;
}


template<class T> vector<T> operator + (const vector<T>& a, const vector<T>& b)
{
    if(a.size() != b.size()) fatal("vector + : size dismatch");

    vector<T> c(a.size());
    for(int i = 0; i < a.size();++i){
	c[i] = a.v[i] + b.v[i];
    }
    return c;
}

template<class T> vector<T> operator - (const vector<T>& a, const vector<T>& b)
{
    if(a.size() != b.size()) fatal("vector - : size dismatch");

    vector<T> c(a.size());
    for(int i = 0; i < a.size();++i){
	c[i] = a[i] - b[i];
    }
    return c;
}

template<class T> vector<T> operator * (const vector<T>& a, const vector<T>& b)
{
    if(a.size() != b.size()) fatal("vector * : size dismatch");

    vector<T> c(a.size());
    for(int i = 0; i < a.size();++i){
	c[i] = a[i] * b[i];
    }
    return c;
}

template<class T> vector<T> operator / (const vector<T>& a, const vector<T>& b)
{
    if(a.size() != b.size()) fatal("vector / : size dismatch");

    vector<T> c(a.size());
    for(int i = 0; i < a.size();++i){
	c[i] = a[i] / b[i];
    }
    return c;
}


//| cumulative : componentwise operation

template<class T> vector<T> operator += (const vector<T>& a, const vector<T>& b)
{
    vector<T> c(a.size());
    return c = (a + b);
}

template<class T> vector<T> operator -= (const vector<T>& a, const vector<T>& b)
{
    vector<T> c(a.size());
    return c = (a - b);
}

template<class T> vector<T> operator *= (const vector<T>& a, const vector<T>& b)
{
    vector<T> c(a.size());
    return c = (a * b);
}

template<class T> vector<T> operator /= (const vector<T>& a, const vector<T>& b)
{
    vector<T> c(a.size());
    return c = (a / b);
}


//| logical : componentwise operation

template<class T> int operator == (const vector<T>& a, const vector<T>& b)
{
    if(a.size() != b.size()) fatal("vect == : size dismatch");

    int c = 1;
    for(int i = 0; i < a.size();++i){
	c = c && (a[i] == b[i]);
    }
    return c;
}

template<class T> int operator < (const vector<T>& a, const vector<T>& b)
{
    if(a.size() != b.size()) fatal("vect < : size dismatch");

    int c = 1;
    for(int i = 0; i < a.size();++i){
	c = c && (a[i] < b[i]);
    }
    return c;
}

template<class T> int operator <= (const vector<T>& a, const vector<T>& b)
{
    if(a.size() != b.size()) fatal("vect <= : size dismatch");

    int c = 1;
    for(int i = 0; i < a.size();++i){
	c = c && (a[i] <= b[i]);
    }
    return c;
}

template<class T> int operator > (const vector<T>& a, const vector<T>& b)
{
    if(a.size() != b.size()) fatal("vect > : size dismatch");

    int c = 1;
    for(int i = 0; i < a.size();++i){
	c = c && (a[i] > b[i]);
    }
    return c;
}

template<class T> int operator >= (const vector<T>& a, const vector<T>& b)
{
    if(a.size() != b.size()) fatal("vect >= : size dismatch");

    int c = 1;
    for(int i = 0; i < a.size();++i){
	c = c && (a[i] >= b[i]);
    }
    return c;
}


template<class T> int operator != (const vector<T>& a, const vector<T>& b)
{
    return !(a == b);
}

template<class T> T sum(const vector<T>& a)
{
    T s;
    s = 0;
    for(int i = 0; i < a.size();++i){
	s += a[i];
    }
    return s;
}


}; // namespace
#endif
