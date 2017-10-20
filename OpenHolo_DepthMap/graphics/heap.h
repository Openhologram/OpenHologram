#ifndef __heap_h
#define __heap_h

/*|
    i -> 2i + 1		left
    i -> 2i + 2		right
    i -> (i - 1)/2	parent
*/


#include "graphics/sys.h"
#include "graphics/vector.h"

namespace graphics {

template<class T> 
struct heap {

    vector<T> A;

    heap() : A(0, 50) { }
    ~heap() { }

    int size() const 
    { 
	return A.size(); 
    }

    int has_element() 
    {
	return A.size();
    }

    int lt(T a, int i) {
	if(i < A.size()) return a < A[i];
	else return 1;
    }
    int le(T a, int i) {
	if(i < A.size()) return a <= A[i];
	else return 1;
    }
    int _lt(int i, int j)
    {
	if(i < A.size()) {
	    if(j < A.size()) return A[i] < A[j];
	    else return 1;
	} else {
	    return 0;
	}
    }

    int lchild(int i) { return 2 * i + 1; } //| left child
    int rchild(int i) { return 2 * i + 2; } //| right child
    int parent(int i) { return (i-1)/2; } //| parent

    void add(T a)
    {
	A.add(a);
	int i;
	for(i = A.size() - 1; i > 0; ) {
	    int up = (i - 1) / 2;
	    if(lt(a, up)) {
		A[i] = A[up];
	    } else
		break;
	    i = up;
	}
	A[i] = a;
    }

    T del()
    {
	T min = A[0];
	T a = A[A.size() - 1];
	A.resize(A.size() - 1);

	for(int i = 0; i < A.size(); ) {
	    int l = 2 * i + 1;
	    int r = 2 * i + 2;
	    if(le(a, l) && le(a, r)) {
		A[i] = a;
		break;
	    } else if(_lt(l, r)) {
		A[i] = A[l];
		i = l;
	    } else {
		A[i] = A[r];
		i = r;
	    }
	}

	return min;
    }

    void increase(int pos) // percolate up
    {
	for(int i = pos ; i < A.size() ; ) 
	{
	    int l = lchild(i);
	    int r = rchild(i);
	    if(le(A[i], l) && le(A[i], r)) { break; }
	    else if(_lt(l, r)) {
		swap(i, l);
		i = l;
	    } else {
		swap(i, r);
		i = r;
	    }
	}
    }

    void decrease(int pos) // percolate down
    {
	for(int i = pos ; i > 0 ; ) 
	{
	    int up = parent(i);
	    if(lt(A[i], up))
		swap(i,up);
	    else
		break;
	    i = up;
	}
    }

    void swap(int i, int j) {
	T temp = A[i];
	A[i] = A[j];
	A[j] = temp;
    }

};


}; // namespace graphics
#endif
