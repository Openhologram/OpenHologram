#ifndef __pqueue_h
#define __pqueue_h

#include <graphics/sys.h>
#include <graphics/real.h>
#include <graphics/_limits.h>

namespace graphics {

//|
//| Basically it adopts a heap structure for efficiency
//|
//| heap properties are
//| i -> 2i + 1 : left child
//| i -> 2i + 2 : right child
//| i-> (i-1)/2 : parent
//|

struct pqueue{ 

    struct node {
	void* e; //| this would have key for hash table
	real pri;
    }; //| nested class

    int size;
    int n;
    node* A;

    pqueue(int s=0) : size(s?s:1), n(0), A(new node[size]) { }
    virtual ~pqueue() { delete[] A; }
    virtual void reset() { n = 0; }

    inline int lchild(int i) { return 2 * i + 1; } //| left child
    inline int rchild(int i) { return 2 * i + 2; } //| right child
    inline int parent(int i) { return (i-1)/2; } //| parent

    inline int lt(real pri, int i) {
	if(i < n) return pri < A[i].pri;
	else return 1;
    }

    inline int le(real pri, int i) {
	if(i < n) return pri <= A[i].pri;
	else return 1;
    }

    inline int _lt(int i, int j)
    {
	if(i < n) {
	    if(j < n) return A[i].pri < A[j].pri;
	    else return 1;
	} else {
	    return 0;
	}
    }

    virtual  void move(int i, int j)
    ;

    virtual  void make(int i, void* e, real pri)
    ;

    virtual  void swap(int i, int j)
    ;

    //| percolade down :: refer to data structure book
     void increase(int pos, real delta) 
    //| delta should be positive
    ;

    //| percolade up :: refer to data structure book
     void decrease(int pos, real delta) //| delta should be positive
    ;


	 void decrease(void* e, real pri);

     void add(void* e, real pri)
    ;

     void* del(int pos)
    ;

     void* del()
    ;

    inline int is_empty() const 
    { 
	return !n; 
    }

    inline int num() const 
    { 
	return n; 
    }

    //| element which has minimum priority
    inline void* get_of_minp() const
    { 
	if(n != 0) return A[0].e; 
    }   

    //| minimum priority of this queue
    inline real get_minp() const 
    { 
	if(n != 0) return A[0].pri; 
    } 

     void resize(int newsize)
    ;
};

}; // namespace
#endif
