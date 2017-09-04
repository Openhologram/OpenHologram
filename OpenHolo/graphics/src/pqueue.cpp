#include <graphics/pqueue.h>

namespace graphics {

//|
//| Basically it adopts a heap structure for efficiency
//|
//| heap properties are
//| i -> 2i + 1 : left child
//| i -> 2i + 2 : right child
//| i-> (i-1)/2 : parent
//|

 void pqueue::move(int i, int j)
    {
	//| Move j to i.
	//| Hereafter, in a structure associated with this
	//| update a tuple having a key i. remove a
	//| tuple of key j.

	A[i].e = A[j].e;
	A[i].pri = A[j].pri;
    }

 void pqueue::make(int i, void* e, real pri)
    {
	//| Replace A[i] with e and pri.
	//| and create a new correspondant for hash table
	//| which has the key, i, when the heap is associated
	//| with hash or other indexing
	A[i].e = e;
	A[i].pri = pri;
    }

 void pqueue::swap(int i, int j)
    {
	void* te = A[i].e;
	real tpri = A[i].pri;
	A[i].e = A[j].e; A[i].pri = A[j].pri;
	A[j].e = te; A[j].pri = tpri;
    }

 void pqueue::increase(int pos, real delta) 
    //| delta should be positive
    {
	A[pos].pri += delta;

	for(int i = pos ; i < n ; ) 
	{
	    int l = lchild(i);
	    int r = rchild(i);
	    if(le(A[i].pri, l) && le(A[i].pri, r)) { break; }
	    else if(_lt(l, r)) {
		swap(i, l);
		i = l;
	    } else {
		swap(i, r);
		i = r;
	    }
	}
    }

 void pqueue::decrease(int pos, real delta) //| delta should be positive
    {
	A[pos].pri -= delta;

	for(int i = pos ; i > 0 ; ) 
	{
	    int up = parent(i);
	    if(lt(A[i].pri, up))
		swap(i,up);
	    else
		break;
	    i = up;
	}
    }
 
 void pqueue::decrease(void* e, real pri) //| delta should be positive
 {
	 int pos = -1;
	 for (int i = 0 ; i < n ; i++) {
		 if (A[i].e == e) {
			 pos = i;
			 break;
		 }
	 }
	A[pos].pri = pri;

	for(int i = pos ; i > 0 ; ) 
	{
	    int up = parent(i);
	    if(lt(A[i].pri, up))
		swap(i,up);
	    else
		break;
	    i = up;
	}
 }
 void pqueue::add(void* e, real pri)
    {
	if(n == size) resize(size * 2);
	n++;
	make(n-1, e, pri);

	for(int i = n - 1 ; i > 0 ; ) 
	{
	    int up = parent(i);
	    if(lt(pri, up))
		swap(i,up);
	    else
		break;
	    i = up;
	}
    }

 void* pqueue::del(int pos)
    {
	void* ee = A[pos].e;

	//| percolade up
	decrease(pos, kMaxReal);
	del();
	return ee;
    }

 void* pqueue::del()
    {
	void* _min = A[0].e;
	real  _minp = A[0].pri;

	void* end = A[n - 1].e;
	real  endp = A[n - 1].pri;

	n--;

	for(int i = 0 ; i < n ; ) 
	{
	    int l = lchild(i);
	    int r = rchild(i);
	    if(le(endp, l) && le(endp, r))
	    {
		move(i, n);
		break;
	    } else if(_lt(l, r)) {
		move(i, l);
		i = l;
	    } else {
		move(i, r);
		i = r;
	    }
	}
	return _min;
    }

 void pqueue::resize(int newsize)
    {
	pqueue::node* AA = new pqueue::node[size = newsize];
	for (int i = 0 ; i < n ; i++ )
	{
	    AA[i].e = A[i].e;
	    AA[i].pri = A[i].pri;
	}

	delete[] A;
	A = AA;
    }


}; // namespace graphics