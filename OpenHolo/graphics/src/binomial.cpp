#include "graphics/binomial.h"

namespace graphics {

int fact(int n)
{
    return(n ? (n * fact(n-1)) : 1);
}

int Binomial(int a, int b)
{
    if(a < b) fatal("Binomial::can't solve it! \n");
    return fact(a)/(fact(b)*fact(a-b));
}




//| Binomial(2, 2)
void Binomial(vector<ivec2>& b, ivec2& a)
{
    ivec2 temp;
    int i = 0, num;
    if(b.size() != (num=Binomial(2,2))) b.resize(num);
    for(int i1 = 0 ; i1 < 2 ; i1++)
	for(int i2 = i1+1 ; i2 < 2 ; i2++)
	{
	    temp[0] = a[i1]; temp[1] = a[i2];
	    b[i] = temp; 
	   ++i;
	}
}


//| Binomial(3, 2)
void Binomial(vector<ivec2>& b, ivec3& a)
{
    ivec2 temp;
    int i = 0, num;
    if(b.size() != (num=Binomial(3,2))) b.resize(num);
    for(int i1 = 0 ; i1 < 3 ; i1++)
	for(int i2 = i1+1 ; i2 < 3 ; i2++)
	{
	    temp[0] = a[i1]; temp[1] = a[i2];
	    b[i] = temp; 
	   ++i;
	}
}


//| Binomial(4, 2)
void Binomial(vector<ivec2>& b, ivec4& a)
{
    ivec2 temp;
    int i = 0, num;
    if (b.size() != (num=Binomial(4,2))) b.resize(num);
    for (int i1 = 0 ; i1 < 4 ; i1++) {
	for (int i2 = i1+1 ; i2 < 4 ; i2++) {
	    temp[0] = a[i1]; temp[1] = a[i2];
	    b[i] = temp; 
	   ++i;
	}
    }
}

}; // namespace graphics