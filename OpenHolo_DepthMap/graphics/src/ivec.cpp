#include "graphics/ivec.h"

#include "graphics/sys.h"
//|
//| ivec : n-dimensional ivector
//|




namespace graphics {



const int ivec2::n = 2;




//| I/O
void print(const ivec2& a)
{
    LOG("(%g", a[0]);
    for(int i = 1; i < 2;++i){
	LOG(" %g", a[i]);
    }
    LOG(") ");
}

void store(FILE* fp, const ivec2& v)
{
    fprintf(fp, "(%lg", v[0]);
    for(int i = 1; i < 2;++i){
	fprintf(fp, " %lg", v[i]);
    }
    fprintf(fp, ")\n");
}

int scan(FILE* fp, const ivec2& v)
{
    int a = fscanf(fp, " (");
    for(int i = 0; i < 2;++i){
	a += fscanf(fp, " %lg", &v[i]);
    }
    a += fscanf(fp, " )");
    return a;
}





const int ivec3::n = 3;




//| I/O
void print(const ivec3& a)
{
    LOG("(%g", a[0]);
    for(int i = 1; i < 3;++i){
	LOG(" %g", a[i]);
    }
    LOG(") ");
}

void store(FILE* fp, const ivec3& v)
{
    fprintf(fp, "(%lg", v[0]);
    for(int i = 1; i < 3;++i){
	fprintf(fp, " %lg", v[i]);
    }
    fprintf(fp, ")\n");
}

int scan(FILE* fp, const ivec3& v)
{
    int a = fscanf(fp, " (");
    for(int i = 0; i < 3;++i){
	a += fscanf(fp, " %lg", &v[i]);
    }
    a += fscanf(fp, " )");
    return a;
}





const int ivec4::n = 4;




//| I/O
void print(const ivec4& a)
{
    LOG("(%g", a[0]);
    for(int i = 1; i < 4;++i){
	LOG(" %g", a[i]);
    }
    LOG(") ");
}

void store(FILE* fp, const ivec4& v)
{
    fprintf(fp, "(%lg", v[0]);
    for(int i = 1; i < 4;++i){
	fprintf(fp, " %lg", v[i]);
    }
    fprintf(fp, ")\n");
}

int scan(FILE* fp, const ivec4& v)
{
    int a = fscanf(fp, " (");
    for(int i = 0; i < 4;++i){
	a += fscanf(fp, " %lg", &v[i]);
    }
    a += fscanf(fp, " )");
    return a;
}


}; //namespace graphics