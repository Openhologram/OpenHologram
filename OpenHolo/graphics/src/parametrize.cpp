#include "graphics/parametrize.h"


namespace graphics {

template <class T>
vector<real> chord_length_parametrize(const vector<T>& a)
{
    vector<real> u(a.size());

    u[0] = 0.0;
    for (int i = 1; i <= u.last() ;++i)
	u[i] = u[i-1] + norm(a[i] - a[i-1]);

    return(u);
}

template <class T>
vector<real> normalized_chord_length_parametrize(const vector<T>& a)
{
    vector<real> u(a.size());

    u[0] = 0.0;
    int i;
    for (i = 1; i <= u.last() ;++i)
	u[i] = u[i-1] + norm(a[i] - a[i-1]);

    for (i = 1; i <= u.last() ;++i)        u[i] = u[i] / u[u.last()];

    return(u);
}

template <class T>
vector<real> centripetal_parametrize(const vector<T>& d)
{
    vector<real> u(d.size());

    u[0] = 0.0;
    for (int i = 1; i <= u.last() ;++i)
	u[i] = u[i-1] + sqrt(norm(d[i] - d[i-1]));

    return(u);
}

template <class T>
vector<real> normalized_centripetal_parametrize(const vector<T>& d)
{
    vector<real> u(d.size());

    u[0] = 0.0;
    int i;
    for (i = 1; i <= u.last() ;++i)
	u[i] = u[i-1] + sqrt(norm(d[i] - d[i-1]));

    for (i = 1; i <= u.last() ;++i)       u[i] = u[i] / u[u.last()];

    return(u);
}

template 
vector<real> chord_length_parametrize(const vector<vec2>& a)
;

template
vector<real> normalized_chord_length_parametrize(const vector<vec2>& a)
;

template
vector<real> centripetal_parametrize(const vector<vec2>& d)
;

template
vector<real> normalized_centripetal_parametrize(const vector<vec2>& d)
;

template 
vector<real> chord_length_parametrize(const vector<vec3>& a)
;

template
vector<real> normalized_chord_length_parametrize(const vector<vec3>& a)
;

template
vector<real> centripetal_parametrize(const vector<vec3>& d)
;

template
vector<real> normalized_centripetal_parametrize(const vector<vec3>& d)
;


template 
vector<real> chord_length_parametrize(const vector<vec4>& a)
;

template
vector<real> normalized_chord_length_parametrize(const vector<vec4>& a)
;

template
vector<real> centripetal_parametrize(const vector<vec4>& d)
;

template
vector<real> normalized_centripetal_parametrize(const vector<vec4>& d)
;

void scale_parametrization(vector<real>& a, real tmin, real tmax)
{
    if (!a.size()) return;

    real old_min = a[0], old_max = a[a.last()];

    for (int i = 0 ; i < a.size() ;++i)
	a[i] = tmin + ((a[i] - old_min)*(tmax-tmin))/(old_max-old_min);
}
}; //namespace graphics