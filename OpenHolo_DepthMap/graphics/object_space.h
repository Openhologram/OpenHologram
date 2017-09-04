// object_space.h: interface for the object_space class.
//
//////////////////////////////////////////////////////////////////////

#ifndef	    __object_space_h
#define	    __object_space_h

#include "graphics/frame.h"
#include "graphics/vector.h"


namespace graphics {

//-----------------------------------------------------------------------------
// space type : root is the absolute world space, center is located at the
//		absolute origin, i(1, 0, 0), j(0, 1, 0), and k(0, 0, 1) are 
//		its axes. The root is the root to everything and governs
//		all the other space as a root.
//
enum	object_space_type {
    object_space_root,
    object_space_general
};
//___________________


class object_space;
class object_space_path;

class object_space
{

public:
    object_space(object_space_type val = object_space_general, void* reserved = 0);
    object_space(const object_space& cp);
    object_space(const frame& fr, void* reserved = 0);

    virtual ~object_space();

    void set_reserved(void* pnt) { m_reserved = pnt; }
    void* get_reserved() const { return m_reserved; }

    const frame& get_local_space() const;
    frame& get_local_space();

    void set_local_space(const frame& val);
    
    void ref()	 { m_ref_cnt++; }
    void unref() { if (m_ref_cnt > 0) m_ref_cnt--; if (m_ref_cnt == 0) destroy(); }

    object_space* make_child_instance();
    void del_child(object_space* ch);
    void del_children();
    void destroy();

    void add_child(object_space* ch);
    void set_parent(object_space* pr);
    object_space* get_parent() const;

    object_space*  get_root() const { return & object_space::root_space; }

    bool is_root() const;
    object_space_path  get_path()const;
    
    vec3 to_world(const vec3& pnt) const;
    vec3 to_model(const vec3& pnt) const;

    vec3 to_world_normal(const vec3& pnt) const;
    vec3 to_model_normal(const vec3& pnt) const;

    line to_model(const line& a) const;
    line to_world(const line& a) const;
    
    void push_to_world() const;
    void push_to_model() const;
    void pop() const;

    const vector<object_space*>& get_children() const;
    vector<object_space*>& get_children();

    int	get_ref_cnt() const { return m_ref_cnt; }

private:

    frame			m_local_space;
    void*			m_reserved;
    int				m_ref_cnt;
    vector<object_space*>	m_children;
    object_space*		m_parent;
    object_space_type		m_is_root;

public:

    static object_space		root_space;
};

class temporary_object_space 
{
    object_space* root;
    object_space* current;

public:
    temporary_object_space() : root(&object_space::root_space) 
    { 
	//root = new object_space(object_space_root);
	current = root->make_child_instance();
	current->ref();
    }

    ~temporary_object_space()
    {
	current->unref();
    }

    object_space* get_local_space() const { return current; }
};


};  // name_space
#endif