// object_space.cpp: implementation of the object_space class.
//
//////////////////////////////////////////////////////////////////////

#include "graphics/object_space.h"
#include "graphics/object_space_path.h"

namespace graphics {

object_space object_space::root_space = object_space(object_space_root);

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

object_space::object_space(object_space_type val, void* ptr)
    : m_local_space(),
    m_ref_cnt(0),
    m_children(),
    m_parent(0),
    m_is_root(val),
    m_reserved(ptr)
{
}

object_space::~object_space()
{
    del_children();
}

const	frame& object_space::get_local_space() const
{
    return m_local_space;
}

frame&	object_space::get_local_space() 
{
    return m_local_space;
}

void	object_space::set_local_space(const frame& val)
{
    m_local_space = val;
}

void	object_space::add_child(object_space *ch)
{
    m_children.add(ch);
    if (ch) ch->set_parent(this);
}

object_space*	object_space::make_child_instance()
{
    object_space* a_child = new object_space;
    add_child(a_child);
    return a_child;
}

void	object_space::destroy()
{
    if (is_root()) return;
    get_parent()->del_child(this);
}

void	object_space::del_child(object_space* ch)
{
    int idx = m_children.find(0, ch);

    if (idx != -1) {
	delete m_children[idx];
	m_children.del(idx);
    }
}

void	object_space::del_children()
{
    for (int i = 0 ; i < m_children.size() ;++i){
	if (m_children[i]->get_children().size() == 0) {
	    if (m_children[i]->m_ref_cnt == 0) 
		delete m_children[i];
	}
	else 
	    m_children[i]->del_children();
    }
}

void	object_space::set_parent(object_space *pr)
{
    m_parent = pr;
}

vector<object_space*>& object_space::get_children()
{
    return m_children;
}

const	vector<object_space*>& object_space::get_children() const
{
    return m_children;
}


object_space*	object_space::get_parent() const
{
    return m_parent;
}

bool	object_space::is_root() const
{
    if (m_is_root == object_space_root)
	return true;
    
    return false;
}


object_space_path object_space::get_path() const
{
    object_space* current = const_cast<object_space*>(this);
    vector<object_space*> temp_path;
    temp_path.add(current);

    while (!current->is_root()) {
	current = current->get_parent();
	if (current == 0) break;
	temp_path.add(current);
    }

    object_space_path ret;
    for (int i = temp_path.size()-1 ; i >= 0 ; i--) {
	ret.add(temp_path[i]);
    }

    return ret;
}


vec3	object_space::to_world(const vec3& pnt) const
{
    object_space_path mat = get_path();
    vec3 point = pnt;
    
    for (int i = mat.size()-1 ; i >= 0 ; i--) {
	object_space* node = mat.get_node(i);
	if (!node->is_root())
	    point = node->get_local_space().to_world(point);
    }
    return point;
}

vec3	object_space::to_world_normal(const vec3& pnt) const
{
    object_space_path mat = get_path();
    vec3 point = pnt;
    
    for (int i = mat.size()-1 ; i >= 0 ; i--) {
	object_space* node = mat.get_node(i);
	if (!node->is_root())
	    point = node->get_local_space().to_world_normal(point);
    }
    return point;
}

vec3	object_space::to_model(const vec3& pnt) const
{
    object_space_path mat = get_path();
    vec3 point = pnt;
    
    for (int i = 0 ; i < mat.size() ;++i){
	object_space* node = mat.get_node(i);
	if (!node->is_root())
	    point = node->get_local_space().to_model(point);
    }
    return point;
}

vec3	object_space::to_model_normal(const vec3& pnt) const
{
    object_space_path mat = get_path();
    vec3 point = pnt;
    
    for (int i = 0 ; i < mat.size() ;++i){
	object_space* node = mat.get_node(i);
	if (!node->is_root())
	    point = node->get_local_space().to_model_normal(point);
    }
    return point;
}

line	object_space::to_model(const line& a) const
{
    vec3 m_pos = to_model(a.get_position());
    vec3 m_end = to_model(a.get_position()+ a.get_direction());
    line new_l(m_pos, m_end);

    return new_l;
}

line	object_space::to_world(const line& a) const
{
    vec3 m_pos = to_world(a.get_position());
    vec3 m_end = to_world(a.get_position()+ a.get_direction());
    line new_l(m_pos, m_end);

    return new_l;
}
void	object_space::push_to_world() const
{
    object_space_path mat = get_path();

    for (int i = mat.size()-1 ; i >= 0 ; i--) {
	object_space* node = mat.get_node(i);
	if (!node->is_root())
	    node->get_local_space().push_to_world();
    }
}

void	object_space::push_to_model() const
{
    object_space_path mat = get_path();
    
    for (int i = 0 ; i < mat.size() ;++i){
	object_space* node = mat.get_node(i);
	if (!node->is_root())
	    node->get_local_space().push_to_model();
    }
}

void	object_space::pop() const
{
    object_space_path mat = get_path();
    
    for (int i = 0 ; i < mat.size() - 1 ;++i){
	glPopMatrix();
    }
}

};  // nameobject_space