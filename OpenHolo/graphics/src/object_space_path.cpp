#include "graphics/object_space_path.h"

namespace graphics {

//-------------------------------------------------------------------------
// assignment operator
object_space_path& object_space_path::operator = (const object_space_path& rhs)
{
    nodes = rhs.nodes;
    indices = rhs.indices;
    return *this;
}
//________________


//-------------------------------------------------------------------------
// Sets head node (first node in chain)
void    object_space_path::set_head(object_space* node)
{
    resize(0);
    add(node, -1);
}
//________________


//-------------------------------------------------------------------------
// Adds node to end of chain; node is nth child of current last node
void    object_space_path::add(int childIndex)
{
    object_space    *tail;
    tail = nodes[size() - 1];
    add((tail->get_children())[childIndex], childIndex);
}
//________________


//-------------------------------------------------------------------------
// Adds node to end of chain; uses first occurrance of node as
// child of current last node. If path is empty, this is
// equivalent to set_head().
void    object_space_path::add(object_space* childNode)
{
    object_space   *tail;
    int	childIndex;

    if (size() == 0) {
	set_head(childNode);
	return;
    }

    tail = nodes[size() - 1];
    childIndex = tail->get_children().find(0, childNode);

    add(childNode, childIndex);
}
//________________


//-------------------------------------------------------------------------
// Adds all nodes in path to end of chain; head node of fromPath must
// be the same as or a child of current last node
void    object_space_path::add(const object_space_path *fromPath)
{
    if (fromPath->size() == 0)
	return;

    object_space* tailNode = nodes[size() - 1];

    vector<object_space*>& children = tailNode->get_children();

    bool headIsTail = (tailNode == fromPath->get_head());

    int firstIndex;
    if (! headIsTail)
	firstIndex = children.find(0, fromPath->get_head());

    if (! headIsTail)
	add(fromPath->get_head(), firstIndex);
    for (int i = 1; i < fromPath->size();++i)
	add(fromPath->get_node(i), fromPath->get_index(i));
}
//________________



//-------------------------------------------------------------------------
// If the paths have different head nodes, this returns -1.
// Otherwise, it returns the index into the chain of the last node
// (starting at the head) that is the same for both paths.
int	   object_space_path::find_fork(const object_space_path *path) const
{
    if (path->get_head() != get_head())
	return -1;

    int shorterLength = path->size();
    if (size() < shorterLength)
	shorterLength = size();

    for (int i = 1; i < shorterLength;++i)
	if (get_index(i) != path->get_index(i))
	    return i - 1;

    return shorterLength - 1;
}
//________________


//-------------------------------------------------------------------------
// This is called when a node in the path chain has a child added.
// The passed index is the index of the new child
void    object_space_path::insert_index(object_space* parent, int newIndex)
{
    int	i;
    for (i = 0; i < size();++i)
	if (nodes[i] == parent)
	    break;

    if (++i< size() && indices[i] >= newIndex)
	++indices[i];
}
//________________


//-------------------------------------------------------------------------
// This is called when a node in the path chain has a child removed.
// The passed index is the index of the child to be removed
void    object_space_path::remove_index(object_space* parent, int oldIndex)
{
    int	i;

    for (i = 0; i < size();++i)
	if (nodes[i] == parent)
	    break;

    if (++i < size()) {
	if (indices[i] == oldIndex)
	    resize(i);

	else if (indices[i] > oldIndex)
	    --indices[i];
    }
}
//________________


//-------------------------------------------------------------------------
// This is called when a node in the path chain replaces a child.
// The passed index is the index of the child to be removed
void    object_space_path::replace_index(object_space* parent, int index, object_space* newChild)
{
    int	i;

    for (i = 0; i < size();++i)
	if (nodes[i] == parent)
	    break;

    if (++i< size() && indices[i] == index) {
	resize(i);
	add(newChild, index);
    }
}
//________________




bool operator == (const object_space_path& p1, const object_space_path& p2)
{
    if (p1.size() != p2.size())
	return false;

    for (int i = p1.size() - 1; i >= 0; --i)
	if (p1.get_node(i) != p2.get_node(i) || p1.get_index(i) != p2.get_index(i))
	    return false;

    return true;
}

}; // nameobject_space