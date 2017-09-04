#ifndef  __object_space_path_h
#define  __object_space_path_h

#include "graphics/vector.h"
#include "graphics/object_space.h"

namespace graphics {

class object_space_path;

class object_space_path {

public:

    //-------------------------------------------------------------------------
    // Default constructor
    object_space_path() : nodes(), indices() {}
    //________________


    //-------------------------------------------------------------------------
    // Constructor given head node to insert in object_space_path
    object_space_path(object_space* node) : nodes(), indices() { set_head(node); }
    //________________


    //-------------------------------------------------------------------------
    // Copy Constructor
    object_space_path(const object_space_path& rhs) : nodes(rhs.nodes), indices(rhs.indices) {}
    //________________


    virtual ~object_space_path() { resize(0); }

    //-------------------------------------------------------------------------
    // assignment
    object_space_path& operator = (const object_space_path& rhs);
    //________________


    //-------------------------------------------------------------------------
    // Sets head node (first node in chain)
    void    set_head(object_space* node);
    //________________


    //-------------------------------------------------------------------------
    // Adds node to end of chain; node is nth child of current last node
    void    add(int childIndex);
    //________________


    //-------------------------------------------------------------------------
    // Adds node to end of chain; uses first occurrance of node as
    // child of current last node. If path is empty, this is
    // equivalent to set_head().
    void    add(object_space* childNode);
    //________________


    //-------------------------------------------------------------------------
    // Adds all nodes in path to end of chain; head node of fromPath must
    // be the same as or a child of current last node
    void    add(const object_space_path *fromPath);
    //________________


    //-------------------------------------------------------------------------
    // Allows path to be treated as a stack: push a node at the end of
    // the chain and pop the last node off
    void    push(int childIndex) { add(childIndex); }
    void    pop() { resize(size() - 1); }
    //________________


    //-------------------------------------------------------------------------
    // Returns the first/last node in a path chain.
    object_space*   get_head() const	{ return nodes[0]; }
    object_space*   get_tail() const	{ return (nodes[size() - 1]); }
    //________________


    //-------------------------------------------------------------------------
    // Returns pointer to ith node in chain
    object_space*   get_node(int i) const { return (nodes[i]); }
    //________________


    //-------------------------------------------------------------------------
    // Returns index of ith node in chain
    int	    get_index(int i) const { return (indices[i]); }
    //________________


    //-------------------------------------------------------------------------
    // Removes all nodes from indexed node on
    void    resize(int start)
    {
	nodes.resize(start);
	indices.resize(start);
    }
    //________________
    

    //-------------------------------------------------------------------------
    // If the paths have different head nodes, this returns -1.
    // Otherwise, it returns the index into the chain of the last node
    // (starting at the head) that is the same for both paths.
    int	    find_fork(const object_space_path *path) const;
    //________________


    //-------------------------------------------------------------------------
    // This is called when a node in the path chain has a child added.
    // The passed index is the index of the new child
    void    insert_index(object_space* parent, int newIndex);
    //________________


    //-------------------------------------------------------------------------
    // This is called when a node in the path chain has a child removed.
    // The passed index is the index of the child to be removed
    void    remove_index(object_space* parent, int oldIndex);
    //________________


    //-------------------------------------------------------------------------
    // This is called when a node in the path chain replaces a child.
    // The passed index is the index of the child to be removed
    void    replace_index(object_space* parent, int index, object_space* newChild);
    //________________


    //-------------------------------------------------------------------------
    // Returns real length of path, including hidden children
    int	size() const  { return nodes.size(); }
    //________________



private:

    vector<object_space*>   nodes;	    // Pointers to nodes
    vector<int>	     indices;	    // Child indices

    void    add(object_space* node, int index)
    {
	// Append to lists
	nodes.add(node);
	indices.add(index);
    }

};

bool operator == (const object_space_path& p1, const object_space_path& p2);

}; // namespace graphics
#endif