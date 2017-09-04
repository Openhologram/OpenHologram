#include "graphics/slic_hierarchy.h"
#include "graphics/SLIC.h"

namespace graphics {


void slic_hierarchy::build(unsigned int* img, int w, int h, int l0, int l1, int l2, int l3, int compact)
{

	std::vector<int> levels(4);
	levels[0] = l0;
	levels[1] = l1;
	levels[2] = l2;
	levels[3] = l3;
	std::vector<int> num_labels(4);
	int i;
	hierarchy_.resize(4);
	int sz = w*h;

	hierarchy_[0].labels = new int[sz];
	hierarchy_[1].labels = new int[sz];
	hierarchy_[2].labels = new int[sz];
	hierarchy_[3].labels = new int[sz];


#pragma omp parallel for private(i)
	for (i = 0; i < 4; ++i) {
		SLIC slic;
		slic.DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(img, w, h, hierarchy_[i].labels, num_labels[i], levels[i], compact);
		hierarchy_[i].segments.resize(num_labels[i]);
	}

#pragma omp parallel for private(i)
	for (i = 0; i < 4; ++i) {
		int* labels = hierarchy_[i].labels;
		for (int x = 0; x < w; x++) {
			for (int y = 0; y < h; y++) {
				int k = labels[x + y*w];
				hierarchy_[i].segments[k].box.extend(vec2(x, y));
			}
		}
	}

#pragma omp parallel for private(i)
	for (i = 0; i < 3; ++i) {
		int* labels = hierarchy_[i].labels;
		int* up_labels = hierarchy_[i + 1].labels;
		for (int k = 0; k < hierarchy_[i].segments.size(); k++) {
			for (int x = (int)hierarchy_[i].segments[k].box.minimum[0]; x < (int)hierarchy_[i].segments[k].box.maximum[0]; x++) {
				for (int y = (int)hierarchy_[i].segments[k].box.minimum[1]; y < (int)hierarchy_[i].segments[k].box.maximum[1]; y++) {
					int m = labels[x + y*w];
					if (m == k) {
						int parent = up_labels[x + y*w];
						hierarchy_[i].segments[k].parents.insert(parent);
					}
				}
			}
		}
	}
}


void slic_hierarchy::build(unsigned int* img, int w, int h, int l0, int l1, int l2, int compact)
{

	std::vector<int> levels(3);
	levels[0] = l0;
	levels[1] = l1;
	levels[2] = l2;

	std::vector<int> num_labels(3);
	int i;
	hierarchy_.resize(3);
	int sz = w*h;

	hierarchy_[0].labels = new int[sz];
	hierarchy_[1].labels = new int[sz];
	hierarchy_[2].labels = new int[sz];


#pragma omp parallel for private(i)
	for (i = 0; i < 3; ++i) {
		SLIC slic;
		slic.DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(img, w, h, hierarchy_[i].labels, num_labels[i], levels[i], compact);
		hierarchy_[i].segments.resize(num_labels[i]);
	}

#pragma omp parallel for private(i)
	for (i = 0; i < 3; ++i) {
		int* labels = hierarchy_[i].labels;
		for (int x = 0; x < w; x++) {
			for (int y = 0; y < h; y++) {
				int k = labels[x + y*w];
				hierarchy_[i].segments[k].box.extend(vec2(x, y));
			}
		}
	}

#pragma omp parallel for private(i)
	for (i = 0; i < 2; ++i) {
		int* labels = hierarchy_[i].labels;
		int* up_labels = hierarchy_[i + 1].labels;
		for (int k = 0; k < hierarchy_[i].segments.size(); k++) {
			for (int x = (int)hierarchy_[i].segments[k].box.minimum[0]; x < (int)hierarchy_[i].segments[k].box.maximum[0]; x++) {
				for (int y = (int)hierarchy_[i].segments[k].box.minimum[1]; y < (int)hierarchy_[i].segments[k].box.maximum[1]; y++) {
					int m = labels[x + y*w];
					if (m == k) {
						int parent = up_labels[x + y*w];
						hierarchy_[i].segments[k].parents.insert(parent);
					}
				}
			}
		}
	}
}
}