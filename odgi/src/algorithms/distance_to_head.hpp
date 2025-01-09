#ifndef VG_ALGORITHMS_DISTANCE_TO_HEAD_HPP_INCLUDED
#define VG_ALGORITHMS_DISTANCE_TO_HEAD_HPP_INCLUDED

#include <unordered_map>

#include "position.hpp"
#include "cached_position.hpp"
#include "hash_map.hpp"
#include <handlegraph/handle_graph.hpp>

namespace vg {
namespace algorithms {

using namespace std;
using namespace handlegraph;

/// Find all of the nodes with no edges on their left sides.
vector<handle_t> head_nodes(const HandleGraph* g);
int32_t distance_to_head(handle_t h, int32_t limit, const HandleGraph* graph);
/// Get the distance in bases from start of node to start of closest head node of graph, or -1 if that distance exceeds the limit.
/// dist increases by the number of bases of each previous node until you reach the head node
/// seen is a set that holds the nodes that you have already gotten the distance of, but starts off empty
int32_t distance_to_head(handle_t h, int32_t limit, int32_t dist, unordered_set<handle_t>& seen, const HandleGraph* graph);
                                                      
}
}

#endif
