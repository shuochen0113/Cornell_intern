#ifndef VG_ALGORITHMS_ID_SORT_HPP_INCLUDED
#define VG_ALGORITHMS_ID_SORT_HPP_INCLUDED

/**
 * \file id_sort.hpp
 *
 * Defines a by-ID sort algorithm for handle graphs.
 */

#include <unordered_map>

#include <handlegraph/handle_graph.hpp>
#include <handlegraph/mutable_handle_graph.hpp>


namespace vg {
namespace algorithms {

using namespace std;
using namespace handlegraph;


/**
 * Order all the handles in the graph in ID order. All orientations are forward.
 */
vector<handle_t> id_order(const HandleGraph* g);

/**
 * Sort the given handle graph by ID, and then apply that sort to re-order the
 * nodes of the graph.
 */
void id_sort(MutableHandleGraph* g);
                                                      
}
}

#endif
