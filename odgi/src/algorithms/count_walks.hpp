#ifndef VG_ALGORITHMS_COUNT_WALKS_HPP_INCLUDED
#define VG_ALGORITHMS_COUNT_WALKS_HPP_INCLUDED

/**
 * \file count_walks.hpp
 *
 * Defines algorithm for counting the number of distinct walks through a DAG.
 */

#include <handlegraph/handle_graph.hpp>
#include "topological_sort.hpp"

#include <unordered_map>
#include <vector>

namespace vg {
namespace algorithms {

using namespace std;
using namespace handlegraph;

    /// Returns the number of source-to-sink walks through the graph. Assumes that
    /// the graph is a single-stranded DAG. Consider checking these properties with
    /// algorithms::is_single_stranded and algorithms::is_directed_acyclic for safety.
    /// Returns numeric_limits<size_t>::max() if the actual number of walks is larger
    /// than this.
    size_t count_walks(const HandleGraph* graph);

}
}

#endif
