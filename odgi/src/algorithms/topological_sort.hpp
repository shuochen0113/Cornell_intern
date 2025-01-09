#pragma once

/**
 * \file topological_sort.hpp
 *
 * Defines a topological sort algorithm for handle graphs.
 */

//#include <unordered_map>
//#include <set>
#include <map>
#include <iostream>
#include "hash_map.hpp"
#include <handlegraph/handle_graph.hpp>
#include <handlegraph/util.hpp>
#include "dynamic.hpp"
#include "apply_bulk_modifications.hpp"
#include "is_single_stranded.hpp"
#include "bfs.hpp"
#include "dfs.hpp"
#include "progress.hpp"

namespace odgi {
namespace algorithms {

using namespace handlegraph;

/// Find all of the nodes with no edges on their left sides.
std::vector<handle_t> head_nodes(const HandleGraph* g);

/// Find all of the nodes with no edges on their right sides.
std::vector<handle_t> tail_nodes(const HandleGraph* g);

/**
 * Order and orient the nodes in the graph using a topological sort. The sort is
 * guaranteed to be machine-independent given the initial graph's node and edge
 * ordering. The algorithm is well-defined on non-DAG graphs, but the order is
 * necessarily not a topological order.
 * 
 * We use a bidirected adaptation of Kahn's topological sort (1962), which can handle components with no heads or tails.
 * 
 * L ← Empty list that will contain the sorted and oriented elements
 * S ← Set of nodes which have been oriented, but which have not had their downstream edges examined
 * N ← Set of all nodes that have not yet been put into S
 * 
 * while N is nonempty do
 *     remove a node from N, orient it arbitrarily, and add it to S
 *         (In practice, we use "seeds": the heads all in a batch at the start, and any
 *          nodes we have seen that had too many incoming edges)
 *     while S is non-empty do
 *         remove an oriented node n from S
 *         add n to tail of L
 *         for each node m with an edge e from n to m do
 *             remove edge e from the graph
 *             if m has no other edges to that side then
 *                 orient m such that the side the edge comes to is first
 *                 remove m from N
 *                 insert m into S
 *             otherwise
 *                 put an oriented m on the list of arbitrary places to start when S is empty
 *                     (This helps start at natural entry points to cycles)
 *     return L (a topologically sorted order and orientation)
 */
std::vector<handle_t> topological_order(const HandleGraph* g,
                                        bool use_heads = true,
                                        bool use_tails = false,
                                        bool progress_reporting = false);

std::vector<handle_t> two_way_topological_order(const HandleGraph* g);

/**
 * Order the nodes in a graph using a topological sort. The sort is NOT guaranteed
 * to be machine-independent, but it is faster than topological_order(). This algorithm 
 * is invalid in a graph that has any cycles. For safety, consider this property with
 * algorithms::is_directed_acyclic().
 */
std::vector<handle_t> lazy_topological_order(const HandleGraph* g);
    
/**
 * Order the nodes in a graph using a topological sort. Similar to lazy_topological_order
 * but somewhat faster. The algorithm is invalid in a graph that has any cycles or
 * any reversing edges. For safety, consider these properties with algorithms::is_acyclic()
 * and algorithms::is_single_stranded().
 */
std::vector<handle_t> lazier_topological_order(const HandleGraph* g);

void topological_sort(MutableHandleGraph& g, bool compact_ids);

std::vector<handle_t> breadth_first_topological_order(const HandleGraph& g, const uint64_t& chunk_size,
                                                      bool use_heads = true, bool use_tails = false);

std::vector<handle_t> depth_first_topological_order(const HandleGraph& g, const uint64_t& chunk_size,
                                                    bool use_heads = false, bool use_tails = false);

}
}
