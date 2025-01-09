#pragma once

/** \file
 * subgraph.hpp: defines a handle graph implementation of a subgraph
 */

#include "hash_map.hpp"
#include <handlegraph/handle_graph.hpp>
#include <handlegraph/util.hpp>
#include <string>
#include <iostream>

namespace odgi {

using namespace handlegraph;

    /**
     * A HandleGraph implementation that acts as a subgraph of some other HandleGraph
     * using a layer of indirection. Only subsets based on nodes; all edges between
     * the nodes in the super graph are considered part of the subgraph. Subgraph
     * handles can also be used by the super graph.
     */
    class SubHandleGraph : public HandleGraph {
    public:
        
        /// Initialize with a super graph and nodes returned by iterators to handles
        /// from the super graph
        template<typename HandleIter>
        SubHandleGraph(const HandleGraph* super, HandleIter begin, HandleIter end);
        
        /// Initialize as empty subgraph of a super graph
        SubHandleGraph(const HandleGraph* super);
        
        /// Add a node from the super graph to the subgraph. Must be a handle to the
        /// super graph. No effect if the node is already included in the subgraph.
        /// Generally invalidates the results of any previous algorithms.
        void add_handle(const handle_t& handle);
        
        //////////////////////////
        /// HandleGraph interface
        //////////////////////////
        
        // Method to check if a node exists by ID
        virtual bool has_node(nid_t node_id) const;
        
        /// Look up the handle for the node with the given ID in the given orientation
        virtual handle_t get_handle(const nid_t& node_id, bool is_reverse = false) const;
        
        /// Get the ID from a handle
        virtual nid_t get_id(const handle_t& handle) const;
        
        /// Get the orientation of a handle
        virtual bool get_is_reverse(const handle_t& handle) const;
        
        /// Invert the orientation of a handle (potentially without getting its ID)
        virtual handle_t flip(const handle_t& handle) const;
        
        /// Get the length of a node
        virtual size_t get_length(const handle_t& handle) const;
        
        /// Get the sequence of a node, presented in the handle's local forward
        /// orientation.
        virtual std::string get_sequence(const handle_t& handle) const;
        
        /// Loop over all the handles to next/previous (right/left) nodes. Passes
        /// them to a callback which returns false to stop iterating and true to
        /// continue. Returns true if we finished and false if we stopped early.
        virtual bool follow_edges_impl(const handle_t& handle, bool go_left, const std::function<bool(const handle_t&)>& iteratee) const;
        
        /// Loop over all the nodes in the graph in their local forward
        /// orientations, in their internal stored order. Stop if the iteratee
        /// returns false. Can be told to run in parallel, in which case stopping
        /// after a false return value is on a best-effort basis and iteration
        /// order is not defined.
        virtual bool for_each_handle_impl(const std::function<bool(const handle_t&)>& iteratee, bool parallel = false) const;
        
        /// Return the number of nodes in the graph
        /// TODO: can't be node_count because XG has a field named node_count.
        virtual size_t get_node_count() const;
        
        /// Return the smallest ID in the graph, or some smaller number if the
        /// smallest ID is unavailable. Return value is unspecified if the graph is empty.
        virtual nid_t min_node_id() const;
        
        /// Return the largest ID in the graph, or some larger number if the
        /// largest ID is unavailable. Return value is unspecified if the graph is empty.
        virtual nid_t max_node_id() const;
        
    private:
        const HandleGraph* super = nullptr;
        ska::flat_hash_set<nid_t> contents;
        // keep track of these separately rather than use an ordered set
        nid_t min_id = std::numeric_limits<nid_t>::max();
        nid_t max_id = std::numeric_limits<nid_t>::min();
        
    };

    
    // Template constructor
    template<typename HandleIter>
    SubHandleGraph::SubHandleGraph(const HandleGraph* super, HandleIter begin, HandleIter end) : super(super) {
        for (auto iter = begin; iter != end; ++iter) {
            add_handle(*iter);
        }
    }
}
