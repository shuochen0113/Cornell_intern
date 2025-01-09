/**
 * \file simplify_siblings.cpp
 *
 * Defines an algorithm to merge nodes or parts of nodes with the same
 * predecessors/successors.
 */

#include "simplify_siblings.hpp"

namespace odgi {
namespace algorithms {

bool simplify_siblings(
    handlegraph::MutablePathDeletableHandleGraph& graph,
    const std::string &progress_message) {

    // Each handle is part of a "family" of handles with the same parents on
    // the left side and the same leading base. We elide the trivial ones. We
    // also only get one family for either a handle or its flipped counterpart,
    // because if both have families they need to be resolved one after the
    // other.
    std::vector<std::vector<handle_t>> families;
    
    // This tracks all the node IDs we've already put in families we are going
    // to merge. This ignores orientation, to ensure that if we're coming into
    // a node on its local right side in one family, we don't come in on its
    // left side in another.
    ska::flat_hash_set<nid_t> in_family;

    std::unique_ptr<algorithms::progress_meter::ProgressMeter> progress;
    const bool show_progress = !progress_message.empty();
    if (show_progress) {
        progress = std::make_unique<algorithms::progress_meter::ProgressMeter>(
            graph.get_node_count(), progress_message + " over nodes");
    }

    graph.for_each_handle([&](const handle_t& local_forward_node) {
        // For each node local forward
        
        for (bool local_orientation : {false, true}) {
            // For it local forward and local reverse
            const handle_t node = local_orientation ? graph.flip(local_forward_node) : local_forward_node;
            
#ifdef debug
            cerr << "Consider " << graph.get_id(node) << (graph.get_is_reverse(node) ? '-' : '+') << endl;
#endif
            
            if (in_family.count(graph.get_id(node))) {
                // If it is in a family in one orientation, don't find a family for it in the other orientation.
                // We can only merge from one end of a node at a time.
#ifdef debug
                cerr << "Node " << graph.get_id(node) << " is already in a family to merge" << endl;
#endif
                
                return;
            }
            // For each handle where it or its RC isn't already in a superfamily, identify its superfamily.
            ska::flat_hash_set<handle_t> superfamily;
            
            // Look left from the node and make a set of the things you see.
            ska::flat_hash_set<handle_t> correct_parents;
            graph.follow_edges(node, true, [&](const handle_t& parent) {
                correct_parents.insert(parent);
#ifdef debug
                cerr << "Parent set includes: " << graph.get_id(parent) << (graph.get_is_reverse(parent) ? '-' : '+') << endl;
#endif
            });
            
            // Keep a set of things that are partial siblings so we don't have to constantly check them
            ska::flat_hash_set<handle_t> partial_siblings;
            for (auto& parent : correct_parents) {
                graph.follow_edges(parent, false, [&](const handle_t& candidate) {
                    // Look right from parents and for each candidate family member
                    
#ifdef debug
                    cerr << "Parent " << graph.get_id(parent) << (graph.get_is_reverse(parent) ? '-' : '+')
                        << " suggests sibling " << graph.get_id(candidate) << (graph.get_is_reverse(candidate) ? '-' : '+') << endl;
#endif
                    
                    if (partial_siblings.count(candidate)) {
                        // Known non-member
#ifdef debug
                        cerr << "\tAlready checked." << endl;
#endif
                        return;
                    }
                    if (superfamily.count(candidate)) {
                        // Known member
#ifdef debug
                        cerr << "\tAlready taken." << endl;
#endif
                        return;
                    }
                    
                    if (in_family.count(graph.get_id(candidate))) {
                        // If it is in a family in one orientation, don't find a family for it in the other orientation.
                        // We can only merge from one end of a node at a time.
#ifdef debug
                        cerr << "\tAlready in a family to merge." << endl;
#endif
                        return;
                    }
                    
                    // Look left from it and see if it has the right parents.
                    size_t seen_parents = 0;
                    bool bad_parent = false;
                    graph.follow_edges(candidate, true, [&](const handle_t& candidate_parent) {
                        if (!correct_parents.count(candidate_parent)) {
                            // We have a parent we shouldn't
                            bad_parent = true;
                            
#ifdef debug
                            cerr << "\tHas unacceptable parent "
                                << graph.get_id(candidate_parent) << (graph.get_is_reverse(candidate_parent) ? '-' : '+') << endl;
#endif
                            
                            return false;
                        } else {
                            // Otherwise we found one of the right ones.
                            seen_parents++;
                            
#ifdef debug
                            cerr << "\tHas OK parent "
                                << graph.get_id(candidate_parent) << (graph.get_is_reverse(candidate_parent) ? '-' : '+') << endl;
#endif
                            
                            return true;
                        }
                    });
                    
#ifdef debug
                    cerr << "\tHas " << seen_parents << "/" << correct_parents.size() << " required parents" << endl;
#endif
                    
                    if (!bad_parent && seen_parents == correct_parents.size()) {
                        // If it has the correct parents, it is a member of the superfamily
                        superfamily.insert(candidate);
                        
#ifdef debug
                        cerr << "\tBelongs in superfamily" << endl;
#endif
                        
                    } else {
                        // Otherwise, it is out, so don't check it again if we find it from another parent.
                        partial_siblings.insert(candidate);
                        
#ifdef debug
                        cerr << "\tOnly a partial sibling" << endl;
#endif
                        
                    }
                });
            }
            
            // Now we have a family. It can't overap with any existing ones.
            
            if (superfamily.size() > 1) {
                // It is nontrivial
                
                // Make sure no node appears multiple times in the superfamily (in opposite orientations).
                // TODO: somehow deal with merging on different ends of the same node
                ska::flat_hash_set<nid_t> seen;
                bool qualified = true;
                for (auto& h : superfamily) {
                    nid_t id = graph.get_id(h);
                    if (seen.count(id)) {
                        // We need to disqualify this superfamily to avoid parallel merging on the same node.
                        qualified = false;

#ifdef debug
                        cerr << "Disqualify superfamily due to duplicate node " << id << endl;
#endif
                        
                        break;
                    }
                   seen.insert(id);
                }
                
                if (!qualified) {
                    // This may contain two nontrivial families for the same node. Skip it.
                    // TODO: Only disqualify actually-conflicting families
                    continue;
                }
                
                // Now we know all the families in the superfamily can exist together.
                
                // Bucket by leading base
                std::map<char, std::vector<handle_t>> by_base;
                for (auto& h : superfamily) {
                    if (graph.get_length(h) == 0) {
                        // Empty nodes probably shouldn't exist, but skip them.
                        
#ifdef debug
                        cerr << "Empty node: " << graph.get_id(h) << (graph.get_is_reverse(h) ? '-' : '+') << endl;
#endif
                        
                        continue;
                    }
                    
                    // Bucket by base into families
                    by_base[graph.get_base(h, 0)].emplace_back(h);
                }
                
#ifdef debug
                cerr << "Found " << by_base.size() << " distinct start bases" << endl;
#endif
                
                for (auto& base_and_family : by_base) {
                    // For each family we found
                    auto& family = base_and_family.second;
                    
                    if (family.size() == 1) {
                        // Ignore the trivial ones
                        continue;
                    }
                    
#ifdef debug
                    cerr << "Nontrivial family of " << family.size() << " nodes starting with " << base_and_family.first << endl;
#endif

                    for (auto& h : family) {
                        // We're going to do this family, so disqualify all the nodes from other families on the other side.
                        in_family.insert(graph.get_id(h));
                        
#ifdef debug
                        cerr << "Ban node " << graph.get_id(h) << " from subsequent families" << endl;
#endif
                        
                    }
                    
                    // Then save the family as a real family to merge on
                    families.push_back(family);
                }
            }
        }
        if (show_progress) {
            progress->increment(1);
        }
    });

    if (show_progress) {
        progress->finish();
    }
    
    in_family.clear();
    
#ifdef debug
    cerr << "Found " << families.size() << " distinct nontrivial families" << endl;
#endif

    if (show_progress) {
        progress.reset();
        progress = std::make_unique<algorithms::progress_meter::ProgressMeter>(
            families.size(), progress_message + " over families");
    }

    // Now we have a bunch of families that won't invalidate each others' handles.
    
    // We set this tro true if we do any work.
    bool made_progress = false;
    
    for (auto& family : families) {
        // Set up the merge
        // everything needs to start at base 0
        std::vector<std::pair<handle_t, size_t>> merge_from;
        merge_from.reserve(family.size());
        merge_from.emplace_back(family.at(0), 0);
    
        // Work out the length of the longest common prefix
        size_t lcp_length = graph.get_length(family.at(0));
        std::string reference_string = graph.get_sequence(family.at(0));
        for (size_t i = 1; i < family.size(); i++) {
            // Create the merge start position
            merge_from.emplace_back(family.at(i), 0);
            
            // See where the first mismatch is, and min that in with the LCP length
            auto other_string = graph.get_sequence(family.at(i));
            auto mismatch_iters = std::mismatch(reference_string.begin(), reference_string.end(), other_string.begin());
            size_t match_length = mismatch_iters.first - reference_string.begin();
            lcp_length = std::min(lcp_length, match_length);
        }
        
        // There should be at least one base of match because we bucketed by base.
        assert(lcp_length >= 1);
        
        // Do the merge. It can only invalidate handles within this family.
        merge(graph, merge_from, lcp_length);
        
        // We did a merge
        made_progress = true;

        if (show_progress) {
            progress->increment(1);
        }
    }

    if (show_progress) {
        progress->finish();
    }

    // To merge everything on the other side of stuff we just merged, we need to start from the top again.
    // So return if we did anything and more might remain (or have been created) to do.
    return made_progress;
}


}
}

