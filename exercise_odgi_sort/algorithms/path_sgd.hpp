#pragma once

#include <vector>
#include <string>
#include <cassert>
#include <algorithm>
#include <random>
#include <set>
#include <thread>
#include <atomic>
#include <handlegraph/path_handle_graph.hpp>
#include <handlegraph/handle_graph.hpp>
#include "xp.hpp"
#include "sgd_term.hpp"
#include "IITree.h"
#include <zipfian_int_distribution.h>
#include <iomanip>
#include <string>
#include "weakly_connected_components.hpp"
#include <sdsl/bit_vectors.hpp>
#include "dirty_zipfian_int_distribution.h"
#include "XoshiroCpp.hpp"
#include "progress.hpp"
#include "utils.hpp"

#include <fstream>

#ifdef USE_GPU
#include "cuda/sort.h"  // Our new GPU 1D path-guided SGD interface
#endif

namespace odgi {
namespace algorithms {

using namespace handlegraph;

struct handle_layout_t {
    uint64_t weak_component = 0;
    double pos = 0;
    handle_t handle = as_handle(0);
};

/// use SGD driven, by path guided, and partly zipfian distribution sampled pairwise distances to obtain a 1D linear layout of the graph that respects its topology
std::vector<double> path_linear_sgd(const graph_t &graph,
                                    const xp::XP &path_index,
                                    const std::vector<path_handle_t>& path_sgd_use_paths,
                                    const uint64_t &iter_max,
                                    const uint64_t &iter_with_max_learning_rate,
                                    const uint64_t &min_term_updates,
                                    const double &delta,
                                    const double &eps,
                                    const double &eta_max,
                                    const double &theta,
                                    const uint64_t &space,
                                    const uint64_t &space_max,
                                    const uint64_t &space_quantization_step,
                                    const double &cooling_start,
                                    const uint64_t &nthreads,
                                    const bool &progress,
                                    const bool &snapshot,
                                    std::vector<std::string> &snapshots);

#ifdef USE_GPU
/**
 * \brief Runs path-linear SGD on the GPU.  
 * \param graph The ODGI graph_t
 * \param path_index The XP path index
 * \param path_sgd_use_paths The subset of paths used by the SGD
 * \param iter_max Maximum iterations
 * \param iter_with_max_learning_rate The iteration at which the learning rate is highest
 * \param min_term_updates The # of pairwise distance updates per iteration
 * \param delta Early-termination threshold
 * \param eps Epsilon used in the learning rate schedule
 * \param eta_max Maximum learning rate
 * \param theta Zipf exponent
 * \param space Maximum jump along the path for local sampling
 * \param space_max A cutoff for space
 * \param space_quantization_step The step for coarse spacing beyond space_max
 * \param cooling_start A fraction of iter_max after which we do "cooling"
 * \param nthreads CPU threads for path/handle data extraction
 * \param progress If true, prints progress
 * \param snapshot If true, periodically captures snapshots
 * \param snapshots A vector for storing snapshot file paths
 * \param target_sorting If true, allows ignoring certain nodes in the updates
 * \param target_nodes A boolean vector (size = #nodes) to skip updates for those nodes
 * \return A vector of final X-coordinates (1D) for each node (ID-1 order).
 */
std::vector<double> path_linear_sgd_gpu(const graph_t &graph,
                                        const xp::XP &path_index,
                                        const std::vector<path_handle_t> &path_sgd_use_paths,
                                        const uint64_t &iter_max,
                                        const uint64_t &iter_with_max_learning_rate,
                                        const uint64_t &min_term_updates,
                                        const double &delta,
                                        const double &eps,
                                        const double &eta_max,
                                        const double &theta,
                                        const uint64_t &space,
                                        const uint64_t &space_max,
                                        const uint64_t &space_quantization_step,
                                        const double &cooling_start,
                                        const uint64_t &nthreads,
                                        const bool &progress,
                                        const bool &snapshot,
                                        std::vector<std::string> &snapshots,
                                        const bool &target_sorting,
                                        std::vector<bool> &target_nodes);
#endif

/// our learning schedule
std::vector<double> path_linear_sgd_schedule(const double &w_min,
                                             const double &w_max,
                                             const uint64_t &iter_max,
                                             const uint64_t &iter_with_max_learning_rate,
                                             const double &eps);

std::vector<handle_t> path_linear_sgd_order(const graph_t &graph,
                                            const xp::XP &path_index,
                                            const std::vector<path_handle_t>& path_sgd_use_paths,
                                            const uint64_t &iter_max,
                                            const uint64_t &iter_with_max_learning_rate,
                                            const uint64_t &min_term_updates,
                                            const double &delta,
                                            const double &eps,
                                            const double &eta_max,
                                            const double &theta,
                                            const uint64_t &space,
                                            const uint64_t &space_max,
                                            const uint64_t &space_quantization_step,
                                            const double &cooling_start,
                                            const uint64_t &nthreads,
                                            const bool &progress,
                                            const std::string &seed,
                                            const bool &snapshot,
                                            const std::string &snapshot_prefix,
                                            const bool &write_layout,
                                            const std::string &layout_out,
											const bool &target_sorting,
											std::vector<bool>& target_nodes);

}

}
