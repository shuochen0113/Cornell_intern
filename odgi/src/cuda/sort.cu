#include "sort.h"
#include <cuda.h>
#include <assert.h>
#include "cuda_runtime_api.h"

// Uncomment this if you want to enable debug output from this file,
// or define it via the CMakeLists or compiler flags:
#define debug_CUDA

#ifdef debug_CUDA
#include <cstdio> // for printf
#include <iostream>
#endif

#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

namespace cuda {

/**
 * \brief store curandState_t in a coalesced manner for each Streaming Multiprocesser block
 */
__global__ 
void cuda_device_init_sort(curandState_t *rnd_state_tmp, curandStateCoalesced_t *rnd_state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Standard XORWOW initialization
    curand_init(42 + tid, tid, 0, &rnd_state_tmp[tid]);

    // Move to the coalesced structure
    rnd_state[blockIdx.x].d[threadIdx.x]   = rnd_state_tmp[tid].d;
    rnd_state[blockIdx.x].w0[threadIdx.x]  = rnd_state_tmp[tid].v[0];
    rnd_state[blockIdx.x].w1[threadIdx.x]  = rnd_state_tmp[tid].v[1];
    rnd_state[blockIdx.x].w2[threadIdx.x]  = rnd_state_tmp[tid].v[2];
    rnd_state[blockIdx.x].w3[threadIdx.x]  = rnd_state_tmp[tid].v[3];
    rnd_state[blockIdx.x].w4[threadIdx.x]  = rnd_state_tmp[tid].v[4];

#ifdef debug_CUDA
    // Debug print from the device init, only once (first block, first thread)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("[cuda_device_init_sort] First thread -> tid=%d. Initializing random states...\n", tid);
    }
#endif
}

/**
 * \brief Pseudorandom 32-bit generator from our coalesced XORWOW state.
 */
__device__ 
unsigned int curand_coalesced(curandStateCoalesced_t *state, int lane_id) {
    uint32_t t = (state->w0[lane_id] ^ (state->w0[lane_id] >> 2));
    state->w0[lane_id] = state->w1[lane_id];
    state->w1[lane_id] = state->w2[lane_id];
    state->w2[lane_id] = state->w3[lane_id];
    state->w3[lane_id] = state->w4[lane_id];
    state->w4[lane_id] = (state->w4[lane_id] ^ (state->w4[lane_id] << 4)) ^ (t ^ (t << 1));
    state->d[lane_id] += 362437;
    return state->w4[lane_id] + state->d[lane_id];
}

/**
 * \brief Returns a uniform float in (0,1].
 */
__device__ 
float curand_uniform_coalesced(curandStateCoalesced_t *state, int lane_id) {
    uint32_t r = curand_coalesced(state, lane_id);
    return _curand_uniform(r);
}

/**
 * \brief A minimal Zipf sampler in CUDA for 1D usage (uses fast power function __powf).
 */
__device__ 
uint32_t cuda_rnd_zipf(curandStateCoalesced_t *rnd_state, uint32_t n, double theta,
                       double zeta2, double zetan, int lane_id) 
{
    double alpha = 1.0 / (1.0 - theta);
    double denom = 1.0 - zeta2 / zetan;
    if (fabs(denom) < 1e-14) {
        denom = 1e-14;
    }
    double eta = (1.0 - __powf(2.0f / float(n), 1.0f - float(theta))) / denom;
    double u  = 1.0 - (double)curand_uniform_coalesced(rnd_state, lane_id);
    double uz = u * zetan;
    int64_t val = 0;
    if (uz < 1.0) {
        val = 1;
    } 
    else if (uz < 1.0 + __powf(0.5f, float(theta))) {
        val = 2;
    } 
    else {
        val = 1 + int64_t(double(n) * __powf(eta * u - eta + 1.0, alpha));
    }
    if (val > n) {
        val = n;
    }
    return uint32_t(val);
}

/**
 * \brief Helper for the 1D update of two node positions.
 */
__device__ 
void update_pos_gpu_1d(double pos1, double pos2, float *x1, float *x2, double eta) {
    double term_dist = fabs(pos1 - pos2);
    if (term_dist < 1e-9) {
        return; // effectively 0 distance in path space
    }
    double w_ij = 1.0 / term_dist;
    double mu = eta * w_ij;
    if (mu > 1.0) mu = 1.0;

    float x1_val = *x1;
    float x2_val = *x2;
    double dx = double(x1_val) - double(x2_val);
    if (fabs(dx) < 1e-9) {
        dx = 1e-9; // avoid zero
    }
    double mag   = fabs(dx);
    double delta = mu * (mag - term_dist) / 2.0;
    double r   = delta / mag;
    double r_x = r * dx;

    // Try to control r_x
    // if (fabs(r_x) > 1e35) {
    //     r_x = (r_x > 0 ? 1e35 : -1e35);
    // }

    // Potential debug message
    // #ifdef debug_CUDA
    //   printf("[update_pos_gpu_1d] pos1=%.3f pos2=%.3f x1=%.3f x2=%.3f dx=%.3f r_x=%.3f mu=%.3f\n",
    //          pos1, pos2, float(x1_val), float(x2_val), dx, r_x, mu);
    // #endif

    // Atomic updates in float precision
    atomicAdd(x1, float(-r_x));
    atomicAdd(x2, float(r_x));
}

/**
 * \brief Per-iteration kernel for path-based 1D SGD.
 */
__global__
void gpu_sort_kernel(int iter,
                     sort_config_t config,
                     curandStateCoalesced_t *rnd_state,
                     double eta,
                     double *zetas,
                     node_data_t node_data,
                     path_data_t path_data,
                     int sm_count) 
{
    uint32_t tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x;

    // figure out which SM we are on
    int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    if (smid >= sm_count) return;
    curandStateCoalesced_t *thread_rnd_state = &rnd_state[smid];

    bool in_cooling = (iter >= config.first_cooling_iteration);

    // pick a random step index overall
    uint32_t step_idx = curand_coalesced(thread_rnd_state, lane) % (path_data.total_path_steps);

    // find which path that belongs to
    uint32_t path_i = path_data.element_array[step_idx].pidx;
    path_t p = path_data.paths[path_i];

    if (p.step_count < 2) {
        return;
    }

    // pick step s1
    uint32_t s1_idx = curand_coalesced(thread_rnd_state, lane) % p.step_count;
    uint32_t s2_idx = s1_idx;  // initialize

    // if in cooling, pick near s1 using a Zipf-based jump
    // else pick randomly
    if (in_cooling) {
        bool coin_flip = (curand_coalesced(thread_rnd_state, lane) & 1) == 0;
        if ((s1_idx > 0 && coin_flip) || s1_idx == p.step_count - 1) {
            // go backward
            uint32_t jump = min(config.space, s1_idx);
            uint32_t space_idx = jump;
            if (jump > config.space_max) {
                space_idx = config.space_max 
                            + (jump - config.space_max) / config.space_quantization_step
                            + 1;
            }
            uint32_t z_i = cuda_rnd_zipf(thread_rnd_state,
                                         jump, config.theta, zetas[2],
                                         zetas[space_idx], lane);
            if (z_i > s1_idx) z_i = s1_idx;
            s2_idx = s1_idx - z_i;
        } else {
            // go forward
            uint32_t jump = min(config.space, p.step_count - s1_idx - 1);
            uint32_t space_idx = jump;
            if (jump > config.space_max) {
                space_idx = config.space_max 
                            + (jump - config.space_max) / config.space_quantization_step
                            + 1;
            }
            uint32_t z_i = cuda_rnd_zipf(thread_rnd_state,
                                         jump, config.theta, zetas[2],
                                         zetas[space_idx], lane);
            if (z_i > (p.step_count - s1_idx - 1)) {
                z_i = (p.step_count - s1_idx - 1);
            }
            s2_idx = s1_idx + z_i;
        }
    } 
    else {
        // pick randomly, ensuring s2_idx != s1_idx
        while (s2_idx == s1_idx) {
            s2_idx = curand_coalesced(thread_rnd_state, lane) % p.step_count;
        }
    }

    path_element_t pe1 = p.elements[s1_idx];
    path_element_t pe2 = p.elements[s2_idx];

#ifdef debug_CUDA
    if (blockIdx.x == 0 && threadIdx.x == 0 && iter < 5) {
        printf("[gpu_sort_kernel] iter=%d tid=%u step_idx=%u path_i=%u s1_idx=%u s2_idx=%u\n",
               iter, tid, step_idx, path_i, s1_idx, s2_idx);
    }
#endif

    // node IDs, 0-based
    uint32_t n1_id = pe1.node_id;
    uint32_t n2_id = pe2.node_id;
    if (n1_id >= node_data.node_count || n2_id >= node_data.node_count) {
#ifdef debug_CUDA
        if (blockIdx.x == 0 && threadIdx.x == 0 && iter < 2) {
            printf("[gpu_sort_kernel] out-of-bounds node index! n1_id=%u n2_id=%u node_count=%u\n", n1_id, n2_id, node_data.node_count);
        }
#endif
        return;
    }

    double pos1 = double(llabs(pe1.pos));
    if (pe1.pos < 0) {
        pos1 += node_data.nodes[n1_id].seq_length;
    }
    bool use_other_end_1 = (curand_coalesced(thread_rnd_state, lane) % 2) == 0;
    if (use_other_end_1) {
        pos1 += node_data.nodes[n1_id].seq_length;
    }

    double pos2 = double(llabs(pe2.pos));
    if (pe2.pos < 0) {
        pos2 += node_data.nodes[n2_id].seq_length;
    }
    bool use_other_end_2 = (curand_coalesced(thread_rnd_state, lane) % 2) == 0;
    if (use_other_end_2) {
        pos2 += node_data.nodes[n2_id].seq_length;
    }

    float *x1 = &node_data.nodes[n1_id].x;
    float *x2 = &node_data.nodes[n2_id].x;

    // We'll capture old positions for debug
    float old_x1 = *x1;
    float old_x2 = *x2;

    // 1D update
    update_pos_gpu_1d(pos1, pos2, x1, x2, eta);

#ifdef debug_CUDA
    if (blockIdx.x == 0 && threadIdx.x == 0 && iter < 5) {
        float new_x1 = *x1;
        float new_x2 = *x2;
        if (isnan(new_x1) || isnan(new_x2)) {
            printf("[gpu_sort_kernel WARNING] iter=%d tid=%u n1_id=%u n2_id=%u pos1=%.2f pos2=%.2f old_x1=%.2f old_x2=%.2f => new_x1=%.2f new_x2=%.2f => NaN!\n",
                   iter, tid, n1_id, n2_id, pos1, pos2, old_x1, old_x2, new_x1, new_x2);
        } else {
            printf("[gpu_sort_kernel] iter=%d tid=%u n1_id=%u n2_id=%u pos1=%.2f pos2=%.2f old_x1=%.2f old_x2=%.2f => new_x1=%.2f new_x2=%.2f\n",
                   iter, tid, n1_id, n2_id, pos1, pos2, old_x1, old_x2, new_x1, new_x2);
        }
    }
#endif
}

/**
 * \brief The main function: sets up data on the GPU, executes iterations, 
 *        and copies the final results back to X.
 */
void gpu_sort(sort_config_t config,
              const odgi::graph_t &graph,
              std::vector<std::atomic<double>> &X) 
{
#ifdef debug_CUDA
    std::cerr << "[debug_CUDA] GPU path-based SGD invoked.\n";
    std::cerr << "[debug_CUDA] iter_max=" << config.iter_max
              << " min_term_updates=" << config.min_term_updates
              << " delta=??"  // if needed
              << " eps=" << config.eps
              << " eta_max=" << config.eta_max
              << " theta=" << config.theta
              << " space=" << config.space
              << " space_max=" << config.space_max
              << " space_quantization_step=" << config.space_quantization_step
              << " cooling_start=" << (double)config.first_cooling_iteration / (double)config.iter_max
              << "\n";
#endif

    // Basic GPU device info
    cudaDeviceProp prop;
    CUDACHECK(cudaGetDeviceProperties(&prop, 0));
    int sm_count = prop.multiProcessorCount;

#ifdef debug_CUDA
    std::cerr << "[gpu_sort] GPU name: " << prop.name 
              << " with " << sm_count << " SMs.\n";
#endif

    // Build the learning-rate schedule
    double *etas;
    CUDACHECK(cudaMallocManaged(&etas, config.iter_max * sizeof(double)));
    int32_t iter_max = config.iter_max;
    int32_t iter_max_lr = config.iter_with_max_learning_rate;
    double w_max = 1.0;
    double eps = config.eps;
    double eta_max = config.eta_max;
    double eta_min = eps / w_max;

#ifdef debug_CUDA
    std::cerr << "[gpu_sort] Building learning-rate schedule for " 
              << iter_max << " iterations.\n";
#endif

    if (iter_max < 2) {
        for (int32_t i = 0; i < iter_max; i++) {
            etas[i] = eta_max;
        }
    } 
    else {
        double lambda = log(eta_max / eta_min) / (double(iter_max) - 1.0);
        for (int32_t i = 0; i < iter_max; i++) {
            double e = eta_max * exp(-lambda * fabs(double(i) - double(iter_max_lr)));
            if (std::isnan(e)) e = eta_min;
            etas[i] = e;
        }
    }

    uint32_t node_count = graph.get_node_count();
    node_data_t node_data;
    node_data.node_count = node_count;
    CUDACHECK(cudaMallocManaged(&node_data.nodes, node_count * sizeof(node_t)));

    // Copy node lengths and initial X positions
    for (uint32_t i = 0; i < node_count; i++) {
        auto h = graph.get_handle(i + 1, false);
        node_data.nodes[i].seq_length = graph.get_length(h);
        node_data.nodes[i].x = float(X[i].load());
    }

    // Build path_data
    uint32_t path_count = graph.get_path_count();
    path_data_t path_data;
    path_data.path_count = path_count;
    path_data.total_path_steps = 0;
    CUDACHECK(cudaMallocManaged(&path_data.paths, path_count * sizeof(path_t)));

    std::vector<odgi::path_handle_t> path_handles;
    path_handles.reserve(path_count);
    graph.for_each_path_handle([&](const odgi::path_handle_t &p) {
        path_handles.push_back(p);
        path_data.total_path_steps += graph.get_step_count(p);
    });
    CUDACHECK(cudaMallocManaged(&path_data.element_array,
                                path_data.total_path_steps * sizeof(path_element_t)));

    // fill out the path array
    uint64_t first_step_counter = 0;
    for (uint32_t pid = 0; pid < path_count; pid++) {
        auto p = path_handles[pid];
        int step_count = graph.get_step_count(p);
        path_data.paths[pid].step_count = step_count;
        path_data.paths[pid].first_step_in_path = first_step_counter;
        first_step_counter += step_count;
    }

    // This parallel block enumerates each path’s steps
#pragma omp parallel for num_threads(config.nthreads)
    for (int pid = 0; pid < (int)path_count; pid++) {
        auto p = path_handles[pid];
        uint32_t step_count = path_data.paths[pid].step_count;
        uint64_t base_idx   = path_data.paths[pid].first_step_in_path;
        if (step_count == 0) {
            path_data.paths[pid].elements = nullptr;
        } else {
            path_element_t *p_el = &path_data.element_array[base_idx];
            path_data.paths[pid].elements = p_el;

            odgi::step_handle_t s   = graph.path_begin(p);
            int64_t pos_along_path  = 1;
            for (uint32_t s_i = 0; s_i < step_count; s_i++) {
                odgi::handle_t h = graph.get_handle_of_step(s);
                p_el[s_i].node_id = graph.get_id(h) - 1; // zero-based
                p_el[s_i].pidx    = pid;
                if (graph.get_is_reverse(h)) {
                    p_el[s_i].pos = -pos_along_path;
                } else {
                    p_el[s_i].pos = pos_along_path;
                }
                pos_along_path += graph.get_length(h);

                if (graph.has_next_step(s)) {
                    s = graph.get_next_step(s);
                }
            }
        }
    }

    // Build a partial array of zetas for Zipf
    uint64_t zetas_cnt = ((config.space <= config.space_max)
                          ? config.space
                          : (config.space_max + (config.space - config.space_max)
                             / config.space_quantization_step + 1)) + 1;
    double *zetas;
    CUDACHECK(cudaMallocManaged(&zetas, zetas_cnt * sizeof(double)));
    double zeta_tmp = 0.0;
    for (uint64_t i = 1; i <= config.space; i++) {
        zeta_tmp += dirtyzipf::fast_precise_pow(1.0 / double(i), config.theta);
        if (i <= config.space_max) {
            zetas[i] = zeta_tmp;
        }
        if (i >= config.space_max && 
           (i - config.space_max) % config.space_quantization_step == 0)
        {
            uint64_t idx = config.space_max + 1 +
                           (i - config.space_max) / config.space_quantization_step;
            if (idx < zetas_cnt) {
                zetas[idx] = zeta_tmp;
            }
        }
    }

#ifdef debug_CUDA
    std::cerr << "[gpu_sort] Allocating random states for " << sm_count << " SM blocks.\n";
#endif
    curandState_t *rnd_state_tmp;
    curandStateCoalesced_t *rnd_state;
    CUDACHECK(cudaMallocManaged(&rnd_state_tmp, sm_count * BLOCK_SIZE * sizeof(curandState_t)));
    CUDACHECK(cudaMallocManaged(&rnd_state,     sm_count * sizeof(curandStateCoalesced_t)));

    cuda_device_init_sort<<<sm_count, BLOCK_SIZE>>>(rnd_state_tmp, rnd_state);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());
#ifdef debug_CUDA
    std::cerr << "[gpu_sort] Finished cuda_device_init_sort.\n";
#endif

    cudaFree(rnd_state_tmp);

    // pick # of blocks to cover min_term_updates threads each iteration
    uint64_t block_nbr = (config.min_term_updates + BLOCK_SIZE - 1) / BLOCK_SIZE;
#ifdef debug_CUDA
    std::cerr << "[gpu_sort] block_nbr = " << block_nbr 
              << " for min_term_updates=" << config.min_term_updates << "\n";
#endif

    // Main iteration loop
    for (int iter = 0; iter < (int)config.iter_max; iter++) {
#ifdef debug_CUDA
        if (iter < 5) {
            std::cerr << "[gpu_sort] Launching iteration " << iter
                      << " with eta=" << etas[iter] << "\n";
        }
#endif
        gpu_sort_kernel<<<block_nbr, BLOCK_SIZE>>>(iter,
                                                   config,
                                                   rnd_state,
                                                   etas[iter],
                                                   zetas,
                                                   node_data,
                                                   path_data,
                                                   sm_count);
        CUDACHECK(cudaGetLastError());
        CUDACHECK(cudaDeviceSynchronize());
#ifdef debug_CUDA
    uint64_t updates_this_iter = block_nbr * BLOCK_SIZE; 
    // or some variation if you do more than 1 update per thread
    uint64_t total_gpu_updates = (uint64_t)(iter + 1) * updates_this_iter;
    std::cerr << "[GPU] iteration=" << iter+1 
              << ", total updates so far=" << total_gpu_updates 
              << "\n";
#endif
    }

#ifdef debug_CUDA
    std::cerr << "[gpu_sort] All iterations done.\n";
#endif

    // Copy final positions back
    for (uint32_t i = 0; i < node_count; i++) {
        X[i].store(float(node_data.nodes[i].x));
    }

#ifdef debug_CUDA
    std::cerr << "[gpu_sort] Copied final positions back to host X.\n";
#endif

    // Cleanup
    cudaFree(etas);
    cudaFree(node_data.nodes);
    cudaFree(path_data.paths);
    cudaFree(path_data.element_array);
    cudaFree(zetas);
    cudaFree(rnd_state);

#ifdef debug_CUDA
    std::cerr << "[gpu_sort] Freed all GPU memory.\n";
#endif
    return;
}

} // end namespace cuda