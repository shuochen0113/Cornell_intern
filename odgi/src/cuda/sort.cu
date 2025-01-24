#include "sort.h"
#include <cuda.h>
#include <cmath>
#include <iostream>
#include <chrono>
#include <assert.h>
#include "cuda_runtime_api.h"

// #define debug_CUDA
#ifdef debug_CUDA
#include <cstdio>
#endif

#define CUDACHECK(cmd) do {                                     \
  cudaError_t err = cmd;                                        \
  if (err != cudaSuccess) {                                     \
    std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__   \
              << " : " << cudaGetErrorString(err) << std::endl; \
    exit(EXIT_FAILURE);                                         \
  }                                                             \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


/**
 * \brief A device function to initialize the coalesced curand states.
 */
__global__ 
void cuda_device_init(curandState_t *rnd_state_tmp, cuda::curandStateCoalesced_t *rnd_state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Initialize older (non-coalesced) CURAND state
    curand_init(42ULL + tid, tid, 0, &rnd_state_tmp[tid]);
    // Convert to coalesced form
    rnd_state[blockIdx.x].d[threadIdx.x] = rnd_state_tmp[tid].d;
    rnd_state[blockIdx.x].w0[threadIdx.x] = rnd_state_tmp[tid].v[0];
    rnd_state[blockIdx.x].w1[threadIdx.x] = rnd_state_tmp[tid].v[1];
    rnd_state[blockIdx.x].w2[threadIdx.x] = rnd_state_tmp[tid].v[2];
    rnd_state[blockIdx.x].w3[threadIdx.x] = rnd_state_tmp[tid].v[3];
    rnd_state[blockIdx.x].w4[threadIdx.x] = rnd_state_tmp[tid].v[4];
}


__device__
unsigned int curand_coalesced(cuda::curandStateCoalesced_t *state, uint32_t t_idx) {
    // XORWOW step
    unsigned int t = (state->w0[t_idx] ^ (state->w0[t_idx] >> 2));
    state->w0[t_idx] = state->w1[t_idx];
    state->w1[t_idx] = state->w2[t_idx];
    state->w2[t_idx] = state->w3[t_idx];
    state->w3[t_idx] = state->w4[t_idx];
    state->w4[t_idx] = (state->w4[t_idx] ^ (state->w4[t_idx] << 4)) ^ (t ^ (t << 1));
    state->d[t_idx] += 362437U;
    return (state->w4[t_idx] + state->d[t_idx]);
}

__device__
float curand_uniform_coalesced(cuda::curandStateCoalesced_t *state, uint32_t t_idx) {
    // We just call curand_coalesced() and map the result to (0,1]
    unsigned int r = curand_coalesced(state, t_idx);
    // same approach as _curand_uniform in official curand headers
    return (float)((double)r * (1.0 / 4294967296.0));
}


/**
 * \brief Zipf sampling, same approximate logic as your 2D path_sgd code.
 */
__device__
uint32_t cuda_rnd_zipf(cuda::curandStateCoalesced_t *rnd_state,
                       uint32_t t_idx,
                       uint32_t n,
                       double theta,
                       double zeta2,
                       double zetan) {
    double alpha = 1.0 / (1.0 - theta);
    double denom = 1.0 - (zeta2 / zetan);
    if (fabs(denom) < 1e-9) {
        denom = 1e-9;
    }
    double eta = (1.0 - pow(2.0 / (double)n, (1.0 - theta))) / denom;
    double u = 1.0 - curand_uniform_coalesced(rnd_state, t_idx); 
    double uz = u * zetan;

    int64_t val;
    if (uz < 1.0) {
        val = 1;
    } else if (uz < (1.0 + pow(0.5, theta))) {
        val = 2;
    } else {
        val = 1 + (int64_t)((double)n * pow((eta * u - eta + 1.0), 1.0 / (1.0 - theta)));
    }
    if (val > (int64_t)n) {
        val = n; // clamp
    }
    assert(val >= 0);
    assert(val <= (int64_t)n);
    return (uint32_t)val;
}

/**
 * \brief Update two node positions in 1D, if they are not "locked" by target-sorting.
 * If both are locked, we skip. If one is locked, we only move the other side.
 */
__device__
void update_pos_gpu_1D(int64_t n1_pos_in_path, uint32_t n1_id,
                       int64_t n2_pos_in_path, uint32_t n2_id,
                       double eta,
                       cuda::node_data_t &node_data,
                       bool use_target_sorting,
                       const bool* device_target_nodes) 
{
    // If target-sorting is on, check if each node is locked
    bool update_n1 = true;
    bool update_n2 = true;
    if (use_target_sorting && device_target_nodes) {
        if (device_target_nodes[n1_id]) {
            update_n1 = false;
        }
        if (device_target_nodes[n2_id]) {
            update_n2 = false;
        }
        // If both locked => skip entirely
        if (!update_n1 && !update_n2) {
            return;
        }
    }

    double term_dist = fabs((double)n1_pos_in_path - (double)n2_pos_in_path);
    if (term_dist < 1e-9) {
        term_dist = 1e-9;
    }

    double w_ij = 1.0 / term_dist;
    double mu = eta * w_ij;
    if (mu > 1.0) {
        mu = 1.0;
    }

    float *x1 = &(node_data.nodes[n1_id].x);
    float *x2 = &(node_data.nodes[n2_id].x);

    double x1_val = (double)*x1;
    double x2_val = (double)*x2;

    double dx = x1_val - x2_val;
    if (fabs(dx) < 1e-9) {
        dx = 1e-9;
    }
    double mag = fabs(dx);
    double delta = mu * (mag - term_dist) / 2.0;
    double r = delta / mag;
    double r_x = r * dx;

    // Move whichever node(s) are not locked
    if (update_n1) {
        float new_x1 = (float)(x1_val - r_x);
        atomicExch(x1, new_x1);
    }
    if (update_n2) {
        float new_x2 = (float)(x2_val + r_x);
        atomicExch(x2, new_x2);
    }
}

// The CUDA kernel for doing 1D path SGD
__global__
void gpu_sort_kernel(int iter,
                     cuda::sort_config_t config,
                     cuda::curandStateCoalesced_t *rnd_state,
                     double eta,
                     double *zetas,
                     cuda::node_data_t node_data,
                     cuda::path_data_t path_data,
                     bool in_cooling_phase,
                     int sm_count,
                     bool use_target_sorting,
                     const bool* device_target_nodes)
{
    // global thread index
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // read SM ID
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    assert(smid < (uint32_t)sm_count);

    // only proceed if tid < min_term_updates
    if (tid >= config.min_term_updates) {
        return;
    }

#ifdef debug_CUDA
    // Print info for the first few threads
    if (tid < 8 && iter < 2) {
        printf("[debug_CUDA kernel] iter=%d, tid=%llu, smid=%u in_cooling=%d\n",
               iter, (unsigned long long)tid, smid, (int)in_cooling_phase);
    }
#endif

    // pick a random generator based on SM ID
    cuda::curandStateCoalesced_t *rng_block = &rnd_state[smid];

    // pick a random step
    uint32_t random_val = curand_coalesced(rng_block, threadIdx.x);
    if (path_data.total_path_steps == 0) {
        return; 
    }
    uint64_t step_idx = random_val % path_data.total_path_steps;

    // fetch stepA
    cuda::path_element_t stepA = path_data.element_array[step_idx];
    uint32_t path_idx = stepA.pidx;
    cuda::path_t p = path_data.paths[path_idx];

    // skip if degenerate path
    if (p.step_count < 2) {
        return;
    }
    assert(p.step_count > 1);

    uint32_t s1_idx = step_idx - p.first_step_in_path; // index within path
    uint32_t s2_idx = 0;

    // do we use "cooling" or random approach? 
    bool coin = ((curand_coalesced(rng_block, threadIdx.x) & 1U) == 0);

    if (in_cooling_phase || coin) {
        // zipf-based approach
        bool go_backward = false;
        uint32_t flip_val = (curand_coalesced(rng_block, threadIdx.x) & 1U); // 0 or 1

        if ((s1_idx > 0 && flip_val == 0) || (s1_idx == p.step_count - 1)) {
            go_backward = true;
        }
        uint32_t jump_space = go_backward ? s1_idx : (p.step_count - 1 - s1_idx);
        if (jump_space > config.space) {
            jump_space = config.space; 
        }
        if (jump_space == 0) {
            // no move possible
            return;
        }

        uint32_t z_idx = jump_space;
        if (jump_space > config.space_max) {
            z_idx = config.space_max + 
                    (jump_space - config.space_max) / config.space_quantization_step + 1;
        }
        uint32_t z_i = cuda_rnd_zipf(rng_block, threadIdx.x,
                                     jump_space,
                                     config.theta,
                                     zetas[2],
                                     zetas[z_idx]);
        if (go_backward) {
            s2_idx = s1_idx - z_i;
        } else {
            s2_idx = s1_idx + z_i;
        }
    } else {
        // purely random s2
        uint32_t r2 = curand_coalesced(rng_block, threadIdx.x) % p.step_count;
        while (r2 == s1_idx && p.step_count > 1) {
            r2 = (r2 + 1) % p.step_count;
        }
        s2_idx = r2;
    }
    assert(s1_idx < p.step_count);
    assert(s2_idx < p.step_count);

    if (s1_idx == s2_idx) {
        // should never happen, but let's guard
        return;
    }

    // fetch stepB
    cuda::path_element_t stepB = p.elements[s2_idx];

    // get node IDs & positions
    uint32_t n1_id = stepA.node_id;
    int64_t n1_pos = stepA.pos;
    if (n1_pos < 0) n1_pos = -n1_pos;

    uint32_t n2_id = stepB.node_id;
    int64_t n2_pos = stepB.pos;
    if (n2_pos < 0) n2_pos = -n2_pos;

    // update 1D (with possible target-sorting skip)
    update_pos_gpu_1D(n1_pos, n1_id,
                      n2_pos, n2_id,
                      eta,
                      node_data,
                      use_target_sorting,
                      device_target_nodes);
}


// The main function that’s declared in sort.h
namespace cuda {

void gpu_sort(sort_config_t config,
              const odgi::graph_t &graph,
              std::vector<std::atomic<double>> &X,
              bool target_sorting,
              const std::vector<bool> &target_nodes)
{
    std::cout << "[gpu_sort] Using GPU to compute 1D path-SGD..." << std::endl;

    // 1) Basic GPU properties
    cudaDeviceProp prop;
    CUDACHECK(cudaGetDeviceProperties(&prop, 0));
    int sm_count = prop.multiProcessorCount;

#ifdef debug_CUDA
    std::cerr << "[debug_CUDA][gpu_sort] sm_count=" << sm_count
              << ", iter_max=" << config.iter_max
              << ", min_term_updates=" << config.min_term_updates 
              << ", target_sorting=" << target_sorting << std::endl;
#endif

    // 2) Precompute the learning rates (eta) for each iteration
    double *etas = nullptr;
    CUDACHECK(cudaMallocManaged(&etas, config.iter_max * sizeof(double)));

    const int32_t iter_max = config.iter_max;
    const int32_t iter_with_max_learning_rate = config.iter_with_max_learning_rate;
    const double w_max = 1.0;
    const double eps = config.eps;
    const double eta_max = config.eta_max;
    const double eta_min = eps / w_max;
    const double lambda  = log(eta_max / eta_min) / (double(iter_max) - 1.0);

    for (uint64_t i = 0; i < config.iter_max; i++) {
        double val = eta_max * exp(-lambda * (fabs(double(i) - double(iter_with_max_learning_rate))));
        if (std::isnan(val)) val = eta_min;
        etas[i] = val;
    }

    // 3) Build node_data (copy from X)
    uint32_t node_count = graph.get_node_count();
    assert(graph.min_node_id() == 1);
    assert(graph.max_node_id() == node_count);
    assert(graph.max_node_id() - graph.min_node_id() + 1 == node_count);

    node_data_t node_data;
    node_data.node_count = node_count;
    CUDACHECK(cudaMallocManaged(&node_data.nodes, node_count * sizeof(node_t)));

    // Fill node_data from X
    uint64_t i_node = 0;
    graph.for_each_handle([&](const handlegraph::handle_t &h) {
        uint64_t idx = odgi::number_bool_packing::unpack_number(h);
        node_data.nodes[idx].x = float(X[idx].load());
        node_data.nodes[idx].seq_length = (int32_t)graph.get_length(h); 
        i_node++;
    });

    // 4) Build path_data
    uint32_t path_count = graph.get_path_count();
    path_data_t path_data;
    path_data.path_count = path_count;
    path_data.total_path_steps = 0;
    CUDACHECK(cudaMallocManaged(&path_data.paths, path_count * sizeof(path_t)));

    // gather path handles
    std::vector<odgi::path_handle_t> path_handles;
    path_handles.reserve(path_count);
    graph.for_each_path_handle([&](const odgi::path_handle_t &p) {
        path_handles.push_back(p);
        path_data.total_path_steps += graph.get_step_count(p);
    });

    CUDACHECK(cudaMallocManaged(&path_data.element_array, 
                                path_data.total_path_steps * sizeof(path_element_t)));

    // fill in step_count + first_step_in_path
    uint64_t first_step_counter = 0;
    for (uint32_t p_i = 0; p_i < path_count; p_i++) {
        odgi::path_handle_t p = path_handles[p_i];
        uint64_t sc = graph.get_step_count(p);
        path_data.paths[p_i].step_count = sc;
        path_data.paths[p_i].first_step_in_path = first_step_counter;
        first_step_counter += sc;
    }

    // fill path_data.element_array in parallel
    #pragma omp parallel for num_threads(config.nthreads)
    for (int p_i = 0; p_i < (int)path_count; p_i++) {
        odgi::path_handle_t path_h = path_handles[p_i];
        uint64_t step_count = path_data.paths[p_i].step_count;
        uint64_t first_step = path_data.paths[p_i].first_step_in_path;

        if (step_count == 0) {
            path_data.paths[p_i].elements = nullptr;
        } else {
            path_element_t *base = &path_data.element_array[first_step];
            path_data.paths[p_i].elements = base;

            odgi::step_handle_t s = graph.path_begin(path_h);
            int64_t pos = 1;
            for (uint64_t s_i = 0; s_i < step_count; s_i++) {
                odgi::handle_t h = graph.get_handle_of_step(s);
                uint64_t node_id = graph.get_id(h) - 1;
                bool is_rev = graph.get_is_reverse(h);

                base[s_i].pidx    = p_i;
                base[s_i].node_id = node_id;
                base[s_i].pos     = is_rev ? -pos : pos;

                pos += graph.get_length(h);

                if (graph.has_next_step(s)) {
                    s = graph.get_next_step(s);
                }
            }
        }
    }

    // 5) Precompute zetas for zipf
    auto start_zeta = std::chrono::high_resolution_clock::now();
    uint64_t zetas_cnt = (config.space <= config.space_max)
                         ? (config.space + 1)
                         : (config.space_max + 
                            (config.space - config.space_max) / config.space_quantization_step + 1 + 1);
    double *zetas = nullptr;
    CUDACHECK(cudaMallocManaged(&zetas, zetas_cnt * sizeof(double)));

    double zeta_accum = 0.0;
    for (uint64_t k = 1; k <= config.space; k++) {
        zeta_accum += dirtyzipf::fast_precise_pow(1.0 / double(k), config.theta);
        if (k <= config.space_max) {
            zetas[k] = zeta_accum;
        }
        if (k >= config.space_max && (k - config.space_max) % config.space_quantization_step == 0) {
            zetas[config.space_max + 1 + (k - config.space_max) / config.space_quantization_step] = zeta_accum;
        }
    }
    auto end_zeta = std::chrono::high_resolution_clock::now();
    uint32_t duration_zeta_ms = 
        std::chrono::duration_cast<std::chrono::milliseconds>(end_zeta - start_zeta).count();

    // 6) Allocate RNG states
    curandState_t *rnd_tmp = nullptr;
    curandStateCoalesced_t *rnd_state = nullptr;
    CUDACHECK(cudaMallocManaged(&rnd_tmp,   sm_count * BLOCK_SIZE * sizeof(curandState_t)));
    CUDACHECK(cudaMallocManaged(&rnd_state, sm_count * sizeof(curandStateCoalesced_t)));

    cuda_device_init<<<sm_count, BLOCK_SIZE>>>(rnd_tmp, rnd_state);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaFree(rnd_tmp));

    // 7) If we are doing target sorting, copy that array to device
    bool *device_target_nodes = nullptr;
    if (target_sorting && !target_nodes.empty()) {
        CUDACHECK(cudaMallocManaged(&device_target_nodes, node_count * sizeof(bool)));
        for (uint32_t i = 0; i < node_count; i++) {
            device_target_nodes[i] = target_nodes[i];
        }
        // device_target_nodes is now a bool array on GPU
#ifdef debug_CUDA
        std::cerr << "[debug_CUDA][gpu_sort] Copied target_nodes to device.\n";
#endif
    }

    // 8) Main loop: launch kernel for each iteration
    uint64_t block_nbr = (config.min_term_updates + BLOCK_SIZE - 1ULL) / BLOCK_SIZE;

    for (uint64_t iter = 0; iter < config.iter_max; iter++) {
        double cur_eta = etas[iter];
        bool in_cooling_phase = (iter >= config.first_cooling_iteration);

#ifdef debug_CUDA
        // Print every 10 iterations or the first few
        if (iter < 5 || iter % 10 == 0) {
            std::cerr << "[debug_CUDA][gpu_sort] iteration=" << iter
                      << ", eta=" << cur_eta
                      << ", in_cooling=" << in_cooling_phase
                      << std::endl;
        }
#endif

        gpu_sort_kernel<<<block_nbr, BLOCK_SIZE>>>(
            (int)iter,
            config,
            rnd_state,
            cur_eta,
            zetas,
            node_data,
            path_data,
            in_cooling_phase,
            sm_count,
            target_sorting,
            device_target_nodes
        );
        CUDACHECK(cudaGetLastError());
        CUDACHECK(cudaDeviceSynchronize());
    }

    // 9) Copy final positions back to X
    for (uint64_t idx = 0; idx < node_count; idx++) {
        X[idx].store(double(node_data.nodes[idx].x));
    }

#ifdef debug_CUDA
    std::cerr << "[debug_CUDA][gpu_sort] Done with all iterations; copying positions back.\n";
#endif

    // 10) Clean up
    cudaFree(etas);
    cudaFree(node_data.nodes);
    cudaFree(path_data.paths);
    cudaFree(path_data.element_array);
    cudaFree(zetas);
    cudaFree(rnd_state);

    if (device_target_nodes != nullptr) {
        cudaFree(device_target_nodes);
    }

    std::cout << "[gpu_sort] Finished GPU-based 1D path-SGD.\n";
}

} // end namespace cuda