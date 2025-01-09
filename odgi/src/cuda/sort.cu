#include "sort.h"
#include <cuda.h>
#include <assert.h>
#include "cuda_runtime_api.h"

#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

namespace cuda {


// store curandState_t in a coalesced manner for each Streaming Multiprocesser block.
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
}

// Pseudorandom 32-bit generator from our coalesced XORWOW state.
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

// Returns a uniform float in (0,1].
__device__ 
float curand_uniform_coalesced(curandStateCoalesced_t *state, int lane_id) {
    uint32_t r = curand_coalesced(state, lane_id);
    return _curand_uniform(r);
}

/*
 A minimal Zipf sampler in CUDA for 1D usage (uses fast power function __powf).
 - n: domain of Zipf
 - theta: exponent
 - zeta2, zetan: partial sums for normalization
 */
__device__ 
uint32_t cuda_rnd_zipf(curandStateCoalesced_t *rnd_state, uint32_t n, double theta, double zeta2, 
double zetan, int lane_id) {
    double alpha = 1.0 / (1.0 - theta);
    double denom = 1.0 - zeta2 / zetan;
    if (fabs(denom) < 1e-14) {
        denom = 1e-14;
    }
    double eta = (1.0 - __powf(2.0f / float(n), 1.0f - float(theta))) / denom;

    // invert a uniform sample
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
    if (val > n) val = n;
    return uint32_t(val);
}

/*
 Helper for the 1D update of two node positions.
 - pos1: path position of node1 in path space
 - pos2: path position of node2 in path space
 - x1: pointer to node1’s coordinate
 - x2: pointer to node2’s coordinate
 - eta: current learning rate factor
 */
__device__ 
void update_pos_gpu_1d(double pos1, double pos2, float *x1, float *x2, double eta) {
    double term_dist = fabs(pos1 - pos2);
    if (term_dist < 1e-9) {
        return; // effectively 0 distance in path space
    }
    double w_ij = 1.0 / term_dist;
    double mu = eta * w_ij;
    if (mu > 1.0){
        mu = 1.0;
    }

    float x1_val = *x1;
    float x2_val = *x2;
    double dx = double(x1_val) - double(x2_val);
    if (fabs(dx) < 1e-9) {
        dx = 1e-9; // avoid zero
    }
    double mag = fabs(dx);
    double delta = mu * (mag - term_dist) / 2.0;
    // The actual movement
    double r = delta / mag;
    double r_x = r * dx;

    // Atomic updates in float precision
    atomicAdd(x1, float(-r_x));
    atomicAdd(x2, float(r_x));
}

/*
 Per-iteration kernel for path-based 1D SGD.
 Picks random path steps for each thread, finds a partner step in the path 
 (possibly using a Zipf distribution if in “cooling” mode). 
 Then applies a 1D distance update to the node positions.
 */
__global__
void gpu_sort_kernel(int iter,
                     sort_config_t config,
                     curandStateCoalesced_t *rnd_state,
                     double eta,
                     double *zetas,
                     node_data_t node_data,
                     path_data_t path_data,
                     int sm_count) {
    uint32_t tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x;  // local lane ID within the block

    // figure out which SM we are on, to pick the correct curandStateCoalesced_t
    int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    if (smid >= sm_count) return;
    curandStateCoalesced_t *thread_rnd_state = &rnd_state[smid];

    // Decide if we are in “cooling” mode
    bool in_cooling = (iter >= config.first_cooling_iteration);

    // 1) pick a random step index overall
    uint32_t step_idx = curand_coalesced(thread_rnd_state, lane)% (path_data.total_path_steps);

    // 2) find which path that belongs to
    uint32_t path_i = path_data.element_array[step_idx].pidx;
    path_t p = path_data.paths[path_i];

    if (p.step_count < 2) {
        // trivial path, skip
        return;
    }

    // pick step s1 randomly
    uint32_t s1_idx = curand_coalesced(thread_rnd_state, lane) % p.step_count;

    // pick step s2
    uint32_t s2_idx = s1_idx;  // initialize
    if (in_cooling) {
        // pick near s1 using a Zipf-based jump
        bool go_backward = false;
        bool coin_flip = (curand_coalesced(thread_rnd_state, lane) & 1) == 0;
        if ((s1_idx > 0 && coin_flip) || s1_idx == p.step_count - 1) {
            // go backward
            go_backward = true;
            uint32_t jump = min(config.space, s1_idx);
            // quantize
            uint32_t space_idx = jump;
            if (jump > config.space_max) {
                space_idx = config.space_max + (jump - config.space_max) / config.space_quantization_step + 1;
            }
            uint32_t z_i = cuda_rnd_zipf(thread_rnd_state,
                                         jump,
                                         config.theta,
                                         zetas[2],      // zetas[2] ~ zeta(2)
                                         zetas[space_idx],
                                         lane);
            if (z_i > s1_idx) z_i = s1_idx;  // clamp
            s2_idx = s1_idx - z_i;
        } 
        else {
            // go forward
            uint32_t jump = min(config.space, p.step_count - s1_idx - 1);
            uint32_t space_idx = jump;
            if (jump > config.space_max) {
                space_idx = config.space_max + (jump - config.space_max) / config.space_quantization_step + 1;
            }
            uint32_t z_i = cuda_rnd_zipf(thread_rnd_state,
                                         jump,
                                         config.theta,
                                         zetas[2],
                                         zetas[space_idx],
                                         lane);
            if (z_i > (p.step_count - s1_idx - 1)) {
                z_i = (p.step_count - s1_idx - 1);
            }
            s2_idx = s1_idx + z_i;
        }
    } 
    else {
        // pick randomly
        while (s2_idx == s1_idx) {
            s2_idx = curand_coalesced(thread_rnd_state, lane) % p.step_count;
        }
    }

    // gather the path positions
    path_element_t pe1 = p.elements[s1_idx];
    path_element_t pe2 = p.elements[s2_idx];

    // node IDs, 0-based
    uint32_t n1_id = pe1.node_id;
    uint32_t n2_id = pe2.node_id;
    if (n1_id >= node_data.node_count || n2_id >= node_data.node_count) return;

    double pos1 = double(llabs(pe1.pos)); // path-based position
    if (pe1.pos < 0) {
        // orientation is reversed
        pos1 += node_data.nodes[n1_id].seq_length;
    }
    // random flip for “use other end” in CPU code:
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

    // 3) update in 1D
    update_pos_gpu_1d(pos1, pos2, x1, x2, eta);
}

/*
 The main function: sets up data on the GPU, executes iterations,
 and copies the final results back to X (the atomic<double> vector).
 */
void gpu_sort(sort_config_t config,
              const odgi::graph_t &graph,
              std::vector<std::atomic<double>> &X) {

    std::cerr << "===== [GPU] Running 1D path-guided SGD =====" << std::endl;

    // Basic GPU device info
    cudaDeviceProp prop;
    CUDACHECK(cudaGetDeviceProperties(&prop, 0));
    int sm_count = prop.multiProcessorCount;

    // Build the learning-rate schedule in host memory
    double *etas;
    cudaMallocManaged(&etas, config.iter_max * sizeof(double));

    int32_t iter_max = config.iter_max;
    int32_t iter_max_lr = config.iter_with_max_learning_rate;
    double w_max = 1.0;
    double eps = config.eps;
    double eta_max = config.eta_max;
    double eta_min = eps / w_max;
    if (iter_max < 2) {
        // trivial fallback
        for (int32_t i = 0; i < iter_max; i++) etas[i] = eta_max;
    } 
    else {
        double lambda = log(eta_max / eta_min) / (double(iter_max) - 1.0);
        for (int32_t i = 0; i < iter_max; i++) {
            double e = eta_max * exp(-lambda * fabs(double(i) - double(iter_max_lr)));
            if (std::isnan(e)) e = eta_min;
            etas[i] = e;
        }
    }

    // Build node_data
    uint32_t node_count = graph.get_node_count();
    assert(graph.min_node_id() == 1);
    assert(graph.max_node_id() - graph.min_node_id() + 1 == node_count);

    node_data_t node_data;
    node_data.node_count = node_count;
    cudaMallocManaged(&node_data.nodes, node_count * sizeof(node_t));

    // Copy node lengths and initial X positions
    for (uint32_t i = 0; i < node_count; i++) {
        // Node IDs in the ODGI graph go from 1..node_count
        auto h = graph.get_handle(i + 1, false);
        node_data.nodes[i].seq_length = graph.get_length(h);
        node_data.nodes[i].x = float(X[i].load());  // single-precision copy
    }

    // Build path_data
    uint32_t path_count = graph.get_path_count();
    path_data_t path_data;
    path_data.path_count = path_count;
    path_data.total_path_steps = 0;
    cudaMallocManaged(&path_data.paths, path_count * sizeof(path_t));

    std::vector<odgi::path_handle_t> path_handles;
    path_handles.reserve(path_count);
    graph.for_each_path_handle([&](const odgi::path_handle_t &p) {
        path_handles.push_back(p);
        path_data.total_path_steps += graph.get_step_count(p);
    });
    cudaMallocManaged(&path_data.element_array, path_data.total_path_steps * sizeof(path_element_t));

    // Fill out the path array
    uint64_t first_step_counter = 0;
    for (uint32_t pid = 0; pid < path_count; pid++) {
        auto p = path_handles[pid];
        int step_count = graph.get_step_count(p);
        path_data.paths[pid].step_count = step_count;
        path_data.paths[pid].first_step_in_path = first_step_counter;
        first_step_counter += step_count;
    }

    // Question: what is that part used for?
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
                // negative if reversed
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
                          : (config.space_max + (config.space - config.space_max) / config.space_quantization_step + 1)) + 1;
    double *zetas;
    cudaMallocManaged(&zetas, zetas_cnt * sizeof(double));
    double zeta_tmp = 0.0;
    for (uint64_t i = 1; i <= config.space; i++) {
        zeta_tmp += dirtyzipf::fast_precise_pow(1.0 / double(i), config.theta);
        // store in the same manner as the CPU code
        if (i <= config.space_max) {
            zetas[i] = zeta_tmp;
        }
        if (i >= config.space_max && (i - config.space_max) % config.space_quantization_step == 0) {
            uint64_t idx = config.space_max + 1 + (i - config.space_max) / config.space_quantization_step;
            if (idx < zetas_cnt) {
                zetas[idx] = zeta_tmp;
            }
        }
    }

    // Prepare random states for each SM block
    curandState_t *rnd_state_tmp;
    curandStateCoalesced_t *rnd_state;
    CUDACHECK(cudaMallocManaged(&rnd_state_tmp, sm_count * BLOCK_SIZE * sizeof(curandState_t)));
    CUDACHECK(cudaMallocManaged(&rnd_state, sm_count * sizeof(curandStateCoalesced_t)));
    cuda_device_init_sort<<<sm_count, BLOCK_SIZE>>>(rnd_state_tmp, rnd_state);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());
    cudaFree(rnd_state_tmp);

    // pick # of blocks to cover min_term_updates threads each iteration
    uint64_t block_nbr = (config.min_term_updates + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Main iteration loop
    for (int iter = 0; iter < (int)config.iter_max; iter++) {
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
    }

    // Copy final positions back
    for (uint32_t i = 0; i < node_count; i++) {
        X[i].store(double(node_data.nodes[i].x));
    }

    // Cleanup
    cudaFree(etas);
    cudaFree(node_data.nodes);
    cudaFree(path_data.paths);
    cudaFree(path_data.element_array);
    cudaFree(zetas);
    cudaFree(rnd_state);

    return;
}

}