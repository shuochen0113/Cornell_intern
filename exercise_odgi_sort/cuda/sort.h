#pragma once

#include <iostream>
#include <chrono>
#include <vector>
#include <omp.h>
#include <random>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include <sstream>
#include <iomanip>

#include "odgi.hpp"
#include "XoshiroCpp.hpp"
#include "dirty_zipfian_int_distribution.h"

namespace cuda {

// Data structures for 1D path-guided SGD

/* 
 for each node:
 - x: the 1D coordinate of the node’s center
 - length: the node’s length in nucleotides
 */
struct __align__(8) node_t {
    float x;
    int32_t seq_length;
};

struct node_data_t {
    uint32_t node_count;
    node_t *nodes;
};

/*
 for path element:
 - pidx: the path index
 - node_id: which node this step references
 - pos: the path-based position of the step (if negative, reversed)
 */
struct __align__(8) path_element_t {
    uint32_t pidx;
    uint32_t node_id;
    int64_t pos;
};

/*
 a path's discription:
 - step_count: how many steps
 - first_step_in_path: offset into a global array of path_element_t
 - elements: pointer to the first step
 */
struct path_t {
    uint32_t step_count;
    uint64_t first_step_in_path;
    path_element_t *elements;
};

struct path_data_t {
    uint32_t path_count;
    uint64_t total_path_steps;
    path_t *paths;
    path_element_t *element_array;
};

#define BLOCK_SIZE 1024
#define WARP_SIZE 32

struct curandStateXORWOWCoalesced_t {
    unsigned int d[BLOCK_SIZE];
    unsigned int w0[BLOCK_SIZE];
    unsigned int w1[BLOCK_SIZE];
    unsigned int w2[BLOCK_SIZE];
    unsigned int w3[BLOCK_SIZE];
    unsigned int w4[BLOCK_SIZE];
};
typedef struct curandStateXORWOWCoalesced_t curandStateCoalesced_t;

struct sort_config_t {
    uint64_t iter_max; // total # of iterations
    uint64_t min_term_updates; // # of pairwise updates per iteration
    double eta_max; // maximum learning rate
    double eps; // min learning rate factor
    int32_t iter_with_max_learning_rate;
    uint32_t first_cooling_iteration;
    double theta;
    uint32_t space;
    uint32_t space_max;
    uint32_t space_quantization_step;
    int nthreads;
};

void gpu_sort(sort_config_t config,
              const odgi::graph_t &graph,
              std::vector<std::atomic<double>> &X);

}