// Added features:
// 1. For each query, check whether a vertex is already collected
// 2. For each query, collecte all the vertices along the hops, then construct
// matrix and compute square distance in CUDA

#include <cmath>
#include <cstring>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <limits>
#include <queue>
#include <stdio.h>

using namespace std;

int V, D, E, L, K, A, B, C, M, Q;
int *X;
int *edges;
int *collected;

__global__ void gpu_l2_distance(int *a, int *b, int *c, int n, int D) {
  int n_index = blockDim.x * blockIdx.x + threadIdx.x;

  if (n_index < n * D) {
    c[n_index] = (a[n_index] - b[n_index]) * (a[n_index] - b[n_index]);
  }
}

// For each query
int nearest_id(int start_point, int max_hop, int *query_data) {
  std::queue<std::pair<int, int>> q;
  q.push(std::make_pair(start_point, 0));
  int min_l2 = std::numeric_limits<int>::max();
  int min_id = -1;

  // Collected vertices
  int *collected_set_id = new int[V];
  int *collected_set_matrix = new int[V * D];
  int collected_number = 0;

  // BFS for hops
  while (!q.empty()) {
    auto now = q.front();
    q.pop();
    int id = now.first;
    int hop = now.second;

    if (collected[id] == 0) {
      collected[id] = 1;
      collected_set_id[collected_number] = id;
      std::memcpy(collected_set_matrix + collected_number * D, X + id * D,
                  D * sizeof(int));
      collected_number++;
    }

    if (hop + 1 <= max_hop) {
      // vertices set in a hop
      int degree = edges[id * (L + 1)];
      for (int i = 1; i <= degree; ++i) {
        int v = edges[id * (L + 1) + i];
        q.push(std::make_pair(v, hop + 1));
      }
    }
  }

  // GPU computation
  int *cpu_c = new int[collected_number * D];
  int *a, *b, *c;

  int *query_extend = new int[collected_number * D];
  for (int i = 0; i < collected_number; i++) {
    std::memcpy(query_extend + i * D, query_data, D * sizeof(int));
  }

  cudaMalloc((void **)&a, sizeof(int) * collected_number * D);
  cudaMalloc((void **)&b, sizeof(int) * collected_number * D);
  cudaMalloc((void **)&c, sizeof(int) * collected_number * D);

  cudaMemcpy(a, collected_set_matrix, sizeof(int) * collected_number * D,
             cudaMemcpyHostToDevice);
  cudaMemcpy(b, query_extend, sizeof(int) * collected_number * D,
             cudaMemcpyHostToDevice);

  int block_size = 256;
  int grid_size = ceil(collected_number * D / block_size) + 1;
  gpu_l2_distance<<<grid_size, block_size>>>(a, b, c, collected_number, D);

  cudaMemcpy(cpu_c, c, sizeof(int) * collected_number * D,
             cudaMemcpyDeviceToHost);
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);

  // Return the min id
  // std::cout << "Square distance:" << std::endl;
  for (int i = 0; i < collected_number; i++) {
    int sum = 0;
    for (int j = 0; j < D; ++j) {
      sum += cpu_c[i * D + j];
    }
    // std::cout << sum << std::endl;
    if (sum < min_l2 || (sum == min_l2 && collected_set_id[i] < min_id)) {
      min_l2 = sum;
      min_id = collected_set_id[i];
    }
  }
  return min_id;
}

int main(int argc, char **argv) {
  FILE *fin = fopen(argv[1], "r");
  FILE *fout = fopen(argv[2], "w");
  fscanf(fin, "%d%d%d%d%d%d%d%d%d%d", &V, &D, &E, &L, &K, &A, &B, &C, &M, &Q);
  // X[i]: randomly generated D-dim vectors for V vertices
  X = new int[V * D];
  for (int i = 0; i < K; ++i)
    fscanf(fin, "%d", &X[i]);
  for (int i = K; i < V * D; ++i)
    X[i] = ((long long)A * X[i - 1] + (long long)B * X[i - 2] + C) % M;
  // graph one-dimensional representation
  edges = new int[V * (L + 1)];
  for (int i = 0; i < V; ++i) {
    edges[i * (L + 1)] = 0;
  }
  for (int i = 0; i < E; ++i) {
    int u, v;
    fscanf(fin, "%d%d", &u, &v);
    int degree = edges[u * (L + 1)];
    edges[u * (L + 1) + degree + 1] = v;
    ++edges[u * (L + 1)];
  }
  int *query_data = new int[D];
  collected = new int[V];

  // compute NN for Q queries
  for (int i = 0; i < Q; ++i) {
    int start_point, hop;
    fscanf(fin, "%d%d", &start_point, &hop);
    for (int i = 0; i < D; ++i) {
      fscanf(fin, "%d", &query_data[i]);
    }

    std::fill(collected, collected + V, 0);
    fprintf(fout, "%d\n", nearest_id(start_point, hop, query_data));
  }
  fclose(fin);
  fclose(fout);

  delete[] X;
  delete[] edges;
  delete[] query_data;
  delete[] collected;

  return 0;
}
