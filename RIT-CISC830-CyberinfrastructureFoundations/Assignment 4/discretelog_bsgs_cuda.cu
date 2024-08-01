#include <cmath>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <omp.h>
#include <stdio.h>

__device__ long long fast_power(long long base, long long exp, long long M) {
  __int128_t power_res = 1, base_tmp = base;
  while (exp > 0) {
    if (exp & 1) {
      power_res = power_res * base_tmp % M;
    }
    base_tmp = base_tmp * base_tmp % M;
    exp >>= 1;
  }
  return power_res;
}

__global__ void discretelog_cuda(long long base, long long B, long long M,
                                 long long n, long long *baby_val,
                                 long long *giant_val) {
  long long id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < n) {
    baby_val[id] = fast_power(base, id, M);
    // need to guarantee n*(M-2) not overflow
    long long step_inv = fast_power(base, n * (M - 2), M);
    giant_val[id] = ((__int128_t)B * fast_power(step_inv, id, M)) % M;
  }
}

int main(int argc, char **argv) {
  // long long A = 51514, B = 5722767456, M = 40929166261;

  long long A, B, M;
  FILE *fin = fopen(argv[1], "r");
  FILE *fout = fopen(argv[2], "w");
  fscanf(fin, "%lld%lld%lld", &A, &B, &M);

  long long n = (long long)ceil(sqrt((double)M));
  long long *baby_val, *giant_val;
  cudaMallocManaged((void **)&baby_val, sizeof(long long) * n);
  cudaMallocManaged((void **)&giant_val, sizeof(long long) * n);

  int BLOCK_SIZE = 1024;
  long long gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  discretelog_cuda<<<gridSize, BLOCK_SIZE>>>(A, B, M, n, baby_val, giant_val);
  cudaDeviceSynchronize();

  long long x = -1;
#pragma omp parallel for collapse(2)
  for (long long i = 0; i < n; i++) {
    for (long long j = 0; j < n; j++) {
      if (baby_val[i] == giant_val[j]) {
        x = i + j * n;
        fprintf(fout, "%lld\n", x);
      }
    }
  }
  if (x == -1) {
    printf("Solution not found.\n");
  }

  cudaFree(baby_val);
  cudaFree(giant_val);
  fclose(fin);
  fclose(fout);
  return 0;
}
