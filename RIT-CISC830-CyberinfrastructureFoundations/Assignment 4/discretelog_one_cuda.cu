// Enumberate x to compute A^x % M using Binary Exponentiation

#include <cstdio>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <omp.h>


__device__ int stop_flag = 0;

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

__global__ void discretelog_cuda(long long base, long long B, long long M, long long *result) {
  long long id = blockIdx.x * blockDim.x + threadIdx.x;
  if (stop_flag == 1) {
    return;
  }
  if (id < M) {
    if (fast_power(base, id, M) == B) {
      *result = id;
      stop_flag = 1;
    }
    __syncthreads();
  }  
}

int main(int argc, char **argv) {
  long long A = 57116, B = 3206248404, M = 20184892661;

  long long *result_cpu = (long long*) malloc(sizeof(long long)); 
  long long *result_cuda;

  cudaMalloc((void **)&result_cuda, sizeof(long long));

  int BLOCK_SIZE = 1024;
  long long gridSize = (long long) M / BLOCK_SIZE + 1;

  discretelog_cuda<<<gridSize, BLOCK_SIZE>>>(A, B, M, result_cuda);


  cudaMemcpy(result_cpu, result_cuda, sizeof(long long), cudaMemcpyDeviceToHost);
  if (result_cpu != 0) {
      printf("%lld\n", *result_cpu);
    }

  cudaFree(result_cuda);
  free(result_cpu);
  return 0;
}
