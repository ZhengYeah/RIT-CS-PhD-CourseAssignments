// Add features:
// 1. CUDA computation: matmul, transpose
// 2. OMP computation: others

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <stdexcept>
#include <vector>
#include <cuda_runtime_api.h>



__global__ void matmul_cuda(double *ret, double *a, double *b, size_t a_row, size_t b_col, size_t k_tmp) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  double tmp_sum = 0;
  if ((row < a_row) && (col < b_col)) {
    for (size_t k = 0; k < k_tmp; ++k) {
      tmp_sum += a[row * k_tmp + k] * b[k * b_col + col];
    }
    ret[row * b_col + col] = tmp_sum;
    // for (int i = 0; i < a_row * b_col; ++i) {
    //   printf("%f\n", ret[i]);
    // }
  }
}
__global__ void matmul_cuda_batch(double *ret, double *a, double *b, size_t batch, size_t a_row, size_t b_col, size_t k_tmp) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int bat = blockIdx.z;
  double tmp_sum = 0;
  int a_size = a_row * k_tmp;
  if ((row < a_row) && (col < b_col)) {
      for (int k = 0; k < k_tmp; ++k) {
        tmp_sum += a[bat * a_size + row * k_tmp + k] * b[k * b_col + col];
      }
      ret[bat * a_row * b_col + row * b_col + col] = tmp_sum;
    }
}
__global__ void transpose_cuda(double *ret, double *a, int a_row, int a_col) {
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < a_col && idy < a_row) {
    int pos = idy * a_col + idx;
    int trans_pos = idx * a_row + idy;
    ret[trans_pos] = a[pos];
  }
}

__global__ void transpose_cuda_batch(double *ret, double *a, int batch, int a_row, int a_col) {
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int bat = blockIdx.z;
  int a_size = a_row * a_col;
  if (idx < a_col && idy < a_row) {
    int pos = (bat * a_size) + idy * a_col + idx;
    int trans_pos = (bat * a_size) + idx * a_row + idy;
    ret[trans_pos] = a[pos];
  }
}

class Tensor {
public:
  std::vector<double> data;
  std::vector<size_t> dims;

  Tensor(std::vector<size_t> dims) : dims(dims) {
    size_t len = 1;
    for (auto d : dims)
      len *= d;
    data.resize(len);
  }

  Tensor(std::vector<size_t> dims, std::vector<std::vector<size_t>> idx,
         std::vector<double> val)
      : dims(dims) {
    size_t len = 1;
    for (auto d : dims)
      len *= d;
    data.resize(len);
    if (idx.size() != val.size())
      throw std::runtime_error("Mismatched idx and val size");
#pragma omp parallel for
    for (size_t i = 0; i < idx.size(); ++i) {
      data[index(idx[i])] = val[i];
    }
  }

  // better opm
  static Tensor ones(std::vector<size_t> dims) {
    Tensor ret(dims);
#pragma omp parallel for
    for (size_t i = 0; i < ret.data.size(); ++i)
      ret.data[i] = 1;
    return ret;
  }

  size_t index(std::vector<size_t> x) {
    if (x.size() != dims.size())
      throw std::runtime_error("Mismatched dims in index");
    size_t ret = 0;
    size_t prod = 1;
    for (int i = dims.size() - 1; i >= 0; --i) {
      if (x[i] >= dims[i])
        throw std::runtime_error("Index out of bound");
      ret += x[i] * prod;
      prod *= dims[i];
    }
    return ret;
  }

  Tensor reshape(std::vector<size_t> new_dims) {
    size_t len = 1;
    for (auto d : new_dims)
      len *= d;
    if (len != data.size())
      throw std::runtime_error("Mismatched dims in reshape");
    Tensor ret(new_dims);
    ret.data = data;
    return ret;
  }

  Tensor transpose() {
    if (dims.size() == 2) {
      Tensor ret({dims[1], dims[0]});
      double *ret_cuda, *data_cuda;

      cudaMalloc((void **)&ret_cuda, sizeof(double) * data.size());
      cudaMalloc((void **)&data_cuda, sizeof(double) * data.size());

      double *data_array = &data[0];
      cudaMemcpy(data_cuda, data_array, sizeof(double) * data.size(), cudaMemcpyHostToDevice);

      int BLOCK_SIZE = 16;
      int grid_row = (int)ceil(dims[0] / BLOCK_SIZE) + 1;
      int grid_col = (int)ceil(dims[1] / BLOCK_SIZE) + 1;
      dim3 dimGrid(grid_col, grid_row);
      dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

      transpose_cuda<<<dimGrid, dimBlock>>>(ret_cuda, data_cuda, dims[0], dims[1]);

      double *ret_array = &ret.data[0];
      cudaMemcpy(ret_array, ret_cuda, sizeof(double) * dims[0] * dims[1], cudaMemcpyDeviceToHost);
      cudaFree(ret_cuda);
      cudaFree(data_cuda);

      return ret;
    } else if (dims.size() == 3) {
      Tensor ret({dims[0], dims[2], dims[1]});
      double *ret_cuda, *data_cuda;

      cudaMalloc((void **)&ret_cuda, sizeof(double) * data.size());
      cudaMalloc((void **)&data_cuda, sizeof(double) * data.size());

      double *data_array = &data[0];
      cudaMemcpy(data_cuda, data_array, sizeof(double) * data.size(), cudaMemcpyHostToDevice);

      int BLOCK_SIZE = 16;
      int grid_row = (int)ceil(dims[1] / BLOCK_SIZE) + 1;
      int grid_col = (int)ceil(dims[2] / BLOCK_SIZE) + 1;
      int grid_batch = dims[0];
      dim3 dimGrid(grid_col, grid_row, grid_batch);
      dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

      transpose_cuda_batch<<<dimGrid, dimBlock>>>(ret_cuda, data_cuda, dims[0], dims[1], dims[2]);

      double *ret_array = &ret.data[0];
      cudaMemcpy(ret_array, ret_cuda, sizeof(double) * data.size(), cudaMemcpyDeviceToHost);
      cudaFree(ret_cuda);
      cudaFree(data_cuda);

      return ret;
    } else {
      throw std::runtime_error("The tensor must be 2D or batched 2D tensors");
    }
  }

  // better cuda
  // Tensor transpose() {
  //   if (dims.size() == 2) {
  //     Tensor ret({dims[1], dims[0]});
  //     for (size_t i = 0; i < dims[0]; ++i) {
  //       for (size_t j = 0; j < dims[1]; ++j) {
  //         ret.data[ret.index({j, i})] = data[index({i, j})];
  //       }
  //     }
  //     return ret;
  //   } else if (dims.size() == 3) {
  //     Tensor ret({dims[0], dims[2], dims[1]});
  //     for (size_t b = 0; b < dims[0]; ++b) {
  //       for (size_t i = 0; i < dims[1]; ++i) {
  //         for (size_t j = 0; j < dims[2]; ++j) {
  //           ret.data[ret.index({b, j, i})] = data[index({b, i, j})];
  //         }
  //       }
  //     }
  //     return ret;
  //   } else {
  //     throw std::runtime_error("The tensor must be 2D or batched 2D tensors");
  //   }
  // }

  // better opm
  Tensor neg() {
    Tensor ret(dims);
#pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i)
      ret.data[i] = -data[i];
    return ret;
  }

  // better opm
  Tensor reciprocal() {
    Tensor ret(dims);
#pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i)
      ret.data[i] = 1.0 / data[i];
    return ret;
  }

  // better cuda
  Tensor add(Tensor x) {
    if (dims != x.dims)
      throw std::runtime_error("Mismatched shape in add");
    Tensor ret(dims);
#pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i)
      ret.data[i] = data[i] + x.data[i];
    return ret;
  }

  // better cuda
  Tensor subtract(Tensor x) {
    if (dims != x.dims)
      throw std::runtime_error("Mismatched shape in subtract");
    return add(x.neg());
  }

  // better cuda
  Tensor mult(double x) {
    Tensor ret(dims);
#pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i)
      ret.data[i] = data[i] * x;
    return ret;
  }

  // better cuda
  Tensor elementwise_mult(Tensor x) {
    if (dims != x.dims)
      throw std::runtime_error("Mismatched shape in elementwise_mult");
    Tensor ret(dims);
#pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i)
      ret.data[i] = data[i] * x.data[i];
    return ret;
  }

  // better opm
  Tensor pow(double x) {
    Tensor ret(dims);
#pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i)
      ret.data[i] = std::pow(data[i], x);
    return ret;
  }

  // better opm
  Tensor relu() {
    Tensor ret(dims);
#pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i)
      ret.data[i] = data[i] > 0 ? data[i] : 0;
    return ret;
  }

  // better opm
  Tensor binarilize() {
    Tensor ret(dims);
#pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i)
      ret.data[i] = data[i] > 0 ? 1 : 0;
    return ret;
  }

  // better opm
  Tensor exp() {
    Tensor ret(dims);
#pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i)
      ret.data[i] = std::exp(data[i]);
    return ret;
  }

  // better cuda
  Tensor matmul(Tensor x) {
    if (x.dims.size() != 2) {
      throw std::runtime_error(
          "The right operand of matmul must be 2D tensors");
    }
    if (dims.size() != 2 && dims.size() != 3) {
      throw std::runtime_error("The left operand of matmul must be 2D tensors "
                               "or batched 2D tensors");
    }
    if (dims[dims.size() - 1] != x.dims[0]) {
      throw std::runtime_error("Mismatched matmul matrix dimentions");
    }
    if (dims.size() == 2) {
      Tensor ret({dims[0], x.dims[1]});
      double *ret_cuda, *data_cuda, *x_cuda;

      cudaMalloc((void **)&ret_cuda, sizeof(double) * dims[0] * x.dims[1]);
      cudaMalloc((void **)&data_cuda, sizeof(double) * dims[0] * dims[1]);
      cudaMalloc((void **)&x_cuda, sizeof(double) * x.dims[0] * x.dims[1]);

      double *data_array = &data[0];
      double *x_array = &x.data[0];
      cudaMemcpy(data_cuda, data_array, sizeof(double) * data.size(), cudaMemcpyHostToDevice);
      cudaMemcpy(x_cuda, x_array, sizeof(double) * x.data.size(), cudaMemcpyHostToDevice);

      int BLOCK_SIZE = 16;
      int grid_row = (int)ceil(dims[0] / BLOCK_SIZE) + 1;
      int grid_col = (int)ceil(x.dims[1] / BLOCK_SIZE) + 1;
      dim3 dimGrid(grid_col, grid_row);
      dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

      matmul_cuda<<<dimGrid, dimBlock>>>(ret_cuda, data_cuda, x_cuda, dims[0], x.dims[1], dims[1]);

      double *ret_array = &ret.data[0];
      cudaMemcpy(ret_array, ret_cuda, sizeof(double) * dims[0] * x.dims[1], cudaMemcpyDeviceToHost);
      cudaFree(ret_cuda);
      cudaFree(data_cuda);
      cudaFree(x_cuda);

      return ret;
    } else {
      Tensor ret({dims[0], dims[1], x.dims[1]});
      double *ret_cuda, *data_cuda, *x_cuda;

      cudaMalloc((void **)&ret_cuda, sizeof(double) * dims[0] * dims[1] * x.dims[1]);
      cudaMalloc((void **)&data_cuda, sizeof(double) * data.size());
      cudaMalloc((void **)&x_cuda, sizeof(double) * x.data.size());

      double *data_array = &data[0];
      double *x_array = &x.data[0];
      cudaMemcpy(data_cuda, data_array, sizeof(double) * data.size(), cudaMemcpyHostToDevice);
      cudaMemcpy(x_cuda, x_array, sizeof(double) * x.data.size(), cudaMemcpyHostToDevice);

      int BLOCK_SIZE = 16;
      int grid_row = (int)ceil(dims[1] / BLOCK_SIZE) + 1;
      int grid_col = (int)ceil(x.dims[1] / BLOCK_SIZE) + 1;
      int grid_batch = dims[0];
      dim3 dimGrid(grid_col, grid_row, grid_batch);
      dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

      matmul_cuda_batch<<<dimGrid, dimBlock>>>(ret_cuda, data_cuda, x_cuda, dims[0], dims[1], x.dims[1], dims[2]);

      double *ret_array = &ret.data[0];
      cudaMemcpy(ret_array, ret_cuda, sizeof(double) * dims[0] * dims[1] * x.dims[1], cudaMemcpyDeviceToHost);
      cudaFree(ret_cuda);
      cudaFree(data_cuda);
      cudaFree(x_cuda);

      return ret;
    }
  }

//   Tensor matmul(Tensor x) {
//     if (x.dims.size() != 2) {
//       throw std::runtime_error(
//           "The right operand of matmul must be 2D tensors");
//     }
//     if (dims.size() != 2 && dims.size() != 3) {
//       throw std::runtime_error("The left operand of matmul must be 2D tensors "
//                                "or batched 2D tensors");
//     }
//     if (dims[dims.size() - 1] != x.dims[0]) {
//       throw std::runtime_error("Mismatched matmul matrix dimentions");
//     }
//     if (dims.size() == 2) {
//       Tensor ret({dims[0], x.dims[1]});
// #pragma omp parallel for
//       for (size_t i = 0; i < dims[0]; ++i) {
//         for (size_t j = 0; j < x.dims[1]; ++j) {
//           for (size_t k = 0; k < dims[1]; ++k) {
//             ret.data[ret.index({i, j})] +=
//                 data[index({i, k})] * x.data[x.index({k, j})];
//           }
//         }
//       }
//       return ret;
//     } else {
//       Tensor ret({dims[0], dims[1], x.dims[1]});
// #pragma omp parallel for
//       for (size_t b = 0; b < dims[0]; ++b) {
//         for (size_t i = 0; i < dims[1]; ++i) {
//           for (size_t j = 0; j < x.dims[1]; ++j) {
//             for (size_t k = 0; k < dims[2]; ++k) {
//               ret.data[ret.index({b, i, j})] +=
//                   data[index({b, i, k})] * x.data[x.index({k, j})];
//             }
//           }
//         }
//       }
//       return ret;
//     }
//   }

  void print() {
    for (auto x : data)
      printf("%s\n", std::to_string(x).c_str());
  }

  std::vector<double> get_data() { return data; }

  std::vector<size_t> get_dims() { return dims; }
};



// Test code

// double rand_mod() {
//   return (double)rand() / (double)RAND_MAX;
// }
// int main() {
//   unsigned long N = 3, M = 5, P = 7;
//   Tensor x({P, N, M}), y({M,P}), z({P, 1});
//   srand(1);
//   std::generate(x.data.begin(), x.data.end(), rand_mod);
//   std::generate(y.data.begin(), y.data.end(), rand_mod);
//   std::generate(z.data.begin(), z.data.end(), rand_mod);
//   Tensor ret({y.dims[0], z.dims[1]});
//   // ret = x.matmul(y).relu().matmul(z);
//   ret = x.transpose();
//   ret.print();

//   Tensor ret_test({y.dims[0], z.dims[1]});
//   // ret = x.matmul(y).relu().matmul(z);
//   ret_test = x.transpose_test();
//   ret_test.print();
//   return 0;
// }
