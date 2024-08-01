#include <cmath>
#include <omp.h>
#include <stdexcept>
#include <vector>

class Tensor {
public:
  std::vector<float> data;
  std::vector<size_t> dims;

  Tensor(std::vector<size_t> dims) : dims(dims) {
    size_t len = 1;
    for (auto d : dims)
      len *= d;
    data.resize(len);
  }

  Tensor(std::vector<size_t> dims, std::vector<std::vector<size_t>> idx, std::vector<float> val) : dims(dims) {
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
#pragma omp parallel for
      for (size_t i = 0; i < dims[0]; ++i) {
        for (size_t j = 0; j < dims[1]; ++j) {
          ret.data[ret.index({j, i})] = data[index({i, j})];
        }
      }
      return ret;
    } else if (dims.size() == 3) {
      Tensor ret({dims[0], dims[2], dims[1]});
#pragma omp parallel for
      for (size_t b = 0; b < dims[0]; ++b) {
        for (size_t i = 0; i < dims[1]; ++i) {
          for (size_t j = 0; j < dims[2]; ++j) {
            ret.data[ret.index({b, j, i})] = data[index({b, i, j})];
          }
        }
      }
      return ret;
    } else {
      throw std::runtime_error("The tensor must be 2D or batched 2D tensors");
    }
  }

  Tensor neg() {
    Tensor ret(dims);
#pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i)
      ret.data[i] = -data[i];
    return ret;
  }

  Tensor reciprocal() {
    Tensor ret(dims);
#pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i)
      ret.data[i] = 1.0 / data[i];
    return ret;
  }

  Tensor add(Tensor x) {
    if (dims != x.dims)
      throw std::runtime_error("Mismatched shape in add");
    Tensor ret(dims);
#pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i)
      ret.data[i] = data[i] + x.data[i];
    return ret;
  }

  Tensor subtract(Tensor x) {
    if (dims != x.dims)
      throw std::runtime_error("Mismatched shape in subtract");
    return add(x.neg());
  }

  Tensor mult(float x) {
    Tensor ret(dims);
#pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i)
      ret.data[i] = data[i] * x;
    return ret;
  }

  Tensor elementwise_mult(Tensor x) {
    if (dims != x.dims)
      throw std::runtime_error("Mismatched shape in elementwise_mult");
    Tensor ret(dims);
#pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i)
      ret.data[i] = data[i] * x.data[i];
    return ret;
  }

  Tensor pow(float x) {
    Tensor ret(dims);
#pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i)
      ret.data[i] = std::pow(data[i], x);
    return ret;
  }

  Tensor relu() {
    Tensor ret(dims);
#pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i)
      ret.data[i] = data[i] > 0 ? data[i] : 0;
    return ret;
  }

  Tensor binarilize() {
    Tensor ret(dims);
#pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i)
      ret.data[i] = data[i] > 0 ? 1 : 0;
    return ret;
  }

  Tensor exp() {
    Tensor ret(dims);
#pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i)
      ret.data[i] = std::exp(data[i]);
    return ret;
  }

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
#pragma omp parallel for
      for (size_t i = 0; i < dims[0]; ++i) {
        for (size_t j = 0; j < x.dims[1]; ++j) {
          for (size_t k = 0; k < dims[1]; ++k) {
            ret.data[ret.index({i, j})] +=
                data[index({i, k})] * x.data[x.index({k, j})];
          }
        }
      }
      return ret;
    } else {
      Tensor ret({dims[0], dims[1], x.dims[1]});
#pragma omp parallel for
      for (size_t b = 0; b < dims[0]; ++b) {
        for (size_t i = 0; i < dims[1]; ++i) {
          for (size_t j = 0; j < x.dims[1]; ++j) {
            for (size_t k = 0; k < dims[2]; ++k) {
              ret.data[ret.index({b, i, j})] +=
                  data[index({b, i, k})] * x.data[x.index({k, j})];
            }
          }
        }
      }
      return ret;
    }
  }

  void print() {
    for (auto x : data)
      printf("%s\n", std::to_string(x).c_str());
  }

  std::vector<float> get_data() { return data; }

  std::vector<size_t> get_dims() { return dims; }
};
