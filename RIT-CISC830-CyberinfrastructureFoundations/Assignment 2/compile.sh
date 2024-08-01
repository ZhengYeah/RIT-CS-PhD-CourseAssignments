# g++ hop_ann.cc -o hop_ann -std=c++11
nvcc -O3 -Xcompiler -march=native hop_nn_cuda.cu -o hop_nn_cuda -Xptxas -O3