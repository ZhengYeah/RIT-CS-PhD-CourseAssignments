# g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) hw3tensor.cc -o hw3tensor$(python3-config --extension-suffix)

nvcc --compiler-options=-O3,-fopenmp,-Wall,-shared,-march=native,-std=c++11,-fPIC $(python3 -m pybind11 --includes) hw3tensor.cu -o hw3tensor$(python3-config --extension-suffix) -Xptxas -O3 -arch=sm_52



# test files

# g++ -fopenmp -O3 -march=native -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) hw3tensor.cc -o hw3tensor_omp_float$(python3-config --extension-suffix)

# nvcc -g --compiler-options=-fopenmp,-Wall,-march=native,-fPIC -G tensor_omp_cuda_test.cu -o tensor_omp_cuda 
