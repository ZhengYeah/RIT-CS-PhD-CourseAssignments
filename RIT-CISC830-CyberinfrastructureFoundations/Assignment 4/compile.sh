# g++ -march=native -O3 -fopenmp discretelog.cc -o discretelog 

nvcc --compiler-options=-O3,-fopenmp,-Wall,-march=native,-std=c++11,-fPIC discretelog_bsgs_cuda.cu -o discretelog_bsgs_cuda -Xptxas -O3 -arch=sm_52