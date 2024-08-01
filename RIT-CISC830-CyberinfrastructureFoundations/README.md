# RIT-CISC830-Cyberinfrastructure-Foundation-Assignments

Solutions for assignments of CISC-830 Cyberinfrastructure Foundations (2024 Spring).

# Contents

* Assignment 1: Integer sorting
  * Passes all test cases within the time limit.
  * Faster solution: $10^6$ numbers a thread using `std::sort, then merges all the threads. $O(\log n * \log n)$ time complexity.
  * The complexity of counting sort under multiple threads is limited by the `max()` and `write(index++)` processes.

* Assignment 2: Nearest neighbor on DAG
  * Passes the first 8 test cases in the time limit.
  * This assignment has a strict timeout limitation; there are two large test cases that easily time out.
  * For less running time, you need:
    * combine all the queries in one matrix;
    * and move the whole distance computation into GPU.    

* Assignment 3: Tensor Library
  * Passes all test cases within the time limit.
  * Faster solution: for a sequence of operation of each tensor in the Python script, maintain the last result in CUDA memory (no `cudafree()`), e.g. using a pair of "old-new" variables.

* Assignment 4: Discrete Logarithm
  * Passes all test cases within the time limit.
  * Faster solution: CPU BSGS using unordered container.
