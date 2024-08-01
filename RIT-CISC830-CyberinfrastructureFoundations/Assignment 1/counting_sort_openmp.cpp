// Parallel counting sort (rank sort) using OpenMP
// Author: Ye Zheng 
#include <cstdlib>
#include <iostream>
#include <memory>
#include <time.h>
#include <omp.h>

using namespace std;

// initialize static sorting array
const int SIZE = 1e8;
int X[SIZE];


void counting_sort(int* start, int* end) {
	int max_value = 0;
	int* count;
	int index = 0;
	int length = end - start;
	
#pragma omp parallel for reduction(max: max_value)
	for (int i = 0; i < length; i++) {
		max_value = X[i] > max_value ? X[i] : max_value;
	}
	count = new int[max_value + 1];
	fill(count, count + max_value + 1, 0);

#pragma omp parallel for
	for (int i = 0; i < length; i++) {
#pragma omp atomic
		count[X[i]]++;
	}

	// sequential dependency on index
	for (int i = 0; i <= max_value; i++) {
		for (int j = 1; j <= count[i]; j++) {
			X[index++] = i;
		}
	}
}


int main(int argc, char** argv) {
	int thread_num = omp_get_max_threads();
	omp_set_num_threads(thread_num);

	// clock_t time_start;

	int N, K, A, B, C, M;
	FILE* fin = fopen(argv[1], "r");
	fscanf(fin, "%d%d%d%d%d%d", &N, &K, &A, &B, &C, &M);
	for (int i = 0; i < K; ++i)
		fscanf(fin, "%d", &X[i]);
	fclose(fin);

	for (int i = K; i < N; ++i)
		X[i] = ((long long)A * X[i - 1] + (long long)B * X[i - 2] + C) % M;

	// time_start = clock();
	// // std::sort(X, X + N);
	counting_sort(X, X + N);
	// clock_t clock_elapsed = clock() - time_start;
	// double time_elapsed = clock_elapsed / (double) CLOCKS_PER_SEC;
	// cout << "Time elapsed: " << time_elapsed << " seconds." << endl;

	FILE* fout = fopen(argv[2], "w");
	for (int i = 0; i < N; ++i)
		fprintf(fout, "%d\n", X[i]);
	fclose(fout);
	return 0;
}
