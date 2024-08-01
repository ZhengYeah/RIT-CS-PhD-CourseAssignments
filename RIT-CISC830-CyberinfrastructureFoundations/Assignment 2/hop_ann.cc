#include <iostream>
#include <limits>
#include <queue>
#include <stdio.h>

using namespace std;

int V, D, E, L, K, A, B, C, M, Q;
int *X;
int *edges;

int squared_l2_dist(int *x, int *y, int D) {
  int sum2 = 0;
  for (int i = 0; i < D; ++i)
    sum2 += (x[i] - y[i]) * (x[i] - y[i]);
  return sum2;
}

int nearest_id(int start_point, int max_hop, int *query_data) {
  std::queue<std::pair<int, int>> q;
  q.push(std::make_pair(start_point, 0));
  int min_d = std::numeric_limits<int>::max();
  int min_id = -1;
  cout << "Square distance:" << endl;
  while (!q.empty()) {
    auto now = q.front();
    q.pop();
    int id = now.first;
    int hop = now.second;
    // cout << id << endl;
    int d = squared_l2_dist(X + id * D, query_data, D);
    cout << d << endl;
    if ((d < min_d) || (d == min_d && id < min_id)) {
      min_d = d;
      min_id = id;
    }
    if (hop + 1 <= max_hop) {
      int degree = edges[id * (L + 1)];
      for (int i = 1; i <= degree; ++i) {
        int v = edges[id * (L + 1) + i];
        q.push(std::make_pair(v, hop + 1));
      }
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
  for (int i = 0; i < Q; ++i) {
    int start_point, hop;
    fscanf(fin, "%d%d", &start_point, &hop);
    for (int i = 0; i < D; ++i) {
      fscanf(fin, "%d", &query_data[i]);
    }
    fprintf(fout, "%d\n", nearest_id(start_point, hop, query_data));
  }
  fclose(fin);
  fclose(fout);

  delete[] X;
  delete[] edges;
  delete[] query_data;

  return 0;
}
