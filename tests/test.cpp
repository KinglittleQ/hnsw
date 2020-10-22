#include <iostream>
#include <random>
#include <chrono>
#include <vector>
#include <set>

#include "hnsw.hpp"
#include "bruteforce.hpp"
#include "distance.hpp"

using std::cout;
using std::endl;
using namespace std::chrono;

using PointSet = std::vector<std::pair<uint32_t, float>>;

void GenerateRandomFloat(float *data, size_t num);
double ComputeRecall(const std::vector<PointSet> &groundtruth,
                     const std::vector<PointSet> &points);

int main(void) {
  const uint32_t dim = 8;
  const uint32_t n_points = 100000;
  const uint32_t n_queries = 1000;
  float *points = new float[dim * n_points];
  GenerateRandomFloat(points, dim * n_points);

  const uint32_t M = 16;
  const uint32_t ef = 10;
  const uint32_t ef_construction = 50;
  const uint32_t K = 100;

  hnsw::L2Distance distance(dim);
  hnsw::HNSWIndex hnsw_index(points, n_points, dim, distance, M, ef, ef_construction);
  hnsw::BruteForceIndex bf_index(points, n_points, dim, distance);

  // Test build
  auto t0 = std::chrono::steady_clock::now();
  hnsw_index.Build();
  auto t1 = std::chrono::steady_clock::now();
  auto duration = duration_cast<milliseconds>(t1 - t0).count();
  cout << "HNSW Build time: "<< duration << "ms" << endl;
  cout << "Top layer: " << hnsw_index.TopLayer() << endl;
  cout << "Distance calculations: " << distance.num << endl;

  float *queries = new float[dim * n_queries];
  GenerateRandomFloat(queries, dim * n_queries);

  hnsw_index.SetEfConstruction(256);

  std::vector<PointSet> result1(n_queries);
  t0 = std::chrono::steady_clock::now();
  for (uint32_t i = 0; i < n_queries; i++) {
    result1[i] = hnsw_index.Search(K, &queries[i * dim]);
  }
  t1 = std::chrono::steady_clock::now();
  duration = duration_cast<microseconds>(t1 - t0).count();
  cout << "HNSW: " << duration / n_queries << " µs/query" << endl;

  std::vector<PointSet> result2(n_queries);
  t0 = std::chrono::steady_clock::now();
  for (uint32_t i = 0; i < n_queries; i++) {
    result2[i] = bf_index.Search(K, &queries[i * dim]);
  }
  t1 = std::chrono::steady_clock::now();
  duration = duration_cast<microseconds>(t1 - t0).count();
  cout << "BruteForce: "<< duration / n_queries << " µs/query" << endl;

  cout << "Recall@100: " << ComputeRecall(result2, result1) << endl;

  delete points;
  delete queries;

  return 0;
}

void GenerateRandomFloat(float *data, size_t num) {
  std::random_device rd;
  std::mt19937 e(rd());
  std::uniform_real_distribution<> dist(0, 10);

  for (size_t i = 0; i < num; i++) {
    data[i] = dist(e);
  }
}

double ComputeRecall(const std::vector<PointSet> &groundtruth,
                     const std::vector<PointSet> &points) {

  int num = 0;
  for  (size_t i = 0; i < groundtruth.size(); i++) {
    std::set<uint32_t> label;
    for (const auto &p : groundtruth[i]) {
      label.insert(p.first);
    }
    for (const auto &p : points[i]) {
      if (label.count(p.first) != 0) {
        num += 1;
      }
    }
  }

  return 1.0 * num / (groundtruth.size() * groundtruth[0].size());
}