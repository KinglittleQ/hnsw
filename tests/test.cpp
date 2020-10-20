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
  uint32_t dim = 8;  // mutiple of 8
  uint32_t n_points = 100000;
  uint32_t n_queries = 10;
  float *points = new float[dim * n_points];
  GenerateRandomFloat(points, dim * n_points);

  uint32_t M = 16;
  uint32_t ef = 1;
  uint32_t ef_construction = 200;

  hnsw::L2Distance<float> distance(dim);
  hnsw::HNSWIndex<float> hnsw_index(points, n_points, dim, distance, M, ef, ef_construction);
  hnsw::BruteForceIndex<float> bf_index(points, n_points, dim, distance);

  // Test build
  auto t0 = std::chrono::steady_clock::now();
  hnsw_index.Build();
  auto t1 = std::chrono::steady_clock::now();
  auto duration = duration_cast<milliseconds>(t1 - t0).count();
  cout << "HNSW Build time: "<< duration << "ms" << endl;
  cout << "Top layer: " << hnsw_index.TopLayer() << endl;

  float *queries = new float[dim * n_queries];
  GenerateRandomFloat(queries, dim * n_queries);

  std::vector<PointSet> result1(n_queries);
  t0 = std::chrono::steady_clock::now();
  for (uint32_t i = 0; i < n_queries; i++) {
    result1[i] = hnsw_index.Search(100, &queries[i * dim]);
  }
  t1 = std::chrono::steady_clock::now();
  duration = duration_cast<microseconds>(t1 - t0).count();
  cout << "Elapsed time: "<< duration << "µs" << endl;

  std::vector<PointSet> result2(n_queries);
  t0 = std::chrono::steady_clock::now();
  for (uint32_t i = 0; i < n_queries; i++) {
    result2[i] = bf_index.Search(100, &queries[i * dim]);
  }
  t1 = std::chrono::steady_clock::now();
  duration = duration_cast<microseconds>(t1 - t0).count();
  cout << "Elapsed time: "<< duration << "µs" << endl;

  cout << "Recall: " << ComputeRecall(result2, result1) << endl;

  delete points;

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