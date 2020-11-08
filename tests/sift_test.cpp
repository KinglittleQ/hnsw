#include "bruteforce.hpp"
#include "distance.hpp"
#include "hnsw.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <vector>

void read_fvecs(char *filename, float *&data, uint32_t &num_vectors, int &dim);
void read_ivecs(char *filename, int *&data, uint32_t &num_vectors, int &dim);

using namespace std::chrono;

using PointSet = std::vector<std::pair<uint32_t, float>>;

void GenerateRandomFloat(float *data, size_t num);
double ComputeRecall(uint32_t num, int K, const int *gt, const std::vector<PointSet> &points);

int main(int argc, char **argv) {
  float *base_data, *query_data;
  int *gt_data;
  uint32_t num_bases, num_queries, num_gts;
  int dim, K;

  assert(argc >= 4 && "Usage: sift_test dataset_path query_path groundtruth_path index_path");
  read_fvecs(argv[1], base_data, num_bases, dim);
  read_fvecs(argv[2], query_data, num_queries, dim);
  read_ivecs(argv[3], gt_data, num_gts, K);
  assert(num_gts == num_queries && "#GT must be equal to #queries");

  const uint32_t M = 32;
  const uint32_t ef_search = 256;
  const uint32_t ef_construction = 40;

  hnsw::L2Distance distance(dim);
  hnsw::HNSWIndex hnsw_index(base_data, num_bases, dim, distance, M, ef_construction);

  if (argc == 5) {
    hnsw_index.LoadIndex(argv[4]);
  } else {
    // Test build
    auto t0 = steady_clock::now();
    hnsw_index.Build();
    auto t1 = steady_clock::now();
    auto duration = duration_cast<milliseconds>(t1 - t0).count();
    printf("HNSW Build time: %ldms\n", duration);
    printf("Top layer: %d\n", hnsw_index.TopLayer());
    printf("Distance calculations: %ld\n", distance.num);

    hnsw_index.SaveIndex("index.bin");
  }

  printf("Searching ...\n");
  hnsw_index.SetEfSearch(ef_search);
  std::vector<PointSet> result(num_queries);
  auto t0 = steady_clock::now();
  for (uint32_t i = 0; i < num_queries; i++) {
    result[i] = hnsw_index.Search(K, &query_data[i * dim]);
  }
  auto t1 = steady_clock::now();
  auto duration = duration_cast<microseconds>(t1 - t0).count();
  printf("HNSW search speed: %ld µs/query\n", duration / num_queries);

  printf("Evaluating ...\n");
  double recall = ComputeRecall(num_queries, K, gt_data, result);
  printf("Recall@100: %f\n", recall);

  delete gt_data;
  delete base_data;
  delete query_data;

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

double ComputeRecall(uint32_t num, int K, const int *gt, const std::vector<PointSet> &points) {
  int recalls = 0;
  for (size_t i = 0; i < num; i++) {
    std::set<uint32_t> label;
    for (int j = 0; j < K; j++) {
      label.insert(gt[i * K + j]);
    }
    for (const auto &p : points[i]) {
      if (label.count(p.first) != 0) {
        recalls += 1;
      }
    }
  }

  return 1.0 * recalls / (num * K);
}

void read_fvecs(char *filename, float *&data, uint32_t &num_vectors, int &dim) {
  printf("Loading data from %s\n", filename);
  std::ifstream in(filename, std::ios::binary);

  if (!in.is_open()) {
    printf("Cannot open file: %s\n", filename);
    exit(1);
    return;
  }

  // Get #dimensions
  in.read((char *)&dim, 4);

  // Get file size
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;

  // fisze = (4 + 4d) * n
  num_vectors = fsize / (1 + dim) / 4;
  data = new float[num_vectors * dim];

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num_vectors; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char *)(data + i * dim), dim * 4);
  }
  in.close();

  printf("Data size: (%u, %d)\n", num_vectors, dim);
  return;
}

void read_ivecs(char *filename, int *&data, uint32_t &num_vectors, int &dim) {
  float *data_float;
  read_fvecs(filename, data_float, num_vectors, dim);
  data = reinterpret_cast<int *>(data_float);
  return;
}
