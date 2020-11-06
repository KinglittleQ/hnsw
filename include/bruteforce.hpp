#ifndef __HNSW_BRUTEFORCE_HPP__
#define __HNSW_BRUTEFORCE_HPP__

#include "distance.hpp"
#include "index.hpp"
#include "matrix.hpp"

#include <queue>

namespace hnsw {

class BruteForceIndex : public Index {
  using MaxPointHeap = std::priority_queue<Point, PointSet, PointLessComparator>;

public:
  BruteForceIndex(const float *data, uint32_t n_points, uint32_t dim, const Distance &distance)
      : points_(data, n_points, dim), num_points_(n_points), dim_(dim), distance_(distance) {}

  ~BruteForceIndex() = default;

  void Build() {}

  PointSet Search(uint32_t K, const float *query) {
    MaxPointHeap result;
    for (uint32_t i = 0; i < num_points_; i++) {
      float dist = distance_(query, points_[i]);
      result.emplace(i, dist);
      if (result.size() > K) {
        result.pop();
      }
    }

    PointSet neighbors;
    while (!result.empty()) {
      neighbors.push_back(result.top());
      result.pop();
    }
    return neighbors;
  }

private:
  Matrix points_;
  uint32_t num_points_;
  uint32_t dim_;
  const Distance &distance_;
};

}  // namespace hnsw

#endif