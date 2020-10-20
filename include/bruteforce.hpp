#ifndef __HMSW_BRUTEFORCE_HPP__
#define __HMSW_BRUTEFORCE_HPP__

#include "matrix.hpp"
#include "index.hpp"
#include "distance.hpp"

#include <queue>

namespace hnsw {

template <typename T>
class BruteForceIndex : public Index<T> {
  using typename Index<T>::Point;
  using typename Index<T>::PointSet;
  using typename Index<T>::PointLessComparator;
  using MaxPointHeap = std::priority_queue<Point, PointSet, PointLessComparator>;

public:
  BruteForceIndex(const T *data, uint32_t n_points, uint32_t dim, const Distance<T> &distance) :
                  points_(data, n_points, dim), num_points_(n_points),
                  dim_(dim), distance_(distance) {}

  ~BruteForceIndex() = default;

  void Build() {}

  PointSet Search(uint32_t K, const T *query) {
    MaxPointHeap result;
    for (uint32_t i = 0; i < num_points_; i++) {
      T dist = distance_(query, points_[i]);
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
  Matrix<T> points_;
  uint32_t num_points_;
  uint32_t dim_;
  const Distance<T> &distance_;
};

}  // end hnsw

#endif