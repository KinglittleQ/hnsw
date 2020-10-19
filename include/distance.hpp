#ifndef __HMSW_DISTANCE_HPP__
#define __HMSW_DISTANCE_HPP__

#include <cmath>

namespace hnsw {

template <typename T>
class Distance {
public:
  Distance(size_t dim) : dim_(dim) {}
  virtual T operator() (const T *p1, const T *p2) const = 0;
  virtual ~Distance() = default;

protected:
  size_t dim_;
};

template <typename T>
class L2Distance : public Distance<T> {
  using Distance<T>::dim_;

public:
  L2Distance(size_t dim) : Distance<T>(dim) {}
  T operator() (const T *p1, const T *p2) const {
    T sum = 0;
    for (size_t i = 0; i < dim_; i++) {
      sum += (p1[i] - p2[i]) * (p1[i] - p2[i]);
    }
    return sqrt(sum);
  }
};

template <typename T>
class L1Distance : public Distance<T> {
  using Distance<T>::dim_;

public:
  L1Distance(size_t dim) : Distance<T>(dim) {}
  T operator()(const T *p1, const T *p2) const {
    T sum = 0;
    for (size_t i = 0; i < dim_; i++) {
      sum += fabs(p1[i] - p2[i]);
    }
    return sum;
  }
};

}  // hnsw

#endif