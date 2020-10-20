#ifndef __HMSW_DISTANCE_HPP__
#define __HMSW_DISTANCE_HPP__

#include <cmath>
#include <cassert>
#include <immintrin.h>

#define AVX_L2SQR(addr1, addr2, dest, tmp1, tmp2) \
    tmp1 = _mm256_loadu_ps(addr1); \
    tmp2 = _mm256_loadu_ps(addr2); \
    tmp1 = _mm256_sub_ps(tmp1, tmp2); \
    tmp1 = _mm256_mul_ps(tmp1, tmp1); \
    dest = _mm256_add_ps(dest, tmp1);

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

#ifndef __AVX__
    // Auto vectorized with -O3 flag
    T sum = 0;
    for (size_t i = 0; i < dim_; i++) {
      T tmp = p1[i] - p2[i];
      sum += tmp * tmp;
    }
    return sum;
#else
    assert(dim_ % 8 == 0 && "The data dimension must be mutiple of 8 to use ARX");
    float result = 0;

    __m256 sum;
    __m256 tmp0, tmp1;
    // unsigned D = (dim_ + 7) & ~7U;  // mutiple of 8
    unsigned residual_size = dim_ % 16;
    unsigned aligned_size = dim_ - residual_size;
    const float *residual_start1 = p1 + aligned_size;
    const float *residual_start2 = p2 + aligned_size;
    float unpack[8] __attribute__ ((aligned (32))) = {0, 0, 0, 0, 0, 0, 0, 0};

    sum = _mm256_loadu_ps(unpack);
    if ( residual_size != 0 ) {
      AVX_L2SQR(residual_start1, residual_start2, sum, tmp0, tmp1);
    }

    const float *ptr1 = p1;
    const float *ptr2 = p2;
    for (unsigned i = 0; i < aligned_size; i += 16, ptr1 += 16, ptr2 += 16) {
      AVX_L2SQR(ptr1, ptr2, sum, tmp0, tmp1);
      AVX_L2SQR(ptr1 + 8, ptr2 + 8, sum, tmp0, tmp1);
    }
    _mm256_storeu_ps(unpack, sum);
    result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + \
             unpack[4] + unpack[5] + unpack[6] + unpack[7];

    return result;
#endif
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