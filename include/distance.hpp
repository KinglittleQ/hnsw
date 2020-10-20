#ifndef __HMSW_DISTANCE_HPP__
#define __HMSW_DISTANCE_HPP__

#include <cmath>
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
    T sum = 0;
    for (size_t i = 0; i < dim_; i++) {
      sum += (p1[i] - p2[i]) * (p1[i] - p2[i]);
    }
    return sqrt(sum);
#else
    float result = 0;

    __m256 sum;
    __m256 l0, l1;
    __m256 r0, r1;
    unsigned D = (dim_ + 7) & ~7U;  // mutiple of 8
    unsigned DR = D % 16;
    unsigned DD = D - DR;
    const float *l = p1;
    const float *r = p2;
    const float *e_l = l + DD;
    const float *e_r = r + DD;
    float unpack[8] __attribute__ ((aligned (32))) = {0, 0, 0, 0, 0, 0, 0, 0};

    sum = _mm256_loadu_ps(unpack);
    if ( DR ) { AVX_L2SQR(e_l, e_r, sum, l0, r0); }

    for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16) {
      AVX_L2SQR(l, r, sum, l0, r0);
      AVX_L2SQR(l + 8, r + 8, sum, l1, r1);
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