#ifndef __HMSW_INDEX_HPP__
#define __HMSW_INDEX_HPP__

#include <cstdint>
#include <vector>

namespace hnsw {

using index_t = uint32_t;

class Index {
protected:
  using Point = std::pair<index_t, float>;
  using PointSet = std::vector<Point>;

  struct PointLessComparator {
    bool operator() (const Point &a, const Point &b) {
      return a.second < b.second;
    }
  };

  struct PointGreaterComparator {
    bool operator() (const Point &a, const Point &b) {
      return a.second > b.second;
    }
  };

public:
  Index() = default;
  virtual void Build() = 0;
  virtual PointSet Search(uint32_t K, const float *query) = 0;
  virtual ~Index() = default;
};

} // end hnsw

#endif