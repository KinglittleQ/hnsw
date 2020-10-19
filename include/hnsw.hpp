#ifndef __HMSW_HNSW_HPP__
#define __HMSW_HNSW_HPP__

#include "distance.hpp"
#include "matrix.hpp"
#include "index.hpp"

#include <cassert>
#include <memory>
#include <cmath>
#include <vector>
#include <random>
#include <queue>
#include <unordered_map>
#include <set>
#include <iostream>

#define LOG_LEVEL ERROR_LEVEL
#include "logger.hpp"

namespace hnsw {

using std::vector;
using std::pair;
using std::priority_queue;
using std::unordered_map;
using std::set;

using layer_t = int32_t;


template<typename T>
class HNSWIndex : public Index<T> {
  using typename Index<T>::Point;
  using typename Index<T>::PointSet;
  using typename Index<T>::PointLessComparator;
  using typename Index<T>::PointGreaterComparator;
  using MaxPointHeap = priority_queue<Point, PointSet, PointLessComparator>;
  using MinPointHeap = priority_queue<Point, PointSet, PointGreaterComparator>;

  struct Vertex {
    vector<PointSet> neighbors;  // edges in layers
    layer_t layer = 0;  // maximumn layer

    void ConnectTo(const index_t neighbor, T distance, layer_t l) {
      LOG_ASSERT(l <= layer, "Too many layers");
      neighbors[l].emplace_back(neighbor, distance);
    }
  };

public:
  HNSWIndex(T *data, size_t n_points, size_t dim, const Distance<T> &distance,
            int M, int ef, int ef_construction) :
            points_(data, n_points, dim), distance_(distance) {

    top_layer_ = 0;
    M_ = M;
    maxM_ = M;
    maxM0_ = 2 * M;
    ml_ = 1 / log(M);
    ef_ = ef;
    ef_construction_ = std::max(ef_construction, M);
    dim_ = dim;

    vertices_.resize(n_points);
  }

  ~HNSWIndex() = default;

  void Build() {
    for (index_t i = 0; i < points_.Rows(); i++) {
      Insert(i);
    }
  }

  void Insert(index_t q) {
    Vertex &vertex = vertices_[q];
    LOG_ASSERT(vertex.layer == 0 && vertex.neighbors.size() == 0,
               "Vertex has already been inserted");
    vertex.layer = RandomChoiceLayer();
    vertex.neighbors.resize(vertex.layer + 1);

    LOG_INFO("Insert #%u, layer=%d", q, vertex.layer);

    num_points_ += 1;
    if (num_points_ == 1) {
      top_layer_ = vertex.layer;
      ep_ = q;
      return;
    }

    const T *query = points_[q];
    PointSet ep;
    ep.emplace_back(ep_, distance_(query, points_[ep_]));

    for (layer_t l = top_layer_; l > vertex.layer; l--) {
      MaxPointHeap candidates = SearchLayer(query, ep, ef_, l);
      ep = SelectNeighbors(query, candidates, 1);
    }

    // Search neighbors and connect to them
    for (layer_t l = std::min(vertex.layer, top_layer_); l >= 0; l--) {
      size_t maxM = (l == 0) ? maxM0_ : maxM_;
      MaxPointHeap candidates = SearchLayer(query, ep, ef_construction_, l);
      ep = SelectNeighbors(query, candidates, M_);  // next ep

      for (const Point &neighbor : ep) {

        LOG_INFO("Connect #%u to #%u at layer %d", q, neighbor.first, l);
        vertices_[neighbor.first].ConnectTo(q, neighbor.second, l);
        vertex.ConnectTo(neighbor.first, neighbor.second, l);

        // Shrink neighbors
        auto &edges = vertices_[neighbor.first].neighbors[l];
        if (edges.size() > maxM) {
          edges = SelectNeighbors(points_[neighbor.first], edges, maxM);
        }
      }
    }

    if (vertex.layer > top_layer_) {
      top_layer_ = vertex.layer;
      ep_ = q;
    }

    return;
  }

  PointSet Search(size_t K, const T *query) {
    PointSet ep;
    ep.emplace_back(ep_, distance_(query, points_[ep_]));  

    for (layer_t l = top_layer_; l >= 0; l--) {
      MaxPointHeap candidates = SearchLayer(query, ep, ef_construction_, l);
      if (l == 0) {
        ep = SelectNeighbors(query, candidates, K);
      } else {
        ep = SelectNeighbors(query, candidates, M_);
      }
    }

    return ep;
  }

  MaxPointHeap SearchLayer(const T *q, const PointSet &ep, size_t ef, size_t layer) {
    MaxPointHeap result;  // max heap
    MinPointHeap candidates; // min heap
    unordered_map<index_t, bool> visited;

    for (size_t i = 0; i < ep.size(); i++) {
      visited[ep[i].first] = true;
      candidates.push(ep[i]);
      result.push(ep[i]);
    }

    while (!candidates.empty()) {
      Point p = candidates.top();
      candidates.pop();

      if (p.second > result.top().second) {
        break;
      }

      for (const Point &neighbor : vertices_[p.first].neighbors[layer]) {
        index_t edge = neighbor.first;
        if (visited.count(edge) != 0) {
          continue;
        }
        visited[edge] = true;
        T dist = distance_(points_[edge], q);
        if (dist < result.top().second) {
          candidates.emplace(edge, dist);
          result.emplace(edge, dist);
          if (result.size() > ef) {
            result.pop();
          }
        }
      }
    }

    return result;
  }

  // simple select
  PointSet SelectNeighbors(const T *q, MaxPointHeap &candidates, size_t M) {
    while (candidates.size() > M) {
      candidates.pop();
    }
    PointSet result;
    while (!candidates.empty()) {
      result.push_back(candidates.top());
      candidates.pop();
    }
    return result;
  }

  // simple select
  PointSet SelectNeighbors(const T *q, PointSet &candidates, size_t M) {
    std::make_heap(candidates.begin(), candidates.end(), PointLessComparator());
    while (candidates.size() > M) {
      std::pop_heap(candidates.begin(), candidates.end(), PointLessComparator());
      candidates.pop_back();
    }
    return candidates;
  }

  layer_t RandomChoiceLayer() {
    double x = distribution_(generator_);
    x = -std::log(x) * ml_;
    layer_t layer = static_cast<layer_t>(std::floor(x));
    return layer;
  }

private:
  // private parameters
  int M_;      // number of connections to be inserted to one node at one layer
  int maxM_;   // maximumn number of neighbors of one node at one layer except layer 0
  int maxM0_;  // maximumn number of neighbors of one node at layer 0
  double ml_;  // parameter of distribution to decide maximumn layer l. Autoselect ml = 1 / ln(M);
  int ef_;     // number of neighbors to find while searching layer l < lc <= L
  int ef_construction_;  // number of neighbors to find while searching layer 0 <= lc <= l

  // inner variables
  index_t ep_;  // enter point at the top layer
  layer_t top_layer_{0};    // top layer
  vector<Vertex> vertices_;

  std::default_random_engine generator_;
  std::uniform_real_distribution<double> distribution_;

  // private data members
  Matrix<T> points_;
  size_t num_points_{0};
  size_t dim_;
  const Distance<T> &distance_;
};


}  // end hnsw


#endif