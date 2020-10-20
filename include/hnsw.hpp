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
      assert(l <= layer && "Too many layers");
      neighbors[l].emplace_back(neighbor, distance);
    }
  };

public:
  HNSWIndex(T *data, uint32_t n_points, uint32_t dim, const Distance<T> &distance,
            int M, int ef, int ef_construction) :
            points_(data, n_points, dim), distance_(distance) {

    top_layer_ = 0;
    M_ = M;
    maxM_ = M;
    maxM0_ = 2 * M;
    ml_ = 1 / log(1.0 * M);
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
    assert(vertex.layer == 0 && vertex.neighbors.size() == 0
           && "Vertex has already been inserted");
    vertex.layer = RandomChoiceLayer();
    vertex.neighbors.resize(vertex.layer + 1);
    for (auto iter = vertex.neighbors.begin() + 1; iter != vertex.neighbors.end(); iter++) {
      iter->reserve(maxM_);
    }
    vertex.neighbors[0].reserve(maxM0_);

    num_points_ += 1;
    if (num_points_ == 1) {
      top_layer_ = vertex.layer;
      ep_ = q;
      return;
    }

    const T *query = points_[q];
    PointSet enterpoints;
    enterpoints.emplace_back(ep_, distance_(query, points_[ep_]));

    for (layer_t l = top_layer_; l > vertex.layer; l--) {
      MaxPointHeap candidates = SearchLayer(query, enterpoints, ef_, l);
      enterpoints = SelectNeighborsHeuristic(query, candidates, 1);
    }

    // Search neighbors and connect to them
    for (layer_t l = std::min(vertex.layer, top_layer_); l >= 0; l--) {
      uint32_t maxM = (l == 0) ? maxM0_ : maxM_;
      MaxPointHeap candidates = SearchLayer(query, enterpoints, ef_construction_, l);
      enterpoints = SelectNeighborsHeuristic(query, candidates, M_);  // next ep

      for (const Point &neighbor : enterpoints) {
        vertices_[neighbor.first].ConnectTo(q, neighbor.second, l);
        vertex.ConnectTo(neighbor.first, neighbor.second, l);

        // Shrink neighbors
        auto &edges = vertices_[neighbor.first].neighbors[l];
        if (edges.size() > maxM) {
          edges = SelectNeighborsHeuristic(points_[neighbor.first], edges, maxM);
        }
      }
    }

    if (vertex.layer > top_layer_) {
      top_layer_ = vertex.layer;
      ep_ = q;
    }

    return;
  }

  PointSet Search(uint32_t K, const T *query) {
    PointSet enterpoints;
    enterpoints.emplace_back(ep_, distance_(query, points_[ep_]));

    for (layer_t l = top_layer_; l >= 0; l--) {
      MaxPointHeap candidates = SearchLayer(query, enterpoints, ef_construction_, l);
      if (l == 0) {
        enterpoints = SelectNeighborsHeuristic(query, candidates, K);
      } else {
        enterpoints = SelectNeighborsHeuristic(query, candidates, M_);
      }
    }

    return enterpoints;
  }

  MaxPointHeap SearchLayer(const T *q, const PointSet &ep, uint32_t ef, uint32_t layer) {
    MaxPointHeap result;  // max heap
    MinPointHeap candidates; // min heap
    unordered_map<index_t, bool> visited;

    for (uint32_t i = 0; i < ep.size(); i++) {
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
  PointSet SelectNeighbors(const T *q, MaxPointHeap &candidates, uint32_t M) {
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
  PointSet SelectNeighbors(const T *q, PointSet &candidates, uint32_t M) {
    if (candidates.size() <= M) {
      return candidates;
    }
    MaxPointHeap candidates_heap(candidates.begin(), candidates.end());
    return SelectNeighbors(q, candidates_heap, M);
  }

  // heuristic algo
  PointSet SelectNeighborsHeuristic(const T *q, MaxPointHeap &candidates,
                                    uint32_t M, bool keep_pruned = true) {
    if (candidates.size() <= M) {
      PointSet selected_points;
      while (!candidates.empty()) {
        selected_points.push_back(candidates.top());
        candidates.pop();
      }
      return selected_points;
    }

    MinPointHeap closest_candidates;
    while (!candidates.empty()) {
      closest_candidates.push(candidates.top());
      candidates.pop();
    }

    PointSet selected_points;
    MinPointHeap discarded_points;
    while (!closest_candidates.empty()) {
      Point p = closest_candidates.top();
      closest_candidates.pop();
      bool is_nearest = true;
      for (const auto &neighbor : selected_points) {
        T dist = distance_(points_[p.first], points_[neighbor.first]);
        if (dist < p.second) {
          is_nearest = false;
          break;
        }
      }
      if (is_nearest) {
        selected_points.push_back(p);
      } else {
        discarded_points.push(p);
      }
    }

    if (keep_pruned) {
      while (selected_points.size() < M && !discarded_points.empty()) {
        selected_points.push_back(discarded_points.top());
        discarded_points.pop();
      }
    }

    return selected_points;
  }

  PointSet SelectNeighborsHeuristic(const T *q, PointSet &candidates, uint32_t M,
                                    bool keep_pruned = true) {
    if (candidates.size() <= M) {
      return candidates;
    }
    MaxPointHeap candidates_heap(candidates.begin(), candidates.end());
    return SelectNeighborsHeuristic(q, candidates_heap, M, keep_pruned);
  }

  layer_t RandomChoiceLayer() {
    double x = distribution_(generator_);
    x = -std::log(x) * ml_;
    layer_t layer = static_cast<layer_t>(std::floor(x));
    return layer;
  }

  layer_t TopLayer() { return top_layer_; }

private:
  // private parameters
  uint32_t M_;      // number of connections to be inserted to one node at one layer
  uint32_t maxM_;   // maximumn number of neighbors of one node at one layer except layer 0
  uint32_t maxM0_;  // maximumn number of neighbors of one node at layer 0
  double ml_;  // parameter of distribution to decide maximumn layer l. Autoselect ml = 1 / ln(M);
  uint32_t ef_;     // number of neighbors to find while searching layer l < lc <= L
  uint32_t ef_construction_;  // number of neighbors to find while searching layer 0 <= lc <= l

  // inner variables
  index_t ep_;  // enter point at the top layer
  layer_t top_layer_{0};    // top layer
  vector<Vertex> vertices_;

  std::default_random_engine generator_;
  std::uniform_real_distribution<double> distribution_;

  // private data members
  Matrix<T> points_;
  uint32_t num_points_{0};
  uint32_t dim_;
  const Distance<T> &distance_;
};


}  // end hnsw


#endif