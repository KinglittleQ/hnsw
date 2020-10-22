#ifndef __HMSW_MATRIX_HPP__
#define __HMSW_MATRIX_HPP__

#include <iostream>
#include <vector>

namespace hnsw {

class Matrix {
public:
  explicit Matrix(const float *data_ptr, size_t n_rows, size_t n_cols) {
    data_ = data_ptr;
    rows_ = n_rows;
    cols_ = n_cols;
    for (size_t i = 0; i < n_rows; i++) {
      row_ptrs_.push_back(data_ + i * n_cols);
    }
  }

  const float *operator[](size_t row) const {
    return row_ptrs_[row];
  }

  size_t Rows() { return rows_; }
  size_t Cols() { return cols_; }

  void PrintMatrix() const {
    for (size_t i = 0; i < rows_; i++) {
      for (size_t j = 0; j < cols_; j++) {
        std::cout << (*this)[i][j] << " ";
      }
      std::cout << std::endl;
    }
  }

private:
  const float *data_;
  size_t rows_, cols_;
  std::vector<const float *> row_ptrs_; 
};

}  // end hnsw

#endif