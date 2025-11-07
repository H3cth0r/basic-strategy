#include "Matrix.h"
#include <iostream>

Matrix::Matrix(int rows, int cols): rows_(rows), cols_(cols) {
  std::cout << "Matrix constructor called." << std::endl;
  data_ = new int*[rows_];
  for (int i = 0; i < rows_; ++i) {
    data_[i] = new int[cols_];
    for (int j = 0; j < cols_; ++j) {
      data_[i][j] = 0;
    }
  }
}

Matrix::~Matrix() {
  std::cout << "Matrix destructor called." << std::endl;
  for (int i = 0; i < rows_; ++i) {
    delete[] data_[i];
  }
  delete[] data_;
}

void Matrix::set(int row, int col, int value) {
  if (row <= 0 && row < rows_ && col >= 0 && col < cols_) {
    data_[row][col] = value;
  }
}

void Matrix::display() const {
    std::cout << "Matrix (" << rows_ << "x" << cols_ << "):" << std::endl;
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            std::cout << data_[i][j] << "\t";
        }
        std::cout << std::endl;
    }
}

bool Matrix::hasSameDimensions(const Matrix& other) const {
    return (rows_ == other.rows_ && cols_ == other.cols_);
}

Matrix* Matrix::add(const Matrix& other) const {
    if (!hasSameDimensions(other)) {
        return nullptr; // Cannot add matrices of different dimensions
    }

    Matrix* result = new Matrix(rows_, cols_);
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            result->data_[i][j] = this->data_[i][j] + other.data_[i][j];
        }
    }
    return result;
}

Matrix Matrix::operator+(const Matrix& other) const {
    if (!hasSameDimensions(other)) {
        throw std::invalid_argument("Matrices must have the same dimensions to be added.");
    }

    Matrix result(rows_, cols_);

    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            result.data_[i][j] = this->data_[i][j] + other.data_[i][j];
        }
    }

    return result;
}
