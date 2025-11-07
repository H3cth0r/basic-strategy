#ifndef MATRIX_H
#define MATRIX_H

#include "Displayable.h"

class Matrix : public Displayable {
  public:
    Matrix(int rows, int cols);
    ~Matrix();

    void set(int row, int col, int value);
    void display() const override;
    bool hasSameDimensions(const Matrix& other) const;
    Matrix* add(const Matrix& other) const;

    Matrix operator+(const Matrix& other) const;

  private:
    int rows_;
    int cols_;
    int** data_;
};

#endif
