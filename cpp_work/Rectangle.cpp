#include "Rectangle.h"
#include <iostream>

Rectangle::Rectangle(double width, double height) : width_(width), height_(height) {
  std::cout << "Rectangle created with width " << width_ << " and height " << height_ << std::endl;
}

Rectangle::~Rectangle() {
  std::cout << "Rectangle destroyed." << std::endl;
}

void Rectangle::print() const {
  std::cout << "Rectangle - Width: " << width_ << ", Height: " << height_ << std::endl;
}

double Rectangle::area() const {
  return width_ * height_;
}
