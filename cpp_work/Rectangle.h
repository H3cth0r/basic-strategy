#ifndef RECTANGLE_H
#define RECTANGLE_H

#include "Printable.h"
#include "Shape.h"

class Rectangle : public Printable, public Shape {
  public:
    Rectangle(double width, double height);
    ~Rectangle();

    void print() const override;

    double area() const override;

  private:
    double width_;
    double height_;
};

#endif
