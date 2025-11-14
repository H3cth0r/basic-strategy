#include <iostream>
#include <cmath>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


namespace GeometricShapes
{
  struct Point
  {
    double x, y;
  };

  class Shape
  {
    public:
      virtual double getArea() const = 0;
      virtual double getPerimeter() const = 0;
      virtual std::string getName() const = 0;
      virtual ~Shape() {}
  };

  class Circle : public Shape
  {
    private:
      Point center;
      double radius;

    public:
      Circle(Point center_t, double radius_t) : center(center_t), radius(radius_t) {}

      double getArea() const override 
      {
        return M_PI*(std::pow(this->radius, 2));
      }
      double getPerimeter() const override 
      {
        return 2*M_PI*this->radius;
      }
      std::string getName() const override
      {
        return "Circle";
      }
  };

  class ShapePortfolio
  {
    private:
      Shape** shapes;
      size_t size;
      size_t capacity;

    public:
      ShapePortfolio() : shapes(nullptr), size(0), capacity(0)
      {
        capacity = 10;
        shapes = new Shape*[capacity]; // allocate array on the heap
      }
      ~ShapePortfolio ()
      {
      }

      void addShape(Shape* shape)
      {
        if (size >= capacity)
        {
          // double the capacity (a common strategy)
          size_t newCapacity = (capacity == 0) ? 10 : capacity * 2;
          Shape** newShapes = new Shape*[newCapacity];

          for (size_t i = 0; i < size; ++i)
          {
            newShapes[i] = shapes[i];
          }
          delete[] shapes;

          // Point to the new array and update capacity
          shapes = newShapes;
          capacity = newCapacity;
        }
        shapes[size] = shape;
        size++;
      }

      void displayPortfolio() const
      {
        std::cout << "--- Shape Portfolio ---\n";
        if (size == 0) {
            std::cout << "The portfolio is empty.\n";
            return;
        }
        for (size_t i = 0; i < size; ++i)
        {
            std::cout << "Shape " << i + 1 << ": " << shapes[i]->getName() << "\n"
                      << "  Area: " << shapes[i]->getArea() << "\n"
                      << "  Perimeter: " << shapes[i]->getPerimeter() << "\n\n";
        }
        std::cout << "-----------------------\n";
      }
  };

}

int main() {
  GeometricShapes::ShapePortfolio myPortfolio;

  myPortfolio.addShape(new GeometricShapes::Circle(GeometricShapes::Point{0, 0}, 5.0));

  myPortfolio.displayPortfolio();

  return 0;
}
