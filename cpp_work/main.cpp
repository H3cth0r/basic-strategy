#include "Rectangle.h"

int main() {
  // Dynamic memory allocation for a rectangle object
  Rectangle* rect = new Rectangle(10.0, 5.0);

  // Using the object through a Printable pointer (Polymorphism)
  Printable* printableShape = rect;

  std::cout << "Printing through Printable pointer: ";
  printableShape->print();

  // Using the object through a Shape pointer (Polymorphism)
  Shape* geometricShape = rect;
  std::cout << "Area calculated through Shape pointer: " << geometricShape->area() << std::endl;

  std::cout << "Directly accessing methods: " << std::endl;
  rect->print();
  std::cout << "Area: " << rect->area() << std::endl;

  delete rect;

  return 0;

}
