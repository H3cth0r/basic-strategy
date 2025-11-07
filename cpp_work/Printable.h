#ifndef PRINTABLE_H
#define PRINTABLE_H

#include <iostream>
#include <string>

class Printable {
  public:
    virtual ~Printable() {}
    virtual void print() const = 0;
};

#endif
