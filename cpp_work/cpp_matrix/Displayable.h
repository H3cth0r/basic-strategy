#ifndef DISPLAYABLE_H
#define DISPLAYABLE_H

class Displayable {
  public:
    virtual ~Displayable() {}
    virtual void display() const = 0;
};

#endif
