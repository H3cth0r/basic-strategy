- class with at least one pure virtual function becomes an abstract class and object of abstract classes cannot be created directly.
- Abstract classes are used to define interfaces and ensure common structure among derived classes.
- Useful in polymorphism where different classes share the same interface but have different behaviors.
- A pure virtual function forces derived classes to override it.
- `virtual void draw() = 0;` declares a pure virtual function, forcing derived classes to provide their own implementation.

```
g++ shapes.cpp -o shapes -std=c++17 -Wall
```
