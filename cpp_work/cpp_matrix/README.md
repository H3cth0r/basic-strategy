## What is a virtual function?
Member function is a member function that you expect to be redefined in derived classes. When you refer to a derived class object using a pointer or a reference to the base class, you can call a virtual function for that object and execute the derived class version of the function.

## Meaning of const in parent functions
 When you declare a member function as `const`, you are making a promise that this function will not modify any member variables of the object it is called on.

 For example, in `double area() const;` the `const` at the end ensures that calling the area function will not change the `width_` or `height_` of a Rectangle object.

 ## Allocate Memory for array of integers

 ```
 int* myArray = new int[size];
 delete[] myArray;
 ```

 ```
 g++ main.cpp Matrix.cpp -o matrix_app
 ./matrix_app
 ```
