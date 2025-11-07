#include <iostream>
#include "Matrix.h"

int main() {
    int size = 5;
    int* simpleArray = new int[size];
    for (int i = 0; i < size; ++i) {
        simpleArray[i] = i * 10;
        std::cout << "simpleArray[" << i << "] = " << simpleArray[i] << std::endl;
    }
    delete[] simpleArray; // Deallocate the array
    std::cout << std::endl;

    Matrix* m1 = new Matrix(2, 3);
    m1->set(0, 0, 1);
    m1->set(0, 1, 2);
    m1->set(0, 2, 3);
    m1->set(1, 0, 4);
    m1->set(1, 1, 5);
    m1->set(1, 2, 6);

    Displayable* displayableObject = m1;
    displayableObject->display();
    std::cout << std::endl;

    Matrix* m2 = new Matrix(2, 3);
    m2->set(0, 0, 7);
    m2->set(0, 1, 8);
    m2->set(0, 2, 9);
    m2->set(1, 0, 10);
    m2->set(1, 1, 11);
    m2->set(1, 2, 12);

    m1->display();
    m2->display();

    Matrix* sumMatrix = m1->add(*m2);
    
    if (sumMatrix) {
        sumMatrix->display();
        delete sumMatrix; 
    }

    delete m1;
    delete m2;

    return 0;

}
