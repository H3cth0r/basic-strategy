#include <iostream>
#include "Matrix.h"

int main() {
    Matrix* m1 = new Matrix(2, 3);
    m1->set(0, 0, 1); m1->set(0, 1, 2); m1->set(0, 2, 3);
    m1->set(1, 0, 4); m1->set(1, 1, 5); m1->set(1, 2, 6);

    Matrix* m2 = new Matrix(2, 3);
    m2->set(0, 0, 7); m2->set(0, 1, 8); m2->set(0, 2, 9);
    m2->set(1, 0, 10); m2->set(1, 1, 11); m2->set(1, 2, 12);

    std::cout << "Matrix 1:" << std::endl;
    m1->display();
    std::cout << "Matrix 2:" << std::endl;
    m2->display();

    try {
        Matrix sumMatrix = *m1 + *m2;

        std::cout << "Sum of Matrix 1 and Matrix 2:" << std::endl;
        sumMatrix.display();

    } catch (const std::invalid_argument& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    delete m1;
    delete m2;

    return 0;
}
