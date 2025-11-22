#include <iostream>

using namespace std;

int main() {
  int rows, cols;
  cout << "Enter number of rows: ";
  cin >> rows;

  cout << "Enter number of columns: ";
  cin >> cols;

  int A[100][100], B[100][100], C[100][100];

  cout << "\nEnter elements of Matrix A: \n";
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      cin >> A[i][j];
    }
  }

  cout << "\nEnter elements of Matrix B: \n";
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      cin >> B[i][j];
    }
  }

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      C[i][j] = A[i][j] + B[i][j];
    }
  }

  cout << "\nSum of matrices:\n";
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      cout << C[i][j] << " ";
    }
    cout << endl;
  }

  return 0;

}
