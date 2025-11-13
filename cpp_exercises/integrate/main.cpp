#include <iostream>
#include <functional>
#include <cmath>
#include <string>
#include <sstream>
#include <stdexcept>

/*
 * std::function comes from functional
 * std::sqrt, log comes from cmath
 * std::ostringstream comes from sstream. allows for manupulation of strings as if they were output streams. 
 * insiert variouys data types into an ostringstream object and the data will be formatted and stored as an string.
 * */

namespace 
{
  enum class INTYPE {
    LEFT, RIGHT, TRAPEZOID
  };

  double computeIntegral(const std::function<double(double)>& f, double ll, double ul, size_t n, INTYPE theType) {
    if (ul < ll) throw std::out_of_range("Lower limit should be lower than upper limit");
    if (n < 1) throw std::out_of_range("Steps not properly defined");

    const double stepSize = (ul-ll)/n;
    double result = 0;

    switch(theType) {
      case INTYPE::LEFT:
        for (int i = 0; i < n; ++i) {
          double leftPoint = ll + (i*stepSize);
          result += f(leftPoint);
        }
        return result * stepSize;
      case INTYPE::RIGHT:
        for (int i = 1; i <= n; ++i) {
          double rightPoint = ll + (i*stepSize);
          result += f(rightPoint);
        }
        return result * stepSize;
      case INTYPE::TRAPEZOID:
        result += f(ll) + f(ul);
        for (int i = 0; i < n; ++i) {
          double point = ll + (i*stepSize);
          result += 2 * f(point);
        }
        return result * (stepSize/2.0);
      default:
        throw std::out_of_range("Integration type not defined properly");
    }
  }

  void printResults(const std::function<double(double)>& theFunc)
  {
    double const resultLeft = computeIntegral(theFunc, 1.0, 2.0, 50, INTYPE::LEFT);
    double const resultRight = computeIntegral(theFunc, 1.0, 2.0, 50, INTYPE::RIGHT);
    double const resultTrapezoid = computeIntegral(theFunc, 1.0, 2.0, 50, INTYPE::TRAPEZOID);

    std::ostringstream output;
    output << "Result left: " << resultLeft << "\nResult right: " << resultRight << "\nResult TRAPEZOID: " << resultTrapezoid << "\n\n";
    std::cout << output.str() ;
  }

}

void writeFunctionName(const std:: string& strng) {
  std::ostringstream output;
  output << strng << "\n";
  std::cout << output.str();
}


int main() {
  try 
  {
    auto functionIdentity = [] (const double x) {
      return x;
    };
    auto functionPolynom = [] (const double x) {
      return 2 * x*x + 5;
    };
    auto functionSqrt = [] (const double x) {
      return std::sqrt(x);
    };
    auto functionLog = [] (const double x) {
      return std::log(x);
    };

    writeFunctionName("identity");
    printResults(functionIdentity);
  } catch (const std::exception& e) {
    std::cerr << "An error occurred: " << e.what() << "\n";
    return 1;
  }
  return 0;
};
