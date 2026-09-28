#ifndef MATHOPERATIONS_HPP
#define MATHOPERATIONS_HPP

#include <Eigen/Dense>

unsigned int factorial(unsigned int val) {
    unsigned int result = 1;

    if (val == 0) {
        return result;
    }

    for (unsigned int i = 0; i < val - 1; i++) {
        result *= val - i;
    }
    return result;
}

unsigned int binomial(unsigned int n, unsigned int k) {
    if (k > n) {
        return 0;
    }
    
    unsigned int result = 1;
    for (unsigned int i = n; i > n - k; i--) {
        result *= i;
    }
    for (unsigned int i = 1; i <= k; i++) {
        result /= i;
    }
    return result;
}

int signInt(int val) {
    if (val >= 0) {
        return 1;
    } else {
        return -1;
    }
}

double productDiagMatrices(const Eigen::MatrixXd& m1, const Eigen::MatrixXd& m2) {
    if (m1.rows() != m2.rows() || m1.cols() != m2.cols()) {
        throw std::invalid_argument("Matrices must have the same dimensions");
    }

    double product = 0.0;
    for (int i = 0; i < m1.rows(); ++i) {
        product += m1(i, i) * m2(i, i);
    }
    return product;
}

double normDiagMatrix(const Eigen::MatrixXd& m) {
    if (m.rows() != m.cols()) {
        throw std::invalid_argument("Matrix must be square");
    }

    double norm = 0.0;
    for (int i = 0; i < m.rows(); ++i) {
        norm += m(i, i) * m(i, i);
    }
    return std::sqrt(norm);
}

#endif