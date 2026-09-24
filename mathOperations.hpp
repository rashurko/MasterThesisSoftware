#ifndef MATHOPERATIONS_HPP
#define MATHOPERATIONS_HPP

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



#endif