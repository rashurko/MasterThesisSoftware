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



#endif