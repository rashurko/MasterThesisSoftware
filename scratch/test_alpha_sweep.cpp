
#include <iostream>
#include <iomanip>
#include "scratch/test_randomSystem.hpp"

class SubTester : public randomSystem {
public:
    SubTester(unsigned int N, unsigned int M, double g, unsigned int seed)
        : randomSystem(N, M, g, seed) {
        diagonalizeH();
    }
    double getE() const { return system.Eground_proj; }
};

int main() {
    for (double alpha : {1e-3, 5e-4, 2e-4, 1e-4}) {
        SubTester sys(5, 7, 1.0, 52);
        sys.performProjectionMethod_Dykstra(alpha, 1e-3, 1e-3, "/dev/null");
        double E = sys.getE();
        std::cout << "alpha = " << std::scientific << std::setprecision(1) << alpha 
                  << " -> E = " << std::fixed << std::setprecision(5) << E 
                  << " (exact = -6.82049)" << std::endl;
    }
    return 0;
}
