#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include "randomSystem.hpp"

int main() {
    // We create a system with N=5, M=7, g=1, seed=52
    randomSystem sys(5, 7, 1.0, 52);
    
    // Test standard Dykstra vs modified Dykstra
    std::cout << "Testing standard Dykstra..." << std::endl;
    double alpha = 1e-3;
    sys.performProjectionMethod_Dykstra(alpha, 1e-3, 1e-3, "test_proj_orig.txt");
    
    return 0;
}
