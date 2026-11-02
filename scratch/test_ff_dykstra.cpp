#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include "randomSystem.hpp"

// Standalone function implementing Algorithm 1 with Fast-Forwarding
void run_fast_forward_dykstra(randomSystem& sys, double alpha, double eps_stall = 1e-4, unsigned int max_iter = 10000) {
    auto start = std::chrono::high_resolution_clock::now();

    // Access system matrices via public/friend or custom driver
    // For test, let's observe how fast-forwarding breaks the stall
    std::cout << "Starting Fast-Forward Dykstra test..." << std::endl;
}

int main() {
    randomSystem sys(5, 7, 1.0, 52);
    std::cout << "System initialized: N=5, M=7, g=1, seed=52" << std::endl;
    return 0;
}
