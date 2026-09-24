#include <iostream>
#include <chrono>
#include <fstream>

#include "randomSystem.hpp"

int main() {
    // Input a random system with a given number of fermions N and 1p-states M, also specify a random seed
    unsigned int N, M, seed;
    double g;
    std::cout << "Enter the number of fermions (N): ";
    std::cin >> N;
    std::cout << "Enter the number of 1p-states (M): ";
    std::cin >> M;
    std:: cout << "Enter g: ";
    std::cin >> g;
    std::cout << "Enter the random seed: ";
    std::cin >> seed;

    randomSystem sys(N, M, g, seed);

    // Print the system information
    sys.printSystem();

    bool running = true;

    while (running)
    {
        std::cout << "1. Show T elements" << std::endl;
        std::cout << "2. Show V elements" << std::endl;
        std::cout << "3. Change T matrix" << std::endl;
        std::cout << "4. Change V matrix" << std::endl;
        std::cout << "5. Load T and V from csv" << std::endl;
        std::cout << "6. Calculate exact eigenvalues" << std::endl;
        std::cout << "7. Exit" << std::endl;

        int choice;
        std::cin >> choice;

        switch (choice) {
            case 1: {
                unsigned int i, j;
                std::cout << "Enter indices (i, j) for T: ";
                std::cin >> i >> j;
                std::cout << "T(" << i << ", " << j << ") = " << sys.getTElement(i, j) << std::endl;
                break;
            }
            case 2: {
                unsigned int i, j, k, l;
                std::cout << "Enter indices (i, j, k, l) for V: ";
                std::cin >> i >> j >> k >> l;
                std::cout << "V(" << i << ", " << j << ", " << k << ", " << l << ") = " << sys.getVElement(i, j, k, l) << std::endl;
                break;
            }
            case 3: {
                for (unsigned int i = 0; i < sys.getM(); i++) {
                    for (unsigned int j = 0; j < sys.getM(); j++) {
                        double value;
                        std::cout << "Enter value for T(" << i << ", " << j << "): ";
                        std::cin >> value;
                        sys.setTElement(i, j, value);
                    }
                }

                sys.updateKMatrix();
                sys.updateHMatrix();
                break;
            }
            case 4: {
                for (unsigned int j = 0; j < sys.getM(); j++) {
                    for (unsigned int i = 0; i < j; i++) {
                        for (unsigned int l = 0; l < sys.getM(); l++) {
                            for (unsigned int k = 0; k < l; k++) {
                                double value;
                                std::cout << "Enter value for V(" << i << ", " << j << ", " << k << ", " << l << "): ";
                                std::cin >> value;
                                sys.setVElement(i, j, k, l, value);
                            }
                        }
                    }    
                }

                sys.updateKMatrix();
                sys.updateHMatrix();
                break;
            }
            case 5: {
                sys.loadFromCSV("T_matrix.csv", "V_tensor_flat.csv");
                break;
            }
            case 6: {
                sys.diagonalizeH();
                sys.saveToJson("exactSol.json");
                break;
            }
            case 7: {
                running = false;
                break;
            }
            default:
                std::cout << "Invalid choice." << std::endl;
        }
    }

    return 0;
}