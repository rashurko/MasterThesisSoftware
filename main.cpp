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

    // Save the system to a JSON file for SDPT3
    // sys.saveToSDPT3Json("system_SDPT3.json");

    // Print the system information
    sys.printSystem();

    bool running = true;

    while (running)
    {
        std::cout << "1. Load T and V from csv" << std::endl;
        std::cout << "2. Calculate exact eigenvalues" << std::endl;
        std::cout << "3. Perform Potential Reduction Method" << std::endl;
        std::cout << "4. Perform Projection Method" << std::endl;
        std::cout << "5. Run PR and ED comparison" << std::endl;
        std::cout << "6. Run PR and ED comparison for multiple g values" << std::endl;
        std::cout << "7. Run Projection Method for multiple alpha values" << std::endl;
        std::cout << "8. Exit" << std::endl;

        int choice;
        std::cin >> choice;

        switch (choice) {
            case 1: {
                sys.loadFromCSV("T_matrix.csv", "V_tensor_flat.csv");
                break;
            }
            case 2: {
                sys.diagonalizeH();
                sys.saveToJson("exactSol.json");
                break;
            }
            case 3: {
                sys.performPotentialReduction();
                sys.saveToJsonPR("potentialReduction.json");
                break;
            }
            case 4: {
                double alpha = 1e-3;
                sys.performProjectionMethod_Dykstra(alpha, 1e-3, 1e-3);
                sys.saveToJsonProjection("projectionMethod.json");
                break;
            }

            case 5: {
                // Inputs for the comparison
                unsigned int M_min;
                unsigned int M_max;
                unsigned int M_step;
                std::cout << "Enter the minimum number of 1p-states (M_min): ";
                std::cin >> M_min;
                std::cout << "Enter the maximum number of 1p-states (M_max): ";
                std::cin >> M_max;
                std::cout << "Enter the step size for 1p-states (M_step): ";
                std::cin >> M_step;

                sys.runExactDiagAndSDPMultiple(N, g, M_min, M_max, M_step);
                break;
            }
            case 6: {
                // Inputs for the comparison
                double g_min;
                double g_max;
                double n_g;
                std::cout << "Enter the minimum value of g (g_min): ";
                std::cin >> g_min;
                std::cout << "Enter the maximum value of g (g_max): ";
                std::cin >> g_max;
                std::cout << "Enter the number of g values (n_g): ";
                std::cin >> n_g;

                n_g = std::round(n_g);
                if (n_g < 1) {
                    std::cerr << "Invalid number of g values." << std::endl;
                    break;
                }

                sys.runExactDiagAndSDPMultipleG(N, M, g_min, g_max, n_g);
                break;
            }
            case 7: {
                // Inputs for the comparison
                double alpha_min;
                double alpha_max;
                double n_alpha;
                std::cout << "Enter the minimum value of alpha (alpha_min): ";
                std::cin >> alpha_min;
                std::cout << "Enter the maximum value of alpha (alpha_max): ";
                std::cin >> alpha_max;
                std::cout << "Enter the number of alpha values (n_alpha): ";
                std::cin >> n_alpha;

                n_alpha = std::round(n_alpha);
                if (n_alpha < 1) {
                    std::cerr << "Invalid number of alpha values." << std::endl;
                    break;
                }

                sys.runProjectionMethodMultipleAlphas(N, M, g, alpha_min, alpha_max, n_alpha);
                break;
            }
            case 8: {
                running = false;
                break;
            }
            default:
                std::cout << "Invalid choice." << std::endl;
        }
    }

    return 0;
}