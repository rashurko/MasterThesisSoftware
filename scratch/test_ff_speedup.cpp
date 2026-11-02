#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include "randomSystem.hpp"

// We create a test harness where performProjectionMethod_Dykstra_FF is called
class TestSystem : public randomSystem {
public:
    TestSystem(unsigned int N, unsigned int M, double g, unsigned int seed) 
        : randomSystem(N, M, g, seed) {}

    void performProjectionMethod_Dykstra_FastForward(const double alpha, double eps_stall = 1e-4, double N_skip_init = 50.0, unsigned int max_iter = 1000) {
        auto start = std::chrono::high_resolution_clock::now();

        // Initialize
        system.alpha_p = alpha;
        system.Gamma_p = -(1.0 / alpha) * system.K;

        Eigen::MatrixXd D_n = system.Gamma_p;
        Eigen::MatrixXd D_n_1 = system.Gamma_p;
        Eigen::MatrixXd C = Eigen::MatrixXd::Zero(D_n.rows(), D_n.cols());
        Eigen::MatrixXd X_Gamma = Eigen::MatrixXd::Zero(D_n.rows(), D_n.cols());
        Eigen::MatrixXd X_Q = Eigen::MatrixXd::Zero(D_n.rows(), D_n.cols());
        Eigen::MatrixXd X_G = Eigen::MatrixXd::Zero(D_n.rows(), D_n.cols());

        Eigen::MatrixXd X_Gamma_prev = X_Gamma;
        Eigen::MatrixXd X_Q_prev = X_Q;
        Eigen::MatrixXd X_G_prev = X_G;

        std::cout << "--- Start Fast-Forwarded Dykstra (eps_stall=" << eps_stall << ") ---" << std::endl;

        unsigned int i = 0;
        unsigned int stall_count = 0;
        unsigned int total_fast_forwards = 0;
        double N_skip = N_skip_init;

        while (i < max_iter) {
            D_n = D_n_1;

            X_Gamma_prev = X_Gamma;
            X_Q_prev = X_Q;
            X_G_prev = X_G;

            // 1. Gamma projection
            C = D_n - X_Gamma;
            D_n_1 = projectGamma(C);
            X_Gamma = (D_n_1 - C);

            // 2. Q projection
            C = D_n_1 - X_Q;
            D_n_1 = projectGammaQ(C);
            X_Q = (D_n_1 - C);

            // 3. G projection
            C = D_n_1 - X_G;
            D_n_1 = projectGammaG(C);
            X_G = (D_n_1 - C);

            double norm_diff = (D_n - D_n_1).norm();
            double energy = (system.K.array() * D_n_1.array()).sum();

            // Dual increments across cycle
            Eigen::MatrixXd dX_Gamma = X_Gamma - X_Gamma_prev;
            Eigen::MatrixXd dX_Q = X_Q - X_Q_prev;
            Eigen::MatrixXd dX_G = X_G - X_G_prev;
            double dual_drift_norm = dX_Gamma.norm() + dX_Q.norm() + dX_G.norm();

            // Algorithm 1 Line 5: Check if stalled
            if (i >= 2 && norm_diff < eps_stall && dual_drift_norm > 1e-7) {
                stall_count++;
                // Apply Fast-Forwarding update (Algorithm 1 Line 8-9)
                X_Gamma += N_skip * dX_Gamma;
                X_Q += N_skip * dX_Q;
                X_G += N_skip * dX_G;
                total_fast_forwards++;

                std::cout << ">>> [Iter " << i << "] STALL DETECTED! Fast-forwarding duals by N_skip = " 
                          << N_skip << " cycles (diff = " << norm_diff << ", energy = " << energy << ")" << std::endl;
            } else {
                if (norm_diff >= eps_stall) {
                    stall_count = 0;
                }
            }

            if (i % 50 == 0 || norm_diff < 1e-6) {
                std::cout << "Iter " << i << ", Energy: " << energy << ", Norm diff: " << norm_diff << std::endl;
            }

            if (norm_diff < 1e-6 && i > 5) {
                std::cout << "Converged at iteration " << i << std::endl;
                break;
            }

            i++;
        }

        auto elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
        std::cout << "Completed in " << elapsed << "s with " << total_fast_forwards << " fast-forward jumps." << std::endl;
        std::cout << "Final Energy: " << (system.K.array() * D_n_1.array()).sum() << std::endl;
    }
};

int main() {
    TestSystem sys(5, 7, 1.0, 52);
    double alpha = 1e-3;
    sys.performProjectionMethod_Dykstra_FastForward(alpha, 1e-4, 50.0, 500);
    return 0;
}
