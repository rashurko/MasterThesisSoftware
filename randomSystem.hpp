#ifndef RANDOMSYSTEM_HPP
#define RANDOMSYSTEM_HPP

#include <iostream>
#include <cmath>

#include <fstream>
#include <sstream>
#include <string>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "mathOperations.hpp"

struct System {
    // Constants
    unsigned int N;         // number of fermions
    unsigned int M;         // number of fermionic states
    double g;               // 2-particle interaction term, H = T + gV

    // Basis vectors for the states
    unsigned int dimBasis;
    std::vector<std::vector<unsigned int>> basis;

    // Exact diagonalization results
    Eigen::VectorXd exactEnergies;
    Eigen::MatrixXd exactEigenvectors;

    // Single-particle interaction matrix T, two-partcile interaction matrix V and the Hamiltonian matrix H
    Eigen::MatrixXd T;
    Eigen::MatrixXd V;
    Eigen::MatrixXd H;

    // Reduced Hamiltonian K
    Eigen::MatrixXd K;
};

class randomSystem {
    private:
        System system;
        // Seed for random number generation
        unsigned int seed;

        void initializeRandomNumberGenerator() {
            std::srand(seed);
        }

        // Returns the index I of pair (i,j) with the sign; i<j => +, i>j => -
        std::pair<unsigned int, int> getIdxV(unsigned int i, unsigned int j) const {
            if (i < j) {
                return {(j - i - 1) + (i * (2 * system.M - i - 1)) / 2, 1};
            } else {
                return {(i - j - 1) + (j * (2 * system.M - j - 1)) / 2, -1};
            }
        }

        void generateBasis() {
            unsigned dimBasis = binomial(system.M, system.N);
            system.dimBasis = dimBasis;
            system.basis.reserve(dimBasis);

            // Initial basis vector
            std::vector<unsigned int> basisVector(system.M, 0);
            for (unsigned int i = 0; i < system.N; i++) {
                basisVector[i] = 1;
            }

            // std::prev_permutation slides iteratively to the right
            do {
                system.basis.push_back(basisVector);

                // print basis vector
                // for (unsigned int i = 0; i < system.M; i++) {
                //     std::cout << basisVector[i];
                // }
                // std::cout << std::endl;

            } while (std::prev_permutation(basisVector.begin(), basisVector.end()));            
        }

        void generateKMatrix() {
            unsigned int P = system.M * (system.M - 1) / 2;
            system.K = Eigen::MatrixXd(P, P);
            system.K = (system.g) * system.V;
            // Calculate the elements of K
            for (unsigned int j = 0; j < system.M; j++) {
                for (unsigned int i = 0; i < j; i++) {
                    auto [idx1, sign1] = getIdxV(i, j);
                    for (unsigned int l = 0; l < system.M; l++) {
                        for (unsigned int k = 0; k < l; k++) {
                            double factorT = 0.0;
                            if (j == l) {
                                factorT += system.T(i, k);
                            }
                            if (i == k) {
                                factorT += system.T(j, l);
                            }
                            if (i == l) {
                                factorT -= system.T(j, k);
                            }
                            if (j == k) {
                                factorT -= system.T(i, l);
                            }
                            auto [idx2, sign2] = getIdxV(k, l);
                            system.K(idx1, idx2) += (1 / ((system.N - 1))) * sign1 * sign2 * factorT;
                        }
                    }
                }
            }
            system.K *= 0.5;
        }

        void generateHMatrix() {
            unsigned int dimH = system.dimBasis;
            system.H = Eigen::MatrixXd(dimH, dimH);
            system.H.setZero();

            // Fill the H matrix
            for (unsigned int i = 0; i < dimH; i++) {
                for (unsigned int j = 0; j < dimH; j++) {
                    auto ket = system.basis[i];
                    auto bra = system.basis[j];
                    double H_ij = 0.0;

                    // Count number of different elements in eigenstates
                    unsigned int diff = 0;
                    for (unsigned int idx = 0; idx < system.M; idx++) {
                        if (ket[idx] != bra[idx]) {
                            diff++;
                        }
                    }

                    if (diff > 4) {
                        continue;
                    }

                    for (unsigned int l = 0; l < system.M; l++) {
                        for (unsigned int k = 0; k < l; k++) {
                            if (k == l || ket[k] == 0 || ket[l] == 0) {
                                continue;
                            }
                            for (unsigned int n = 0; n < system.M; n++) {
                                for (unsigned int m = 0; m < n; m++) {
                                    if (m == n || bra[n] == 0 || bra[m] == 0) {
                                        continue;
                                    }

                                    auto ketNew = ket;
                                    auto braNew = bra;
                                    for (unsigned int p = 0; p < system.M; p++) {
                                        if (p == k) {
                                            ketNew[p] -= 1;
                                        }
                                        if (p == l) {
                                            ketNew[p] -= 1;
                                        }
                                        if (p == m) {
                                            braNew[p] -= 1;
                                        }
                                        if (p == n) {
                                            braNew[p] -= 1;
                                        }
                                    }
                                    // Are ketNew and braNew equal?
                                    if (ketNew == braNew) {
                                        int sign = pow(-1, countFront(ket, k) + countFront(ket, l) + countFront(bra, m) + countFront(bra, n)) * signInt(l-k) * signInt(n-m);
                                        H_ij += 2 * sign * getKElement(k, l, m, n);
                                    }
                                }
                            }
                        }
                    }

                    system.H(i, j) = H_ij;
                }
            }

            // print H
            std::cout << "Matrix H:\n" << system.H << std::endl;
        }

        void generateRandomMatrices() {
            Eigen::MatrixXd rngMatrixT = Eigen::MatrixXd::Random(system.M, system.M);
            system.T = 0.5 * (rngMatrixT + rngMatrixT.transpose());

            unsigned int P = system.M * (system.M - 1) / 2;
            Eigen::MatrixXd rngMatrixV = Eigen::MatrixXd::Random(P, P);
            system.V = 0.5 * (rngMatrixV + rngMatrixV.transpose());

            // Generate the K matrix
            generateKMatrix();

            // print K
            // std::cout << "Matrix K:\n" << system.K << std::endl;

            // Generate the H matrix
            generateHMatrix();
        }

        // Count occupied states before m
        unsigned int countFront(std::vector<unsigned int> &vec, unsigned int m) {
            unsigned int count = 0;
            for (unsigned int i = 0; i < m; i++) {
                if (vec[i] == 1) {
                    count++;
                }
            }
            return count;
        }

    public:
        randomSystem(unsigned int N, unsigned int M, double g, unsigned int seed) {
            system.N = N;
            system.M = M;
            system.g = g;
            this->seed = seed;
            initializeRandomNumberGenerator();
            generateBasis();
            generateRandomMatrices();
        }

        // Diagonalizes the Hamiltonian H
        void diagonalizeH() {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(system.H);
            if (solver.info() != Eigen::Success) {
                std::cerr << "Error diagonalizing Hamiltonian H" << std::endl;
                return;
            }
            system.exactEnergies = solver.eigenvalues();
            system.exactEigenvectors = solver.eigenvectors();

            // print eigenvalues
            std::cout << "Exact eigenvalues:\n" << system.exactEnergies << std::endl;
        }

        void setTElement(unsigned int i, unsigned int j, double value) {
            system.T(i, j) = value;
        }

        void setVElement(unsigned int i, unsigned int j, unsigned int k, unsigned int l, double value) {
            auto [idx1, sign1] = getIdxV(i, j);
            auto [idx2, sign2] = getIdxV(k, l);
            system.V(idx1, idx2) = sign1 * sign2 * value;
        }

        void updateKMatrix() {
            generateKMatrix();
        }

        void updateHMatrix() {
            generateHMatrix();
        }

        unsigned int getM() const {
            return system.M;
        }

        unsigned int getN() const {
            return system.N;
        }

        double getTElement(unsigned int i, unsigned int j) const {
            return system.T(i, j);
        }

        double getVElement(unsigned int i, unsigned int j, unsigned int k, unsigned int l) const {
            auto [idx1, sign1] = getIdxV(i, j);
            auto [idx2, sign2] = getIdxV(k, l);
            return sign1 * sign2 * system.V(idx1, idx2);
        }

        double getKElement(unsigned int i, unsigned int j, unsigned int k, unsigned int l) const {
            auto [idx1, sign1] = getIdxV(i, j);
            auto [idx2, sign2] = getIdxV(k, l);
            return sign1 * sign2 * system.K(idx1, idx2);
        }

        void printSystem() const {
            std::cout << "Number of fermions (N): " << system.N << std::endl;
            std::cout << "Number of fermionic states (M): " << system.M << std::endl;
            std::cout << "g: " << system.g << std::endl;
            std::cout << "Dimensions single-particle interaction matrix T:\n" << system.T.rows() << " x " << system.T.cols() << std::endl;
            std::cout << "Dimensions two-particle interaction matrix V:\n" << system.V.rows() << " x " << system.V.cols() << std::endl;
        }

        void loadFromCSV(const std::string& t_file, const std::string& v_file) {
            // 1. Read T matrix
            system.T = Eigen::MatrixXd::Zero(system.M, system.M);
            std::ifstream t_stream(t_file);
            std::string line, cell;
            unsigned int row = 0;
            
            while (std::getline(t_stream, line) && row < system.M) {
                std::stringstream lineStream(line);
                unsigned int col = 0;
                while (std::getline(lineStream, cell, ',') && col < system.M) {
                    system.T(row, col) = std::stod(cell);
                    col++;
                }
                row++;
            }

            // 2. Read flat V tensor
            std::vector<double> v_flat;
            std::ifstream v_stream(v_file);
            while (std::getline(v_stream, line)) {
                std::stringstream lineStream(line);
                while (std::getline(lineStream, cell, ',')) {
                    v_flat.push_back(std::stod(cell));
                }
            }

            // 3. Map 4D V tensor to your 2D antisymmetric V matrix
            unsigned int P = system.M * (system.M - 1) / 2;
            system.V = Eigen::MatrixXd::Zero(P, P);

            for (unsigned int l = 0; l < system.M; l++) {
                for (unsigned int k = 0; k < l; k++) {
                    auto [idx2, sign2] = getIdxV(k, l);
                    for (unsigned int j = 0; j < system.M; j++) {
                        for (unsigned int i = 0; i < j; i++) {
                            auto [idx1, sign1] = getIdxV(i, j);
                            
                            // Calculate 1D index for the flat 4D array (i, j, k, l)
                            unsigned int flat_idx = i * (system.M * system.M * system.M) + 
                                                    j * (system.M * system.M) + 
                                                    k * system.M + 
                                                    l;
                            
                            // Calculate 1D index for the exchange term (i, j, l, k)
                            unsigned int flat_idx_exch = i * (system.M * system.M * system.M) + 
                                                        j * (system.M * system.M) + 
                                                        l * system.M + 
                                                        k;

                            // Antisymmetrize the physical integral
                            double v_val = v_flat[flat_idx] - v_flat[flat_idx_exch];
                            
                            // Assign to the super-index matrix
                            system.V(idx1, idx2) = sign1 * sign2 * v_val;
                        }
                    }
                }
            }
            
            // 4. Rebuild K and H with the newly loaded physics
            generateKMatrix();
            generateHMatrix();
        }



        
};


#endif