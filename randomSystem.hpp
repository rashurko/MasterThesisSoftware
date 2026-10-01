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

    // ----------------------
    // SDP

    // Basis of traceless matrices
    std::vector<std::vector<Eigen::MatrixXd>> fBasis;
    std::vector<std::vector<double>> fCoeffs;

    // 2-RDM and conditions
    Eigen::MatrixXd Gamma;
    Eigen::MatrixXd rho;
    Eigen::MatrixXd Q;
    Eigen::MatrixXd G;
    Eigen::MatrixXd L; // Matrix representability conditions 

    // Representability conditions in f_i basis
    Eigen::MatrixXd Gamma_1;
    std::vector<std::vector<Eigen::MatrixXd>> Gamma_f_i;
    Eigen::MatrixXd Q_1;
    std::vector<std::vector<Eigen::MatrixXd>> Q_f_i;
    Eigen::MatrixXd G_1;
    std::vector<std::vector<Eigen::MatrixXd>> G_f_i;
    Eigen::MatrixXd L_1;
    std::vector<std::vector<Eigen::MatrixXd>> L_f_i;
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
                            system.K(idx1, idx2) += (1.0 / ((system.N - 1))) * sign1 * sign2 * factorT;
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
                                        int sign = pow(-1, countFront(ket, k) + countFront(ket, l) + countFront(bra, m) + countFront(bra, n));
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
            // std::cout << "Matrix H:\n" << system.H << std::endl;
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

        // -----------------------------------------------------------------------
        // SDP

        // Returns the index I of pair (i, j) for the G matrix
        unsigned int getGIndex(unsigned int i, unsigned int j) const {
            return j + system.M * i;
        }

        double getGammaElement(unsigned int i, unsigned int j, unsigned int k, unsigned int l) const {
            if (i == j || k == l) {
                return 0.0; // V is antisymmetric, so V(i,i,k,l) = V(i,j,k,k) = 0
            }

            auto [idx1, sign1] = getIdxV(i, j);
            auto [idx2, sign2] = getIdxV(k, l);
            return sign1 * sign2 * system.Gamma(idx1, idx2);
        }

        double calcRhoElement(unsigned int i, unsigned int j) const {
            double prefactor = 1.0 / (system.N - 1);
            double result = 0.0;
            for (unsigned int k = 0; k < system.M; k++) {
                result += getGammaElement(i, k, j, k);
            }
            return prefactor * result;
        }

        void calcRho() {
            system.rho = Eigen::MatrixXd::Zero(system.M, system.M);
            for (unsigned int i = 0; i < system.M; i++) {
                for (unsigned int j = 0; j <= i; j++) {
                    double rho_ij = calcRhoElement(i, j);
                    system.rho(i, j) = rho_ij;
                    system.rho(j, i) = rho_ij; // Exploit symmetry
                }
            }
        }

        double calcQElement(unsigned int i, unsigned int j, unsigned int k, unsigned int l) const {
            if (i == j || k == l) {
                return 0.0;
            }

            double result = getGammaElement(i, j, k, l);

            if (i == k && j == l) {
                result += 1 - system.rho(j, l) - system.rho(i, k);
            } else if (i == k) {
                result += -system.rho(j, l);
            } else if (j == l) {
                result += -system.rho(i, k);
            }

            if (i == l && j == k) {
                result += -1 + system.rho(j, k) + system.rho(i, l);
            } else if (i == l) {
                result += system.rho(j, k);
            } else if (j == k) {
                result += system.rho(i, l);
            }

            return result;
        }

        void calcQ() {
            unsigned int D = system.M * (system.M  - 1) / 2;
            system.Q = Eigen::MatrixXd::Zero(D, D);
            for (unsigned int j = 0; j < system.M; j++) {
                for (unsigned int i = 0; i < j; i++) {
                    auto [idx1, sign1] = getIdxV(i, j);
                    for (unsigned int l = 0; l < system.M; l++) {
                        for (unsigned int k = 0; k < l; k++) {
                            auto [idx2, sign2] = getIdxV(k, l);

                            double Q_ijkl = calcQElement(i, j, k, l);
                            system.Q(idx1, idx2) = sign1 * sign2 * Q_ijkl;
                        }
                    }
                }
            }
        }
        
        double calcGElement(unsigned int i, unsigned int j, unsigned int k, unsigned int l) const {
            double G_ijkl = -getGammaElement(i, l, k, j);
            if (j == l) {
                G_ijkl += system.rho(i, k);
            }

            return G_ijkl;
        }

        void calcG() {
            for (unsigned int i = 0; i < system.M; i++) {
                for (unsigned int j = 0; j < system.M; j++) {
                    unsigned int idx1 = getGIndex(i, j);
                    for (unsigned int k = 0; k < system.M; k++) {
                        for (unsigned int l = 0; l < system.M; l++) {
                            unsigned int idx2 = getGIndex(k, l);

                            system.G(idx1, idx2) = calcGElement(i, j, k, l);
                        }
                    }
                }
            }
        }

        void initGamma() {
            unsigned int D = system.M * (system.M - 1) / 2;
            double prefactor = static_cast<double>(system.N * (system.N - 1)) / (2 * D);
            system.Gamma = Eigen::MatrixXd::Identity(D, D) * prefactor;
        }

        void initQ () {
            unsigned int D = system.M * (system.M  - 1) / 2;
            double prefactor = 1.0 - 2*static_cast<double>(system.N) / (system.M) + static_cast<double>(system.N * (system.N - 1)) / (system.M * (system.M - 1));
            system.Q = Eigen::MatrixXd::Identity(D, D) * prefactor;
        }

        void initG() {
            unsigned int D = system.M * system.M;
            double prefactor = static_cast<double>(system.N * (system.M - system.N)) / (system.M * (system.M - 1));
            system.G = Eigen::MatrixXd::Identity(D, D) * prefactor;

            double prefactor2 = static_cast<double>(system.N * (system.N - 1)) / (system.M * (system.M - 1));
            for (unsigned int j = 0; j < system.M; j++) {
                for (unsigned int i = 0; i <= j; i++) {
                    unsigned int idx1 = getGIndex(i, i);
                    unsigned int idx2 = getGIndex(j, j);
                    system.G(idx1, idx2) += prefactor2;
                    if (idx1 != idx2) {
                        system.G(idx2, idx1) += prefactor2;
                    }
                }
            }
        }

        void calculateGamma() {
            initGamma();

            for (unsigned int i = 0; i < system.fBasis.size(); i++) {
                for (unsigned int j = 0; j < system.fBasis[i].size(); j++) {
                    double coeff = system.fCoeffs[i][j];
                    system.Gamma += coeff * system.fBasis[i][j];
                }
            }
        }

        void generateL() {
            unsigned int dimGamma = system.Gamma.rows();
            unsigned int dimQ = system.Q.rows();
            unsigned int dimG = system.G.rows();
            
            unsigned int D = dimGamma + dimQ + dimG;
            system.L = Eigen::MatrixXd::Zero(D, D);

            system.L.block(0, 0, dimGamma, dimGamma) = system.Gamma;
            system.L.block(dimGamma, dimGamma, dimQ, dimQ) = system.Q;
            system.L.block(dimGamma + dimQ, dimGamma + dimQ, dimG, dimG) = system.G;
        }

        // f_i basis
        // -------------------------------------------------------------------------

        // Generate a basis of traceless matrices
        void generateBasisF() {
            system.fBasis.clear();
            unsigned int D = system.M * (system.M - 1) / 2;

            for (unsigned int j = 0; j < D; j++) {
                std::vector<Eigen::MatrixXd> fBasis_j;
                std::vector<double> fCoeffs_j;
                for (unsigned int i = 0; i <= j; i++) {
                    Eigen::MatrixXd f_ij = Eigen::MatrixXd::Zero(D, D);
                    // Generate orthonormal traceless matrices: f_ii = 0
                    if (i != j) {
                        f_ij(i, j) = 1.0;
                        f_ij(j, i) = 1.0;
                        f_ij /= f_ij.norm();
                        fBasis_j.push_back(f_ij);
                        fCoeffs_j.push_back(0.0);
                    }
                    // Generate orthonormal traceless matrices with elements only on the diagonal (using Gram-Schmidt algorithm)
                    else if (i != D - 1) {
                        // Initialize initial diagonal matrix
                        f_ij(0, 0) = 1.0;
                        f_ij(i+1, i+1) = -1.0;

                        // Use Gram-Schmidt algorithm to construct orthonormal vectors
                        Eigen::MatrixXd f_ij_orth = f_ij;
                        for (unsigned int k = 0; k < i; k++) {
                            f_ij_orth -= productDiagMatrices(system.fBasis[k][k], f_ij) * system.fBasis[k][k];
                        }
                        f_ij_orth /= normDiagMatrix(f_ij_orth);

                        fBasis_j.push_back(f_ij_orth);
                        fCoeffs_j.push_back(0.0);
                    }
                }
                system.fBasis.push_back(fBasis_j);
                system.fCoeffs.push_back(fCoeffs_j);
            }
        }

        // Test that the fBasis is traceless.
        void testFBasisTraceless() const {
            for (const auto& fBasis_j : system.fBasis) {
                for (const auto& f_ij : fBasis_j) {
                    double trace = f_ij.trace();
                    if (std::abs(trace) > 1e-10) {
                        std::cerr << "fBasis is not traceless!" << std::endl;
                        return;
                    }
                }
            }
            std::cout << "fBasis is traceless." << std::endl;
        }

        // Test that the fBasis is orthonormal
        void testFBasisOrthonormal() const {
            for (const auto& fBasis_j : system.fBasis) {
                for (const auto& f_ij : fBasis_j) {
                    double norm = f_ij.norm();
                    if (std::abs(norm - 1.0) > 1e-10) {
                        std::cout << norm << std::endl;
                        std::cerr << "fBasis is not orthonormal!" << std::endl;
                        return;
                    }
                    for (const auto& fBasis_k : system.fBasis) {
                        for (const auto& f_kl : fBasis_k) {
                            if (&f_ij != &f_kl) {
                                double innerProduct = (f_ij.array() * f_kl.array()).sum();
                                if (std::abs(innerProduct) > 1e-10) {
                                    std::cerr << "fBasis is not orthonormal!" << std::endl;
                                    return;
                                }
                            }
                        }
                    }
                }
            }
            std::cout << "fBasis is orthonormal." << std::endl;
        }

        double get_fi_element(unsigned int i, unsigned int j, unsigned int k, unsigned int l, Eigen::MatrixXd f_i) const {
            if (i == j || k == l) {
                return 0.0; // f_i is antisymmetric, so f_i(i,i,k,k) = -V(i,i,k,k) = 0
            }

            auto [idx1, sign1] = getIdxV(i, j);
            auto [idx2, sign2] = getIdxV(k, l);
            return sign1 * sign2 * f_i(idx1, idx2);
        }

        double F_ij(unsigned int i, unsigned int j, Eigen::MatrixXd f_i) const {
            double result = 0.0;
            for (unsigned int k = 0; k < system.M; k++) {
                result += get_fi_element(i, k, j, k, f_i);
            }
            result *= 1.0 / (system.N - 1);
            return result;
        }

        void generateGamma_1() {
            unsigned int D = system.M * (system.M - 1) / 2;
            double prefactor = static_cast<double>(system.N * (system.N - 1)) / (2 * D);
            system.Gamma_1 = Eigen::MatrixXd::Identity(D, D) * prefactor;

        }

        void generateGamma_f_i() {
            system.Gamma_f_i.clear();
            system.Gamma_f_i = system.fBasis;
        }

        double calcGElementf_i(unsigned int i, unsigned int j, unsigned int k, unsigned int l, Eigen::MatrixXd f_i) {
            double G_ijkl = -get_fi_element(i, l, k, j, f_i);
            if (j == l) {
                double sum = F_ij(i, k, f_i);
                G_ijkl += sum;
            }
            return G_ijkl;

        }

        Eigen::MatrixXd calcGf_i(Eigen::MatrixXd f_i) {
            unsigned int D = system.M * system.M;
            Eigen::MatrixXd Gf_i = Eigen::MatrixXd::Zero(D, D);

            for (unsigned int i = 0; i < system.M; i++) {
                for (unsigned int j = 0; j < system.M; j++) {
                    unsigned int idx1 = getGIndex(i, j);
                    for (unsigned int k = 0; k < system.M; k++) {
                        for (unsigned int l = 0; l < system.M; l++) {
                            unsigned int idx2 = getGIndex(k, l);

                            Gf_i(idx1, idx2) = calcGElementf_i(i, j, k, l, f_i);
                        }
                    }
                }
            }
            return Gf_i;
        }

        void generateG_1() {
            unsigned int D_I = system.M * (system.M - 1) / 2;
            Eigen::MatrixXd I = Eigen::MatrixXd::Identity(D_I, D_I);

            system.G_1 = calcGf_i(I);
        }

        void generateG_f_i() {
            system.G_f_i.clear();
            system.G_f_i.resize(system.fBasis.size());
            for (unsigned int i = 0; i < system.fBasis.size(); i++) {
                system.G_f_i[i].resize(system.fBasis[i].size());
                for (unsigned int j = 0; j < system.fBasis[i].size(); j++) {
                    system.G_f_i[i][j] = calcGf_i(system.fBasis[i][j]);
                }
            }
        }

        double calcQElementf_i(unsigned int i, unsigned int j, unsigned int k, unsigned int l, Eigen::MatrixXd f_i) {
            if (i == j || k == l) {
                return 0.0;
            }

            double result = get_fi_element(i, j, k, l, f_i);

            if (i == k && j == l) {
                result += 1 - F_ij(j, l, f_i) - F_ij(i, k, f_i);
            } else if (i == k) {
                result += -F_ij(j, l, f_i);
            } else if (j == l) {
                result += -F_ij(i, k, f_i);
            }

            if (i == l && j == k) {
                result += -1 + F_ij(j, k, f_i) + F_ij(i, l, f_i);
            } else if (i == l) {
                result += F_ij(j, k, f_i);
            } else if (j == k) {
                result += F_ij(i, l, f_i);
            }

            return result;
        }

        Eigen::MatrixXd calcQf_i(Eigen::MatrixXd f_i) {
            unsigned int D = system.M * (system.M  - 1) / 2;
            Eigen::MatrixXd Qf_i = Eigen::MatrixXd::Zero(D, D);
            for (unsigned int j = 0; j < system.M; j++) {
                for (unsigned int i = 0; i < j; i++) {
                    auto [idx1, sign1] = getIdxV(i, j);
                    for (unsigned int l = 0; l < system.M; l++) {
                        for (unsigned int k = 0; k < l; k++) {
                            auto [idx2, sign2] = getIdxV(k, l);

                            double Q_ijkl = calcQElementf_i(i, j, k, l, f_i);
                            Qf_i(idx1, idx2) = sign1 * sign2 * Q_ijkl;
                        }
                    }
                }
            }
            return Qf_i;
        }

        void generateQ_1() {
            unsigned int D_I = system.M * (system.M - 1) / 2;
            Eigen::MatrixXd I = Eigen::MatrixXd::Identity(D_I, D_I);

            system.Q_1 = calcQf_i(I);
        }

        void generateQ_f_i() {
            system.Q_f_i.clear();
            system.Q_f_i.resize(system.fBasis.size());
            for (unsigned int i = 0; i < system.fBasis.size(); i++) {
                system.Q_f_i[i].resize(system.fBasis[i].size());
                for (unsigned int j = 0; j < system.fBasis[i].size(); j++) {
                    system.Q_f_i[i][j] = calcQf_i(system.fBasis[i][j]);
                }
            }
        }

        void generateL_1() {
            unsigned int dimGamma = system.Gamma_1.rows();
            unsigned int dimQ = system.Q_1.rows();
            unsigned int dimG = system.G_1.rows();

            unsigned int D = dimGamma + dimQ + dimG;
            system.L_1 = Eigen::MatrixXd::Zero(D, D);

            system.L_1.block(0, 0, dimGamma, dimGamma) = system.Gamma_1;
            system.L_1.block(dimGamma, dimGamma, dimQ, dimQ) = system.Q_1;
            system.L_1.block(dimGamma + dimQ, dimGamma + dimQ, dimG, dimG) = system.G_1;
        }

        void generateL_f_i() {
            system.L_f_i.clear();
            system.L_f_i.resize(system.fBasis.size());
            for (unsigned int i = 0; i < system.fBasis.size(); i++) {
                system.L_f_i[i].resize(system.fBasis[i].size());
                for (unsigned int j = 0; j < system.fBasis[i].size(); j++) {
                    unsigned int dimGamma = system.Gamma_f_i[i][j].rows();
                    unsigned int dimQ = system.Q_f_i[i][j].rows();
                    unsigned int dimG = system.G_f_i[i][j].rows();

                    system.L_f_i[i][j] = Eigen::MatrixXd::Zero(dimGamma + dimQ + dimG, dimGamma + dimQ + dimG);
                    system.L_f_i[i][j].block(0, 0, dimGamma, dimGamma) = system.Gamma_f_i[i][j];
                    system.L_f_i[i][j].block(dimGamma, dimGamma, dimQ, dimQ) = system.Q_f_i[i][j];
                    system.L_f_i[i][j].block(dimGamma + dimQ, dimGamma + dimQ, dimG, dimG) = system.G_f_i[i][j];
                }
            }
        }

        // Calculate L from f basis
        void calcL_f() {
            system.L = static_cast<double>(system.N * (system.N - 1)) / (system.M * (system.M - 1)) * system.L_1;
            for (unsigned int i = 0; i < system.fBasis.size(); i++) {
                for (unsigned int j = 0; j < system.fBasis[i].size(); j++) {
                    system.L += system.fCoeffs[i][j] * system.L_f_i[i][j];
                }
            }
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
            // std::cout << "Exact eigenvalues:\n" << system.exactEnergies << std::endl;
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
            if (i == j || k == l) {
                return 0.0; // V is antisymmetric, so V(i,i,k,l) = V(i,j,k,k) = 0
            }

            auto [idx1, sign1] = getIdxV(i, j);
            auto [idx2, sign2] = getIdxV(k, l);
            return sign1 * sign2 * system.V(idx1, idx2);
        }

        double getKElement(unsigned int i, unsigned int j, unsigned int k, unsigned int l) const {
            if (i == j || k == l) {
                return 0.0; // K is antisymmetric, so K(i,i,k,l) = K(i,j,k,k) = 0
            }

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

        void saveToJson(const std::string& filename) const {
            std::ofstream file(filename);
            if (!file.is_open()) {
                std::cerr << "Could not open file for writing: " << filename << std::endl;
                return;
            }

            file << "{\n";
            file << "  \"N\": " << system.N << ",\n";
            file << "  \"M\": " << system.M << ",\n";
            file << "  \"g\": " << system.g << ",\n";

            // Save T matrix
            file << "  \"T\": [\n";
            for (unsigned int i = 0; i < system.M; ++i) {
                file << "    [";
                for (unsigned int j = 0; j < system.M; ++j) {
                    file << system.T(i, j);
                    if (j < system.M - 1) file << ", ";
                }
                file << "]";
                if (i < system.M - 1) file << ",";
                file << "\n";
            }
            file << "  ],\n";

            // Save V matrix
            unsigned int P = system.M * (system.M - 1) / 2;
            file << "  \"V\": [\n";
            for (unsigned int i = 0; i < P; ++i) {
                file << "    [";
                for (unsigned int j = 0; j < P; ++j) {
                    file << system.V(i, j);
                    if (j < P - 1) file << ", ";
                }
                file << "]";
                if (i < P - 1) file << ",";
                file << "\n";
            }
            file << "  ],\n";

            // Save K matrix
            file << "  \"K\": [\n";
            for (unsigned int i = 0; i < P; ++i) {
                file << "    [";
                for (unsigned int j = 0; j < P; ++j) {
                    file << system.K(i, j);
                    if (j < P - 1) file << ", ";
                }
                file << "]";
                if (i < P - 1) file << ",";
                file << "\n";
            }
            file << "  ],\n";

            // Save H matrix
            file << "  \"H\": [\n";
            for (unsigned int i = 0; i < system.dimBasis; ++i) {
                file << "    [";
                for (unsigned int j = 0; j < system.dimBasis; ++j) {
                    file << system.H(i, j);
                    if (j < system.dimBasis - 1) file << ", ";
                }
                file << "]";
                if (i < system.dimBasis - 1) file << ",";
                file << "\n";
            }
            file << "  ],\n";

            // Save eigenenergies
            file << "  \"eigenenergies\": [\n";
            for (unsigned int i = 0; i < system.dimBasis; ++i) {
                file << "    " << system.exactEnergies[i];
                if (i < system.dimBasis - 1) file << ",";
                file << "\n";
            }
            file << "  ],\n";

            // Save eigenvectors
            file << "  \"eigenvectors\": [\n";
            for (unsigned int i = 0; i < system.dimBasis; ++i) {
                file << "    [";
                for (unsigned int j = 0; j < system.dimBasis; ++j) {
                    file << system.exactEigenvectors(i, j);
                    if (j < system.dimBasis - 1) file << ", ";
                }
                file << "]";
                if (i < system.dimBasis - 1) file << ",";
                file << "\n";
            }
            file << "  ]\n";

            file << "}\n";

        }

        // -----------------------------------------------
        // SDP

        void performPotentialReduction() {
            // Generate basis for traceless matrices
            generateBasisF();

            // Test that the fBasis is traceless
            // testFBasisTraceless();
            // Test that the fBasis is orthonormal
            // testFBasisOrthonormal();

            // Initialize 2-RDM and conditions
            initGamma();
            initQ();
            initG();
            generateL();

            // Test
            //calculateGamma();
            //calcRho();
            //calcQ();
            //calcG();

            generateGamma_1();
            generateGamma_f_i();
            generateG_1();
            generateG_f_i();
            generateQ_1();
            generateQ_f_i();
            generateL_1();
            generateL_f_i();
            calcL_f();

        }

        // Save the result of Potential Reduction to a Json
        void saveToJsonPR(const std::string& filename) const {
            std::ofstream file(filename);
            if (!file) {
                std::cerr << "Error opening file for writing: " << filename << std::endl;
                return;
            }

            file << "{\n";

            // Basis {f}
            file << "  \"f Basis\": [\n";
            for (unsigned int i = 0; i < system.fBasis.size(); i++) {
                for (unsigned int j = 0; j < system.fBasis[i].size(); j++) {
                    file << "[";
                    for (unsigned int k = 0; k < system.fBasis[i][j].rows(); k++) {
                        file << "[";
                        for (unsigned int l = 0; l < system.fBasis[i][j].cols(); l++) {
                            file << system.fBasis[i][j](k, l);
                            if (l < system.fBasis[i][j].cols() - 1) file << ", ";
                        }
                        file << "]";
                        if (k < system.fBasis[i][j].rows() - 1) file << ",\n ";
                    }
                    file << "]";
                    if (not(i == system.fBasis.size() - 1 && j == system.fBasis[i].size() - 1)) file << ",";
                    file << "\n";
                }
            }
            file << "  ],\n";

            // Gamma
            file << "  \"Gamma\": [\n";
            for (unsigned int i = 0; i < system.Gamma.rows(); i++) {
                file << "    [";
                for (unsigned int j = 0; j < system.Gamma.cols(); j++) {
                    file << system.Gamma(i, j);
                    if (j < system.Gamma.cols() - 1) file << ", ";
                }
                file << "]";
                if (i < system.Gamma.rows() - 1) file << ",";
                file << "\n";
            }
            file << "  ],\n";

            // Q matrix
            file << "  \"Q\": [\n";
            for (unsigned int i = 0; i < system.Q.rows(); i++) {
                file << "    [";
                for (unsigned int j = 0; j < system.Q.cols(); j++) {
                    file << system.Q(i, j);
                    if (j < system.Q.cols() - 1) file << ", ";
                }
                file << "]";
                if (i < system.Q.rows() - 1) file << ",";
                file << "\n";
            }
            file << "  ],\n";

            // G matrix
            file << "  \"G\": [\n";
            for (unsigned int i = 0; i < system.G.rows(); i++) {
                file << "    [";
                for (unsigned int j = 0; j < system.G.cols(); j++) {
                    file << system.G(i, j);
                    if (j < system.G.cols() - 1) file << ", ";
                }
                file << "]";
                if (i < system.G.rows() - 1) file << ",";
                file << "\n";
            }
            file << "  ],\n";

            // L matrix
            file << "  \"L\": [\n";
            for (unsigned int i = 0; i < system.L.rows(); i++) {
                file << "    [";
                for (unsigned int j = 0; j < system.L.cols(); j++) {
                    file << system.L(i, j);
                    if (j < system.L.cols() - 1) file << ", ";
                }
                file << "]";
                if (i < system.L.rows() - 1) file << ",";
                file << "\n";
            }
            file << "  ]\n";

            file << "}\n";
        }
};


#endif