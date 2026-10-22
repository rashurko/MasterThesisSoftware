#ifndef RANDOMSYSTEM_HPP
#define RANDOMSYSTEM_HPP

#include <iostream>
#include <cmath>

#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <unistd.h>
#include <stdexcept>
#include <thread>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/IterativeLinearSolvers>

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

    // 2-RDM
    Eigen::MatrixXd Gamma_exact;

    // ----------------------
    // SDP

    // Basis of traceless matrices
    std::vector<std::vector<Eigen::MatrixXd>> fBasis;
    std::vector<std::vector<double>> fCoeffs;

    // 2-RDM and conditions
    Eigen::MatrixXd Gamma;
    Eigen::MatrixXd Gamma_inv;
    Eigen::MatrixXd rho;
    Eigen::MatrixXd Q;
    Eigen::MatrixXd Q_inv;
    Eigen::MatrixXd G;
    Eigen::MatrixXd G_inv;
    Eigen::MatrixXd L; // Matrix representability conditions 

    // Representability conditions in f_i basis
    Eigen::MatrixXd Gamma_1;
    std::vector<std::vector<Eigen::MatrixXd>> Gamma_f_i;
    Eigen::MatrixXd Qlin_1;
    std::vector<std::vector<Eigen::MatrixXd>> Qlin_f_i;
    Eigen::MatrixXd G_1;
    std::vector<std::vector<Eigen::MatrixXd>> G_f_i;
    Eigen::MatrixXd Llin_1;
    std::vector<std::vector<Eigen::MatrixXd>> Llin_f_i;

    Eigen::MatrixXd Hessian;
    std::vector<std::vector<Eigen::MatrixXd>> Hessian_f_i;
    // Conjugate Gradients
    Eigen::MatrixXd A;
    Eigen::VectorXd x;
    Eigen::VectorXd b;

    double Eground_pr;
    double duality_gap;

    // Projection method
    double alpha_p;
    Eigen::MatrixXd Gamma_p;


    double durationPr;
    double durationDiag;
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
            system.basis.reserve(system.dimBasis);

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
            //system.K *= 0.5;
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
                                        H_ij += sign * getKElement(k, l, m, n);
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

        void calcExactGammaGroundState() {
            unsigned int dimGamma = system.M * (system.M - 1) / 2;
            system.Gamma_exact = Eigen::MatrixXd::Zero(dimGamma, dimGamma);
            Eigen::VectorXd groundState = system.exactEigenvectors.col(0);

            for (unsigned int j = 0; j < system.M; j++) {
                for (unsigned int i = 0; i < j; i++) {
                    for (unsigned int l = 0; l < system.M; l++) {
                        for (unsigned int k = 0; k < l; k++) {

                            // Loop over basis states
                            for (unsigned int m = 0; m < system.dimBasis; m++) {
                                for (unsigned int n = 0; n < system.dimBasis; n++) {
                                    auto ket = system.basis[m];
                                    auto bra = system.basis[n];
                                    double Cm = groundState[m];
                                    double Cn = groundState[n];
                                    double Gamma_ijkl = 0.0;

                                    // Count the number of different elements in eigenstates
                                    unsigned int diff = 0;
                                    for (unsigned int idx = 0; idx < system.M; idx++) {
                                        if (ket[idx] != bra[idx]) {
                                            diff++;
                                        }
                                    }

                                    if (diff > 4) {
                                        continue;
                                    }

                                    if (i == j || ket[i] == 0 || ket[j] == 0) {
                                        continue;
                                    }

                                    if (k == l || bra[k] == 0 || bra[l] == 0) {
                                        continue;
                                    }

                                    auto ketNew = ket;
                                    auto braNew = bra;
                                    for (unsigned int p = 0; p < system.M; p++) {
                                        if (p == i) {
                                            ketNew[p] -= 1;
                                        }
                                        if (p == j) {
                                            ketNew[p] -= 1;
                                        }
                                        if (p == k) {
                                            braNew[p] -= 1;
                                        }
                                        if (p == l) {
                                            braNew[p] -= 1;
                                        }
                                    }

                                    // Are ketNew and braNew equal?
                                    if (ketNew == braNew) {
                                        int sign = pow(-1, countFront(ket, i) + countFront(ket, j) + countFront(bra, k) + countFront(bra, l));
                                        Gamma_ijkl += sign * Cm * Cn;
                                    }
                                    auto [idx1, sign1] = getIdxV(i, j);
                                    auto [idx2, sign2] = getIdxV(k, l);
                                    system.Gamma_exact(idx1, idx2) += sign1 * sign2 * Gamma_ijkl;
                                }
                            }
                        }
                    }
                }
            }
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

        double getShortElement(unsigned int i, unsigned int j, unsigned int k, unsigned int l, Eigen::MatrixXd& mat) const {
            if (i == j || k == l) {
                return 0.0; // V is antisymmetric, so V(i,i,k,l) = V(i,j,k,k) = 0
            }

            auto [idx1, sign1] = getIdxV(i, j);
            auto [idx2, sign2] = getIdxV(k, l);
            return sign1 * sign2 * mat(idx1, idx2);
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

        void calcGamma_inv() {
            system.Gamma_inv = system.Gamma.ldlt().solve(Eigen::MatrixXd::Identity(system.Gamma.rows(), system.Gamma.cols()));
        }

        void calcQ_inv() {
            system.Q_inv = system.Q.ldlt().solve(Eigen::MatrixXd::Identity(system.Q.rows(), system.Q.cols()));
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

        void calcG_inv() {
            system.G_inv = system.G.ldlt().solve(Eigen::MatrixXd::Identity(system.G.rows(), system.G.cols()));
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

        double get_fi_element(unsigned int i, unsigned int j, unsigned int k, unsigned int l, const Eigen::MatrixXd& f_i) const {
            if (i == j || k == l) {
                return 0.0; // f_i is antisymmetric, so f_i(i,i,k,k) = -V(i,i,k,k) = 0
            }

            auto [idx1, sign1] = getIdxV(i, j);
            auto [idx2, sign2] = getIdxV(k, l);
            return sign1 * sign2 * f_i(idx1, idx2);
        }

        double F_ij(unsigned int i, unsigned int j, const Eigen::MatrixXd& f_i) const {
            double result = 0.0;
            for (unsigned int k = 0; k < system.M; k++) {
                result += get_fi_element(i, k, j, k, f_i);
            }
            result *= 1.0 / (system.N - 1);
            return result;
        }

        void generateGamma_1() {
            unsigned int D = system.M * (system.M - 1) / 2;
            system.Gamma_1 = Eigen::MatrixXd::Identity(D, D);

        }

        void generateGamma_f_i() {
            system.Gamma_f_i.clear();
            system.Gamma_f_i = system.fBasis;
        }

        double calcGElementf_i(unsigned int i, unsigned int j, unsigned int k, unsigned int l, const Eigen::MatrixXd& f_i) const {
            double G_ijkl = -get_fi_element(i, l, k, j, f_i);
            if (j == l) {
                double sum = F_ij(i, k, f_i);
                G_ijkl += sum;
            }
            return G_ijkl;

        }

        Eigen::MatrixXd calcGf_i(const Eigen::MatrixXd& f_i) const {
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

        double calcGAdjElement(unsigned int i, unsigned int j, unsigned int k, unsigned int l, const Eigen::MatrixXd& mat) const {
            unsigned int I1 = getGIndex(i, l);
            unsigned int J1 = getGIndex(k, j);
            unsigned int I2 = getGIndex(j, l);
            unsigned int J2 = getGIndex(k, i);
            unsigned int I3 = getGIndex(i, k);
            unsigned int J3 = getGIndex(l, j);
            unsigned int I4 = getGIndex(j, k);
            unsigned int J4 = getGIndex(l, i);
            double G_ijkl = (-mat(I1, J1) + mat(I2, J2) + mat(I3, J3) - mat(I4, J4));

            double sum = 0.0;
            if (j == l) {
                for (unsigned int m = 0; m < system.M; m++) {
                    unsigned int idx1 = getGIndex(i, m);
                    unsigned int idx2 = getGIndex(k, m);
                    sum += mat(idx1, idx2);
                }
            }
            if (i == l) {
                for (unsigned int m = 0; m < system.M; m++) {
                    unsigned int idx1 = getGIndex(j, m);
                    unsigned int idx2 = getGIndex(k, m);
                    sum -= mat(idx1, idx2);
                }
            }
            if (j == k) {
                for (unsigned int m = 0; m < system.M; m++) {
                    unsigned int idx1 = getGIndex(i, m);
                    unsigned int idx2 = getGIndex(l, m);
                    sum -= mat(idx1, idx2);
                }
            }
            if (i == k) {
                for (unsigned int m = 0; m < system.M; m++) {
                    unsigned int idx1 = getGIndex(j, m);
                    unsigned int idx2 = getGIndex(l, m);
                    sum += mat(idx1, idx2);
                }
            }

            G_ijkl += sum / ((system.N - 1));

            return G_ijkl;
        }

        Eigen::MatrixXd calcGAdj(const Eigen::MatrixXd& mat) const {
            unsigned int D = system.M * (system.M - 1) / 2;
            Eigen::MatrixXd GAdj = Eigen::MatrixXd::Zero(D, D);

            for (unsigned int i = 0; i < system.M; i++) {
                for (unsigned int j = 0; j < i; j++) {
                    auto [idx1, sign1] = getIdxV(i, j);
                    for (unsigned int k = 0; k < system.M; k++) {
                        for (unsigned int l = 0; l < k; l++) {
                            auto [idx2, sign2] = getIdxV(k, l);
                            GAdj(idx1, idx2) = calcGAdjElement(i, j, k, l, mat);
                        }
                    }
                }
            }

            return GAdj;
        }

        double calcGAdjElement_new(unsigned int i, unsigned int j, unsigned int k, unsigned int l, const Eigen::MatrixXd& mat) const {
            unsigned int I = getGIndex(i, l);
            unsigned int J = getGIndex(k, j);
            double G_ijkl = -mat(I, J);

            if (j == l) {
                double sum = 0.0;
                for (unsigned int m = 0; m < system.M; m++) {
                    unsigned int idx1 = getGIndex(i, m);
                    unsigned int idx2 = getGIndex(k, m);
                    sum += mat(idx1, idx2);
                }
                G_ijkl += sum / ((system.N - 1));
            }

            return G_ijkl;
        }

        Eigen::MatrixXd calcGAdj_new(const Eigen::MatrixXd& mat) const {
            unsigned int D = system.M * (system.M - 1) / 2;
            Eigen::MatrixXd GAdj = Eigen::MatrixXd::Zero(D, D);

            for (unsigned int j = 0; j < system.M; j++) {
                for (unsigned int i = 0; i < j; i++) {
                    auto [idx1, sign1] = getIdxV(i, j);
                    for (unsigned int l = 0; l < system.M; l++) {
                        for (unsigned int k = 0; k < l; k++) {
                            auto [idx2, sign2] = getIdxV(k, l);
                            GAdj(idx1, idx2) = calcGAdjElement_new(i, j, k, l, mat);
                        }
                    }
                }
            }

            return GAdj;
        }

        

        void test_Gadj() const {
            // Test matrix Y
            Eigen::MatrixXd Y = Eigen::MatrixXd::Random(system.M * system.M, system.M * system.M);

            // Test for all basis matrices f_i
            for (unsigned int i = 0; i < system.fBasis.size(); i++) {
                for (unsigned int j = 0; j < system.fBasis[i].size(); j++) {
                    Eigen::MatrixXd f_i = system.fBasis[i][j];
                    Eigen::MatrixXd G_f_i = calcGf_i(f_i);
                    Eigen::MatrixXd GAdj_Y = calcGAdj(Y);

                    // Check if the adjoint property holds: <G_f_i, Y> = <f_i, GAdj_Y>
                    double lhs = (G_f_i.array() * Y.array()).sum();
                    double rhs = (f_i.array() * GAdj_Y.array()).sum();

                    std::cout<< lhs << " vs " << rhs << std::endl;

                    if (std::abs(lhs - rhs) > 1e-10) {
                        std::cerr << "Gadj test failed for fBasis[" << i << "][" << j << "]!" << std::endl;
                        return;
                    }
                }
            }
            std::cout << "Gadj test passed for all fBasis matrices." << std::endl;
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

        double calcQlinElementf_i(unsigned int i, unsigned int j, unsigned int k, unsigned int l, const Eigen::MatrixXd& f_i) const {
            if (i == j || k == l) {
                return 0.0;
            }

            double result = get_fi_element(i, j, k, l, f_i);

            if (i == k && j == l) {
                result += -F_ij(j, l, f_i) - F_ij(i, k, f_i);
            } else if (i == k) {
                result += -F_ij(j, l, f_i);
            } else if (j == l) {
                result += -F_ij(i, k, f_i);
            }

            if (i == l && j == k) {
                result += F_ij(j, k, f_i) + F_ij(i, l, f_i);
            } else if (i == l) {
                result += F_ij(j, k, f_i);
            } else if (j == k) {
                result += F_ij(i, l, f_i);
            }

            return result;
        }
        

        Eigen::MatrixXd calcQlinf_i(const Eigen::MatrixXd& f_i) const {
            unsigned int D = system.M * (system.M  - 1) / 2;
            Eigen::MatrixXd Qf_i = Eigen::MatrixXd::Zero(D, D);
            for (unsigned int j = 0; j < system.M; j++) {
                for (unsigned int i = 0; i < j; i++) {
                    auto [idx1, sign1] = getIdxV(i, j);
                    for (unsigned int l = 0; l < system.M; l++) {
                        for (unsigned int k = 0; k < l; k++) {
                            auto [idx2, sign2] = getIdxV(k, l);

                            double Q_ijkl = calcQlinElementf_i(i, j, k, l, f_i);
                            Qf_i(idx1, idx2) = sign1 * sign2 * Q_ijkl;
                        }
                    }
                }
            }
            return Qf_i;
        }

        Eigen::MatrixXd calcQf_i(const Eigen::MatrixXd& f_i) const {
            unsigned int D = system.M * (system.M - 1) / 2;
            Eigen::MatrixXd Qf_i = calcQlinf_i(f_i);
            for (unsigned int j = 0; j < system.M; j++) {
                for (unsigned int i = 0; i < j; i++) {
                    auto [idx1, sign1] = getIdxV(i, j);
                    for (unsigned int l = 0; l < system.M; l++) {
                        for (unsigned int k = 0; k < l; k++) {
                            auto [idx2, sign2] = getIdxV(k, l);
                            
                            if (i == k && j == l) {
                                Qf_i(idx1, idx2) += 1 * ((2 * f_i.trace()) / (system.N * (system.N - 1)));
                            } else if (i == l && j == k) {
                                Qf_i(idx1, idx2) -= 1 * ((2 * f_i.trace()) / (system.N * (system.N - 1)));
                            }
                        }
                    }
                }
            }

            return Qf_i;
        }

        double calcQAdjElement(unsigned int i, unsigned int j, unsigned int k, unsigned int l, const Eigen::MatrixXd& mat) const {
            double sum = 0.0;

            if (j == l) {
                for (unsigned int m = 0; m < system.M; m++) {
                    if (i == m || m == k) continue;

                    auto [I, signI] = getIdxV(i, m);
                    auto [J, signJ] = getIdxV(m, k);
                    sum += signI * signJ * mat(I, J);
                }
                sum *= 4.0 / ((system.N - 1));
            }

            return sum;
        }

        Eigen::MatrixXd calcQAdj(const Eigen::MatrixXd& mat) const {
            unsigned int D = system.M * (system.M - 1) / 2;
            Eigen::MatrixXd QAdj = mat;

            for (unsigned int j = 0; j < system.M; j++) {
                for (unsigned int i = 0; i < j; i++) {
                    auto [idx1, sign1] = getIdxV(i, j);
                    for (unsigned int l = 0; l < system.M; l++) {
                        for (unsigned int k = 0; k < l; k++) {
                            auto [idx2, sign2] = getIdxV(k, l);
                            QAdj(idx1, idx2) += calcQAdjElement(i, j, k, l, mat);
                        }
                    }
                }
            }

            return QAdj; 
        }

        void test_Qadj() const {
            // Test matrix Y
            Eigen::MatrixXd Y = Eigen::MatrixXd::Random(system.M * (system.M - 1) / 2, system.M * (system.M - 1) / 2);

            // Test for all basis matrices f_i
            for (unsigned int i = 0; i < system.fBasis.size(); i++) {
                for (unsigned int j = 0; j < system.fBasis[i].size(); j++) {
                    Eigen::MatrixXd f_i = system.fBasis[i][j];
                    Eigen::MatrixXd Qlin_f_i = calcQlinf_i(f_i);
                    Eigen::MatrixXd QlinAdj_Y = calcQlinf_i(Y);

                    // Check if the adjoint property holds: <Qlin_f_i, Y> = <f_i, QlinAdj_Y>
                    double lhs = (Qlin_f_i.array() * Y.array()).sum();
                    double rhs = (f_i.array() * QlinAdj_Y.array()).sum();
                    std::cout << lhs << " vs " << rhs << std::endl;
                }
            }
        }

        void generateQlin_1() {
            unsigned int D_I = system.M * (system.M - 1) / 2;
            Eigen::MatrixXd I = Eigen::MatrixXd::Identity(D_I, D_I);

            system.Qlin_1 = calcQlinf_i(I);
        }

        void generateQlin_f_i() {
            system.Qlin_f_i.clear();
            system.Qlin_f_i.resize(system.fBasis.size());
            for (unsigned int i = 0; i < system.fBasis.size(); i++) {
                system.Qlin_f_i[i].resize(system.fBasis[i].size());
                for (unsigned int j = 0; j < system.fBasis[i].size(); j++) {
                    system.Qlin_f_i[i][j] = calcQlinf_i(system.fBasis[i][j]);
                }
            }
        }

        void generateLlin_1() {
            unsigned int dimGamma = system.Gamma_1.rows();
            unsigned int dimQ = system.Qlin_1.rows();
            unsigned int dimG = system.G_1.rows();

            unsigned int D = dimGamma + dimQ + dimG;
            system.Llin_1 = Eigen::MatrixXd::Zero(D, D);

            system.Llin_1.block(0, 0, dimGamma, dimGamma) = system.Gamma_1;
            system.Llin_1.block(dimGamma, dimGamma, dimQ, dimQ) = system.Qlin_1;
            system.Llin_1.block(dimGamma + dimQ, dimGamma + dimQ, dimG, dimG) = system.G_1;
        }

        void generateLlin_f_i() {
            system.Llin_f_i.clear();
            system.Llin_f_i.resize(system.fBasis.size());
            for (unsigned int i = 0; i < system.fBasis.size(); i++) {
                system.Llin_f_i[i].resize(system.fBasis[i].size());
                for (unsigned int j = 0; j < system.fBasis[i].size(); j++) {
                    unsigned int dimGamma = system.Gamma_f_i[i][j].rows();
                    unsigned int dimQ = system.Qlin_f_i[i][j].rows();
                    unsigned int dimG = system.G_f_i[i][j].rows();

                    system.Llin_f_i[i][j] = Eigen::MatrixXd::Zero(dimGamma + dimQ + dimG, dimGamma + dimQ + dimG);
                    system.Llin_f_i[i][j].block(0, 0, dimGamma, dimGamma) = system.Gamma_f_i[i][j];
                    system.Llin_f_i[i][j].block(dimGamma, dimGamma, dimQ, dimQ) = system.Qlin_f_i[i][j];
                    system.Llin_f_i[i][j].block(dimGamma + dimQ, dimGamma + dimQ, dimG, dimG) = system.G_f_i[i][j];
                }
            }
        }

        // Calculate L from f basis
        void calcL_f() {
            system.L = static_cast<double>(system.N * (system.N - 1)) / (system.M * (system.M - 1)) * system.Llin_1;

            // add affine part due to Q to L
            unsigned int dimGamma = system.Gamma_1.rows();
            unsigned int dimQ = system.Qlin_1.rows();
            system.L.block(dimGamma, dimGamma, dimQ, dimQ) += Eigen::MatrixXd::Identity(dimQ, dimQ);

            for (unsigned int i = 0; i < system.fBasis.size(); i++) {
                for (unsigned int j = 0; j < system.fBasis[i].size(); j++) {
                    system.L += system.fCoeffs[i][j] * system.Llin_f_i[i][j];
                }
            }
        }

        Eigen::MatrixXd shortToFull(Eigen::MatrixXd& matShort) const {
            unsigned int D = system.M * system.M;
            Eigen::MatrixXd full = Eigen::MatrixXd::Zero(D, D);
            for (unsigned int i = 0; i < system.M; i++) {
                for (unsigned int j = 0; j < system.M; j++) {
                    for (unsigned int k = 0; k < system.M; k++) {
                        for (unsigned int l = 0; l < system.M; l++) {
                            unsigned int I = getGIndex(i, j);
                            unsigned int J = getGIndex(k, l);
                            full(I, J) = getShortElement(i, j, k, l, matShort);
                        }
                    }
                }
            }
            return full;
        }

        Eigen::MatrixXd calcGradBarrier(double t) const {
            unsigned int D = system.M * (system.M - 1) / 2;
            Eigen::MatrixXd gradBarrier = system.K;
            for (unsigned int k = 0; k < 3; k++) {
                if (k == 0) {
                    Eigen::MatrixXd Lk_Gamma_inverse = system.Gamma.ldlt().solve(Eigen::MatrixXd::Identity(D, D));
                    Eigen::MatrixXd Lk_adj = Lk_Gamma_inverse;
                    gradBarrier -= t * Lk_adj;
                } else if (k == 1) {
                    Eigen::MatrixXd Lk_Gamma_inverse = system.Q.ldlt().solve(Eigen::MatrixXd::Identity(D, D));
                    Eigen::MatrixXd Lk_adj = calcQlinf_i(Lk_Gamma_inverse);
                    gradBarrier -= t * Lk_adj;
                } else {
                    Eigen::MatrixXd Lk_Gamma_inverse = system.G.ldlt().solve(Eigen::MatrixXd::Identity(system.M * system.M, system.M * system.M));
                    Eigen::MatrixXd Lk_adj = calcGAdj(Lk_Gamma_inverse);
                    gradBarrier -= t * Lk_adj;
                }
            }

            Eigen::MatrixXd gradBarrierPr = gradBarrier - (gradBarrier.trace() / D) * Eigen::MatrixXd::Identity(D, D);

            return gradBarrierPr;

        }

        Eigen::MatrixXd calcHessianAction(double t, const Eigen::MatrixXd& deltaGamma) const {
            unsigned int D = system.M * (system.M - 1) / 2;
            Eigen::MatrixXd HessianAction = Eigen::MatrixXd::Zero(D, D);

            // Gamma
            HessianAction += system.Gamma_inv * deltaGamma * system.Gamma_inv;
            
            // Q
            Eigen::MatrixXd Lk_gamma_Q = calcQlinf_i(deltaGamma);
            Eigen::MatrixXd product_Q = system.Q_inv * Lk_gamma_Q * system.Q_inv;
            HessianAction += calcQlinf_i(product_Q);
            
            // G
            Eigen::MatrixXd Lk_gamma_G = calcGf_i(deltaGamma);
            Eigen::MatrixXd product_G = system.G_inv * Lk_gamma_G * system.G_inv;
            HessianAction += calcGAdj(product_G);

            // Traceless projection
            double trace = HessianAction.trace();
            Eigen::MatrixXd I = Eigen::MatrixXd::Identity(D, D);
            Eigen::MatrixXd HessianActionPr = HessianAction - (trace / D) * I;

            return t * 0.5 * (HessianActionPr + HessianActionPr.transpose());
        }

        void generateHessian_f_i() {
            system.Hessian_f_i.clear();
            system.Hessian_f_i.resize(system.fBasis.size());
            for (unsigned int i = 0; i < system.fBasis.size(); i++) {
                system.Hessian_f_i[i].resize(system.fBasis[i].size());
                for (unsigned int j = 0; j < system.fBasis[i].size(); j++) {
                    system.Hessian_f_i[i][j] = calcHessianAction(1, system.fBasis[i][j]);
                }
            }
        }

        void generateA(double t) {
            unsigned int D = 0;
            for (unsigned int i = 0; i < system.fBasis.size(); i++) {
                for (unsigned int j = 0; j < system.fBasis[i].size(); j++) {
                    D++;
                }
            }
            system.A = Eigen::MatrixXd::Zero(D, D);

            unsigned int idx1 = 0;
            for (unsigned int i = 0; i < system.fBasis.size(); i++) {
                for (unsigned int j = 0; j < system.fBasis[i].size(); j++) {
                    
                    unsigned int idx2 = 0;
                    for (unsigned int k = 0; k < system.fBasis.size(); k++) {
                        for (unsigned int l = 0; l < system.fBasis[k].size(); l++) {
                            system.A(idx1, idx2) = t * (system.fBasis[i][j].array() * system.Hessian_f_i[k][l].array()).sum();
                            
                            idx2++;
                        }
                    }   
                    idx1++; 
                }
            }
            system.A = 0.5 * (system.A + system.A.transpose());
        }

        void generateB(const Eigen::MatrixXd& gradBarrier) {
            unsigned int D = 0;
            for (unsigned int i = 0; i < system.fBasis.size(); i++) {
                D += system.fBasis[i].size();
            }
            system.b = Eigen::VectorXd::Zero(D);
            unsigned int idx = 0;
            for (unsigned int i = 0; i < system.fBasis.size(); i++) {
                for (unsigned int j = 0; j < system.fBasis[i].size(); j++) {
                    system.b(idx) = -(system.fBasis[i][j].array() * gradBarrier.array()).sum();
                    idx++;
                }
            }
        }

        void solveCG(double t, double tol = 1e-6, int max_iter = 1000) {
            // 1. Initial guess x_0 = 0
            system.x = Eigen::VectorXd::Zero(system.b.size()); 
            
            // 2. Initial residual r_0 = b - A*x_0 = b (since x_0 is 0)
            Eigen::VectorXd r = system.b; 
            Eigen::VectorXd p = r;
            double rsold = r.squaredNorm();

            int iter = 0;
            for (iter = 0; iter < max_iter; ++iter) {
                // Matrix-free Hessian-vector product
                Eigen::VectorXd Ap = applyHessianVector(p, t);

                double pAp = p.dot(Ap);
                if (pAp <= 0.0) {
                    std::cerr << "CG Warning: Hessian is not positive definite (pAp = " << pAp << ")." << std::endl;
                    break; 
                }

                double alpha = rsold / pAp;
                system.x += alpha * p;
                r -= alpha * Ap;

                double rsnew = r.squaredNorm();
                if (std::sqrt(rsnew) < tol) {
                    break;
                }

                p = r + (rsnew / rsold) * p;
                rsold = rsnew;
            }

            std::cout << "Custom CG converged in " << iter << " iterations." << std::endl;
            std::cout << "Approximate error: " << std::sqrt(rsold) << std::endl;
        }

        Eigen::MatrixXd solveCG_MatrixFree(double t, const Eigen::MatrixXd& gradBarrier, double tol = 1e-6, int max_iter = 1000) {
            unsigned int D = system.K.rows();
            
            // The right-hand side is the negative gradient. 
            // It is already traceless from calcGradBarrier.
            Eigen::MatrixXd B_mat = -gradBarrier; 

            // 1. Initial guess X_0 = 0
            Eigen::MatrixXd X = Eigen::MatrixXd::Zero(D, D);
            
            // 2. Initial residual R_0 = B_mat - H(X_0) = B_mat
            Eigen::MatrixXd R = B_mat;
            Eigen::MatrixXd P = R;
            double rsold = (R.array() * R.array()).sum(); // Matrix Frobenius inner product

            int iter = 0;
            for (iter = 0; iter < max_iter; ++iter) {
                // Apply Hessian action directly to the matrix P
                Eigen::MatrixXd AP = calcHessianAction(t, P);

                double pAp = (P.array() * AP.array()).sum();
                if (pAp <= 0.0) {
                    std::cerr << "CG Warning: Hessian is not positive definite (pAp = " << pAp << ")." << std::endl;
                    break; 
                }

                double alpha = rsold / pAp;
                X += alpha * P;
                R -= alpha * AP;

                double rsnew = (R.array() * R.array()).sum();
                if (std::sqrt(rsnew) < tol) {
                    break;
                }

                P = R + (rsnew / rsold) * P;
                rsold = rsnew;
            }

            std::cout << "  Matrix-Free CG converged in " << iter << " iterations." << std::endl;
            return X; 
        }

        double getAlpha(const double t, const Eigen::MatrixXd& deltaGamma, const double tolerance) const {
            // Find the eigenvalues of the symmetric eigenvalue problems S_j*x = lambda_j*x
            // Gamma
            Eigen::MatrixXd Gamma_deltaGamma = deltaGamma;
            // Calculate X_Gamma(Gamma) = Gamma^{-1/2} using Eigenvalue decomposition
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver_Gamma(system.Gamma);
            if (solver_Gamma.info() != Eigen::Success) {
                std::cerr << "Error diagonalizing Gamma" << std::endl;
                return 0.0;
            }
            Eigen::MatrixXd X_Gamma = solver_Gamma.operatorInverseSqrt();
            Eigen::MatrixXd S_Gamma = X_Gamma * Gamma_deltaGamma * X_Gamma;
            // Calculate the eigenvalues of S_Gamma
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver_S_Gamma(S_Gamma);
            if (solver_S_Gamma.info() != Eigen::Success) {
                std::cerr << "Error diagonalizing S_Gamma" << std::endl;
                return 0.0;
            }
            Eigen::VectorXd eigenvalues_Gamma = solver_S_Gamma.eigenvalues();

            // Q
            Eigen::MatrixXd Q_deltaGamma = calcQlinf_i(deltaGamma);
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver_Q(system.Q);
            if (solver_Q.info() != Eigen::Success) {
                std::cerr << "Error diagonalizing Q" << std::endl;
                return 0.0;
            }
            Eigen::MatrixXd X_Q = solver_Q.operatorInverseSqrt();
            Eigen::MatrixXd S_Q = X_Q * Q_deltaGamma * X_Q;
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver_S_Q(S_Q);
            if (solver_S_Q.info() != Eigen::Success) {
                std::cerr << "Error diagonalizing S_Q" << std::endl;
                return 0.0;
            }
            Eigen::VectorXd eigenvalues_Q = solver_S_Q.eigenvalues();

            // G
            Eigen::MatrixXd G_deltaGamma = calcGf_i(deltaGamma);
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver_G(system.G);
            if (solver_G.info() != Eigen::Success) {
                std::cerr << "Error diagonalizing G" << std::endl;
                return 0.0;
            }
            Eigen::MatrixXd X_G = solver_G.operatorInverseSqrt();
            Eigen::MatrixXd S_G = X_G * G_deltaGamma * X_G;
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver_S_G(S_G);
            if (solver_S_G.info() != Eigen::Success) {
                std::cerr << "Error diagonalizing S_G" << std::endl;
                return 0.0;
            }
            Eigen::VectorXd eigenvalues_G = solver_S_G.eigenvalues();

            // Solve Tr(K * deltaGamma) = t\sum_j (\sum_i lambda_i^j / (1 + alpha * lambda_i^j)) for alpha using the bisection method
            double alpha_min = 0.0;
            double alpha_max = 1.0;
            double alpha = 0.0;
            double trace_K_deltaGamma = (system.K.array() * deltaGamma.array()).sum();
            double f_alpha_min = 0.0;
            double f_alpha_max = 0.0;
            for (auto eigenvalues : {eigenvalues_Gamma, eigenvalues_Q, eigenvalues_G}) {
                for (unsigned int i = 0; i < eigenvalues.size(); i++) {
                    f_alpha_min += eigenvalues(i) / (1.0 + alpha_min * eigenvalues(i));
                    f_alpha_max += eigenvalues(i) / (1.0 + alpha_max * eigenvalues(i));
                }
            }

            while (alpha_max - alpha_min > tolerance) {
                alpha = 0.5 * (alpha_min + alpha_max);
                double f_alpha = 0.0;
                for (auto eigenvalues : {eigenvalues_Gamma, eigenvalues_Q, eigenvalues_G}) {
                    for (unsigned int i = 0; i < eigenvalues.size(); i++) {
                        f_alpha += eigenvalues(i) / (1.0 + alpha * eigenvalues(i));
                    }
                }
                if (trace_K_deltaGamma - t * f_alpha > 0) {
                    alpha_max = alpha;
                } else {
                    alpha_min = alpha;
                }
            }

            return alpha;
        }

        void initHessian() {
            unsigned int D = 0;
            for (unsigned int i = 0; i < system.fBasis.size(); i++) {
                for (unsigned int j = 0; j < system.fBasis[i].size(); j++) {
                    D++;
                }
            }
            system.Hessian = Eigen::MatrixXd::Zero(D, D);
        }

        double calcHessianElement(unsigned int i, unsigned int j, unsigned int k, unsigned int l, double t, const Eigen::MatrixXd& L_inverse) const {
            Eigen::MatrixXd Lf_i = system.Llin_f_i[i][j];
            Eigen::MatrixXd Lf_j = system.Llin_f_i[k][l];

            Eigen::MatrixXd W_i = L_inverse * Lf_i;
            Eigen::MatrixXd W_j = L_inverse * Lf_j;

            return t * (W_i.array() * W_j.transpose().array()).sum();

        }

        void calcHessian(double t) {
            unsigned int dimGamma = system.Gamma_1.rows();
            unsigned int dimQ = system.Qlin_1.rows();
            unsigned int dimG = system.G_1.rows();
            unsigned int D_L = dimGamma + dimQ + dimG;

            Eigen::MatrixXd L_inverse = Eigen::MatrixXd::Zero(D_L, D_L);

            Eigen::MatrixXd L_Gamma = system.L.block(0, 0, dimGamma, dimGamma);
            Eigen::MatrixXd L_Q     = system.L.block(dimGamma, dimGamma, dimQ, dimQ);
            Eigen::MatrixXd L_G     = system.L.block(dimGamma + dimQ, dimGamma + dimQ, dimG, dimG);

            L_inverse.block(0, 0, dimGamma, dimGamma) = 
                L_Gamma.ldlt().solve(Eigen::MatrixXd::Identity(dimGamma, dimGamma));
                
            L_inverse.block(dimGamma, dimGamma, dimQ, dimQ) = 
                L_Q.ldlt().solve(Eigen::MatrixXd::Identity(dimQ, dimQ));
                
            L_inverse.block(dimGamma + dimQ, dimGamma + dimQ, dimG, dimG) = 
                L_G.ldlt().solve(Eigen::MatrixXd::Identity(dimG, dimG));

            unsigned int row_index = 0;
            for (unsigned int i = 0; i < system.fBasis.size(); i++) {
                for (unsigned int j = 0; j < system.fBasis[i].size(); j++) {
                    unsigned int col_index = 0;
                    for (unsigned int k = 0; k < system.fBasis.size(); k++) {
                        for (unsigned int l = 0; l < system.fBasis[k].size(); l++) {
                            
                            if (col_index >= row_index) {
                                double val = calcHessianElement(i, j, k, l, t, L_inverse);
                                system.Hessian(row_index, col_index) = val;
                                system.Hessian(col_index, row_index) = val; 
                            }
                            col_index++;
                        }
                    }
                    row_index++;
                }
            }
        }

        // --------------------------------------------------------------------------------
        // Projection method
        void calcGammaClosest(const double alpha, const Eigen::MatrixXd& K, const double fmu = 1e-3, const double trace_tol = 1e-3) {
            system.alpha_p = alpha;
            Eigen::MatrixXd alphaK = (1.0 / system.alpha_p) * K;
            system.Gamma_p = getFClosest(alphaK, system.N * (system.N - 1) / 2);
        }

        Eigen::MatrixXd getPsdClosest(const Eigen::MatrixXd& F) const {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(F);
            Eigen::VectorXd evals = solver.eigenvalues();
            
            // Simply truncate negative eigenvalues to 0 (no shifting/trace matching required)
            for (unsigned int i = 0; i < evals.size(); i++) {
                if (evals(i) < 0.0) {
                    evals(i) = 0.0;
                }
            }
            
            return solver.eigenvectors() * evals.asDiagonal() * solver.eigenvectors().transpose();
        }

        Eigen::MatrixXd getFClosest(const Eigen::MatrixXd& F, double Trace_F, const double bis_tol = 1e-6) const {
            unsigned int D = F.rows();

            // Calculate the eigenvalues and eigenvectors of F
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(F);
            if (solver.info() != Eigen::Success) {
                std::cerr << "Error diagonalizing F" << std::endl;
                exit(EXIT_FAILURE);
            }
            Eigen::VectorXd evals = solver.eigenvalues();

            double current_trace = 0.0;
            for (unsigned int i = 0; i < D; i++) {
                current_trace += std::max(0.0, evals(i));
            }
            // if (current_trace <= Trace_F) {
            //     Eigen::MatrixXd F_closest = solver.eigenvectors() * evals.cwiseMax(0.0).asDiagonal() * solver.eigenvectors().transpose();
            //     return F_closest; 
            // }

            // Bisection method to find mu
            double mu_min = evals.maxCoeff() - Trace_F; 
            double mu_max = evals.maxCoeff(); 
            double mu = 0.0;

            while ((mu_max - mu_min) > bis_tol) {
                mu = 0.5 * (mu_min + mu_max);
                
                current_trace = 0.0;
                for (unsigned int i = 0; i < D; i++) {
                    current_trace += std::max(0.0, evals(i) - mu);
                }

                if (current_trace > Trace_F) {
                    mu_min = mu; 
                } else {
                    mu_max = mu; 
                }
            }

            Eigen::VectorXd shifted_evals(D);
            for (unsigned int i = 0; i < D; i++) {
                shifted_evals(i) = std::max(0.0, evals(i) - mu);
            }

            Eigen::MatrixXd F_closest = solver.eigenvectors() * shifted_evals.asDiagonal() * solver.eigenvectors().transpose();
            return F_closest;
        }       

        Eigen::MatrixXd projectGamma(const Eigen::MatrixXd& Gamma) const {
            Eigen::MatrixXd FGamma = Gamma;
            double Trace_FGamma = system.N * (system.N - 1) / 2;
            Eigen::MatrixXd FGamma_closest = getFClosest(FGamma, Trace_FGamma);
            return FGamma_closest;
        }

        //Practical Hastenes-Stiefel CG solver
        Eigen::MatrixXd solveCG_Projection_Q(const Eigen::MatrixXd& B, double tol = 1e-6, unsigned int max_iter = 100000) const {
            unsigned int D = B.rows();
            Eigen::MatrixXd Lambda = Eigen::MatrixXd::Zero(D, D);
            Eigen::MatrixXd ALambda = calcQf_i(calcQlinf_i(Lambda));
            Eigen::MatrixXd R = B - ALambda;
            Eigen::MatrixXd Rnew = R;
            Eigen::MatrixXd P = R;
            double R_norm = R.norm();

            double n_iter = 0;
            while (R_norm > tol && n_iter < max_iter) {
                double alpha = (R_norm * R_norm) / (P.transpose() * (calcQf_i(calcQlinf_i(P)))).trace();
                Lambda += alpha * P;
                Rnew = B - calcQf_i(calcQlinf_i(Lambda));
                double eta = (Rnew.norm() * Rnew.norm()) / (R_norm * R_norm);
                P *= eta;
                P += Rnew;
                R = Rnew;
                R_norm = R.norm();

                n_iter++;
            }

            std::cout << "  Practical CG (Q-Projection) converged in " << n_iter << " iterations." << std::endl;
            std::cout << "  Final residual norm: " << R_norm << std::endl;

            if (n_iter >= max_iter) {
                std::cerr << "Warning: Practical CG (Q-Projection) reached maximum iterations without convergence." << std::endl;
                return Eigen::MatrixXd::Zero(D, D); // Return zero matrix to indicate failure
            }

            return Lambda;
        }

        Eigen::MatrixXd solveCG_Projection_G(const Eigen::MatrixXd& B, double tol = 1e-6, unsigned int max_iter = 10000) const {
            unsigned int D = B.rows();
            Eigen::MatrixXd GAdjB = calcGAdj(B);

            Eigen::MatrixXd Lambda = Eigen::MatrixXd::Zero(D, D);
            Eigen::MatrixXd GAdjLambda = calcGAdj(Lambda);
            Eigen::MatrixXd ALambda = calcGf_i(calcGAdj(Lambda));
            Eigen::MatrixXd GAdjALambda = calcGAdj(ALambda);

            Eigen::MatrixXd R = GAdjB - GAdjALambda;
            Eigen::MatrixXd Rnew = R;
            Eigen::MatrixXd P = R;
            double R_norm = R.norm();

            Eigen::MatrixXd AP = calcGAdj(calcGf_i(P));

            double n_iter = 0;
            while (R_norm > tol && n_iter < max_iter) {
                double alpha = (R_norm * R_norm) / (P.transpose() * AP).trace();
                GAdjLambda += alpha * P;
                Rnew -= alpha * AP;
                double eta = (Rnew.norm() * Rnew.norm()) / (R_norm * R_norm);
                P *= eta;
                P += Rnew;
                AP = calcGAdj(calcGf_i(P));
                R = Rnew;
                R_norm = R.norm();

                //print alpha, Rnorm, Pnorm, Rnewnorm, Lambdanorm
                std::cout << "  Alpha: " << alpha << ", Rnorm: " << R_norm << ", Pnorm: " << P.norm() << ", Rnewnorm: " << Rnew.norm() << ", Lambdanorm: " << Lambda.norm() << std::endl;

                n_iter++;
            }

            std::cout << "  Practical CG (G-Projection) converged in " << n_iter << " iterations." << std::endl;
            std::cout << "  Final residual norm: " << R_norm << std::endl;

            if (n_iter >= max_iter || !std::isfinite(R_norm) || R_norm > tol) {
                std::cerr << "Warning: Practical CG (G-Projection) reached maximum iterations without convergence." << std::endl;
                return Eigen::MatrixXd::Zero(D, D); // Return zero matrix to indicate failure
            }

            return GAdjLambda;
        }

        Eigen::MatrixXd projectGammaQ(const Eigen::MatrixXd& Gamma) const {
            unsigned int D = Gamma.rows();
            Eigen::MatrixXd FQ = calcQf_i(Gamma); 
            double Trace_FQ = static_cast<double>((system.M - system.N) * (system.M - system.N - 1)) / 2.0; // Target trace for Q-condition
            Eigen::MatrixXd FQ_closest = getFClosest(FQ, Trace_FQ); // Project FQ onto the PSD cone
            
            // Calculate Lambda using CG
            Eigen::MatrixXd B_mat = FQ_closest - FQ;

            Eigen::MatrixXd Lambda = solveCG_Projection_Q(B_mat);
            Eigen::MatrixXd deltaGamma = calcQlinf_i(Lambda);

            Eigen::MatrixXd GammaQ = Gamma + deltaGamma;

            std::cout << "Target trace for Q-condition: " << Trace_FQ << std::endl;
            std::cout << "Trace of GammaQ: " << calcQf_i(GammaQ).trace() << std::endl;
            std::cout << "Trace of Gamma: " << GammaQ.trace() << std::endl;
            std::cout << "Q lifting residual: " << (FQ_closest - calcQf_i(GammaQ)).norm() << std::endl;
            std::cout << "Gamma correction norm: " << deltaGamma.norm() << std::endl;

            return GammaQ;
        }

        Eigen::MatrixXd projectGammaG(const Eigen::MatrixXd& Gamma) const {
            Eigen::MatrixXd FG = calcGf_i(Gamma);
            double Trace_FG = system.N * (system.M - system.N + 1);
            Eigen::MatrixXd FG_closest = getFClosest(FG, Trace_FG);

            double trace_FG_closest = FG_closest.trace();
            std::cout << "Trace of FG_closest: " << trace_FG_closest << std::endl;
            
            Eigen::MatrixXd B_mat = FG_closest - FG;
    
            Eigen::MatrixXd deltaGamma = solveCG_Projection_G(B_mat);

            // Check the residual
            Eigen::MatrixXd residual = B_mat - calcGf_i(deltaGamma);
            std::cout << "Residual norm: " << residual.norm() << std::endl;

            Eigen::MatrixXd GammaG = Gamma + deltaGamma;

            std::cout << "Target trace for G-condition: " << Trace_FG << std::endl;
            std::cout << "Trace of GammaG: " << calcGf_i(GammaG).trace() << std::endl;
            std::cout << "Trace of Gamma: " << GammaG.trace() << std::endl;
            std::cout << "G lifting residual: " << (FG_closest - calcGf_i(GammaG)).norm() << std::endl;
            //std::cout << "Minimum eigenvalue of GammaG: " << GammaG.eigenvalues().minCoeff() << std::endl;
            std::cout << "Gamma correction norm: " << deltaGamma.norm() << std::endl;

            // Stop if the trace of GammaG is not close to the target trace
            // if (std::abs(calcGf_i(GammaG).trace() - Trace_FG) > 1e-3) {
            //     throw std::runtime_error("Trace of GammaG is not close to the target trace for G-condition.");
            // }

            return GammaG;
        }

    public:
        randomSystem(unsigned int N, unsigned int M, double g, unsigned int seed) {
            system.N = N;
            system.M = M;
            system.g = g;
            this->seed = seed;

            unsigned dimBasis = binomial(system.M, system.N);
            system.dimBasis = dimBasis;

            initializeRandomNumberGenerator();
            generateRandomMatrices();
        }

        // Diagonalizes the Hamiltonian H
        void diagonalizeH() {
            // Generate the basis for the system
            generateBasis();
            // Generate the H matrix
            generateHMatrix();
            
            // Timer
            auto start = std::chrono::high_resolution_clock::now();

            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(system.H);
            if (solver.info() != Eigen::Success) {
                std::cerr << "Error diagonalizing Hamiltonian H" << std::endl;
                return;
            }
            system.exactEnergies = solver.eigenvalues();
            system.exactEigenvectors = solver.eigenvectors();

            // print eigenvalues
            // std::cout << "Exact eigenvalues:\n" << system.exactEnergies << std::endl;
            system.durationDiag = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();

            // Calcualte 2-RDM from exact eigenvectors
            calcExactGammaGroundState();
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
            //generateHMatrix();
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
            file << "  \"seed\": " << seed << ",\n";

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
            file << "  ],\n";

            // Save the 2-RDM Gamma
            file << "  \"Gamma\": [\n";
            for (unsigned int i = 0; i < system.Gamma_exact.rows(); ++i) {
                file << "    [";
                for (unsigned int j = 0; j < system.Gamma_exact.cols(); ++j) {
                    file << system.Gamma_exact(i, j);
                    if (j < system.Gamma_exact.cols() - 1) file << ", ";
                }
                file << "]";
                if (i < system.Gamma_exact.rows() - 1) file << ",";
                file << "\n";
            }
            file << "  ],\n";

            // Save the duration of diagonalization
            file << "  \"durationDiag\": " << system.durationDiag << "\n";

            file << "}\n";

        }

        // -----------------------------------------------
        // SDP

        Eigen::VectorXd applyHessianVector(const Eigen::VectorXd& x, double t) const {
            // 1. Reconstruct DeltaGamma from vector x
            Eigen::MatrixXd deltaGamma = Eigen::MatrixXd::Zero(system.Gamma.rows(), system.Gamma.cols());
            unsigned int idx = 0;
            for (unsigned int i = 0; i < system.fBasis.size(); i++) {
                for (unsigned int j = 0; j < system.fBasis[i].size(); j++) {
                    deltaGamma += x(idx) * system.fBasis[i][j];
                    idx++;
                }
            }

            // 2. Apply Hessian Action (Matrix-Free)
            Eigen::MatrixXd H_action = calcHessianAction(t, deltaGamma);

            // 3. Project back to vector form
            Eigen::VectorXd y = Eigen::VectorXd::Zero(x.size());
            idx = 0;
            for (unsigned int i = 0; i < system.fBasis.size(); i++) {
                for (unsigned int j = 0; j < system.fBasis[i].size(); j++) {
                    y(idx) = (system.fBasis[i][j].array() * H_action.array()).sum();
                    idx++;
                }
            }
            return y;
        }

        void performPotentialReduction(double tInit = 1.0, double tDecay = 0.5, double eps_tol = 1e-3, double duality_gap_tol = 1e-6, std::string logFile = "pr_log.txt") {
            // Initialize the clock
            auto start = std::chrono::high_resolution_clock::now();

            // Generate basis for traceless matrices
            // generateBasisF();

            // Initialize 2-RDM and conditions
            initGamma();
            calcGamma_inv();
            calcRho();
            initQ();
            calcQ_inv();
            initG();
            calcG_inv();


            double t = tInit;
            std::ofstream log(logFile);
            log << "--- Start Potential Reduction Solver ---" << std::endl;

            for (unsigned int outer = 0; outer < 100; outer++) {
                double elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
                log << "--- Outer iteration " << outer << " (t = " << t << ", timer=" << elapsed << "s)" << " ---" << std::endl;

                for (unsigned int inner = 0; inner < 100; inner++) {
                    Eigen::MatrixXd gradBarrier = calcGradBarrier(t);
                    
                    if (gradBarrier.norm() < eps_tol) {
                        break;
                    }

                    // Solve for deltaGamma directly using the matrix-free CG
                    Eigen::MatrixXd deltaGamma = solveCG_MatrixFree(t, gradBarrier, 1e-6, 1000000);

                    // Get step size and apply update
                    double alpha = getAlpha(t, deltaGamma, 1e-6); 

                    // Check if Gamma, Q and G remain positive semidefinite after the update using cholesky decomposition
                    Eigen::LLT<Eigen::MatrixXd> llt_Gamma(system.Gamma + alpha * deltaGamma);
                    Eigen::LLT<Eigen::MatrixXd> llt_Q(system.Q + alpha * calcQlinf_i(deltaGamma));
                    Eigen::LLT<Eigen::MatrixXd> llt_G(system.G + alpha * calcGf_i(deltaGamma));
                    while (llt_Gamma.info() != Eigen::Success || llt_Q.info() != Eigen::Success || llt_G.info() != Eigen::Success) {
                        alpha *= 0.5; // Reduce step size
                        llt_Gamma.compute(system.Gamma + alpha * deltaGamma);
                        llt_Q.compute(system.Q + alpha * calcQlinf_i(deltaGamma));
                        llt_G.compute(system.G + alpha * calcGf_i(deltaGamma));

                        if (alpha < 1e-10) {
                            alpha = 0.0;
                            break;
                        }
                    }
                

                    system.Gamma += alpha * deltaGamma;
                    
                    calcGamma_inv();
                    calcRho();
                    calcQ();
                    calcQ_inv();
                    calcG();
                    calcG_inv();
                }

                double duality_gap = (system.Gamma.rows() + system.Q.rows() + system.G.rows()) * t; // Duality gap = dim(\mathcal{L}) * t
                log << "Duality gap: " << duality_gap << std::endl;
                if (duality_gap < duality_gap_tol) {
                    log << "Duality gap below tolerance, stopping." << std::endl;
                    break;
                }

                t *= tDecay;

                double energy = (system.K.array() * system.Gamma.array()).sum();
                system.Eground_pr = energy;
                system.duality_gap = duality_gap;

                log << "Ground state energy (SDP): " << energy << std::endl;
            }

            system.durationPr = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
        }

        // Save the result of Potential Reduction to a Json
        void saveToJsonPR(const std::string& filename) const {
            std::ofstream file(filename);
            if (!file) {
                std::cerr << "Error opening file for writing: " << filename << std::endl;
                return;
            }

            file << "{\n";

            // N
            file << "  \"N\": " << system.N << ",\n";
            // M
            file << "  \"M\": " << system.M << ",\n";
            // g
            file << "  \"g\": " << system.g << ",\n";
            // seed 
            file << "  \"seed\": " << seed << ",\n";

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

            // Ground state energy
            file << "  \"Ground state energy (SDP)\": " << system.Eground_pr << ",\n";

            // Duality gap
            file << "  \"Duality gap\": " << system.duality_gap << ",\n";

            // Duration of Potential Reduction
            file << "  \"durationPr\": " << system.durationPr << "\n";

            file << "}\n";
        }

        // --------------------------------------------------------------------------
        // Projection method
        void performProjectionMethod(const double alpha, const double fmu = 1e-3, const double trace_tol = 1e-3) {
            // Calculate Gamma_0
            //calcGammaClosest(alpha, system.K, fmu, trace_tol);
            system.alpha_p = alpha;
            system.Gamma_p = -(1.0 / alpha) * system.K;

            // Halpern alogirthm
            unsigned int n = 1;
            Eigen::MatrixXd A = system.Gamma_p;
            Eigen::MatrixXd D_n = system.Gamma_p;
            Eigen::MatrixXd D_n_1 = system.Gamma_p;
            while ((D_n - D_n_1).norm() > 1e-6 || n == 1) {
                D_n = D_n_1;
                // Project Gamma_0 onto Q and G
                Eigen::MatrixXd Gamma = projectGamma(D_n);
                Eigen::MatrixXd GammaQ = projectGammaQ(Gamma);
                Eigen::MatrixXd Cn = projectGammaG(GammaQ);
                // Eigen::MatrixXd GammaQ = projectGammaQ(Gamma);
                // Eigen::MatrixXd Cn = GammaQ;
                //Eigen::MatrixXd Cn = projectGammaG(GammaQ);
                // Eigen::MatrixXd GammaG = projectGammaG(Gamma);
                // Eigen::MatrixXd Cn = projectGammaQ(GammaG);

                D_n_1 = (1.0 / (n + 1)) * A + (static_cast<double>(n) / static_cast<double>(n + 1)) * Cn;

                // Print energy
                double energy = (system.K.array() * D_n_1.array()).sum();
                std::cout << "Iteration " << n << ", Energy: " << energy << ", Norm diff: " << (D_n - D_n_1).norm() << std::endl;

                n++;
            }
            system.Gamma_p = D_n_1;
        }

        void performProjectionMethod_Dykstra(const double alpha, const double fmu = 1e-3, const double trace_tol = 1e-3) {
            // Calculate Gamma_0
            //calcGammaClosest(alpha, system.K, fmu, trace_tol);
            system.alpha_p = alpha;
            system.Gamma_p = -(1.0 / alpha) * system.K;

            // Dykstra's algorithm
            Eigen::MatrixXd D_n = system.Gamma_p;
            Eigen::MatrixXd D_n_1 = system.Gamma_p;
            Eigen::MatrixXd C = Eigen::MatrixXd::Zero(D_n.rows(), D_n.cols());
            Eigen::MatrixXd X_Gamma = Eigen::MatrixXd::Zero(D_n.rows(), D_n.cols());
            Eigen::MatrixXd X_Q = Eigen::MatrixXd::Zero(D_n.rows(), D_n.cols());
            Eigen::MatrixXd X_G = Eigen::MatrixXd::Zero(D_n.rows(), D_n.cols());

            unsigned int i = 0;
            while ((D_n - D_n_1).norm() > 1e-6 || i == 0) {
                D_n = D_n_1;

                // Gamma projection
                C = D_n - X_Gamma;
                D_n_1 = projectGamma(C);
                X_Gamma = D_n_1 - C;

                // Q projection
                C = D_n_1 - X_Q;
                D_n_1 = projectGammaQ(C);
                X_Q = D_n_1 - C;

                // G projection
                C = D_n_1 - X_G;
                D_n_1 = projectGammaG(C);
                X_G = D_n_1 - C;

                // Print energy
                double energy = (system.K.array() * D_n_1.array()).sum();
                std::cout << "Iteration " << i << ", Energy: " << energy << ", Norm diff: " << (D_n - D_n_1).norm() << std::endl;

                i++;
            }

            system.Gamma_p = D_n_1;
        }

        void saveToJsonProjection(const std::string& filename) const {
            std::ofstream file(filename);
            if (!file) {
                std::cerr << "Error opening file for writing: " << filename << std::endl;
                return;
            }

            file << "{\n";

            // N
            file << "  \"N\": " << system.N << ",\n";
            // M
            file << "  \"M\": " << system.M << ",\n";
            // g
            file << "  \"g\": " << system.g << ",\n";
            // seed 
            file << "  \"seed\": " << seed << ",\n";

            // Gamma closest
            file << "  \"Gamma closest\": [\n";
            for (unsigned int i = 0; i < system.Gamma_p.rows(); i++) {
                file << "    [";
                for (unsigned int j = 0; j < system.Gamma_p.cols(); j++) {
                    file << system.Gamma_p(i, j);
                    if (j < system.Gamma_p.cols() - 1) file << ", ";
                }
                file << "]";
                if (i < system.Gamma_p.rows() - 1) file << ",";
                file << "\n";
            }
            file << "  ],\n";

            // alpha^{-1}K
            file << "  \"alpha^{-1}K\": [\n";
            Eigen::MatrixXd Ksc = (1.0 / system.alpha_p) * system.K;
            for (unsigned int i = 0; i < Ksc.rows(); i++) {
                file << "    [";
                for (unsigned int j = 0; j < Ksc.cols(); j++) {
                    file << Ksc(i, j);
                    if (j < Ksc.cols() - 1) file << ", ";
                }
                file << "]";
                if (i < Ksc.rows() - 1) file << ",";
                file << "\n";
            }
            file << "  ],\n";

            // Ground state energy
            double energy = (system.K.array() * system.Gamma_p.array()).sum();
            file << "  \"Ground state energy (Projection method)\": " << energy << "\n";

            file << "}\n";
        }
        
        // ----------------------------------------------------------------------------
        // Comparison of exact diagonalization and SDP
        void runExactDiagAndSDPMultiple(unsigned int N, double g, unsigned int M_min, unsigned int M_max, unsigned int M_step, std::string outputFileJson = "exactDiagSDPcomparison.json") {
            std::ofstream file(outputFileJson);
            if (!file.is_open()) {
                std::cerr << "Could not open file for writing: " << outputFileJson << std::endl;
                return;
            }

            file << "{\n";
            file << "  \"N\": " << N << ",\n";
            file << "  \"g\": " << g << ",\n";
            file << "  \"results\": [\n";
            for (unsigned int M = M_min; M <= M_max; M += M_step) {
                randomSystem system(N, M, g, seed);
                system.generateRandomMatrices();
                system.diagonalizeH();
                system.performPotentialReduction();

                file << "    {\n";
                file << "      \"M\": " << M << ",\n";
                file << "      \"Gamma_exact\": [\n";
                for (unsigned int i = 0; i < system.system.Gamma_exact.rows(); ++i) {
                    file << "        [";
                    for (unsigned int j = 0; j < system.system.Gamma_exact.cols(); ++j) {
                        file << system.system.Gamma_exact(i, j);
                        if (j < system.system.Gamma_exact.cols() - 1) file << ", ";
                    }
                    file << "]";
                    if (i < system.system.Gamma_exact.rows() - 1) file << ",";
                    file << "\n";
                }
                file << "      ],\n";
                file << "      \"Gamma_SDP\": [\n";
                for (unsigned int i = 0; i < system.system.Gamma.rows(); ++i) {
                    file << "        [";
                    for (unsigned int j = 0; j < system.system.Gamma.cols(); ++j) {
                        file << system.system.Gamma(i, j);
                        if (j < system.system.Gamma.cols() - 1) file << ", ";
                    }
                    file << "]";
                    if (i < system.system.Gamma.rows() - 1) file << ",";
                    file << "\n";
                }
                file << "      ],\n";
                file << "      \"E_exact\": " << system.system.exactEnergies.minCoeff() << ",\n";
                file << "      \"E_SDP\": " << system.system.Eground_pr << ",\n";
                file << "      \"Duality gap\": " << system.system.duality_gap << ",\n";
                file << "      \"durationDiag\": " << system.system.durationDiag << ",\n";
                file << "      \"durationPr\": " << system.system.durationPr << "\n";
                file << "    }";
                if (M + M_step <= M_max) {
                    file << ",";
                }
                file << "\n";

                file.flush();
            }

            file << "  ]\n";
            file << "}\n";
        }

        // ----------------------------------------------------------------------------
        // Comparison of exact diagonalization and SDP for a single system with different g values
        void runExactDiagAndSDPMultipleG(unsigned int N, unsigned int M, double g_min, double g_max, double n_g, std::string outputFileJson = "exactDiagSDPcomparison_g.json") {
            std::ofstream file(outputFileJson);
            if (!file.is_open()) {
                std::cerr << "Could not open file for writing: " << outputFileJson << std::endl;
                return;
            }

            file << "{\n";
            file << "  \"N\": " << N << ",\n";
            file << "  \"M\": " << M << ",\n";
            file << "  \"results\": [\n";

            double g_init = system.g; // Store the initial g value to restore later

            for (unsigned int i = 0; i < n_g; ++i) {
                system.g = g_min + i * (g_max - g_min) / (n_g - 1);
                // Re-generate H and K matrices with the new g value
                generateKMatrix();
                generateHMatrix();
                diagonalizeH();
                performPotentialReduction();

                file << "    {\n";
                file << "      \"g\": " << system.g << ",\n";
                file << "      \"Gamma_exact\": [\n";
                for (unsigned int i = 0; i < system.Gamma_exact.rows(); ++i) {
                    file << "        [";
                    for (unsigned int j = 0; j < system.Gamma_exact.cols(); ++j) {
                        file << system.Gamma_exact(i, j);
                        if (j < system.Gamma_exact.cols() - 1) file << ", ";
                    }
                    file << "]";
                    if (i < system.Gamma_exact.rows() - 1) file << ",";
                    file << "\n";
                }
                file << "      ],\n";
                file << "      \"Gamma_SDP\": [\n";
                for (unsigned int i = 0; i < system.Gamma.rows(); ++i) {
                    file << "        [";
                    for (unsigned int j = 0; j < system.Gamma.cols(); ++j) {
                        file << system.Gamma(i, j);
                        if (j < system.Gamma.cols() - 1) file << ", ";
                    }
                    file << "]";
                    if (i < system.Gamma.rows() - 1) file << ",";
                    file << "\n";
                }
                file << "      ],\n";
                file << "      \"E_exact\": " << system.exactEnergies.minCoeff() << ",\n";
                file << "      \"E_SDP\": " << system.Eground_pr << ",\n";
                file << "      \"Duality gap\": " << system.duality_gap << ",\n";
                file << "      \"durationDiag\": " << system.durationDiag << ",\n";
                file << "      \"durationPr\": " << system.durationPr << "\n";
                file << "    }";
                if (i + 1 < n_g) {
                    file << ",";
                }
                file << "\n";
                file.flush();
            }
            file << "\n  ]\n";
            file << "}\n";
            file.flush();

            system.g = g_init; // Restore original g value
        }

        // ----------------------------------------------------------------------------
        // Export to SDPT3 format
        void saveToSDPT3Json(const std::string& filename) {
            std::ofstream file(filename);
            if (!file) {
                std::cerr << "Error opening file for writing: " << filename << std::endl;
                return;
            }

            generateBasisF();

            // Re-initialize offsets to guarantee purity
            initGamma(); calcRho(); initQ(); initG();

            // Helper lambda to print matrices
            auto printMat = [&](const Eigen::MatrixXd& mat) {
                file << "[\n";
                for (unsigned int i = 0; i < mat.rows(); i++) {
                    file << "      [";
                    for (unsigned int j = 0; j < mat.cols(); j++) {
                        file << mat(i, j) << (j < mat.cols() - 1 ? ", " : "");
                    }
                    file << "]" << (i < mat.rows() - 1 ? "," : "") << "\n";
                }
                file << "    ]";
            };

            file << "{\n";

            // Count total D
            unsigned int D = 0;
            for (unsigned int i = 0; i < system.fBasis.size(); i++) {
                D += system.fBasis[i].size();
            }

            // 1. Objective Vector b
            file << "  \"b\": [\n    ";
            unsigned int idx = 0;
            for (unsigned int i = 0; i < system.fBasis.size(); i++) {
                for (unsigned int j = 0; j < system.fBasis[i].size(); j++) {
                    double b_val = -(system.fBasis[i][j].array() * system.K.array()).sum();
                    file << b_val << (idx < D - 1 ? ", " : "");
                    idx++;
                }
            }
            file << "\n  ],\n";

            // 2. Constant matrices C
            file << "  \"C_Gamma\": "; printMat(system.Gamma); file << ",\n";
            file << "  \"C_Q\": "; printMat(system.Q); file << ",\n";
            file << "  \"C_G\": "; printMat(system.G); file << ",\n";
            file << "  \"K_mat\": "; printMat(system.K); file << ",\n";

            // 3. Operator matrices A
            file << "  \"A_Gamma\": [\n";
            idx = 0;
            for (unsigned int i = 0; i < system.fBasis.size(); i++) {
                for (unsigned int j = 0; j < system.fBasis[i].size(); j++) {
                    Eigen::MatrixXd A = -system.fBasis[i][j];
                    printMat(A); file << (idx < D - 1 ? ",\n" : "\n");
                    idx++;
                }
            }
            file << "  ],\n";

            file << "  \"A_Q\": [\n";
            idx = 0;
            for (unsigned int i = 0; i < system.fBasis.size(); i++) {
                for (unsigned int j = 0; j < system.fBasis[i].size(); j++) {
                    Eigen::MatrixXd A = -calcQlinf_i(system.fBasis[i][j]);
                    printMat(A); file << (idx < D - 1 ? ",\n" : "\n");
                    idx++;
                }
            }
            file << "  ],\n";

            file << "  \"A_G\": [\n";
            idx = 0;
            for (unsigned int i = 0; i < system.fBasis.size(); i++) {
                for (unsigned int j = 0; j < system.fBasis[i].size(); j++) {
                    Eigen::MatrixXd A = -calcGf_i(system.fBasis[i][j]);
                    printMat(A); file << (idx < D - 1 ? ",\n" : "\n");
                    idx++;
                }
            }
            file << "  ]\n";

            file << "}\n";
        }
};


#endif