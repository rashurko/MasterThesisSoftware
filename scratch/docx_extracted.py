import matplotlib.pyplot as plt

def convergence_plots(tensor, algorithm, noise_list, constraints_lists, iterations):
    for i, noise in enumerate(noise_list):
        plt.figure()
        plt.title(f"Noise = {noise:.4f}")

        for j, constraints_list in enumerate(constraints_lists):
            initial_guess = tensor + np.random.rand(8, 8, 8, 8) * noise
            result = algorithm(initial_guess, constraints_list, traces, N=4, alpha=1.0, eps=1e-6, max_iter = iterations)
            label = f'Constraint: {j+1}'

            # Print numerical values
            print(f"Difference between projected tensor and rdm2 for constraint list {j+1}: {result['error'][-1]:.6f}")

            energy_diff = np.einsum('ijkl, ijkl', result['D'], h) - e
            print(f"Energy difference for constraint list {j+1}: {energy_diff:.6f}")

            plt.plot(result['error'], label=label)

        plt.yscale('log')
        plt.ylabel('Error')
        plt.xlabel('Iterations')
        plt.legend()
        plt.show()

        
        plt.figure()
        plt.title(f"Noise = {noise:.4f}")
        for j, constraints_list in enumerate(constraints_lists):
            
            plt.scatter(rdm2.flatten(), result['D'].flatten(), label=f'Scatter: Constraint {j+1}')

        plt.plot([-1, 1], [-1, 1], 'r--')
        plt.xlabel('rdm2')
        plt.ylabel('projected tensor')
        plt.legend()
        plt.show()
        
#define constraints lists
DM2constraints = [calc_Q, calc_G]
allconstraints = [calc_Q, calc_G, calc_T2]

constraints_lists = [DM2constraints, allconstraints]
noise_list = [0]


r"""RDMREP: A Python package for projecting density matrices onto the set of constraines"""

from .tools import find_closest_sdp, calculate_shift
from .constraints import calc_G, calc_Q, calc_T1
from .algorithms import dykstra, halpern

__all__ = ['find_closest_sdp', 'calculate_shift', 'calc_G', 'calc_Q', 'calc_T1', 'dykstra', 'halpern']


from scipy.optimize import root
import numpy as np
 
    

def tensor2matrix(gamma):
    shape = gamma.shape
    rank = gamma.ndim
    
    if rank == 6:
        n = shape[0]
        gamma_new = np.reshape(gamma, (n**3, n**3))
    elif rank == 4:
        n = shape[0]
        gamma_new = np.reshape(gamma, (n**2, n**2))
    else:
        raise ValueError("Tensor rank not supported. Only rank-4 and rank-6 tensors are supported.")
    
    return gamma_new, n


def matrix2tensor(matrix, n):
    shape = matrix.shape
    
    if len(shape) != 2:
        raise ValueError("Input must be a 2-dimensional matrix.")
    
    if shape == (n**2, n**2):
        return np.reshape(matrix, (n, n, n, n))
    elif shape == (n**3, n**3):
        return np.reshape(matrix, (n, n, n, n, n, n))
    else:
        raise ValueError("Matrix dimensions do not match n^2 or n^3 pattern.")
    

def find_closest_sdp(gamma, trace):
    r"""
    Projection onto a semidefinite constraint.

    Parameters
    ----------
    gamma : np.ndarray
        Density matrix.
    trace : float
        Value of the correct trace.

    """
    #reshape DM (rank-4 or rank-6 tensor) into matrix (rank-2 tensor)
    gamma_reshaped, n = tensor2matrix(gamma)
    #symmetrize if necessary
    L = 0.5 * (gamma_reshaped + gamma_reshaped.conj().T)
    #find eigendecomposition
    vals, vecs = np.linalg.eigh(L)
    #calculate the shift, sigma0
    sigma0 = calculate_shift(vals, trace)
    
    #calculate the closest semidefinite positive matrix with correct trace
    shifted_eigvals = np.heaviside(vals - sigma0, 0.5)*(vals - sigma0)
    L_closest = vecs @ (np.diag(shifted_eigvals)) @ vecs.conj().T 
    #convert back into density matrix via reshaping
    gamma_new = matrix2tensor(L_closest, n)
    # return the reconstructed density matrix
    return gamma_new

def calculate_shift(eigenvalues, alpha):
    r"""
    Calculate the shift for density matrix for it to have the correct trace. 
    This step shifts the spectrum and eliminates the negative eigenvalues.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues of not shifted matrix.
    alpha : float
        Value of the coprrect trace.
        
    """
    #sample code, to be confirmed
    trace = lambda sigma0: np.sum(np.heaviside(eigenvalues - sigma0, 0.5)*(eigenvalues - sigma0))  
    constraint = lambda x: trace(x) - alpha  
    res = root(constraint, 0) 
    return res.x

def spinize_rdm(y):
    """
    Transform a two- or four- index array from spatial to spin representation,
    assuming a closed-shell singlet state (S=0).

    Parameters
    ----------
    y : np.ndarray
        Spatial representation array (ndim == 2 or ndim == 4).

    Returns
    -------
    x : np.ndarray
        Spin representation array.
    """
    n = y.shape[0]
    nsorbs = n * 2
    
    if y.ndim == 2:
        x = np.zeros((nsorbs, nsorbs))
        # Alpha and beta blocks are identical and half of the spatial 1-RDM
        x[:n, :n] = y / 2.0
        x[n:, n:] = y / 2.0
        
    elif y.ndim == 4:
        x = np.zeros((nsorbs, nsorbs, nsorbs, nsorbs))
        
        # y_swap corresponds to y_{ijlk}
        y_swap = y.transpose(0, 1, 3, 2)
        
        # Extract the alpha-beta-alpha-beta block (A) using singlet symmetry
        A = (2 * y + y_swap) / 6.0
        A_swap = A.transpose(0, 1, 3, 2)
        
        # Reconstruct the 6 non-zero spin blocks of the 2-RDM
        x[:n, :n, :n, :n] = A - A_swap        # aaaa
        x[n:, n:, n:, n:] = A - A_swap        # bbbb
        x[:n, n:, :n, n:] = A                 # abab
        x[n:, :n, n:, :n] = A                 # baba
        x[:n, n:, n:, :n] = -A_swap           # abba (spin-flip)
        x[n:, :n, :n, n:] = -A_swap           # baab (spin-flip)
        
    else:
        raise ValueError("Input must have ndim == 2 or ndim == 4")
        
    return x



import numpy as np

from scipy.optimize import root  
from .tools import tensor2matrix, matrix2tensor

N = 4 # change as necessary
N_orbitals = 2 # change as necessary

__all__ = [
    "find_closest_sdp",
    "project_antisymmetric",
    "calc_P",
    "calc_Q",
    "calc_G",
    "calc_T1",
    "calc_T2",
    "calc_T2_prime",
]

def project_antisymmetric(tensor):
    """
    Projects a 2-RDM onto the strict fermionic antisymmetric subspace.
    Enforces Gamma_abcd = -Gamma_bacd = -Gamma_abdc = Gamma_badc
    """
    return 0.25 * (
        tensor 
        - np.einsum('bacd -> abcd', tensor) 
        - np.einsum('abdc -> abcd', tensor) 
        + np.einsum('badc -> abcd', tensor)
    )


def calc_P(gamma, N, M, conjugate=False):
    if not conjugate:
        return gamma
    else:
        return gamma

def calc_Q(tensor, N_electrons, M_dim, conjugate=False, linear_only=False):  
    """
    Applies the Q-condition mapping or its exact numerical adjoint.
    """
    eye = np.eye(M_dim)
    scale = 1.0 / (N_electrons - 1.0)
    
    if not conjugate:
        # 1. Partial trace (rho)
        traced_tensor = np.einsum('abgb -> ag', tensor) * scale
        
        # 2. Linear cross-terms
        term1 =  np.einsum('ag, bd -> abgd', eye, traced_tensor)
        term2 = -np.einsum('ad, bg -> abgd', eye, traced_tensor)
        term3 = -np.einsum('bg, ad -> abgd', eye, traced_tensor)
        term4 =  np.einsum('bd, ag -> abgd', eye, traced_tensor)
        
        wedge_I_rho = term1 + term2 + term3 + term4
        
        # 3. Constant shift
        wedge_I_I = (np.einsum('ag, bd -> abgd', eye, eye) - 
                     np.einsum('ad, bg -> abgd', eye, eye))
                     
        if linear_only:
            return tensor - wedge_I_rho
        else:
            return tensor + wedge_I_I - wedge_I_rho
        
    else:
        # 1. Gather numerical adjoint traces from unstructured matrices
        K1 =  np.einsum('abgd, ag -> bd', tensor, eye) * scale
        K2 = -np.einsum('abgd, ad -> bg', tensor, eye) * scale
        K3 = -np.einsum('abgd, bg -> ad', tensor, eye) * scale
        K4 =  np.einsum('abgd, bd -> ag', tensor, eye) * scale
        
        # The exact sum of all adjoint traces
        K_total = K1 + K2 + K3 + K4
        
        # 2. Scatter back to the Gamma-space structure
        wedge_I_rho_adj = np.einsum('bd, ag -> abgd', eye, K_total)
        
        return tensor - wedge_I_rho_adj

def calc_G(tensor, N_electrons, M_dim, conjugate=False, linear_only=False):
    """
    Applies the self-adjoint G-condition mapping (L = L^dag).
    Can be used for both the forward mapping (Gamma -> G) 
    and the adjoint mapping (D -> Gamma_correction).

    There is no conjugate branch here because the G-condition mapping is self-adjoint by construction.
    """
    eye = np.eye(M_dim)
    scale = 1.0 / (N_electrons - 1.0)
    
    # 1. Partial trace calculation (summing over 2nd and 4th indices)
    # If tensor is Gamma, this is rho. If tensor is D, this is W.
    traced_tensor = np.einsum('abgb -> ag', tensor) * scale
    
    # 2. Linear variations matching the G-condition
    term_1 = np.einsum('bd, ag -> abgd', eye, traced_tensor)
    term_2 = -np.einsum('adgb -> abgd', tensor) 
    
    return term_1 + term_2
    

def calc_T2(tensor, N_electrons, M_dim=None, conjugate=False):
    """
    Calculates the T2 forward mapping or its strict numerical adjoint for unstructured matrices.
    """
    if not conjugate:
        gamma = tensor
        eye = np.eye(gamma.shape[0])
        rho = 1.0 / (N_electrons - 1.0) * np.einsum('abgb -> ag', gamma)
        
        term_1_1 =  np.einsum('ad, be, gz -> abgdez', eye, eye, rho)
        term_1_2 = -np.einsum('ae, bd, gz -> abgdez', eye, eye, rho)
        term_2   =  np.einsum('gz, abde -> abgdez', eye, gamma)
        term_3   = -np.einsum('ad, gezb -> abgdez', eye, gamma)
        term_4   =  np.einsum('bd, geza -> abgdez', eye, gamma)
        term_5   =  np.einsum('ae, gdzb -> abgdez', eye, gamma)
        term_6   = -np.einsum('be, gdza -> abgdez', eye, gamma)
        
        return term_1_1 + term_1_2 + term_2 + term_3 + term_4 + term_5 + term_6
        
    else:
        A = tensor
        eye = np.eye(A.shape[0])
        scale = 1.0 / (N_electrons - 1.0)
        
        # --- Adjoint of rho terms (Term 1.1 and 1.2) ---
        A_tt1 = np.einsum('xyaxyc -> ac', A)
        K1_1 = scale * np.einsum('bd, ac -> abcd', eye, A_tt1)
        
        # THE FIX: Trace 0 with 4 (x) and 1 with 3 (y)
        A_tt2 = np.einsum('xyayxc -> ac', A) 
        K1_2 = -scale * np.einsum('bd, ac -> abcd', eye, A_tt2)
        
        # --- Adjoint of direct Gamma term (Term 2) ---
        K2 = np.einsum('abxcdx -> abcd', A)
        
        # --- Adjoint of cross terms (Terms 3 to 6) ---
        K3 = -np.einsum('xdaxbc -> abcd', A)
        K4 =  np.einsum('dxaxbc -> abcd', A)
        K5 =  np.einsum('xdabxc -> abcd', A)
        K6 = -np.einsum('dxabxc -> abcd', A)
        
        return K1_1 + K1_2 + K2 + K3 + K4 + K5 + K6

def calc_T1(gamma, conjugate=False):
    """
    Calculating T1 tensor

    Parameters
    ----------
    gamma: np.ndarray
        1DM or 2DM tensor
    N: int
        number of electrons in the system
    conjugate: bool
        conjugate or regular condition

    Returns
    -------
    np.ndarray

    Notes
    -----
    T1 is defined as:

    .. math::
        \begin{aligned}
            \mathcal{T}_1^{\dagger}(\Gamma)_{\alpha \beta ; \gamma \delta}=&\frac{2N}{N(N-1)}(\delta_{\alpha \gamma}\delta_{\beta \delta} - \delta_{\alpha \delta}\delta_{\beta \gamma}) \rm{Tr} A +\bar{A}_{\alpha\beta;\gamma\delta} \\
            &-\frac{1}{2(N-1)} [\delta_{\beta \delta} \bar{\bar{A}}_{\alpha \gamma} - \delta_{\alpha \delta}\bar{\bar{A}}_{\beta \gamma} - \delta_{\beta \gamma}\bar{\bar{A}}_{\alpha \delta} + \delta_{\alpha \gamma}\bar{\bar{A}}_{\beta \delta}]
        \end{aligned}

        \begin{aligned}
            \mathcal{T}_1^(\Gamma)_{\alpha \beta \gamma; \delta \epsilon \zeta}=&   \delta_{\gamma\zeta}\delta_{\beta\epsilon}\delta_{\alpha\delta} - \delta_{\gamma\epsilon}\delta_{\alpha\delta}\delta_{\beta\zeta} 
                                                                                  + \delta_{\alpha\zeta}\delta_{\gamma\epsilon}\delta_{\beta\delta} + \delta_{\gamma\zeta}\delta_{\alpha\epsilon}\delta_{\beta\delta}
                                                                                  + \delta_{\beta\zeta}\delta_{\alpha\epsilon}\delta_{\gamma\delta} - \delta_{\alpha\zeta}\delta_{\beta\epsilon}\delta_{\gamma\delta} \\
                                                                                & -(\delta_{\gamma\zeta}\delta_{\beta\epsilon} - \delta_{\beta\zeta}\delta_{gamma\epsilon})\rho_{\alpha\delta}
                                                                                  +(\delta_{\gamma]zeta}\delta_{\aplha\epsilon} - \delta_{\alpha\zeta}\delta_{\gamma\epsilon})\rho_{\beta\delta}
                                                                                  -(\delta_{\beta\zeta}\delta_{\alpha\epsilon} - \delta_{\alpha\zeta}\delta_{\beta\epsilon})\rho_{\gamma\delta} \\
                                                                                & +(\delta_{\gamma\zeta}\delta_{\beta\delta} - \delta_{\beta\zeta}\delta_{\gamma\delta})\rho_{\alpha\epsilon}
                                                                                  -(\delta_{\gamma\zeta}\delta_{\alpha\delta} - \delta_{\alpha\zeta}\delta_{\gamma\delta})\rho_{\epsilon\beta}
                                                                                  +(\delta_{\beta\zeta}\delta_{\alpha\delta} - \delta_{alpha\zeta}\delta_{\beta\delta})\rho_{gamma\epsilon} \\
                                                                                & -(\delta_{\beta\delta}\delta_{\gamma\epsilon} - \delta_{\beta\epsilon}\delta_{\gamma\delta})\rho_{alpha\zeta} 
                                                                                  +(\delta_{\gamma\epsilon}\delta_{\alpha\delta} - \delta_{\alpha\epsilon}\delta_{\gamma\delta})\rho_{\beta\zeta}
                                                                                  -(\delta_{\beta\epsilon}\delta_{\alpha\delta} - \delta_{\alpha\epsilon}\delta_{\beta\delta})\rho_{\gamma\zeta} \\
                                                                                & + \delta_{\gamma\zeta}\Gamma_{\alpha\beta;\delta\epsilon} - \delta_{\beta\zeta}\Gamma_{\alpha\gamma;\delta\epsilon}
                                                                                  + \delta_{\alpha\zeta}\Gamma_{\beta\gamma;\delta\epsilon} - \delta_{\gamma\epsilon}\Gamma_{\alpha\beta;\delta\zeta}
                                                                                  + \delta_{\beta\epsilon}\Gamma_{\alpha\gemma;\delta\zeta} - \delta_{\alpha\epsilon}\Gamma_{\beta\gamma;\delta]zeta} \\
                                                                                & + \delta_{\gamma\delta}\Gamma_{\alpha\beta;\epsilon\zeta} - \delta_{\beta\delta}\Gamma_{\alpha\gamma;\epsilon\zeta}
                                                                                  + \delta_{\alpha\delta}\Gamma_{\beta\gamma;\epsilon\zeta} .                                                        
        \end{aligned}
        

    """
    eye = np.eye(gamma.shape[0])

    if not conjugate:
        rho = 1 / (N-1) * np.einsum('abgb -> ag', gamma)
        term_1 = np.einsum('gz, be, ad -> abgdez', eye, eye, eye) - \
                 np.einsum('ge, ad, bz -> abgdez', eye, eye, eye) + \
                 np.einsum('az, ge, bd -> abgdez', eye, eye, eye) - \
                 np.einsum('gz, ae, bd -> abgdez', eye, eye, eye) + \
                 np.einsum('bz, ae, gd -> abgdez', eye, eye, eye) - \
                 np.einsum('az, be, gd -> abgdez', eye, eye, eye)
        term_2 = - np.einsum('gz, be, ad -> abgdez', eye, eye, rho) + \
                   np.einsum('bz, ge, ad -> abgdez', eye, eye, rho) + \
                   np.einsum('gz, ae, bd -> abgdez', eye, eye, rho) - \
                   np.einsum('az, ge, bd -> abgdez', eye, eye, rho) - \
                   np.einsum('bz, ae, gd -> abgdez', eye, eye, rho) + \
                   np.einsum('az, be, gd -> abgdez', eye, eye, rho)
        term_3 = np.einsum('gz, bd, ae -> abgdez', eye, eye, rho) - \
                 np.einsum('bz, gd, ae -> abgdez', eye, eye, rho) - \
                 np.einsum('gz, ad, eb -> abgdez', eye, eye, rho) + \
                 np.einsum('az, gd, eb -> abgdez', eye, eye, rho) + \
                 np.einsum('bz, ad, ge -> abgdez', eye, eye, rho) - \
                 np.einsum('az, bd, ge -> abgdez', eye, eye, rho)
        term_4 = - np.einsum('bd, ge, az -> abgdez', eye, eye, rho) + \
                   np.einsum('be, gd, az -> abgdez', eye, eye, rho) + \
                   np.einsum('ge, ad, bz -> abgdez', eye, eye, rho) - \
                   np.einsum('ae, gd, bz -> abgdez', eye, eye, rho) - \
                   np.einsum('be, ad, gz -> abgdez', eye, eye, rho) + \
                   np.einsum('ae, bd, gz -> abgdez', eye, eye, rho)
        term_5 = np.einsum('gz, abde -> abgdez', eye, gamma) - np.einsum('bz, agde -> abgdez', eye, gamma) + \
                 np.einsum('az, bgde -> abgdez', eye, gamma) - np.einsum('ge, abdz -> abgdez', eye, gamma) + \
                 np.einsum('be, agdz -> abgdez', eye, gamma) - np.einsum('ae, bgdz -> abgdez', eye, gamma) + \
                 np.einsum('gd, abez -> abgdez', eye, gamma) - np.einsum('bd, agez -> abgdez', eye, gamma) + \
                 np.einsum('ad, bgez -> abgdez', eye, gamma)
        k1 = term_1 + term_2 + term_3 + term_4 + term_5
        return k1

    else:
        tr_gamma = np.einsum('abcabc', gamma)
        gamma_abgd = np.einsum('ablgdl -> abgd', gamma)
        term_1 = 2 / (N*N - N) *\
                (np.einsum('ag, bd -> abgd', eye, eye) - np.einsum('ad, bg -> abgd', eye, eye)) * tr_gamma + gamma_abgd        
        
        gamma_ag = np.einsum('abgb -> ag', gamma_abgd)
        gamma_bg = np.einsum('abag -> bg', gamma_abgd)
        gamma_ad = np.einsum('agdg -> ad', gamma_abgd)
        gamma_bd = np.einsum('abda -> bd', gamma_abgd)
        
        term_2 = - 1 / (2*N - 2)*\
                (np.einsum('bd, ag -> abgd', eye, gamma_ag) - np.einsum('ad, bg -> abgd', eye, gamma_bg) -\
                 np.einsum('bg, ad -> abgd', eye, gamma_ad) + np.einsum('ag, bd -> abgd', eye, gamma_bd))

        return term_1 + term_2
    

def calculate_shift(eigenvalues, alpha):
    r"""
    Calculate the shift for density matrix for it to have the correct trace. 
    This step shifts the spectrum and eliminates the negative eigenvalues.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues of not shifted matrix.
    alpha : float
        Value of the coprrect trace.
        
    """
    #sample code, to be confirmed
    trace = lambda sigma0: np.sum(np.heaviside(eigenvalues - sigma0, 0.5)*(eigenvalues - sigma0))  
    constraint = lambda x: trace(x) - alpha  
    res = root(constraint, 0) 
    return res.x

r"""Set of algorithms for RDMREP package."""

import numpy as np
import scipy.sparse.linalg as spla
from .tools import find_closest_sdp
from . import constraints, tools, algorithms
from functools import partial


def solve_lagrange_multiplier(apply_A, apply_A_dagger, Delta, tol=1e-6, maxiter=None):
    """
    Solves the linear system (A * A_dagger) lambda = Delta using a 
    matrix-free Conjugate Gradient solver.
    
    Parameters:
    -----------
    apply_A : callable
        Function that applies the linear mapping A. 
        Takes a parameter-space tensor (e.g., Gamma) and returns a constraint-space tensor.
    apply_A_dagger : callable
        Function that applies the adjoint mapping A^dagger. 
        Takes a constraint-space tensor (e.g., lambda) and returns a parameter-space tensor.
    Delta : np.ndarray
        The residual tensor in the constraint space (e.g., Q_target - Q_current).
    tol : float
        Relative tolerance for the CG solver.
    maxiter : int, optional
        Maximum number of iterations for the CG solver.
        
    Returns:
    --------
    lambda_tensor : np.ndarray
        The optimized Lagrange multipliers, shaped identically to Delta.
    """
    # 1. Capture the exact shape and size of the constraint space
    constraint_shape = Delta.shape
    vec_size = Delta.size
    
    # 2. Define the composite matvec operator
    def matvec(lambda_vec):
        # a. Reshape the 1D solver vector back to the constraint tensor shape
        lambda_tensor = lambda_vec.reshape(constraint_shape)
        
        # b. Apply A_dagger: Maps from Constraint Space -> Parameter Space
        #    (e.g., from Q-space back to Gamma-space)
        param_tensor = apply_A_dagger(lambda_tensor)
        
        # c. Apply A: Maps from Parameter Space -> Constraint Space
        #    (e.g., from Gamma-space back to Q-space)
        constraint_tensor_new = apply_A(param_tensor)

        # d. appyly regularization if needed (e.g., adding a small multiple of the identity)
        regularized_output = constraint_tensor_new + 1e-8 * lambda_tensor
        
        # e. Flatten the output back to 1D for SciPy
        return regularized_output.reshape(-1) 

    # 3. Instantiate the SciPy LinearOperator
    M_operator = spla.LinearOperator(
        shape=(vec_size, vec_size), 
        matvec=matvec,
        dtype=Delta.dtype
    )
    
    # 4. Flatten the right-hand side (the residual)
    rhs_vec = Delta.reshape(-1)
    
    # 5. Solve using Conjugate Gradient
    lambda_sol_vec, exit_code = spla.cg(M_operator, rhs_vec, tol=tol, maxiter=maxiter)
    
    if exit_code > 0:
        print(f"Warning: CG stopped early after {exit_code} iterations without reaching tolerance.")
    elif exit_code < 0:
        print("Warning: CG encountered an illegal input or breakdown.")
        
    # 6. Reshape the solution back to the original constraint tensor shape
    return lambda_sol_vec.reshape(constraint_shape)

def apply_projection_step(
    tensor_current, 
    forward_map_eval,     # The full affine map for checking the state
    forward_map_linear,   # The purely linear map for the CG solver
    adjoint_map, 
    target_projector, 
    cg_tol=1e-6, 
    cg_maxiter=1000
):
    # 1. Use the full mapping to evaluate the current state
    mapped_current = forward_map_eval(tensor_current)
    
    # 2. Find the non-linear target
    target_tensor = target_projector(mapped_current)
    
    # 3. Calculate the residual 
    delta_constraint = target_tensor - mapped_current
    residual_norm = np.linalg.norm(delta_constraint)
    
    # 4. Solve the linear system using ONLY strictly linear operators
    lmbda = solve_lagrange_multiplier(
        apply_A=forward_map_linear, 
        apply_A_dagger=adjoint_map,
        Delta=delta_constraint,
        tol=cg_tol,
        maxiter=cg_maxiter
    )
    
    # 5. Map back using the linear adjoint
    tensor_corrected = tensor_current + adjoint_map(lmbda)
    
    return tensor_corrected, residual_norm

def enforce_halpern_projections(
    tensor_initial, 
    N_electrons,
    M_dim,
    traces,
    h=None,
    max_outer_iter=100000, 
    outer_tol=1e-5, 
    cg_tol=1e-6, 
    cg_maxiter=2000,
    verbose=True
):
    """
    Executes Halpern iteration with sequential projections to find a fixed point 
    in the intersection of multiple convex constraint sets.
    
    Parameters:
    -----------
    tensor_initial : np.ndarray
        The unpurified 2-RDM guess. This also acts as the "anchor" point for the Halpern step.
    N_electrons : int
        Number of electrons in the system.
    M_dim : int
        Number of spin orbitals.
    traces : dict
        Dictionary of target traces for the constraints (e.g., {'P': 12, 'Q': 12, 'G': 20, 'T2': 152}).
    h : np.ndarray, optional
        The reduced Hamiltonian. Required if you want to log the energy (verbose=True).
    max_outer_iter : int
        Maximum number of full cycles through all constraints.
    outer_tol : float
        Convergence threshold for the maximum residual distance to any cone.
    cg_tol : float
        Tolerance for the internal CG solver.
    cg_maxiter : int
        Maximum iterations for the internal CG solver.
    verbose : bool
        If True, prints logging information.
    
    Returns:
    --------
    tensor_current : np.ndarray
        The fully purified 2-RDM.
    """
    # Define constraints dynamically based on the provided system parameters
    halpern_constraints = [
        {
            'name': 'Antisymmetry',
            'forward_map_eval': lambda x: x,
            'forward_map_linear': lambda x: x,
            'adjoint_map': lambda x: x,
            'target_projector': constraints.project_antisymmetric # Assuming imported from rdmrep.tools
        },
        {
            'name': 'P-Condition',
            'forward_map_eval': lambda x: x,
            'forward_map_linear': lambda x: x,
            'adjoint_map': lambda x: x,
            'target_projector': partial(tools.find_closest_sdp, trace=traces['P'])
        },
        {
            'name': 'Q-Condition',
            'forward_map_eval': partial(constraints.calc_Q, N_electrons=N_electrons, M_dim=M_dim, conjugate=False, linear_only=False),
            'forward_map_linear': partial(constraints.calc_Q, N_electrons=N_electrons, M_dim=M_dim, conjugate=False, linear_only=True),
            'adjoint_map': partial(constraints.calc_Q, N_electrons=N_electrons, M_dim=M_dim, conjugate=True),
            'target_projector': partial(tools.find_closest_sdp, trace=traces['Q'])
        },
        {
            'name': 'G-Condition',
            'forward_map_eval': partial(constraints.calc_G, N_electrons=N_electrons, M_dim=M_dim, conjugate=False),
            'forward_map_linear': partial(constraints.calc_G, N_electrons=N_electrons, M_dim=M_dim, conjugate=False),
            'adjoint_map': partial(constraints.calc_G, N_electrons=N_electrons, M_dim=M_dim, conjugate=True),
            'target_projector': partial(tools.find_closest_sdp, trace=traces['G'])
        }
    ]
    
    # Optionally add T2 if defined in the traces dictionary
    if 'T2' in traces:
        halpern_constraints.append({
            'name': 'T2-Condition',
            'forward_map_eval': partial(constraints.calc_T2, N_electrons=N_electrons, M_dim=M_dim, conjugate=False),
            'forward_map_linear': partial(constraints.calc_T2, N_electrons=N_electrons, M_dim=M_dim, conjugate=False),
            'adjoint_map': partial(constraints.calc_T2, N_electrons=N_electrons, M_dim=M_dim, conjugate=True),
            'target_projector': partial(tools.find_closest_sdp, trace=traces['T2'])
        })
    
    # 1. Initialize the running state and define the Halpern anchor
    anchor_tensor = tensor_initial.copy()
    tensor_current = tensor_initial.copy()
    
    for iteration in range(max_outer_iter):
        # Halpern j-index (1-indexed based on your original script logic)
        j = iteration + 1  
        max_residual_norm = 0.0
        
        gamma_tmp = tensor_current
        
        # 2. Sequential Projections (No memory increments unlike Dykstra)
        for constraint in halpern_constraints:
            
            tensor_projected, residual_norm = algorithms.apply_projection_step(
                tensor_current=gamma_tmp,
                forward_map_eval=constraint['forward_map_eval'],
                forward_map_linear=constraint['forward_map_linear'],
                adjoint_map=constraint['adjoint_map'],
                target_projector=constraint['target_projector'],
                cg_tol=cg_tol,
                cg_maxiter=cg_maxiter
            )
            
            # Track the worst-case violation across all constraints in this cycle
            max_residual_norm = max(max_residual_norm, residual_norm)
            
            # Pass the result directly to the next constraint
            gamma_tmp = tensor_projected
            
        # 3. Halpern step: Anchor the sequentially projected tensor to the initial state
        tensor_current = (1.0 / (j + 1)) * anchor_tensor + (j / (j + 1)) * gamma_tmp
            
        # 4. Logging
        if verbose and (iteration % 100 == 0 or iteration == 0):
            print(f"Iteration {iteration:5d}: Max Cone Residual = {max_residual_norm:.6e}")
            # Only attempt to calculate energy if Hamiltonian is provided
            if h is not None:
                print(f"Energy at iteration {iteration:5d}: {np.sum(tensor_current * h):.6f}")
            
        # 5. Global Convergence Check
        # If the maximum shift required to satisfy ANY constraint is below tolerance, we are done.
        if max_residual_norm < outer_tol:
            if verbose:
                print(f"Halpern algorithm converged strictly in {iteration + 1} cycles.")
            break
            
    else:
        if verbose:
            print(f"Warning: Reached maximum iterations ({max_outer_iter}) without strict convergence. "
                  f"Final Max Residual: {max_residual_norm:.6e}")
            
    return tensor_current


def enforce_dykstra_projections(
    tensor_initial, 
    N_electrons,
    M_dim,
    traces,
    h=None,
    max_outer_iter=100, 
    outer_tol=1e-5, 
    cg_tol=1e-6, 
    cg_maxiter=2000,
    verbose=True
):
    """
    Executes Dykstra's Projection Algorithm to find the exact closest tensor 
    in the intersection of multiple convex constraint sets.
    
    Parameters:
    -----------
    tensor_initial : np.ndarray
        The unpurified 2-RDM guess.
    N_electrons : int
        Number of electrons in the system.
    M_dim : int
        Number of spin orbitals.
    traces : dict
        Dictionary of target traces for the constraints (e.g., {'P': 12, 'Q': 12, 'G': 20}).
    h : np.ndarray, optional
        The reduced Hamiltonian. Required if you want to log the energy (verbose=True).
    max_outer_iter : int
        Maximum number of full cycles through all constraints.
    outer_tol : float
        Convergence threshold for the maximum residual distance to any cone.
    cg_tol : float
        Tolerance for the internal CG solver.
    cg_maxiter : int
        Maximum iterations for the internal CG solver.
    verbose : bool
        If True, prints logging information.
    
    Returns:
    --------
    tensor_current : np.ndarray
        The fully purified 2-RDM.
    """
    # Define constraints dynamically based on the provided system parameters
    dykstra_constraints = [
        {
            'name': 'Antisymmetry',
            'forward_map_eval': lambda x: x,
            'forward_map_linear': lambda x: x,
            'adjoint_map': lambda x: x,
            'target_projector': constraints.project_antisymmetric # Assuming imported from rdmrep.tools
        },
        {
            'name': 'P-Condition',
            'forward_map_eval': lambda x: x,
            'forward_map_linear': lambda x: x,
            'adjoint_map': lambda x: x,
            'target_projector': partial(tools.find_closest_sdp, trace=traces['P'])
        },
        {
            'name': 'Q-Condition',
            'forward_map_eval': partial(constraints.calc_Q, N_electrons=N_electrons, M_dim=M_dim, conjugate=False, linear_only=False),
            'forward_map_linear': partial(constraints.calc_Q, N_electrons=N_electrons, M_dim=M_dim, conjugate=False, linear_only=True),
            'adjoint_map': partial(constraints.calc_Q, N_electrons=N_electrons, M_dim=M_dim, conjugate=True),
            'target_projector': partial(tools.find_closest_sdp, trace=traces['Q'])
        },
        {
            'name': 'G-Condition',
            'forward_map_eval': partial(constraints.calc_G, N_electrons=N_electrons, M_dim=M_dim, conjugate=False),
            'forward_map_linear': partial(constraints.calc_G, N_electrons=N_electrons, M_dim=M_dim, conjugate=False),
            'adjoint_map': partial(constraints.calc_G, N_electrons=N_electrons, M_dim=M_dim, conjugate=True),
            'target_projector': partial(tools.find_closest_sdp, trace=traces['G'])
        }
    ]
    # Optionally add T2 if defined in the traces dictionary
    if 'T2' in traces:
        dykstra_constraints.append({
            'name': 'T2-Condition',
            'forward_map_eval': partial(constraints.calc_T2, N_electrons=N_electrons, M_dim=M_dim, conjugate=False),
            'forward_map_linear': partial(constraints.calc_T2, N_electrons=N_electrons, M_dim=M_dim, conjugate=False),
            'adjoint_map': partial(constraints.calc_T2, N_electrons=N_electrons, M_dim=M_dim, conjugate=True),
            'target_projector': partial(tools.find_closest_sdp, trace=traces['T2'])
        })
    
    # 1. Initialize the running state
    tensor_current = tensor_initial.copy()
    
    # 2. Initialize the memory increments (one for each constraint, starting at zero)
    increments = [np.zeros_like(tensor_initial) for _ in dykstra_constraints]
    
    for iteration in range(max_outer_iter):
        max_residual_norm = 0.0
        tensor_old = tensor_current.copy()
        for c_idx, constraint in enumerate(dykstra_constraints):
            
            # A. Remove the previous increment for this specific constraint
            pre_projection_state = tensor_current - increments[c_idx]
            
            # B. Apply the projection to find the closest point in this cone
            tensor_projected, residual_norm = algorithms.apply_projection_step(
                tensor_current=pre_projection_state,
                forward_map_eval=constraint['forward_map_eval'],
                forward_map_linear=constraint['forward_map_linear'],
                adjoint_map=constraint['adjoint_map'],
                target_projector=constraint['target_projector'],
                cg_tol=cg_tol,
                cg_maxiter=cg_maxiter
            )
            
            # Track the worst-case violation across all constraints in this cycle
            max_residual_norm = max(max_residual_norm, residual_norm)
            
            # C. Update the memory increment for this constraint
            increments[c_idx] = tensor_projected - pre_projection_state
            
            # D. Update the running state for the next constraint in the cycle
            tensor_current = tensor_projected
            
        diff_tensor_norm = np.linalg.norm(tensor_current - tensor_old)
        # Logging
        if verbose and iteration % 10 == 0:
            print("Iteration {:5d}: Diff Norm = {:.6e}".format(iteration, diff_tensor_norm))
            # Only attempt to calculate energy if Hamiltonian is provided
            if h is not None:
                print(f"Energy at iteration {iteration:3d}: ", np.sum(tensor_current * h))
            
        # Global Convergence Check
        if diff_tensor_norm < outer_tol:
            if verbose:
                print(f"Dykstra converged to the intersection in {iteration + 1} cycles (change norm < outer_tol).")
            break
    
    else:
        if verbose:
            print(f"Warning: Reached maximum iterations ({max_outer_iter}) without strict convergence. "
                  f"Final Max Residual: {max_residual_norm:.6e}")
            
    return tensor_current

def dykstra(D, constraints, traces, N=4, alpha=0.9, max_iter=20):
    """
    Dykstra's algorithm for projecting onto the set of constraints.

    Parameters
    ----------
    D : np.ndarray
        Initial Density matrix
    constraints : list
        List of constraints. 
    traces : list
        List of correct traces. The first element is the trace of the density matrix.
        Number of trace elements is equal to the number of constraints + 1.
    N : int
        Number of electrons.
    alpha : float
        Step size.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    res : dict
        Dictionary containing the density matrix, 
        the auxiliary variable, 
        and the history of the auxiliary variable.

    """  
    xj_history = []
    rdm_trace = traces[0]
    X_j = np.zeros(D.shape)

    for i in range(max_iter):
        for constraint, trace in zip(constraints, traces[1:]):
            xj_history.append(np.linalg.norm(X_j))

            C  = D - X_j

            mapping = constraint(C, N, conjugate=False)
            mapping_pos = find_closest_sdp(mapping, trace)

            D = constraint(mapping_pos, N, conjugate=True)
            D = find_closest_sdp(D, rdm_trace)

            X_j = alpha*(D - C)

    res =  {"D": D, "X_j": X_j, "xj_history": xj_history}
    return res


def halpern(D, constraints, traces, N=4, M=8, alpha=0.9, eps=1, max_iter=20):
    """
    Newman's algorithm for projecting onto the set of constraints.

    Parameters
    ----------
    D : np.ndarray
        Initial Density matrix
    constraints : list
        List of constraints. 
    traces : list
        List of correct traces. The first element is the trace of the density matrix.
        Number of trace elements is equal to the number of constraints + 1.
    N : int
        Number of electrons.
    alpha : float
        Hyperparameter.
    eps : float
        Tolerance.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    res : dict
        Dictionary containing the density matrix, 
        and the history of the auxiliary variable.

    """  
    gamma_new = D.copy()
    rdm_trace = traces[0]

    norm = [np.inf]
    for i in range (1, max_iter+1):
        for projection, trace in zip(constraints, traces[1:]):
            A = projection(gamma_new, N=N, M=M, conjugate=False)
            A_pos = find_closest_sdp(A, trace)

            # inverse mapping
            gamma_new = projection(A_pos, N=N, M=M, conjugate=True)
            gamma_new = find_closest_sdp(gamma_new, rdm_trace)

        gamma_new=(1.0/(i+1))*D +(i/(i+1))*gamma_new
        norm.append(np.linalg.norm(gamma_new - D))
        is_stop = alpha * abs(norm[i] - norm[i - 1]) + (1 - alpha) * norm[i] < eps
        if is_stop:
            break
    res = {"D": gamma_new, "norm": norm}
    return res