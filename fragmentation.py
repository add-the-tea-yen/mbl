import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
import warnings
from collections import Counter

# Suppress RankWarning for polynomial fitting which can occur in level statistics

## -----------------------------------------------------------
## --- DIAGNOSTIC AND ANALYSIS FUNCTIONS
## -----------------------------------------------------------

def find_connected_components(operator):
    """Finds dynamically disconnected subspaces by treating the operator as a graph."""
    print("Building graph and finding connected components...")
    graph = sp.csr_matrix(operator != 0, dtype=int)
    n_components, labels = connected_components(
        csgraph=graph, directed=False, return_labels=True
    )
    component_sizes = np.bincount(labels)
    return n_components, component_sizes

def get_krylov_dimension(operator, initial_state, tol=1e-9):
    """Calculates the dimension of the Krylov subspace for a given initial state."""
    dim = operator.shape[0]
    q = initial_state.astype(np.complex128).flatten()
    q /= np.linalg.norm(q)
    krylov_basis = [q]
    for _ in range(dim):
        v = operator.dot(q)
        for basis_vec in krylov_basis:
            v -= np.vdot(basis_vec, v) * basis_vec
        norm_v = np.linalg.norm(v)
        if norm_v < tol:
            break
        q = v / norm_v
        krylov_basis.append(q)
    return len(krylov_basis)

def calculate_ipr(states):
    """Calculates the Inverse Participation Ratio (IPR) for a set of states."""
    # IPR = sum_i |psi_i|^4
    return np.sum(np.abs(states)**4, axis=0)

## -----------------------------------------------------------
## --- MAIN ANALYSIS FUNCTION (EXPANDED)
## -----------------------------------------------------------

def analyze_system_for_fragmentation(operator, system_size):
    """
    Main function to run all diagnostics and determine if a system is fragmented.
    """
    total_dim = operator.shape[0]
    print(f"--- Starting HSF Analysis for a system of Hilbert space dimension D = {total_dim} ---\n")

    # --- 1. Graph Connectivity ---
    print("## 1. Connected Components Analysis ##")
    n_comp, comp_sizes = find_connected_components(operator)
    print(f"Found {n_comp} connected components.")
    if n_comp > 1:
        print(f"Component sizes: {comp_sizes[comp_sizes > 0]}")
        print("-> Fragmentation Signature: ✅ DETECTED")
        is_fragmented_graph = True
    else:
        print("-> Fragmentation Signature: ❌ NOT DETECTED")
        is_fragmented_graph = False
    print("-" * 50)

    # --- Prepare for slow tests ---
    eigenvalues, eigenvectors = None, None
    if system_size <= 14:
        print("Running full diagonalization (for IPR and Level Stats)...")
        eigenvalues, eigenvectors = np.linalg.eig(operator.toarray())
        print("Diagonalization complete.")
    
    # --- 2. IPR Analysis ---
    print("## 2. Inverse Participation Ratio (IPR) Analysis ##")
    if eigenvectors is not None:
        ipr_values = calculate_ipr(eigenvectors)
        mean_ipr = np.mean(ipr_values)
        # For an ergodic system, IPR is ~1/D. We check if it's significantly larger.
        is_localized_ipr = mean_ipr > 5/total_dim 
        print(f"Mean IPR: {mean_ipr:.6f} (Compare to ergodic value ~{1/total_dim:.6f})")
        if is_localized_ipr:
            print("-> Fragmentation Signature: ✅ DETECTED (High IPR suggests localized eigenstates)")
        else:
            print("-> Fragmentation Signature: ❌ NOT DETECTED (Low IPR suggests delocalized eigenstates)")
    else:
        print("System size > 14. Skipping IPR analysis.")
    print("-" * 50)

    # --- 3. Level Statistics ---
    print("## 3. Level Statistics Analysis ##")
    if eigenvalues is not None:
        quasi_energies = np.angle(eigenvalues)
        quasi_energies.sort()
        spacings = np.diff(quasi_energies)
        spacings = spacings[spacings > 1e-9]
        if len(spacings) < 2: avg_r = np.nan
        else:
            r_n = np.minimum(spacings[:-1], spacings[1:]) / np.maximum(spacings[:-1], spacings[1:])
            avg_r = np.mean(r_n)
        print(f"Average quasi-energy spacing ratio <r> = {avg_r:.4f}")
        if avg_r < 0.45:
            print("-> Fragmentation Signature: ✅ DETECTED (Poisson-like, <r> ≈ 0.39)")
            is_poisson = True
        else:
            print("-> Fragmentation Signature: ❌ NOT DETECTED (Wigner-Dyson-like, <r> ≈ 0.53)")
            is_poisson = False
    else:
        is_poisson = None
        print("System size > 14. Skipping level statistics analysis.")
    print("-" * 50)

    # --- 4. Krylov Sector Size Distribution ---
    print("## 4. Krylov Sector Size Distribution ##")
    if system_size <= 12: # This test is very slow
        print("Calculating Krylov dimension for all basis states (this is slow)...")
        all_krylov_dims = []
        for i in range(total_dim):
            initial_state = np.zeros(total_dim)
            initial_state[i] = 1.0
            all_krylov_dims.append(get_krylov_dimension(operator, initial_state))
        
        sector_counts = Counter(all_krylov_dims)
        is_krylov_fragmented = len(sector_counts) > 1
        print(f"Found {len(sector_counts)} distinct sector size(s).")
        print("Counts: {Sector Size: Number of States}")
        for size, count in sector_counts.items():
            print(f"  {{{size}: {count}}}")
            
        if is_krylov_fragmented:
            print("-> Fragmentation Signature: ✅ DETECTED (Multiple sector sizes found)")
        else:
            print("-> Fragmentation Signature: ❌ NOT DETECTED (All states belong to one sector)")
        
        # Plotting the distribution
        plt.figure(figsize=(8, 6))
        plt.bar(sector_counts.keys(), sector_counts.values(), width=0.8)
        plt.xlabel("Krylov Sector Size")
        plt.ylabel("Number of States")
        plt.title("Krylov Sector Size Distribution")
        plt.yscale('log')
        plt.grid(axis='y', linestyle='--')
        plt.show()
    else:
        is_krylov_fragmented = None
        print("System size > 12. Skipping full Krylov sector distribution (too slow).")
    print("-" * 50)
    
    
    print("## 5. Krylov Subspace Dimension Analysis ##")
    initial_state = np.zeros(total_dim)
    initial_state[0] = 1.0 # Start from |...↓↓↓> state
    krylov_dim = get_krylov_dimension(operator, initial_state)
    print(f"Starting from state |...↓↓↓>, the Krylov subspace dimension is: {krylov_dim}")
    is_krylov_fragmented = krylov_dim < total_dim
    if is_krylov_fragmented:
         print(f"-> Fragmentation Signature: ✅ DETECTED (Subspace dim {krylov_dim} < Total dim {total_dim})")
    else:
        print("-> Fragmentation Signature: ❌ NOT DETECTED (State explores the full Hilbert space)")
    print("-" * 50)
    
    
    # --- Final Conclusion ---
    print("\n## 🏁 FINAL CONCLUSION ##")
    if is_fragmented_graph or is_poisson or is_krylov_fragmented:
        print("The system shows strong evidence of Hilbert Space Fragmentation.")
    else:
        print("The system appears to be ERGODIC and not fragmented based on these tests.")

## -----------------------------------------------------------
## --- KICKED FIELD ISING MODEL (KFIM) IMPLEMENTATION
## -----------------------------------------------------------

def create_kfim_operator(L, J, h_stagger, b):
    """Constructs the Floquet operator U_KFIM for the Kicked Field Ising Model."""
    D = 2**L
    print(f"Building KFIM operator for L={L}, D={D}...")
    H_z_diag = np.zeros(D)
    for i in range(D):
        energy = 0
        for j in range(L):
            spin_j = 1 if (i >> j) & 1 else -1
            energy += h_stagger * ((-1)**j) * spin_j
            spin_j_plus_1 = 1 if (i >> ((j + 1) % L)) & 1 else -1
            energy += J * spin_j * spin_j_plus_1
        H_z_diag[i] = energy
    U_z = sp.diags(np.exp(-1j * H_z_diag), format='csr')
    row, col, data = [], [], []
    for i in range(D):
        for j in range(L):
            j_flipped_state = i ^ (1 << j)
            row.append(i); col.append(j_flipped_state); data.append(b)
    H_x = sp.csr_matrix((data, (row, col)), shape=(D, D))
    print("Calculating matrix exponential for U_x (this may take a moment)...")
    U_x = expm(-1j * H_x)
    U_kfim = U_z @ U_x
    print("KFIM operator successfully built.")
    return U_kfim

## -----------------------------------------------------------
## --- MAIN EXECUTION BLOCK
## -----------------------------------------------------------

if __name__ == "__main__":
    
    # Set Parameters for a Fragmented KFIM System
    L = 10                   # System size (L=8 is good for demonstrating all tests)
    J = np.pi / 2.0         # Ising interaction
    b = 1.4                 # Transverse kick strength
    h_stagger = 0.3         # Staggered longitudinal field
    
    # Create the Floquet operator for the KFIM
    U_kfim = create_kfim_operator(L=L, J=J, h_stagger=h_stagger, b=b)
    
    # Analyze the generated system for fragmentation
    analyze_system_for_fragmentation(U_kfim, system_size=L)
