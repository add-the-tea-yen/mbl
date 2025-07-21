import numpy as np
from functools import reduce
from scipy.linalg import expm  # dense matrix exponential
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt

import time

start = time.time()


# ------------------------------ parameters ------------------------------

spins = [1/2, 3/2, 1/2, 3/2, 1/2, 1.5, 1/2, 3/2]

L = len(spins)                      # Chain length
J = np.pi/4               # Ising ZZ coupling strength
b = np.pi/4              # Kick strength
g = np.pi/4         # Transverse field coupling

# Site-dependent longitudinal fields h_j
#np.random.seed(9)
h = np.random.normal(0.6, np.pi / 4, size=L)

bins = 25
# ------------------------------ spin operators -------------------
def spin_matrices(s):
    """Generate Sx, Sz, and Identity matrices for spin s"""
    dim = int(2*s + 1)
    
    # Sz matrix (diagonal)
    sz = np.diag([m for m in np.arange(s, -s-1, -1)])
    
    # Sx matrix (off-diagonal) - optimized construction
    sx = np.zeros((dim, dim), dtype=complex)
    for i in range(dim-1):
        m_i = s - i
        m_j = s - (i+1)
        val = 0.5 * np.sqrt(s*(s+1) - m_i*m_j)
        sx[i, i+1] = val
        sx[i+1, i] = val
    
    # Identity matrix
    I = np.eye(dim, dtype=complex)
    
    return sx, sz, I

# Generate operators for each spin
operators = []
for s in spins:
    sx, sz, I = spin_matrices(s)
    operators.append((sx, sz, I))

print(f"Spin chain: {spins}")
print(f"Hilbert space dimensions: {[op[2].shape[0] for op in operators]}")

# Total Hilbert space dimension
total_dim = np.prod([op[2].shape[0] for op in operators])
print(f"Total Hilbert space dimension: {total_dim}")

# ------------------------------ build H^z -------------------------------
Hz = np.zeros((total_dim, total_dim), dtype=complex)

# ZZ interaction: open boundary (no wrapping)
for j in range(L - 1):
    ops = [operators[i][2] for i in range(L)]  # Start with identities
    ops[j] = operators[j][1]      # sz for site j
    ops[j + 1] = operators[j + 1][1]  # sz for site j+1
    Hz += J * reduce(np.kron, ops)

# On-site longitudinal fields h_j * sz_j
for j in range(L):
    ops = [operators[i][2] for i in range(L)]  # Start with identities
    ops[j] = operators[j][1]      # sz for site j
    Hz += h[j] * reduce(np.kron, ops)

# ------------------------------ build H^x -------------------------------
Hx = np.zeros((total_dim, total_dim), dtype=complex)

# On-site transverse field: b * sx_j
for j in range(L):
    ops = [operators[i][2] for i in range(L)]  # Start with identities
    ops[j] = operators[j][0]      # sx for site j
    Hx += b * reduce(np.kron, ops)

# Optional: Add XX interactions (commented out in original)
"""
# XX interaction: open boundary
for j in range(L - 1):
    ops = [operators[i][2] for i in range(L)]  # Start with identities
    ops[j] = operators[j][0]      # sx for site j
    ops[j + 1] = operators[j + 1][0]  # sx for site j+1
    Hx += reduce(np.kron, ops)

for j in range(L - 3):
    ops = [operators[i][2] for i in range(L)]  # Start with identities
    ops[j] = operators[j][0]      # sx for site j
    ops[j + 3] = operators[j + 3][0]  # sx for site j+3
    Hx += (2/3) * reduce(np.kron, ops)
"""

# ------------------------------ Floquet operator ------------------------
U = expm(-1j * Hz) @ expm(-1j * Hx)   # Floquet unitary for one period

# ------------------------------ Analysis ------------------------------------
print("Floquet operator U built with open boundary conditions. Shape:", U.shape)

# Compute eigenvalues - using full diagonalization for consistency
print("Computing eigenvalues...")
# Use full diagonalization instead of sparse eigs for deterministic results
w = np.linalg.eigvals(U)
print(f"Found {len(w)} eigenvalues")
phases = np.angle(w)
np.savetxt("AHPHASES.csv", phases, delimiter=",")
# ------------------------------ Eigenvalue clipping ------------------------
# Remove eigenvalues that are too close to numerical precision issues
# Clip eigenvalues with very small imaginary parts (numerical errors)
tolerance = 1e-12
w_real_mask = np.abs(w.imag) < tolerance
w_clipped = w[w_real_mask] if np.any(w_real_mask) else w

# Additionally, ensure eigenvalues are on the unit circle (for unitary matrices)
w_magnitudes = np.abs(w_clipped)
unit_circle_mask = np.abs(w_magnitudes - 1.0) < 1e-10
w_clipped = w_clipped[unit_circle_mask]

print(f"Eigenvalues after clipping: {len(w_clipped)} (removed {len(w) - len(w_clipped)})")

# Level spacing analysis
phases = np.angle(w_clipped)
phases = np.sort(phases)
spacings = np.diff(phases)

# Handle the circular nature - add spacing between last and first eigenvalue
circular_spacing = 2*np.pi + phases[0] - phases[-1]
spacings = np.append(spacings, circular_spacing)

# Normalize by mean level spacing
mean_spacing = np.mean(spacings)
spacings = spacings / mean_spacing

# Remove very large spacings that might be artifacts
spacings = spacings[spacings < 3.5]

# Plot histogram
# --- Range for plotting theoretical distributions ---
s_max = np.max(spacings) * 1.2
s_values = np.linspace(0, s_max, 500)

# --- Theoretical PDFs (pre-computed constants) ---
poisson_pdf = np.exp(-s_values)  # Poisson (integrable)
# COE (Circular Orthogonal Ensemble) / GOE (β = 1)
coe_pdf = (np.pi / 2) * s_values * np.exp(- (np.pi / 4) * s_values**2)

# CUE (Circular Unitary Ensemble) / GUE (β = 2)
cue_pdf = (32 / np.pi**2) * s_values**2 * np.exp(- (4 / np.pi) * s_values**2)

# CSE (Circular Symplectic Ensemble) / GSE (β = 4)
cse_pdf = (2**18 / (3**6 * np.pi**3)) * s_values**4 * np.exp(- (64 / (9 * np.pi)) * s_values**2)

# --- Plotting ---
plt.figure(figsize=(12, 8))
plt.hist(spacings, bins=bins, density=True, histtype='step', linewidth=2, label='Empirical Data', color='black')
plt.plot(s_values, poisson_pdf, 'r--', linewidth=2, label='Poisson ($P(s) = e^{-s}$)')
plt.plot(s_values, coe_pdf, 'b-', linewidth=2, label='COE (β=1)', alpha=0.6)
plt.plot(s_values, cue_pdf, 'g-', linewidth=2, label='CUE (β=2)', alpha=0.5)
plt.plot(s_values, cse_pdf, 'm-', linewidth=2, label='CSE (β=4)', alpha=0.5)

plt.title(f"Level Spacing Distribution - Spin Chain {spins}", fontsize=14)
plt.xlabel("Normalized Spacing $s$", fontsize=12)
plt.ylabel("Probability Density $P(s)$", fontsize=12)
plt.grid(True, ls='--', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()

# Print some statistics
print(f"\nSystem Statistics:")
print(f"Number of eigenvalues analyzed: {len(spacings)}")
print(f"Mean spacing: {np.mean(spacings):.4f}")
print(f"Std spacing: {np.std(spacings):.4f}")
print(f"Min spacing: {np.min(spacings):.4f}")
print(f"Max spacing: {np.max(spacings):.4f}")

# Calculate some key statistics for comparison
min_spacings = np.minimum(spacings[:-1], spacings[1:])
max_spacings = np.maximum(spacings[:-1], spacings[1:])
r_ratio = np.mean(min_spacings / max_spacings)
print(f"Average r-ratio: {r_ratio:.4f}")
print(f"Poisson r-ratio ≈ 0.386, COE/CUE r-ratio ≈ 0.536, CSE r-ratio ≈ 0.603")

end = time.time()
print(f"Execution time: {end - start:.6f} seconds")
