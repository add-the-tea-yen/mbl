import numpy as np
from functools import reduce
from scipy.linalg import expm  # dense matrix exponential
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt

import time

start = time.time()


# ------------------------------ parameters ------------------------------

spins = [1,1,1,1,1,1] 

L = len(spins)                      # Chain length
J = np.pi/4               # Ising ZZ coupling strength
b = np.pi/4              # Kick strength
g = np.pi/4         # Transverse field coupling

# Site-dependent longitudinal fields h_j
np.random.seed(6658)
h = np.random.uniform(0.6, np.pi / 4, size=L)


# ------------------------------ spin operators -------------------
def spin_matrices(s):
    """Generate Sx, Sz, and Identity matrices for spin s"""
    dim = int(2*s + 1)
    
    # Sz matrix (diagonal)
    sz = np.diag([m for m in np.arange(s, -s-1, -1)])
    
    # Sx matrix (off-diagonal)
    sx = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            m_i = s - i
            m_j = s - j
            if abs(i - j) == 1:  # Adjacent m values
                sx[i, j] = 0.5 * np.sqrt(s*(s+1) - m_i*m_j)
    
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


