import numpy as np
from functools import reduce
from scipy.linalg import expm
import matplotlib.pyplot as plt
import scipy as sp

def EDU(L,psi,phases):
    # ------------------------------ parameters ------------------------------
    L = L                      # Chain length
    J = np.pi / 4              # Ising ZZ coupling strength
    b = np.pi / 4              # Kick strength
    g = np.pi / 4              # Transverse field coupling
    h = np.random.uniform(0.6, np.pi / 4, size=L)  # Site-dependent fields
    
    # ------------------------------ single-site operators -------------------
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    I2 = np.eye(2, dtype=np.complex128)
    
    # ------------------------------ build H^z -------------------------------
    dim = 2 ** L
    Hz = np.zeros((dim, dim), dtype=np.complex128)
    
    for j in range(L):
        ops = [I2] * L
        ops[j] = sz
        ops[(j + 1) % L] = sz
        Hz += reduce(np.kron, ops)
    Hz *= J
    
    for j in range(L):
        ops = [I2] * L
        ops[j] = sz
        Hz += h[j] * reduce(np.kron, ops)
    
    # ------------------------------ build H^x -------------------------------
    Hx = np.zeros((dim, dim), dtype=np.complex128)
    
    for j in range(L):
        ops = [I2] * L
        ops[j] = sx
        Hx += reduce(np.kron, ops)
    
    for j in range(L):
        ops = [I2] * L
        ops[j] = sx
        ops[(j + 1) % L] = sx
        Hx += reduce(np.kron, ops)
    
    for j in range(L):
        ops = [I2] * L
        ops[j] = sx
        ops[(j + 3) % L] = sx
        Hx += (2 / 3) * reduce(np.kron, ops)
    
    Hx *= b
    
    # ------------------------------ Floquet operator ------------------------
    U = expm(-1j * Hz) @ expm(-1j * Hx)
    print("Floquet operator U built with closed boundary conditions. Shape:", U.shape)
    
    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(U)
    quasienergies = np.angle(eigenvalues)
    sorted_indices = np.argsort(quasienergies)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    # Phases Generation 
    phases = np.sort(np.mod(np.angle(lambdas), 2 * np.pi))

    # Export
    np.savetxt(psi, eigenvectors, delimiter=",")
    
    np.savetxt(phases, phases, delimiter=",")
    
    
    print(f"Computed {len(eigenvalues)} eigenvalues")
    
