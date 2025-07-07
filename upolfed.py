#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 00:36:04 2025
@author: adityan
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla

#-------------------------

# Pauli matrices
I2 = np.eye(2, dtype=np.complex64)
X = np.array([[0, 1], [1, 0]], dtype=np.complex64)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex64)

# --- Gate constructors ---
def rz_gate(angle):
    return np.cos(angle) * I2 - 1j * np.sin(angle) * Z

def rx_gate(angle):
    return np.cos(angle) * I2 - 1j * np.sin(angle) * X

def zz_gate(J):
    return spla.expm(-1j * J * np.kron(Z, Z)).reshape(4, 4).astype(np.complex64)

# --- Apply single-qubit gate ---
def apply_single_qubit_gate(psi, gate, site, L):
    psi = psi.reshape([2] * L)
    psi = np.moveaxis(psi, site, 0)
    shape = psi.shape
    psi = psi.reshape(2, -1)
    psi = (gate @ psi).reshape(2, *shape[1:])
    psi = np.moveaxis(psi, 0, site)
    return psi.reshape(-1)

# --- Apply two-qubit brickwork layer U = Ua + Ub ---
def apply_brick_layer(psi, gate, L, even=True):
    psi = psi.reshape([2] * L)
    for i in range(0 if even else 1, L - 1, 2):
        psi = np.moveaxis(psi, [i, i + 1], [0, 1])
        shape = psi.shape
        psi = psi.reshape(4, -1)
        psi = (gate @ psi).reshape(2, 2, *shape[2:])  # Fixed reshape
        psi = np.moveaxis(psi, [0, 1], [i, i + 1])
    return psi.reshape(-1)

# --- Apply longitudinal field hj ---
def apply_hz_field(psi, h_vec, L):
    for j in range(L):
        Rz = rz_gate(h_vec[j])
        psi = apply_single_qubit_gate(psi, Rz, j, L)
    return psi

# --- Floquet operator application, proper construction of U---
def apply_floquet(psi, L, J, b, h_vec=None, Rx=None, zz=None):
    if zz is None:
        zz = zz_gate(J)
    psi = apply_brick_layer(psi, zz, L, even=True)
    psi = apply_brick_layer(psi, zz, L, even=False)
    if h_vec is not None:
        psi = apply_hz_field(psi, h_vec, L)
    if Rx is None:
        Rx = rx_gate(b)
    for j in range(L):
        psi = apply_single_qubit_gate(psi, Rx, j, L)
    return psi

# --- Geometric filter using weighted complex exponential sum gk(U)---
def apply_geometric_filter(psi, L, J, b, k, phi_tgt, h_vec=None):
    Rx = rx_gate(b)
    zz = zz_gate(J)
    result = np.zeros_like(psi, dtype=np.complex64)
    psi_k = psi.copy()
    for m in range(k + 1):
        weight = np.exp(-1j * m * phi_tgt).astype(np.complex64)
        result += weight * psi_k
        psi_k = apply_floquet(psi_k, L, J, b, h_vec, Rx=Rx, zz=zz)
    return result

# --- Matrix-free operator for g_k(U) ---
class GeometricFilteredOperator(spla.LinearOperator):
    def __init__(self, L, J, b, k, phi_tgt, h_vec=None):
        self.L = L
        self.J = J
        self.b = b
        self.k = k
        self.phi_tgt = phi_tgt
        self.h_vec = h_vec
        self.d = 2 ** L
        super().__init__(dtype=np.complex64, shape=(self.d, self.d))

    def _matvec(self, v):
        return apply_geometric_filter(v, self.L, self.J, self.b, self.k, self.phi_tgt, self.h_vec)

# --- Level spacing plot ---
def run_level_spacing(L=8, J=np.pi/4, b=0.9, phi_tgt=0.0, nev=None, k=None, ncv=None, disorder=False):
    d = 2 ** L
    
    nev = int(min((2**L/10),1000))
    nev=225
    #print(nev)
    ncv = int(2 * nev)
    #print(ncv)
    k = int(0.95 * (2**(L+1))/ncv)
    #print(k)
    
    #nev=225
    #ncv = min(d, 2 * nev + 20)
    #k = 500
    
    
    
    #if k is None: k=700
    
    h_vec = np.random.uniform(0.6, np.pi/4, size=L).astype(np.float32) if disorder else None

    print(f"[POLFED] L={L}, J={J:.3f}, b={b:.3f}, k={k}, nev={nev}, disorder={disorder}")

    G = GeometricFilteredOperator(L, J, b, k, phi_tgt, h_vec)

    v0 = (np.random.randn(d) + 1j * np.random.randn(d)).astype(np.complex64)
    v0 /= np.linalg.norm(v0)

    eigvals, eigvecs = spla.eigs(G, k=nev, which='LM', v0=v0, ncv=ncv)

    Rx = rx_gate(b)
    zz = zz_gate(J)

    U_proj = np.zeros((nev, nev), dtype=np.complex64)
    
    psi_nexts = [apply_floquet(eigvecs[:, i], L, J, b, h_vec, Rx=Rx, zz=zz) for i in range(nev)]
    
    for i in range(nev):
        for j in range(i, nev):
            U_proj[i, j] = np.vdot(eigvecs[:, j], psi_nexts[i])
            if i != j:
                U_proj[j, i] = np.conj(U_proj[i, j])

    # final diagonalisation    
    eigenvalues, eigenvectors = np.linalg.eig(U_proj)
    
    lambdas = eigenvalues
    
    # Sort by phase (quasienergy) sorting for von numen entropy 
    quasienergies = np.angle(eigenvalues)
    sorted_indices = np.argsort(quasienergies)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    #dump eigenvectors
    np.savetxt("psi.csv", eigenvectors, delimiter=",")

    # dump phases
    phases = np.sort(np.mod(np.angle(lambdas), 2 * np.pi))
    np.savetxt("phases.csv", phases, delimiter=",")
    
    #calculate and dump spacings
    spacings = np.diff(phases)
    spacings = np.append(spacings, 2 * np.pi - phases[-1] + phases[0])

    clip_max = 4.0
    spacings = spacings[spacings < clip_max]
    spacings /= np.mean(spacings)
    np.savetxt("spacings.csv", spacings, delimiter=",")
    
# --- Run example ---
if __name__ == "__main__":
    run_level_spacing(L=8, J=np.pi/4, b=np.pi/4, phi_tgt=np.pi/2, disorder=True)


