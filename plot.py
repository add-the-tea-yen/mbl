#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 23:15:42 2025

@author: adityan
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
import math

def load_complex_psi(filename):
    with open(filename, 'r') as f:
        content = f.read()
    # Remove parentheses, commas, newlines
    content = content.replace('(', '').replace(')', '').replace(',', ' ')
    # Split by whitespace
    tokens = content.strip().split()
    # Parse each token as complex
    psi = np.array([complex(s) for s in tokens], dtype=np.complex64)
    return psi

def plotPhases(fphases):
    # Load data
    phases = np.loadtxt(fphases, delimiter=",")
    print(len(phases))
    #eigenvectors = np.loadtxt("./L14/psi.csv", delimiter=",", dtype=np.complex64)  
    
    # --- Plot 1: Histogram of phases ---
    plt.figure(figsize=(8, 4))
    plt.hist(phases, bins=100, density=True, alpha=0.7, color='purple')
    plt.xlabel("Phase (radians)")
    plt.ylabel("Density")
    plt.title("Eigenphase Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./figs/eigenphase_histogram.png")
    plt.show()
    
    # --- Plot 2: Phase spacings ---
    sorted_phases = np.sort(phases)
    spacings = np.diff(sorted_phases)
    spacings = np.append(spacings, 2 * np.pi - sorted_phases[-1] + sorted_phases[0])
    clip_max = 4.0
    spacings = spacings[spacings < clip_max]
    spacings = spacings / np.mean(spacings)
    
    plt.figure(figsize=(8, 4))
    plt.hist(spacings, bins=50, density=True, alpha=0.7, color='gray', label="Numerical spacings")
    s = np.linspace(0, 4, 200)
    plt.plot(s, np.exp(-s), 'k--', label="Poisson")
    coe_pdf = (np.pi / 2) * s * np.exp(- (np.pi / 4) * s**2)  # Wigner-Dyson (COE)
    plt.plot(s, coe_pdf, 'r--', label="COE")
    plt.xlabel("Normalized spacing s")
    plt.ylabel("P(s)")
    plt.title("Level Spacing Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./figs/spacing_from_phases.png")
    plt.show()
    
    angles = phases  # already in [0, 2π)
    
    # === Convert to complex unit circle ===
    eigenvalues = np.exp(1j * angles)
    
    # === Plot on unit circle ===
    fig, ax = plt.subplots(figsize=(6, 6))
    circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--', alpha=0.4)
    
    ax.add_artist(circle)
    ax.plot(eigenvalues.real, eigenvalues.imag, 'o', markersize=5, alpha=0.8, label='Eigenvalues')
    
    # Optionally show target phi
    # phi_tgt = np.pi / 2
    # ax.plot(np.cos(phi_tgt), np.sin(phi_tgt), 'rx', label=r'$\phi_{\rm tgt}$')
    
    # Formatting
    ax.set_title("Eigenvalue Convergence on the Complex Unit Circle")
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    ax.set_aspect('equal')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig("./figs/unit_circle_convergence.png")
    plt.show()




# Fock Baiss 
def plotFockBasis(fpsi):
    def load_complex_csv(filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()
    
        data = []
        for line in lines:
            entries = line.strip().split()
            # remove parentheses and convert to complex
            row = [complex(entry.strip("()")) for entry in entries]
            data.append(row)
    
        return np.array(data, dtype=np.complex64)
    
    # Usage:
    evecs = load_complex_csv(fpsi)
    
    
    
    # If it's complex stored as real+imag parts (shape d x 2*nev), fix this
    if evecs.shape[1] % 2 == 0:
        nev = evecs.shape[1] // 2
        evecs = evecs[:, :nev] + 1j * evecs[:, nev:]
    
    # Compute probability weight in Fock basis
    weights = np.abs(evecs) ** 2  # shape: (d, nev)
    
    # Normalize each column (should already be, but just to be sure)
    weights /= np.sum(weights, axis=0, keepdims=True)
    
    # Sort eigenstates by quasienergy (optional)
    # phis = np.loadtxt("phases.csv", delimiter=",")
    # sorted_idx = np.argsort(phis)
    # weights = weights[:, sorted_idx]
    
    # Plot heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(weights.T, aspect='auto', cmap='inferno', origin='lower')
    plt.xlabel("Fock Basis State Index")
    plt.ylabel("Eigenvector Index")
    plt.title("Fock Basis Weight Distribution")
    plt.colorbar(label="Probability Weight")
    plt.tight_layout()
    plt.show()

def plotAreaVolume(fpsi):
    # Usage:
    evecs = load_complex_psi(fpsi)
    #print(evecs)
    iprs = []
    L = [x for x in range(7,16)]
    for i in evecs:
        print(i)
        #psi = fspi
        ipr = np.sum(np.abs(i)**4)
        iprs.append(ipr)
    plt.plot(L, np.log(iprs),'o-')
    plt.xlabel("System Size L")
    plt.ylabel("log of IPR")



def read_complex_psi(filename, L):
    """Read complex eigenvectors from psi.csv with (a+bj) format."""
    with open(filename, "r") as f:
        content = f.read()

    # Parse complex numbers
    raw = content.replace("(", "").replace(")", "").replace("\n", " ").split()
    data = np.array([complex(s) for s in raw], dtype=np.complex64)

    d = 2 ** L
    nev = len(data) // d
    assert len(data) == d * nev, "Mismatch in psi size"

    psi = data.reshape((d, nev))
    return psi

def plotEchoTime(phases_file, psi_file, L, N_plot=1000):
    # Load data
    phases = np.loadtxt(phases_file, delimiter=",")
    psi = read_complex_psi(psi_file, L)

    nev = psi.shape[1]
    phases_truncated = phases[:nev]  # Ensure matching dimensions

    print(f"Loaded psi shape: {psi.shape}, Phases: {len(phases_truncated)}")

    # Define initial state |psi0⟩ = all spins down
    psi0 = np.zeros(2 ** L, dtype=np.complex64)
    psi0[0] = 1.0

    # Overlaps ⟨ψ_n|ψ0⟩
    proj = np.conj(psi.T) @ psi0

    # Time range (scaled by system size N=L)
    tau_vals = np.linspace(0, 10, N_plot)
    t_vals = L * tau_vals

    L_t = []
    f_t = []

    for t in t_vals:
        amp = np.sum(proj * np.exp(-1j * phases_truncated * t))
        L_t_val = np.abs(amp)**2
        L_t.append(L_t_val)
        f_t.append(-np.log(L_t_val) / (2 * L**2))

    L_t = np.array(L_t)
    f_t = np.array(f_t)

    # --- Plot Loschmidt Echo ---
    plt.figure(figsize=(10, 5))
    plt.plot(tau_vals, L_t, label="Loschmidt Echo |L(t)|²", color='blue')
    plt.xlabel(r"$\tau = t / L$")
    plt.ylabel(r"$|L(t)|^2$")
    plt.title("Dynamical Quantum Phase Transition: Loschmidt Echo")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot Dynamical Free Energy ---
    plt.figure(figsize=(10, 5))
    plt.plot(tau_vals, f_t, label=r"Dynamical Free Energy $f_N(\tau)$", color='red')
    plt.xlabel(r"$\tau = t / L$")
    plt.ylabel(r"$f_N(\tau)$")
    plt.title("Dynamical Free Energy Density")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
















