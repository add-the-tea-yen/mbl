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

def plotPhases(phases):
    # Load data
    phases = np.loadtxt("./L14/phases.csv", delimiter=",")
    print(len(phases))
    eigenvectors = read_complex_csv("./L14/psi.csv")
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
def plotFockBasis(psi):
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
    evecs = load_complex_csv("./L14/psi.csv")
    
    
    
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



















