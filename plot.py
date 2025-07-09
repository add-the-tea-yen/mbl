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


def read_complex_csv(filename):
    with open(filename, "r") as f:
        content = f.read()

    # Step 1: Remove parentheses
    content = content.replace("(", "").replace(")", "")

    # Step 2: Split entries by whitespace or commas
    tokens = content.replace(",", " ").split()

    # Step 3: Convert to complex
    data = np.array([complex(t) for t in tokens], dtype=np.complex128)

    # Step 4: Infer shape if needed — for example, reshape to (d, nev)
    # If unknown, just return flat array
    return data



# Load data
phases = np.loadtxt("./phases.csv", delimiter=",")
eigenvectors = read_complex_csv("./psi.csv")
#eigenvectors = np.loadtxt("./L14/psi.csv", delimiter=",", dtype=np.complex64)  

# --- Plot 1: Histogram of phases ---
plt.figure(figsize=(8, 4))
plt.hist(phases, bins=100, density=True, alpha=0.7, color='purple')
plt.xlabel("Phase (radians)")
plt.ylabel("Density")
plt.title("Eigenphase Distribution")
plt.grid(True)
plt.tight_layout()
plt.savefig("eigenphase_histogram.png")
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
plt.savefig("spacing_from_phases.png")
plt.show()

"""# --- Plot 3: Phase spacings ---
D = len(phase)
result = []
t = [0,(3*D)]
a = 0
for i in t:
    for y in phases:
        a += math.exp()
    g = (1/D)
    
"""



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
plt.savefig("unit_circle_convergence.png")
plt.show()















