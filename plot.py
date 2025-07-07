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
import re


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
phases = np.loadtxt("./L14/phases.csv", delimiter=",")
eigenvectors = read_complex_csv("./L14/psi.csv")
#eigenvectors = np.loadtxt("./L14/psi.csv", delimiter=",", dtype=np.complex64)  # note: '.cvs' is likely a typo

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
spacings = spacings / np.mean(spacings)

plt.figure(figsize=(8, 4))
plt.hist(spacings, bins=70, density=True, alpha=0.7, color='gray', label="Numerical spacings")
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


# --- Plot 3: Heatmap of eigenvectors (magnitude) ---
plt.figure(figsize=(10, 6))
plt.imshow(np.abs(eigenvectors), aspect='auto', cmap='viridis', interpolation='none')
plt.colorbar(label="|ψ|")
plt.xlabel("Eigenvector index")
plt.ylabel("Fock basis state index")
plt.title("Magnitude of Eigenvectors (|ψ|)")
plt.tight_layout()
plt.savefig("eigenvector_magnitude_heatmap.png")
plt.show()


# Choose what to visualize: 'abs', 'real', or 'imag'
mode = 'real'  # or 'real' or 'imag'

if mode == 'abs':
    data = np.abs(eigenvectors)
elif mode == 'real':
    data = eigenvectors.real
elif mode == 'imag':
    data = eigenvectors.imag
else:
    raise ValueError("Invalid mode")

# Axes
num_basis, num_vecs = data.shape
X, Y = np.meshgrid(np.arange(num_vecs), np.arange(num_basis))

# 3D PlotAxes3D
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, data, cmap='plasma', edgecolor='none', linewidth=0, antialiased=True)

ax.set_xlabel("Eigenvector Index")
ax.set_ylabel("Fock Basis Index")
ax.set_zlabel(f"{mode}(ψ)")
ax.set_title(f"3D Plot of Eigenvectors ({mode})")

plt.tight_layout()
plt.savefig("eigenvectors_3d_surface.png")
plt.show()


# Number Vairance
"""
def compute_number_variance(unfolded, L_vals, num_windows=1000):
    #Compute number variance Σ²(L) for different window lengths L
    
    unfolded = np.sort(unfolded)
    min_val, max_val = unfolded[0], unfolded[-1]
    variance = []

    for L in L_vals:
        counts = []
        for _ in range(num_windows):
            x = np.random.uniform(min_val, max_val - L)
            count = np.sum((unfolded >= x) & (unfolded < x + L))
            counts.append(count)
        counts = np.array(counts)
        variance.append(np.var(counts))

    return np.array(variance)

def linear_unfold(phases):
    
    #Linearly unfold eigenphases assuming uniform density over 2π
    
    phases = np.sort(np.mod(phases, 2*np.pi))
    unfolded = (len(phases) / (2 * np.pi)) * phases
    return unfolded

# Load eigenphases from file
phases = np.loadtxt("./L14/phases.csv", delimiter=",")
unfolded = linear_unfold(phases)

# Define L values (window lengths)
L_vals = np.linspace(0.5, 10, 50)
Sigma2 = compute_number_variance(unfolded, L_vals, num_windows=1000)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(L_vals, Sigma2, 'o-', label='Numerical Σ²(L)')
plt.plot(L_vals, L_vals, 'k--', label='Poisson: Σ²(L) = L')
plt.plot(L_vals, (2/np.pi**2)*np.log(2*np.pi*L_vals) + 0.06, 'r--', label='GOE (approx.)')

plt.xlabel("Window length L (in mean level spacing units)")
plt.ylabel("Σ²(L)")
plt.title("Number Variance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("number_variance_plot.png")
plt.show()


"""