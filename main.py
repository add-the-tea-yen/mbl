import numpy as np
from genmulti import EDUM
from plot import *

#plotAreaVolume('./spins/half/L14/psi.csv')
L = [x for x in range(7,16)]
#for i in L:
    #plotPhases(f'./spins/half/L{i}_phases.csv')

#plotEchoTime('./spins/half/L14/phases.csv','./spins/half/L14/psi.csv',14)


import matplotlib.pyplot as plt

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


def loschmidt_echo(phases, psi_eigvecs, psi0, times):
    overlaps = np.conj(psi_eigvecs.T) @ psi0  # ⟨φ_n|ψ₀⟩
    weights = np.abs(overlaps)**2

    G_t = []
    for t in times:
        phase_factors = np.exp(-1j * phases * t)
        G = np.sum(weights * phase_factors)
        G_t.append(G)

    G_t = np.array(G_t)
    L_t = np.abs(G_t)**2
    f_t = -np.log(L_t) / len(psi0)

    return L_t, f_t, G_t
phases = np.loadtxt("./sorted_phases.csv")
psi_eigvecs = load_complex_psi("./sorted_eigenvalues.csv")  # Use your custom loader

L = int(np.log4(psi_eigvecs.shape[0]))  # for spin-1/2
# For spin-3/2 use log base 4

# Initial state (all spins up in computational basis)
dim = 2 ** L
nev = len(psi_eigvecs) // dim
psi_eigvecs = psi_eigvecs.reshape((dim, nev))

psi0 = np.zeros(dim, dtype=complex)
psi0[0] = 1.0

times = np.linspace(0, 100, 500)

L_t, f_t, G_t = loschmidt_echo(phases, psi_eigvecs, psi0, times)

# Plot
plt.figure(figsize=(10,5))
plt.plot(times, f_t, label="Dynamical Free Energy")
plt.xlabel("Time")
plt.ylabel("f(t)")
plt.title("Loschmidt Echo Dynamics")
plt.grid(True)
plt.legend()
plt.show()

