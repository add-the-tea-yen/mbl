
import numpy as np
import matplotlib.pyplot as plt

def load_complex_psi(filename):
    with open(filename, 'r') as f:
        content = f.read()
    content = content.replace('(', '').replace(')', '').replace(',', ' ')
    tokens = content.strip().split()
    psi = np.array([complex(s) for s in tokens], dtype=np.complex64)
    return psi

def loschmidt_echo(phases, psi_eigvecs, psi0, times):
    overlaps = np.conj(psi_eigvecs.T) @ psi0
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

def return_probability(L_t):
    return np.mean(L_t[-50:])  # Long-time average

# --- Load data ---
phases = np.loadtxt("./haha.csv")
psi_raw = load_complex_psi("./Book1.csv")

# --- Determine dimensions ---
nev = len(phases)
dim = len(psi_raw) // nev
print(dim)
assert dim * nev == len(psi_raw), f"Size mismatch! Got {len(psi_raw)}, expected {dim}×{nev}."
psi_eigvecs = psi_raw.reshape((dim, nev))

# --- Initial state in full space ---
psi0 = np.zeros(dim, dtype=np.complex64)
psi0[0] = 1.0  # |↑↑↑...⟩

# --- Time evolution ---
times = np.linspace(0, 100, 500)

L_t, f_t, G_t = loschmidt_echo(phases, psi_eigvecs, psi0, times)

# --- Return Probability ---
ret_prob = return_probability(L_t)
print(f"Return probability: {ret_prob:.4f}")

# --- Plot ---
plt.figure(figsize=(8,6))
plt.plot(times, f_t, label="Dynamical Free Energy")
plt.xlabel("Time")
plt.ylabel("f(t)")
plt.title("Loschmidt Echo")
plt.grid(True)
plt.legend()
plt.show()
