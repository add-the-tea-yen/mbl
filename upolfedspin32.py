import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
from numba import njit, prange

# Constants for spin-3/2
S = 1.5
spin_values = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)
d_spin = len(spin_values)  # 4

# --- Optimized Z + ZZ gate using bitwise logic ---
#@njit(parallel=True)
def apply_z_zz_spin32_diag(psi, h_vec, J, L):
    d = d_spin ** L
    out = np.empty_like(psi)
    for i in prange(d):
        state = i
        hz_sum = 0.0
        zz_sum = 0.0

        sz_prev = spin_values[state % d_spin]
        hz_sum += h_vec[0] * sz_prev
        state //= d_spin

        for site in range(1, L):
            sz = spin_values[state % d_spin]
            hz_sum += h_vec[site] * sz
            zz_sum += sz_prev * sz
            sz_prev = sz
            state //= d_spin

        phase = np.exp(-1j * (hz_sum + J * zz_sum))
        out[i] = phase * psi[i]
    return out

# --- Rx gate ---
def rx_gate_spin32(b):
    # Approximate Rx gate for spin-3/2 (nontrivial to construct exactly)
    # This is a placeholder identity matrix assuming trivial Rx
    return np.eye(d_spin, dtype=np.complex64)

# --- Apply Rx gate to each site ---
def apply_rx_all_sites(psi, Rx, L):
    psi = psi.reshape([d_spin] * L)
    for j in range(L):
        psi = np.moveaxis(psi, j, 0)
        shape = psi.shape
        psi = psi.reshape(d_spin, -1)
        psi = Rx @ psi
        psi = psi.reshape(shape)
        psi = np.moveaxis(psi, 0, j)
    return psi.reshape(-1)

# --- Floquet step ---
def apply_floquet(psi, L, J, b, h_vec=None, Rx=None):
    if h_vec is not None:
        psi = apply_z_zz_spin32_diag(psi, h_vec, J, L)
    if Rx is None:
        Rx = rx_gate_spin32(b)
    psi = apply_rx_all_sites(psi, Rx, L)
    return psi

# --- Geometric filter ---
def apply_geometric_filter(psi, L, J, b, k, phi_tgt, h_vec=None):
    Rx = rx_gate_spin32(b)
    result = np.zeros_like(psi, dtype=np.complex64)
    psi_k = psi.copy()
    for m in range(k + 1):
        result += np.exp(-1j * m * phi_tgt) * psi_k
        psi_k = apply_floquet(psi_k, L, J, b, h_vec, Rx)
    return result

# --- Matrix-free operator ---
class GeometricFilteredOperator(spla.LinearOperator):
    def __init__(self, L, J, b, k, phi_tgt, h_vec=None):
        self.L = L
        self.J = J
        self.b = b
        self.k = k
        self.phi_tgt = phi_tgt
        self.h_vec = h_vec
        self.d = d_spin ** L
        super().__init__(dtype=np.complex64, shape=(self.d, self.d))

    def _matvec(self, v):
        return apply_geometric_filter(v, self.L, self.J, self.b, self.k, self.phi_tgt, self.h_vec)

# --- POLFED Runner ---
def run_level_spacing(L=5, J=np.pi/4, b=np.pi/4, phi_tgt=np.pi/2, nev=300, disorder=True):
    d = d_spin ** L
    ncv = min(d, 2 * nev + 50)
    k = max(3, int(0.9 * (d * 2) / ncv))

    h_vec = np.random.uniform(-1.0, 1.0, size=L).astype(np.float32) if disorder else np.zeros(L, dtype=np.float32)

    print(f"[POLFED-Spin1.5] L={L}, J={J:.3f}, b={b:.3f}, k={k}, nev={nev}, disorder={disorder}")

    G = GeometricFilteredOperator(L, J, b, k, phi_tgt, h_vec)
    v0 = np.random.randn(d) + 1j * np.random.randn(d)
    v0 = v0.astype(np.complex64)
    v0 /= np.linalg.norm(v0)

    eigvals, eigvecs = spla.eigs(G, k=nev, which='LM', v0=v0, ncv=ncv)

    U_proj = np.zeros((nev, nev), dtype=np.complex64)
    psi_nexts = [apply_floquet(eigvecs[:, i], L, J, b, h_vec) for i in range(nev)]
    for i in range(nev):
        for j in range(i, nev):
            U_proj[i, j] = np.vdot(eigvecs[:, j], psi_nexts[i])
            if i != j:
                U_proj[j, i] = np.conj(U_proj[i, j])

    eigvals_final = np.linalg.eigvals(U_proj)
    phases = np.angle(eigvals_final)
    phases = np.sort(np.mod(phases, 2 * np.pi))
    np.savetxt("phases.csv", phases, delimiter=",")

    spacings = np.diff(phases)
    spacings = np.append(spacings, 2 * np.pi - phases[-1] + phases[0])
    spacings /= np.mean(spacings)
    np.savetxt("spacings.csv", spacings, delimiter=",")

    plt.hist(spacings, bins=80, density=True, alpha=0.7)
    plt.title("Level Spacing Distribution (Spin 3/2 Chain)")
    plt.xlabel("s")
    plt.ylabel("P(s)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("spin32_level_spacings.png")
    plt.show()

if __name__ == "__main__":
    run_level_spacing(L=5)
