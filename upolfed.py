import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
from numba import njit, prange

# Pauli matrices
I2 = np.eye(2, dtype=np.complex64)
X = np.array([[0, 1], [1, 0]], dtype=np.complex64)

# --- Gate constructors ---
def rx_gate(angle):
    return np.cos(angle) * I2 - 1j * np.sin(angle) * X

# --- Apply single-qubit gate ---
def apply_single_qubit_gate(psi, gate, site, L):
    psi = psi.reshape([2] * L)
    psi = np.moveaxis(psi, site, 0)
    shape = psi.shape
    psi = psi.reshape(2, -1)
    psi = (gate @ psi).reshape(2, *shape[1:])
    psi = np.moveaxis(psi, 0, site)
    return psi.reshape(-1)

# --- Optimized combined Z + ZZ gate using bitwise logic ---
@njit(parallel=True)
def apply_combined_z_diagonal_numba(psi, h_vec, J, L):
    d = 1 << L
    out = np.empty_like(psi)
    for s in prange(d):
        hz_sum = 0.0
        zz_sum = 0.0
        z_prev = 1.0 if ((s >> 0) & 1) == 0 else -1.0
        hz_sum += h_vec[0] * z_prev
        for j in range(1, L):
            z_curr = 1.0 if ((s >> j) & 1) == 0 else -1.0
            hz_sum += h_vec[j] * z_curr
            zz_sum += z_prev * z_curr
            z_prev = z_curr
        theta = -1j * (hz_sum + J * zz_sum)
        out[s] = np.exp(theta) * psi[s]
    return out

# --- Floquet operator application (optimized) ---
def apply_floquet(psi, L, J, b, h_vec=None, Rx=None):
    if Rx is None:
        Rx = rx_gate(b)
    if h_vec is not None:
        psi = apply_combined_z_diagonal_numba(psi, h_vec, J, L)
    #if Rx is None:
        #Rx = rx_gate(b)
    for j in range(L):
        psi = apply_single_qubit_gate(psi, Rx, j, L)
    return psi

# --- Geometric filter using weighted complex exponential sum gk(U) ---
def apply_geometric_filter(psi, L, J, b, k, phi_tgt, h_vec=None):
    Rx = rx_gate(b)
    result = np.zeros_like(psi, dtype=np.complex64)
    psi_k = psi.copy()
    for m in range(k + 1):
        weight = np.exp(-1j * m * phi_tgt).astype(np.complex64)
        result += weight * psi_k
        psi_k = apply_floquet(psi_k, L, J, b, h_vec, Rx=Rx)
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
    nev = 225
    ncv = int(2 * nev)
    k = int(0.95 * (2**(L+1)) / ncv)
    #k=31
    h_vec = np.random.uniform(0.6, np.pi/4, size=L).astype(np.float32) if disorder else None
    print(f"[POLFED] L={L}, J={J:.3f}, b={b:.3f}, k={k}, nev={nev}, disorder={disorder}")
    G = GeometricFilteredOperator(L, J, b, k, phi_tgt, h_vec)
    v0 = (np.random.randn(d) + 1j * np.random.randn(d)).astype(np.complex64)
    v0 /= np.linalg.norm(v0)
    eigvals, eigvecs = spla.eigs(G, k=nev, which='LM', v0=v0, ncv=ncv)
    Rx = rx_gate(b)
    U_proj = np.zeros((nev, nev), dtype=np.complex64)
    psi_nexts = [apply_floquet(eigvecs[:, i], L, J, b, h_vec, Rx=Rx) for i in range(nev)]
    for i in range(nev):
        for j in range(i, nev):
            U_proj[i, j] = np.vdot(eigvecs[:, j], psi_nexts[i])
            if i != j:
                U_proj[j, i] = np.conj(U_proj[i, j])
    eigenvalues, eigenvectors = np.linalg.eig(U_proj)
    lambdas = eigenvalues
    quasienergies = np.angle(eigenvalues)
    sorted_indices = np.argsort(quasienergies)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    np.savetxt("psi.csv", eigenvectors, delimiter=",")
    phases = np.sort(np.mod(np.angle(lambdas), 2 * np.pi))
    np.savetxt("phases.csv", phases, delimiter=",")
    spacings = np.diff(phases)
    spacings = np.append(spacings, 2 * np.pi - phases[-1] + phases[0])
    clip_max = 4.0
    spacings = spacings[spacings < clip_max]
    spacings /= np.mean(spacings)
    np.savetxt("spacings.csv", spacings, delimiter=",")

# --- Run example ---
if __name__ == "__main__":
    run_level_spacing(L=12, J=np.pi/4, b=np.pi/4, phi_tgt=np.pi/2, disorder=True)
