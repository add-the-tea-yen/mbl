import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
from numba import njit

# --- Spin-1/2 and Spin-3/2 definitions ---
def get_spin_basis(spin):
    if spin == 0.5:
        return np.array([0.5, -0.5], dtype=np.float32)
    elif spin == 1.5:
        return np.array([1.5, 0.5, -0.5, -1.5], dtype=np.float32)
    else:
        raise ValueError("Only spin-1/2 and spin-3/2 supported")

# --- Random Rx gate for each spin ---
def random_rx_gate(angle, dim):
    Sx = np.zeros((dim, dim), dtype=np.complex64)
    for m in range(dim):
        for n in range(dim):
            if abs(m - n) == 1:
                Sx[m, n] = 0.5 * np.sqrt((dim - max(m,n)) * (max(m,n)))
    return np.cos(angle) * np.eye(dim, dtype=np.complex64) - 1j * np.sin(angle) * Sx

# --- Diagonal combined Z + ZZ operation ---
@njit
def apply_diag_mixed(psi, h_vec, J, dims, sz_list):
    d = psi.shape[0]
    out = np.empty_like(psi)
    for s in range(d):
        x = s
        phase = 0.0
        prev_sz = sz_list[x % dims[0]]
        phase += h_vec[0] * prev_sz
        x //= dims[0]
        for j in range(1, len(dims)):
            sz = sz_list[x % dims[j]]
            phase += h_vec[j] * sz
            phase += J * prev_sz * sz
            prev_sz = sz
            x //= dims[j]
        out[s] = psi[s] * np.exp(-1j * phase)
    return out

# --- Apply Rx gates to all spins ---
def apply_rx_all_sites(psi, rx_gates, dims):
    L = len(dims)
    psi = psi.reshape(dims)
    for j in range(L):
        psi = np.moveaxis(psi, j, 0)
        shape = psi.shape
        psi = psi.reshape(dims[j], -1)
        psi = (rx_gates[j] @ psi).reshape((dims[j],) + shape[1:])
        psi = np.moveaxis(psi, 0, j)
    return psi.reshape(-1)

# --- Floquet operator ---
def apply_floquet(psi, L, J, b, h_vec, dims, sz_list, rx_gates):
    psi = apply_diag_mixed(psi, h_vec, J, dims, sz_list)
    psi = apply_rx_all_sites(psi, rx_gates, dims)
    return psi

# --- Geometric filter ---
def apply_geometric_filter(psi, L, J, b, k, phi_tgt, h_vec, dims, sz_list, rx_gates):
    result = np.zeros_like(psi, dtype=np.complex64)
    psi_k = psi.copy()
    for m in range(k + 1):
        weight = np.exp(-1j * m * phi_tgt).astype(np.complex64)
        result += weight * psi_k
        psi_k = apply_floquet(psi_k, L, J, b, h_vec, dims, sz_list, rx_gates)
    return result

# --- Matrix-free operator ---
class GeometricFilteredOperator(spla.LinearOperator):
    def __init__(self, L, J, b, k, phi_tgt, h_vec, dims, sz_list, rx_gates):
        self.L = L
        self.J = J
        self.b = b
        self.k = k
        self.phi_tgt = phi_tgt
        self.h_vec = h_vec
        self.dims = dims
        self.sz_list = sz_list
        self.rx_gates = rx_gates
        self.d = np.prod(dims)
        super().__init__(dtype=np.complex64, shape=(self.d, self.d))
    def _matvec(self, v):
        return apply_geometric_filter(v, self.L, self.J, self.b, self.k, self.phi_tgt, self.h_vec, self.dims, self.sz_list, self.rx_gates)

# --- Main run function ---
def run_level_spacing(L=8, J=np.pi/4, b=0.9, phi_tgt=0.0, disorder=True, fphases="phases.csv", fpsi="psi.csv"):
    spins = [0.5, 1.5] * (L // 2)   # Alternate spin-1/2 and spin-3/2
    dims = np.array([2 if s==0.5 else 4 for s in spins])
    sz_list = np.concatenate([get_spin_basis(s) for s in spins])
    d = np.prod(dims)
    print(d)
    #nev = min(225, d-2)
    #ncv = min(2 * nev, d)
    nev = 1000
    ncv = 2*nev
    k = int(0.95 * (2 ** (L + 1)) / ncv)
    k = 30 
    h_vec = np.random.uniform(0.6, np.pi/4, size=L).astype(np.float32) if disorder else np.zeros(L, dtype=np.float32)
    rx_gates = [random_rx_gate(b, dim) for dim in dims]

    print(f"[Mixed POLFED 1/2+3/2] L={L}, J={J:.3f}, b={b:.3f}, k={k}, nev={nev}, disorder={disorder}")

    G = GeometricFilteredOperator(L, J, b, k, phi_tgt, h_vec, dims, sz_list, rx_gates)

    v0 = (np.random.randn(d) + 1j * np.random.randn(d)).astype(np.complex64)
    v0 /= np.linalg.norm(v0)

    eigvals, eigvecs = spla.eigs(G, k=nev, which='LM', v0=v0, ncv=ncv)

    # Project back to U
    U_proj = np.zeros((nev, nev), dtype=np.complex64)
    psi_nexts = [apply_floquet(eigvecs[:, i], L, J, b, h_vec, dims, sz_list, rx_gates) for i in range(nev)]
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

    np.savetxt(fpsi, eigenvectors, delimiter=",")
    phases = np.sort(np.mod(np.angle(lambdas), 2 * np.pi))
    np.savetxt(fphases, phases, delimiter=",")

# --- Run ---
if __name__ == "__main__":
    run_level_spacing(L=10, J=np.pi/4, b=np.pi/4, phi_tgt=np.pi/2, disorder=True, fphases="phases.csv", fpsi="psi.csv")
