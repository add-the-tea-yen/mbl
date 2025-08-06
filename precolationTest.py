
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import expm
from itertools import product

# System size
L = 6
dim = 2**L

# Pauli matrices
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
id2 = np.eye(2, dtype=complex)

# Helper: build tensor product operator at site i
def embed(op, i, L):
    """Embed a single-site operator 'op' at position 'i' in an L-site chain."""
    ops = [id2] * L
    ops[i] = op
    result = ops[0]
    for j in range(1, L):
        result = np.kron(result, ops[j])
    return result

# Build Hamiltonians
def build_Hz(J=1.0):
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(L):
        Zi = embed(sz, i, L)
        Zj = embed(sz, (i+1)%L, L)  # PBC
        H += -J * Zi @ Zj
    return H

def build_Hx(h=1.0):
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(L):
        H += -h * embed(sx, i, L)
    return H

# Build Floquet operator
Hz = build_Hz(J=1.0)
Hx = build_Hx(h=0.7)
Uz = expm(-1j * Hz)
Ux = expm(-1j * Hx)
U = Uz @ Ux

# Basis: binary strings of length L
basis_states = [''.join(seq) for seq in product('01', repeat=L)]

# Build connectivity graph
G = nx.Graph()
G.add_nodes_from(basis_states)

# Label each basis state with its index in computational basis
state_index = {state: i for i, state in enumerate(basis_states)}

# Add edges if <i|U|j> is non-zero (above some small threshold)
threshold = 1e-6
for i, s1 in enumerate(basis_states):
    for j, s2 in enumerate(basis_states):
        if i < j and np.abs(U[i, j]) > threshold:
            G.add_edge(s1, s2)

# Get connected components
components = list(nx.connected_components(G))
colors = {node: i for i, comp in enumerate(components) for node in comp}
color_list = [colors[node] for node in G.nodes]

# Draw the graph
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color=color_list, cmap=plt.cm.tab20)
plt.title(f"Connectivity Graph for KFIM (L={L}) with {len(components)} Fragment(s)")
plt.show()

# Spectral layout + color by component
components = list(nx.connected_components(G))
colors = {node: i for i, comp in enumerate(components) for node in comp}
color_list = [colors[node] for node in G.nodes]

pos = nx.spectral_layout(G)
plt.figure(figsize=(12, 8))
components = list(nx.connected_components(G))
colors = {n: i for i, comp in enumerate(components) for n in comp}
color_list = [colors[n] for n in G.nodes()]
nx.draw(G, pos, with_labels=True, node_color=color_list, cmap=plt.cm.tab20)
plt.title(f"Connectivity Graph for KFIM (L={L}) with {len(components)} Connected Components")
plt.show()

sizes = [len(c) for c in components]
plt.hist(sizes, bins=range(1, max(sizes)+1))
plt.xlabel("Cluster size")
plt.ylabel("Count")
plt.title("Hilbert space cluster size distribution")
plt.show()

print(nx.number_connected_components(G))
