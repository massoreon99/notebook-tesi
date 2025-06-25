"""func.py
--------------------------------
Utility set per confrontare rapidamente lo spettro di due grafi.:

1. **A** matrice di adiacenza.
2. **bar A** adiacenza del complemento, \(\bar A = J-I-A\).
3. **L** Laplaciana classica, \(L = D - A\).
4. **|L|** Laplaciana senza segno, \(|L| = D + A\).
5. **B** adiacenza del *line-graph* `L(G)`.

"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt



def to_adj(M):

    if isinstance(M, (nx.Graph, nx.DiGraph)):
        return nx.to_numpy_array(M, dtype=float)
    return np.asarray(M, dtype=float)


def sorted_eigvals(A):

    return np.sort(np.linalg.eigvals(to_adj(A)).real)



def laplacian(A):
    """Laplacian  L = D - A."""
    A = to_adj(A)
    D = np.diag(A.sum(axis=1))
    return D - A


def signless_laplacian(A):
    """Signless Laplacian  |L| = D + A."""
    A = to_adj(A)
    D = np.diag(A.sum(axis=1))
    return D + A


def normalized_laplacian(A):
    """Normalized Laplacian  L = I - D^{-1/2} A D^{-1/2}."""
    A = to_adj(A)
    d = A.sum(axis=1)
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.where(d > 0, 1 / np.sqrt(d), 0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return np.eye(len(A)) - D_inv_sqrt @ A @ D_inv_sqrt


def complement_adjacency(A):
    """Adjacency of the complement  bar A = J - I - A."""
    A = to_adj(A)
    n = A.shape[0]
    return np.ones_like(A) - np.eye(n) - A


# ------------------------------------------------------------------
# Line-graph --------------------------------------------------------


def linegraph_adjacency(A):
    """Adjacency **B** of the line-graph *L(G)*.

    Algebraic formula:  *B = NᵀN - 2I*,
    where *N* is the (unoriented) incidence matrix of *G*..
    """
    A = to_adj(A)
    G = nx.from_numpy_array(A)
    N = nx.incidence_matrix(G, oriented=False).toarray()
    m = N.shape[1]
    return N.T @ N - 2 * np.eye(m)


# ------------------------------------------------------------------
# Visual ------------------------------------------------------------


def show_graph_pair(G1, G2, titles=("G₁", "G₂"), layout="circular"):
    """Simple side-by-side drawing of two graphs."""
    pos_fun = nx.circular_layout if layout == "circular" else nx.spring_layout
    pos1, pos2 = pos_fun(G1), pos_fun(G2)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    nx.draw(G1, pos1, with_labels=True, node_color="skyblue")
    plt.title(titles[0])
    plt.subplot(1, 2, 2)
    nx.draw(G2, pos2, with_labels=True, node_color="lightgreen")
    plt.title(titles[1])
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# Report ------------------------------------------------------------


def equiv(s1, s2):

    return "cospettrali" if np.allclose(s1, s2) else "diversi"


def compare_pair(A1, A2, name="Confronto"):

    A1m, A2m = to_adj(A1), to_adj(A2)

    specs = {
        "Adjacency A"   : (sorted_eigvals(A1m), sorted_eigvals(A2m)),
        "Complement Â̄"  : (sorted_eigvals(complement_adjacency(A1m)),
                           sorted_eigvals(complement_adjacency(A2m))),
        "Laplacian L"   : (sorted_eigvals(laplacian(A1m)),
                           sorted_eigvals(laplacian(A2m))),
        "Signless |L|"  : (sorted_eigvals(signless_laplacian(A1m)),
                           sorted_eigvals(signless_laplacian(A2m))),
        "Normalized 𝓛"  : (sorted_eigvals(normalized_laplacian(A1m)),
                           sorted_eigvals(normalized_laplacian(A2m))),
        "Line-graph B"  : (sorted_eigvals(linegraph_adjacency(A1m)),
                           sorted_eigvals(linegraph_adjacency(A2m))),
    }

    print(f"\n{name}")
    for mat_name, (s1, s2) in specs.items():
        print(f"{mat_name:15}: {equiv(s1, s2)}")
