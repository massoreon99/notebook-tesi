import numpy as np
import networkx as nx


def ensure_connected(G):
    if G.is_directed():
        if not nx.is_strongly_connected(G):
            G = G.subgraph(max(nx.strongly_connected_components(G), key=len)).copy()
    else:
        if not nx.is_connected(G):
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    mapping = {old: new for new, old in enumerate(sorted(G.nodes()))}
    return nx.relabel_nodes(G, mapping)

def compute_directed_eig(G, weight= None):
    
    A = nx.to_numpy_array(G, weight=weight)
    degree = A.sum(axis=1)                       # somma gradi in uscita per ogni nodo
    P = A / degree[:, None]                      # ogni riga viene normalizzata per il grado uscente di quel nodo
    
    # Calcolo del vettore di Perron (distribuzione stazionaria)
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmax(np.isclose(eigvals.real, 1))   # eigenvalue più vicino a 1
    phi = np.abs(eigvecs[:, idx].real)             # moduli -> positivo
    phi /= phi.sum()                               # normalizza

    Phi_sqrt = np.diag(np.sqrt(phi))
    Phi_inv_sqrt = np.diag(1 / np.sqrt(phi))

    L = np.identity(len(G)) - 0.5 * (Phi_sqrt @ P @ Phi_inv_sqrt + Phi_inv_sqrt @ P.T @ Phi_sqrt)
    vals, vecs = np.linalg.eigh(L)
    
    return vals, np.real(vecs), G

# ----------- Funzioni di calcolo per grafi non diretti -----------
# In un grafo non diretto, Laplaciani (normali/normalizzati) sono simmetrici => autovalori reali. 
# Walk e lazywalk no => possibili parti immaginarie Walk normalizzata si simmetrizza (->autovalori reali)
def _laplacian(A, D):
    L = D - A
    return np.linalg.eigh(L)  # np.linalg.eigh -> matrice simmetrica, autovalori reali

def _laplacian_normalized(A, D, D_inv_sqrt):
    L = D_inv_sqrt @ (D - A) @ D_inv_sqrt
    return np.linalg.eigh(L)

def _walk(A, D_inv):
    W = A @ D_inv
    vals, vecs = np.linalg.eig(W)
    return vals, np.real(vecs)  

def _lazywalk(A, D_inv, I):
    W = A @ D_inv
    M = 0.5 * (I + W)
    vals, vecs = np.linalg.eig(M)
    return vals, np.real(vecs)

def _walknorm(A, D_inv, D_inv_sqrt, I):
    W = A @ D_inv
    M = 0.5 * (I + W)
    N = D_inv_sqrt @ M @ D_inv_sqrt
    return np.linalg.eigh(N)

def compute_eig(G, mat_type="laplacian", weight=None):

    G = ensure_connected(G)
    
    # Caso grafo diretto
    if G.is_directed():
        if mat_type == "laplacian":
            return compute_directed_eig(G, weight=weight)
        else:
            raise ValueError(f"mat_type '{mat_type}' non supportato su grafo diretto")
    
    # Qui: G è non diretto
    A = nx.to_numpy_array(G, weight=weight)
    degrees = A.sum(axis=1)

    if np.any(degrees == 0):
        raise ValueError("Ci sono nodi di grado 0; il grafo non dovrebbe essere connesso?")

    # Matrici di grado
    D = np.diag(degrees)
    D_inv = np.diag(1 / degrees)
    D_inv_sqrt = np.diag(1 / np.sqrt(degrees))
    I = np.eye(len(G))
    
    # Switch dei calcoli
    matrix_type_switch = {
        "laplacian": lambda: _laplacian(A, D),
        "normalized": lambda: _laplacian_normalized(A, D, D_inv_sqrt),
        "walk": lambda: _walk(A, D_inv),
        "lazywalk": lambda: _lazywalk(A, D_inv, I),
        "walknorm": lambda: _walknorm(A, D_inv, D_inv_sqrt, I)
    }
    
    # Se il tipo non è presente, errore
    if mat_type not in matrix_type_switch:
        raise ValueError(f"Tipo di matrice '{mat_type}' non supportato")
    
    vals, vecs = matrix_type_switch[mat_type]()

    return vals, vecs, G