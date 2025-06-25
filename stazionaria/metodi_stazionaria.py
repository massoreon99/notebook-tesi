import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def my_gth_solve(P):

    """
    Calcola la distribuzione stazionaria π tramite algoritmo GTH.

    Args:
        P (np.ndarray)      :       matrice di transizione o generatore Metzler.

    Returns:
        np.ndarray          :       vettore stazionario π.
    """

    A = P.copy().astype(float)
    n = A.shape[0]
    pi = np.zeros(n)

    # Forward elimination senza pivot 
    for k in range(n - 1):
        scala = A[k, k+1:].sum()
        if scala <= 0:
            raise ValueError("Scala non positiva: catena non ergodica o input non valido.")
        for i in range(k + 1, n):
            A[i, k] /= scala
            for j in range(k + 1, n):
                A[i, j] += A[i, k] * A[k, j]

    # Backward substitution
    pi[-1] = 1.0
    for k in range(n - 2, -1, -1):
        for i in range(k + 1, n):
            pi[k] += A[i, k] * pi[i]

    # Normalizzazione finale
    pi /= pi.sum()
    return pi



def sistema_numpy(P_or_Q, discrete=True):
    """

    Calcola π risolvendo il sistema (I - P^T)x = 0 (DTMC) o Q^T x = 0 (CTMC).

    Args:
        P_or_Q (np.ndarray)     :           matrice di transizione P o generatore Q.
        discrete (bool)         :           True per P, False per Q.

    Returns:
        np.ndarray              :           vettore stazionario π.

    """

    n = P_or_Q.shape[0]
    M = np.eye(n) - P_or_Q.T if discrete else P_or_Q.T.copy()
    M[-1, :] = 1
    b = np.zeros(n)
    b[-1] = 1

    pi = np.linalg.solve(M, b)
    return pi

def sistema_scipy(P_or_Q, discrete=True):
    """
    
    Calcola π risolvendo il sistema (I - P^T)x = 0 (DTMC) o Q^T x = 0 (CTMC).
    Versione con SciPy su matrici sparse.

    Args:
        P_or_Q (np.ndarray)     :           matrice di transizione P o generatore Q.
        discrete (bool)         :           True per P, False per Q.

    Returns:
        np.ndarray              :           vettore stazionario π.

    """

    n = P_or_Q.shape[0]
    M = np.eye(n) - P_or_Q.T if discrete else P_or_Q.T.copy()
    M[-1, :] = 1
    b = np.zeros(n)
    b[-1] = 1

    M_sparse = sp.csc_matrix(M)
    x = spla.spsolve(M_sparse, b)
    return x



def solve_via_eig_numpy(P_or_Q, discrete=True):
    """
    Calcola π come autovettore sinistro dominante tramite NumPy.

    - Per DTMC (P)              :       cerca λ=1
    - Per CTMC (Q)              :       cerca λ=0

    Args:
        P_or_Q (np.ndarray)     :       matrice di transizione (P) o generatore (Q)
        discrete (bool)         :       True se P (DTMC), False se Q (CTMC)

    Returns:
        np.ndarray              :       distribuzione stazionaria π normalizzata
    """
     
    w, v = np.linalg.eig(P_or_Q.T)
    target = 1 if discrete else 0
    idx = np.argmin(np.abs(w - target))
    pi = np.real(v[:, idx])
    pi /= pi.sum()
    return pi

def solve_via_eig_scipy(P_or_Q, discrete=True):
    """
    Calcola π come autovettore sinistro dominante di P o Q tramite ARPACK.
    
    - Per DTMC (P)              :           usa which='LR' (autovalore reale più grande, 1)
    - Per CTMC (Q)              :           usa which='SM' (autovalore reale più piccolo in magnitudine, 0)
    
    Args:
        P_or_Q (np.ndarray)     :           matrice di transizione (P) o generatore (Q)
        discrete (bool)         :           True per DTMC, False per CTMC

    Returns:
        np.ndarray              :           distribuzione stazionaria π normalizzata
    """
    M = sp.csc_matrix(P_or_Q.T)
    which = 'LR' if discrete else 'SM'

    vals, vecs = spla.eigs(M, k=1, which=which)
    pi = np.real(vecs[:, 0])
    pi /= pi.sum()
    return pi



def solve_via_power_numpy(P_or_Q, discrete=True, tol=1e-15, max_iter=500000, test=False):
    """
    Calcola il vettore stazionario π usando il Power Method (dense).

    - Se `discrete=True` (DTMC)     :       π_{n+1} = π_n @ P
    - Se `discrete=False` (CTMC)    :       usa uniformizzazione
                                            → P = I + Q / B con B ≥ max q(i)

    Args:
        P_or_Q (np.ndarray)         :       matrice P (DCTM) o Q (CTMC)
        discrete (bool)             :       True per P, False per Q
        tol (float)                 :       tolleranza su ||π_{n+1} - π_n||₁    {Norma L1}
        max_iter (int)              :       numero massimo di iterazioni

    Returns:
        np.ndarray                  :       vettore stazionario π normalizzato
    """
    n = P_or_Q.shape[0]
    pi = np.ones(n) / n

    if discrete:
        P = P_or_Q
    else:
        B = (-np.diag(P_or_Q)).max()
        if B <= 0:
            raise ValueError("Uniformizzazione fallita: Q ha diagonale non negativa.")

        P = np.eye(n) + P_or_Q / B

    for i in range(max_iter):
        pi_new = pi @ P
        pi_new_sum = pi_new.sum()

        if pi_new_sum == 0 or not np.isfinite(pi_new_sum):
            raise RuntimeError("Power Method fallito: somma non valida in π_new.")

        pi_new /= pi_new_sum

        if np.linalg.norm(pi_new - pi, 1) < tol:
            if test == True:
                return i
            return pi_new

        pi = pi_new

    raise RuntimeError("Power Method non converge entro max_iter.")

def solve_via_power_scipy(P_or_Q, discrete=True, tol=1.1e-15, max_iter=500000):
    """
    Calcola π col Power Method (sparse).

    - Se discrete=True (DTMC)       :           π_{n+1} = π_n @ P
    - Se discrete=False (CTMC)      :           usa uniformizzazione
                                                P = I + Q / B  con B ≥ max q(i)

    Args:
        P_or_Q (array o csr_matrix) :           matrice P (DTMC) o Q (CTMC)
        discrete (bool)             :           True per P, False per Q
        tol (float)                 :           tolleranza ||π_{n+1} - π_n||₁  {Norma L1}
        max_iter (int)              :           iterazioni massime

    Returns:
        np.ndarray                  :           vettore stazionario π
    """
    n = P_or_Q.shape[0]
    pi = np.ones(n) / n
    
    if not sp.issparse(P_or_Q):
        P_or_Q = sp.csr_matrix(P_or_Q)

    if not discrete:
        B = (-P_or_Q.diagonal()).max()
        if B <= 0:
            raise ValueError("Uniformizzazione fallita: Q ha diagonale non negativa.")
        
        I = sp.identity(n, format="csr")
        P = I + P_or_Q / B

    else:
        P = P_or_Q

    for i in range(max_iter):
        pi_new = pi @ P
        pi_new = np.asarray(pi_new).ravel()
        pi_new_sum = pi_new.sum()

        if pi_new_sum <= 0 or not np.isfinite(pi_new_sum):
            raise RuntimeError("Power Method (sparse) fallito: somma non valida.")
        pi_new /= pi_new_sum

        if np.linalg.norm(pi_new - pi, 1) < tol:
            return pi_new
        
        pi = pi_new

    raise RuntimeError("Power Method (sparse) non converge entro max_iter.")

