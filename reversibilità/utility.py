import numpy as np

def debug_uniformizzazione(Q):
    """
    Debug completo della matrice P ottenuta da Q tramite uniformizzazione.
    Mostra lambda B, spettro, spectral gap e proprietà stocastiche.
    """
    B = (-np.diag(Q)).max()
    if B <= 0:
        raise ValueError("Uniformizzazione fallita: Q ha diagonale non negativa.")
    
    P = np.eye(Q.shape[0]) + Q / B

    print("\n🔎 [DEBUG UNIFORMIZZAZIONE]")
    print(f"B (massimo tasso di uscita): {B:.6f}")
    print(f"P.min(): {P.min():.6e}, P.max(): {P.max():.6e}")
    
    row_sums = P.sum(axis=1)
    print(f"Somma righe P → min: {row_sums.min():.6f}, max: {row_sums.max():.6f}")

    eigvals = np.linalg.eigvals(P)
    eigvals = np.real_if_close(eigvals)
    eigvals.sort()
    gap = eigvals[-1] - eigvals[-2]
    print(f"Spectral gap di P (uniformizzata): {gap:.6e}")
    
    print("\nAutovalori P (ultimi):")
    print(eigvals[-5:])
    
def spectral_gap(P):
    eigvals = np.linalg.eigvals(P)
    eigvals = np.real_if_close(eigvals)
    eigvals.sort()
    gap = eigvals[-1] - eigvals[-2]
    print(f"[DEBUG] Spectral gap di P: {gap:.6e}")
    print("Autovalori P (ultimi):")
    print(eigvals[-5:])
    return gap

def stima_k_necessario_precisa(P_or_Q, discrete=True, tol=1e-15):
    """
    Stima precisa del bound superiore di iterazioni k necessarie affinché
    ||pi^(k) - pi||_1 < tol, usando il bound spettrale completo con tutti gli autovalori.
    Supporta sia P (DTMC) che Q (CTMC via uniformizzazione).

    Args:
        P_or_Q (np.ndarray)                 :           matrice P (stocastica per riga) o Q (generatore)
        discrete (bool)                     :           True se P (DTMC), False se Q (CTMC)
        tol (float)                         :           tolleranza desiderata (default 1e-15)

    Returns:
        dict: con chiavi:
            - 'k_upper' (int)               :           stima superiore sul numero di iterazioni
            - 'lambda_star' (float)         :           |λ⋆| (modulo massimo degli autovalori ≠ λ1)
            - 'C_precisa' (float)           :           costante teorica effettiva del bound
            - 'diagonalizzabile' (bool)     :           True se P^T è diagonalizzabile
    """
    n = P_or_Q.shape[0]

    if not discrete:
        B = (-np.diag(P_or_Q)).max()
        if B <= 0:
            raise ValueError("Uniformizzazione fallita: Q ha diagonale non negativa.")
        P = np.eye(n) + P_or_Q / B
        if not np.allclose(P.sum(axis=1), 1, atol=1e-12):
            raise ValueError("La matrice uniformizzata non è stocastica.")
    else:
        P = P_or_Q

    A = P.T
    w, V = np.linalg.eig(A)
    rank = np.linalg.matrix_rank(V)
    idx = np.argsort(-np.abs(w))
    w = w[idx]
    V = V[:, idx]
    V = V / np.linalg.norm(V, axis=0)

    
    diagonalizzabile = (rank == n)
    if not diagonalizzabile:
        return {
            'k_upper': None,
            'lambda_star': None,
            'C_precisa': None,
            'diagonalizzabile': False
        }

    pi0 = np.ones(n) / n
    Vinv = np.linalg.inv(V)
    alpha = Vinv @ pi0
    lambda1 = w[0]
    lambda_others = w[1:]
    alpha_others = alpha[1:]

    def errore_spettrale_teorico(k):
        somma = np.sum(np.abs(alpha_others / alpha[0])**2 * np.abs(lambda_others / lambda1)**(2 * k))
        return np.sqrt(n) * np.sqrt(somma)

    for k in range(1, 100000):
        if errore_spettrale_teorico(k) < tol:
            return {
                'k_upper': k,
                'lambda_star': np.max(np.abs(lambda_others)),
                'diagonalizzabile': True
            }

    return {
        'k_upper': None,
        'lambda_star': np.max(np.abs(lambda_others)),
        'diagonalizzabile': True
    }

