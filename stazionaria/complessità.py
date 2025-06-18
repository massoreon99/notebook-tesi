import metodi_stazionaria, generazione_catene
import time
import copy

import numpy as np

from quantecon import gth_solve

METODI = {
    "gth"                   :                   lambda P, **kw: gth_solve(P),
    "system_numpy"          :                   lambda P, **kw: metodi_stazionaria.sistema_numpy(P, **kw),
    "system_scipy"          :                   lambda P, **kw: metodi_stazionaria.sistema_scipy(P, **kw),
    "eig_numpy"             :                   lambda P, **kw: metodi_stazionaria.solve_via_eig_numpy(P, **kw),
    "eig_scipy"             :                   lambda P, **kw: metodi_stazionaria.solve_via_eig_scipy(P, **kw),
    "power_numpy"           :                   lambda P, **kw: metodi_stazionaria.solve_via_power_numpy(P, **kw),
    "power_scipy"           :                   lambda P, **kw: metodi_stazionaria.solve_via_power_scipy(P, **kw),
}


def risoluzione():
    """Calcola la risoluzione del timer con time.monotonic."""
    start = time.monotonic()
    while time.monotonic() == start:
        pass
    return time.monotonic() - start

def tempo_per_n(P, metodo, resolution, **kwargs):
    if metodo not in METODI:
        raise ValueError(f"Metodo '{metodo}' non riconosciuto")
    tmin = resolution
    count = 0
    start = time.monotonic()
    while time.monotonic() - start < tmin:
        METODI[metodo](P.copy(), **kwargs)
        count += 1
    return (time.monotonic() - start) / count

def errore_su_P_o_Q(P_or_Q, metodo, discrete=True, **kwargs):
    """
    Calcola l'errore della distribuzione stazionaria:
    - ||πP - π||₁ per DTMC
    - ||πQ - 0||₁ per CTMC

    Args:
        P_or_Q (np.ndarray)             :           matrice P o Q
        metodo (str)                    :           nome del metodo
        discrete (bool)                 :           True per DTMC, False per CTMC

    Returns:
        float                           :           errore numerico
    """
    if metodo not in METODI:
        raise ValueError(f"Metodo '{metodo}' non riconosciuto")
    
    pi = METODI[metodo](P_or_Q.copy(), discrete=discrete, **kwargs)
    pi = np.ravel(pi)
    
    if discrete:
        return np.linalg.norm(pi @ P_or_Q - pi, 1)
    else:
        return np.linalg.norm(pi @ P_or_Q, 1)
    
def calcola_costo_e_errore(funzione, C=100, n_punti=20, n_matrici = 10, sparsity=0.98, k=2, tipo_gen_P_orQ=generazione_catene.genera_P_irriducibile, discrete=True):
    """

    Confronta tempi ed errori su matrici connesse, sia P (DTMC) che Q (CTMC).
    
    Se discrete=True                        →           genera e testa P
    Se discrete=False                       →           genera e testa Q 

    Args:
        funzione (str)                      :           nome del metodo da usare (es. 'solve_via_power_numpy')
        C (int)                             :           parametro crescita geometrica
        n_punti (int)                       :           numero di dimensioni da testare
        sparsity (float)                    :           percentuale di zeri (per genera_P_irriducibile o genera_Q_irriducibile)
        k (int)                             :           numero di archi per riga (per genera_P_k_random o genera_Q_k_random)
        tipo_gen_P_orQ (callable)           :           generatore di P o Q
        discrete (bool)                     :           True per DTMC (P), False per CTMC (Q)

    Returns:
        list of (n, tempo medio, errore)

    """
    
    A = 10
    B = C ** (1 / (n_punti - 1))
    risultati = []
    resolution = risoluzione() * 10000

    for i in range(n_punti):
        n = int(A * (B ** i))
        tempi = []
        errori = []

        for j in range(n_matrici):
            # Genera P o Q in base al tipo
            if tipo_gen_P_orQ.__name__.endswith("_irriducibile"):
                max_zeros = n * (n - 1)
                zeros = int(max_zeros * sparsity)
                P_or_Q = tipo_gen_P_orQ(n, zeros=zeros, seed = j)
            else:
                P_or_Q = tipo_gen_P_orQ(n, k, seed = j)

            tempi.append(tempo_per_n(P_or_Q, funzione, resolution, discrete=discrete))
            errori.append(errore_su_P_o_Q(P_or_Q, funzione, discrete=discrete))

        t_median = np.median(tempi)
        e_median = np.median(errori)

        risultati.append((n, t_median, e_median))

    return risultati