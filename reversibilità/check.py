import numpy as np
from quantecon import gth_solve
import metodi_stazionaria as ms
import generazione_catene as gc
from pydtmc import MarkovChain
import time


def risoluzione():
    """Calcola la risoluzione del timer con time.monotonic."""
    start = time.monotonic()
    while time.monotonic() == start:
        pass
    return time.monotonic() - start


def tempo_per_funzione(funzione, resolution):
    tmin = resolution
    count = 0
    start = time.monotonic()
    while time.monotonic() - start < tmin:
        funzione()
        count += 1
    return (time.monotonic() - start) / count

def calcola_tempi(funzione_check, C=100, n_punti=20, n_matrici=10, seed=123, discrete=True, forzata=False):
    """
    Misura il tempo medio del controllo di reversibilità su n_matrici diverse per ogni dimensione.

    Args:
        funzione_check (str)                    :           'nostro' o 'pydtmc'
        C (int)                                 :           valore massimo della dimensione n.
        n_punti (int)                           :           numero di punti da campionare.
        n_matrici (int)                         :           quante matrici generare per ogni n.
        seed (int)                              :           seed di partenza.
        discrete (bool)                         :           True per P (DTMC), False per Q (CTMC).
        forzata (bool)                          :           se True, genera Q forzate (solo se discrete=False)

    Returns:
        list of (n, tempo_medio)
    """

    A = 15
    B = C ** (1 / (n_punti - 1))
    risultati = []
    resolution = risoluzione() * 10000

    for i in range(n_punti):
        n = int(A * (B ** i))
        tempi = []

        for j in range(n_matrici):
            # 1. Genera la matrice
            if discrete:
                P_or_Q = gc.genera_P_reversibile(n, seed + j)
            else:
                if forzata:
                    P_or_Q = gc.genera_Q_reversibile_forzata(n, seed + j)
                else:
                    P_or_Q = gc.genera_Q_reversibile(n, seed + j)

            # 2. Seleziona la funzione da misurare
            if funzione_check == "nostro":
                funzione = lambda: check_reversibility(P_or_Q, discrete=discrete)
            elif funzione_check == "pydtmc":
                funzione = lambda: MarkovChain(P_or_Q).is_reversible
            else:
                raise ValueError(f"Metodo '{funzione_check}' non riconosciuto.")

            # 3. Misura il tempo medio per questa matrice
            t = tempo_per_funzione(funzione, resolution)
            tempi.append(t)
        
        t_median = np.median(tempi)

        # 4. Tempo medio (mediana) su tutte le matrici generate per n
        risultati.append((n, t_median))

    return risultati



def check_reversibility(P_or_Q, discrete=True, tol=1e-15):
    """
    Verifica se una matrice P (DTMC) o Q (CTMC) è reversibile,
    usando il metodo più efficiente in base alla dimensione, alla sparsità
    e al condizionamento numerico.

    Args:
        P_or_Q (np.ndarray)     :           matrice di transizione P o generatore Q.
        discrete (bool)         :           True se P (DTMC), False se Q (CTMC).
        tol (float)             :           tolleranza numerica.

    Returns:
        bool                    :           True se reversibile, False altrimenti.
    """
    n = P_or_Q.shape[0]
    sparsity = np.count_nonzero(P_or_Q) / (n * n)

    if n <= 100:
        pi = gth_solve(P_or_Q)

    else:
        if discrete:
            # Check condizionamento P: righe quasi nulle
            row_mins = P_or_Q.min(axis=1)
            if np.any(row_mins < 1e-12):
                pi = ms.solve_via_eig_scipy(P_or_Q, discrete=True)
            else:
                pi = ms.solve_via_power_scipy(P_or_Q, discrete=True) if sparsity < 0.1 else ms.solve_via_power_numpy(P_or_Q, discrete=True)

        else:
            # Check condizionamento Q: range tassi di uscita
            q_diag = -np.diag(P_or_Q)
            B = q_diag.max()
            min_Q = P_or_Q[P_or_Q > 0].min()
            q_range = B / min_Q
            if q_range > 3*n:
                pi = ms.solve_via_eig_scipy(P_or_Q, discrete=False)
            else:
                pi = ms.solve_via_power_scipy(P_or_Q, discrete=False) if sparsity < 0.1 else ms.solve_via_power_numpy(P_or_Q, discrete=False)

    # Costruzione matrice dei flussi
    F = np.outer(pi, np.ones_like(pi)) * P_or_Q
    return np.allclose(F, F.T, atol=tol)


def test_confronto_reversibilita(n_matrici=100, n=100, seed=123):
    match = 0
    mismatch = 0

    for i in range(n_matrici):
        # Genera una matrice reversibile non banale
        P = gc.genera_P_reversibile(n, seed=seed + i)

        try:
            # Nostro check
            nostro = check_reversibility(P.copy())

            # PyDTMC check
            mc = MarkovChain(P.copy())
            pydtmc = mc.is_reversible

            if nostro == pydtmc:
                match += 1
            else:
                mismatch += 1
                print(f"Mismatch alla matrice {i}: nostro={nostro}, pydtmc={pydtmc}")

        except Exception as e:
            mismatch += 1
            print(f"Errore alla matrice {i}: {e}")

    print(f"\nRisultati su {n_matrici} matrici:")
    print(f"Match: {match}")
    print(f"Mismatch o errori: {mismatch}")