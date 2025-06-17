import numpy as np
import networkx as nx

from quantecon import gth_solve
from pydtmc import MarkovChain




# =============================================================================
#  P  – matrici di transizione (DTMC)
# =============================================================================
def genera_P_irriducibile(n, zeros=0, seed=123, tentativi=50000):
    """Genera una matrice di transizione **P** *(n×n)* casuale, riproducibile e
    **irriducibile**.

    **Nota sulla generazione**
    -------------------------
    Questa funzione è adatta per costruire matrici di transizione P casuali
    con un grado controllabile di sparsità, garantendo l'irriducibilità tramite
    controllo di forte connessione sul grafo diretto `G(P)`.

    Tuttavia, quando il numero di zeri richiesto si avvicina al massimo
    teorico : `n(n-1)`, trovare una matrice fortemente connessa diventa
    sempre più difficile e i tempi di generazione crescono drasticamente.
    Inoltre, anche se P risulta irriducibile, un numero troppo basso di archi
    può portare a periodicità, rendendo il Power Method inefficace.

    In questi casi estremi è preferibile usare una generazione alternativa con
    struttura più controllata (es. `genera_P_k_random`), che garantisce
    a priori la forte connessione e, con un minimo numero di archi extra,
    anche l'aperiodicità.

    Parameters
    ----------
    n : int
        Dimensione della matrice.
    zeros : int, default 0
        Zeri totali (inclusa la diagonale).
    seed : int, default 123
        Seed iniziale per il RNG.
    tentativi : int, default 50000
        Tentativi massimi.

    Returns
    -------
    numpy.ndarray
        Matrice P irriducibile.

    Raises
    ------
    RuntimeError
        Se non viene trovata una matrice connessa entro i tentativi.
    """
    
    for k in range(tentativi):

        P = MarkovChain.random(n, zeros=zeros, seed=seed + k).p

        X = nx.from_numpy_array(P, create_using=nx.DiGraph)

        if nx.is_strongly_connected(X):
            return P
        
    raise RuntimeError(f"Non ho P connessa per n={n}, zeros={zeros}")

def genera_P_k_random(n, k, seed=123):
    """Genera una matrice P irriducibile con k archi in uscita per riga.

    **Nota sulla generazione**
    -------------------------
    Questo approccio garantisce la forte connessione della matrice P per
    costruzione, grazie all'arco obbligato : `(i -> (i+1) \mod n)`
    che forma un ciclo completo tra tutti i nodi.

    Aggiungendo poi `k-1` archi casuali per riga si rompe facilmente anche
    la periodicità del ciclo, rendendo la matrice adatta all'uso del Power
    Method.

    Il metodo è particolarmente utile quando si desidera generare matrici molto
    sparse ma con garanzie strutturali.  A differenza di
    `genera_P_irriducibile`, qui non è necessario alcun controllo
    esplicito sulla connettività e i tempi di generazione restano contenuti
    anche con pochissimi archi (es. ``k = 2``).

    Parameters
    ----------
    n : int
        Dimensione.
    k : int
        Numero totale di archi per riga (``1 ≤ k ≤ n``).
    seed : int, default 123
        Seed per il RNG.

    Returns
    -------
    numpy.ndarray
        Matrice P irriducibile e row-stocastica.
    """
    
    if not (1 <= k <= n):
        raise ValueError("Serve 1 ≤ k ≤ n")
    
    rng = np.random.default_rng(seed)
    P = np.zeros((n, n))
    
    
    for i in range(n):
        # arco obbligato
        targets = {(i + 1) % n}
        if k > 1:
            # scelgo altri k-1 target (possono includere la diagonale i)
            possible_j = [j for j in range(n) if j != (i + 1) % n]
            targets.update(rng.choice(possible_j, size=k-1, replace=False))
      
        for j in targets:
            P[i, j] = rng.uniform(np.nextafter(0, 1), 1)

        # normalizzo la riga i affinché sommi a 1
        row_sum = P[i].sum()
        P[i] /= row_sum

    
    return P

def genera_P_reversibile(n, seed=42):

    """
    Genera una matrice di transizione P reversibile rispetto a un vettore stazionario π.

    La matrice P è:
    - row-stocastica (le righe sommano a 1)
    - reversibile (soddisfa il detailed balance π_i * P_ij = π_j * P_ji)
    - non banale (costruita da una matrice simmetrica casuale)

    Args:
        n (int)             :           dimensione della matrice quadrata P (numero di stati)
        seed (int)          :           seed per la generazione casuale

    Returns:
        np.ndarray          :           matrice P reversibile e stocastica per riga
    """
    rng = np.random.default_rng(seed)

    # 1. Vettore stazionario con valori casuali in {0,1}
    pi = rng.random(n)
    pi /= pi.sum()

    # 2. Matrice simmetrica S ogni elemento in {0,1}
    S = rng.uniform(np.nextafter(0, 1), 1/n, size=(n, n))
    S = (S + S.T) / 2

    # 3. Costruzione di M e normalizzazione riga
    M = S * pi[None, :]
    P = M / M.sum(axis=1, keepdims=True)

    return P



# =============================================================================
#  Q  – matrici generatrici (CTMC)
# =============================================================================
def genera_Q_irriducibile(n, zeros=0, tasso_massimo=1.0, seed=123, tentativi=50000):
    """Genera una matrice generatore **Q** irriducibile.

    **Nota sulla generazione**
    -------------------------
    La funzione genera matrici *Q* **irriducibili** (fortemente connesse)
    partendo da una struttura sparsa con zeri distribuiti casualmente fuori
    diagonale.

    * L'algoritmo impone un massimo di :math:`n(n-2)` zeri off‑diagonali per
      rendere possibile un grafo fortemente connesso.
    * Quando il numero di zeri si avvicina a tale massimo, la probabilità di
      ottenere una *Q* connessa si riduce e possono servire molti tentativi.
    * Un numero troppo basso di archi può indurre **periodicità**, ma nei test
      questo caso è raro.
    * Rispetto alla generazione di *P* sparse, questa funzione è più robusta:
      riesce a garantire connettività anche con elevata sparsità.

    Utile quando si desiderano CTMC irriducibili con controllo sulla sparsità e
    mantenimento dell'ergodicità.
    """
    
    max_zeros = n * (n - 2)
    if zeros > max_zeros:
        raise ValueError(f"Numero di zeri incompatibile: massimo {max_zeros}")

    for k in range(tentativi):
        rng = np.random.default_rng(seed + k)
        Q = np.zeros((n, n))

        # Calcola quanti zeri per riga (distribuzione uniforme)
        zeros_per_row = zeros // n
        extra_zeros = zeros % n  # Zeri extra da distribuire

        # Assegna zeri per ogni riga
        for i in range(n):
            possible_j = [j for j in range(n) if j != i]  # Indici fuori diagonale
            num_zeros = zeros_per_row + (1 if i < extra_zeros else 0)
            
            zero_positions = rng.choice(possible_j, num_zeros, replace=False)
            for j in possible_j:
                if j not in zero_positions:
                    Q[i, j] = rng.uniform(np.nextafter(0, 1), tasso_massimo)

        # Calcola la diagonale
        Q[np.diag_indices(n)] = -Q.sum(axis=1)

        # Verifica irreducibilità
        if nx.is_strongly_connected(nx.from_numpy_array((Q > 0).astype(int), create_using=nx.DiGraph)):
            return Q

    raise RuntimeError(f"Non ho Q irreducibile per n={n}, zeros={zeros}")

def genera_Q_k_random(n, k, tasso_massimo=1.0, seed=123):
    """
    Genera una matrice Q (nxn) irriducibile con:
      - arco obbligato i → (i+1)%n per garantire connessione
      - altri k-1 archi off-diagonali random per riga
      - pesi positivi ∈ (0, tasso_massimo]
      - diagonale negativa, righe a somma zero

    Args:
        n (int)                     :               dimensione della matrice
        k (int)                     :               archi totali per riga (deve essere ≥1 e ≤ n - 1)
        tasso_massimo (float)       :               massimo valore dei tassi
        seed (int)                  :               seed per random

    Returns:
        np.ndarray                  :               matrice generatore Q valida e irriducibile
    """
    if not (1 <= k <= n-1):
        raise ValueError("Serve 1 ≤ k ≤ n-1")

    rng = np.random.default_rng(seed)
    Q = np.zeros((n, n))

    for i in range(n):
        targets = {(i + 1) % n}  # arco obbligato

        pool = [j for j in range(n) if j != i and j != (i + 1) % n]
        num_extra = k - 1
        replace = num_extra > len(pool)

        extra = rng.choice(pool, size=num_extra, replace=False)
        targets.update(extra)

        for j in targets:
            Q[i, j] = rng.uniform(np.nextafter(0, 1), tasso_massimo)

    Q[np.diag_indices(n)] = -Q.sum(axis=1)

    G = nx.from_numpy_array((Q > 0).astype(int), create_using=nx.DiGraph)
    if nx.is_strongly_connected(G):
        return Q

    raise RuntimeError(f"Q non irriducibile con n={n}, k={k}, seed={seed}")

def genera_Q_reversibile(n, seed=42):
    """
    Genera una matrice Q reversibile non banale SENZA accortezze numeriche.

    - La divisione F / π può produrre valori molto grandi o piccoli
        Q può avere cattivo condizionamento numerico

    Args:
        n (int)                 :           dimensione della matrice quadrata Q
        seed (int)              :           seed per la generazione casuale

    Returns:
        np.ndarray              :           matrice generatore CTMC reversibile (Q)
    """
    rng = np.random.default_rng(seed)

    # 1. Vettore stazionario con valori casuali in {0,1}
    pi = rng.random(n)
    pi /= pi.sum()

    # 2. Flussi simmetrici senza controllo di scala ogni elemento in {0,1}
    F = rng.uniform(np.nextafter(0, 1), 1/n, size=(n, n))
    F = (F + F.T) / 2
    np.fill_diagonal(F, 0)

    # 3. Costruzione di Q = F / pi_i, righe a somma zero
    Q = F / pi[:, None]
    np.fill_diagonal(Q, 0)
    np.fill_diagonal(Q, -Q.sum(axis=1))

    return Q

def genera_Q_reversibile_forzata(n, seed=42):
    """
    Genera una matrice Q reversibile non banale, costruita per:

    - garantire stabilità numerica nella divisione Q = F / π
    - mantenere i tassi di uscita q(i) e π dello stesso ordine di grandezza
    - tenere basso il valore di B (massimo tasso d’uscita), così da non degradare l’uniformizzazione in P

    Strategia:
    - π è quasi uniforme, con valori intorno a 1/n
    - F è simmetrica, con valori ∼ 1/(n√n), un ordine più basso di π
      → Q = F / π ha valori moderati e ben bilanciati

    Returns:
        Q (np.ndarray): matrice generatore CTMC reversibile
    """
    np.random.seed(seed)

    # 1. π quasi uniforme ∼ 1/n
    low = 0.75 / n
    high = 1.25 / n
    pi = np.random.uniform(low, high, n)
    pi /= pi.sum()

    # 2. F simmetrica con valori ∼ 1/(n√n)
    F = np.random.uniform(1/(n*np.sqrt(n)), 2.5/(n*np.sqrt(n)), size=(n, n))
    F = (F + F.T) / 2
    np.fill_diagonal(F, 0)

    # 3. Q da detailed balance inverso: Q_ij = F_ij / π_i
    Q = F / pi[:, None]
    np.fill_diagonal(Q, 0)
    np.fill_diagonal(Q, -Q.sum(axis=1))

    return Q