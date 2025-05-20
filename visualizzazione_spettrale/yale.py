import numpy as np
import networkx as nx
from scipy.io import loadmat
from networkx.readwrite import json_graph
import json

# Carica il file .mat
mat_data = loadmat('yale.mat')  

# Estrai la matrice di adiacenza
A = mat_data['A']

# Converti in grafo
G = nx.from_numpy_array(A)

# Salva in JSON (formato compatibile con json_graph)
data = json_graph.node_link_data(G)
with open("yale_graph.json", "w") as f:
    json.dump(data, f)

print("Grafo esportato in yale_graph.json")