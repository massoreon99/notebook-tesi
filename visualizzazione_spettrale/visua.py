import funzioni, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import networkx as nx


def plot_line(G, mat_type="laplacian", title=""):
    # Calcola autovalori e autovettori
    vals, vecs, G = funzioni.compute_eig(G, mat_type)
    
    # Crea figura e asse
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_title(title)
    
    # Array degli indici dei nodi (da 0 a n-1)
    xdata = np.arange(len(G))
    
    # Disegna il primo autovettore come linea con marker circolari
    line, = ax.plot(xdata, vecs[:, 0], marker='o', markersize=3, linewidth=1)
    
    # Crea area per lo slider
    slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
    
    # Imposta lo slider: varia da 0 a len(G)-1, incremento di 1
    slider = Slider(slider_ax, "k", 0, len(G)-1, valinit=0, valstep=1)
    
    # Funzione di callback per aggiornare il grafico
    def update(_):
        k = int(slider.val)         # Indice selezionato
        line.set_ydata(vecs[:, k])  # Aggiorna i dati y con il k-esimo autovettore
        ax.relim()                  # Ricalcola i limiti dell'asse
        ax.autoscale_view()         # Aggiorna la scala
        fig.canvas.draw_idle()      # Ridisegna la figura
        
    slider.on_changed(update)
    
    # Ritorna la figura, l'asse, lo slider, gli autovalori, gli autovettori e il grafo aggiornato
    return fig, ax, slider, vals, vecs

def plot_2d(G, mat_type="laplacian", x=0, y=1, title="", weight=None):
    # Calcola autovalori e autovettori, eventualmente considerando i pesi
    vals, vecs, G = funzioni.compute_eig(G, mat_type, weight=weight)
    
    # Crea figura e asse
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_title(title)
    
    # Calcola le posizioni dei nodi basate sugli autovettori scelti per le coordinate x e y
    pos = {i: (vecs[i, x], vecs[i, y]) for i in range(len(G))}
    nx.draw(G, pos=pos, with_labels=True, ax=ax, font_size=3,
            node_size=70, node_color="skyblue", arrows=True, arrowsize=7,
            edge_color="gray", width=0.15)
    
    # Crea le aree per gli slider
    sx_ax = plt.axes([0.2, 0.08, 0.65, 0.03])
    sy_ax = plt.axes([0.2, 0.03, 0.65, 0.03])
    
    # Crea gli slider per selezionare gli indici degli autovettori da usare per X e Y
    sx = Slider(sx_ax, "X", 0, len(G)-1, valinit=x, valstep=1)
    sy = Slider(sy_ax, "Y", 0, len(G)-1, valinit=y, valstep=1)
    
    # Callback per aggiornare la visualizzazione
    def update(_):
        ax.clear()
        xx = int(sx.val)
        yy = int(sy.val)
        new_pos = {i: (vecs[i, xx], vecs[i, yy]) for i in range(len(G))}
        nx.draw(G, pos=new_pos, with_labels=True, ax=ax, font_size=3,
                node_size=50, node_color="skyblue", arrows=True, arrowsize=7,
                edge_color="gray", width=0.15)
        ax.set_title(title)
        fig.canvas.draw_idle()
    
    sx.on_changed(update)
    sy.on_changed(update)
    
    # Ritorna la figura, l'asse, gli slider e gli autovalori/autovettori
    return fig, ax, (sx, sy), vals, vecs

def plot_3d(G, mat_type="laplacian", x=0, y=1, z=2, title=""):
   
    vals, vecs, G = funzioni.compute_eig(G, mat_type)
    
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    
   
    xs = vecs[:, x]
    ys = vecs[:, y]
    zs = vecs[:, z]
    
    # Disegno nodi
    scat = ax.scatter(xs, ys, zs, c='skyblue', s=50, edgecolors='k', depthshade=True)
    
    # Disegno archi
    lines_dict = {}
    for (u, v) in G.edges():
        xx = [xs[u], xs[v]]
        yy = [ys[u], ys[v]]
        zz = [zs[u], zs[v]]
        ln = ax.plot(xx, yy, zz, color='gray', linewidth=0.5)[0]
        lines_dict[(u, v)] = ln
    
    ax.set_xlabel("Autovettore X")
    ax.set_ylabel("Autovettore Y")
    ax.set_zlabel("Autovettore Z")

    
    text_labels = []

    # Funzione che pulisce i label
    def clear_labels():
        for lbl in text_labels:
            lbl.remove()  # elimina l'oggetto text dallo axes
        text_labels.clear()

    for i in range(len(G)):
        lbl = ax.text(xs[i], ys[i], zs[i], s=str(i), size=3, color="black")
        text_labels.append(lbl)
    
    Slider
    slider_ax_x = plt.axes([0.15, 0.07, 0.65, 0.03])
    slider_ax_y = plt.axes([0.15, 0.04, 0.65, 0.03])
    slider_ax_z = plt.axes([0.15, 0.01, 0.65, 0.03])
    
    x_slider = Slider(slider_ax_x, "X", 0, len(G)-1, valinit=x, valstep=1)
    y_slider = Slider(slider_ax_y, "Y", 0, len(G)-1, valinit=y, valstep=1)
    z_slider = Slider(slider_ax_z, "Z", 0, len(G)-1, valinit=z, valstep=1)
    
    def update(_):
        xx = int(x_slider.val)
        yy = int(y_slider.val)
        zz = int(z_slider.val)
        
        xs_new = vecs[:, xx]
        ys_new = vecs[:, yy]
        zs_new = vecs[:, zz]

        # Aggiorna scatter
        scat._offsets3d = (xs_new, ys_new, zs_new)
        
        # Aggiorna archi
        for (u, v), line3d in lines_dict.items():
            line3d.set_xdata([xs_new[u], xs_new[v]])
            line3d.set_ydata([ys_new[u], ys_new[v]])
            line3d.set_3d_properties([zs_new[u], zs_new[v]])
        
        # Riposiziona label: prima rimuoviamo i vecchi, poi li ricostruiamo
        clear_labels()
        for i in range(len(G)):
            lbl = ax.text(xs_new[i], ys_new[i], zs_new[i], s=str(i), size=3, color="black")
            text_labels.append(lbl)

        fig.canvas.draw_idle()
    
    x_slider.on_changed(update)
    y_slider.on_changed(update)
    z_slider.on_changed(update)
    
    plt.show()
    
    return fig, ax, (x_slider, y_slider, z_slider), vals, vecs

def plot_from_file(path, mat_type="laplacian", mode="2d", x=None, y=None, z=None, title=None, weight=None):
    with open(path) as f:
        data = json.load(f)
    G = nx.node_link_graph(data)

    if mode == "3d":
        return plot_3d(G, mat_type=mat_type, x=x, y=y, z=z, title=title)
    if mode == "2d":
        return plot_2d(G, mat_type=mat_type, x=x, y=y, title=title, weight=weight)
    elif mode == "1d":
        return plot_line(G, mat_type=mat_type, title=title)