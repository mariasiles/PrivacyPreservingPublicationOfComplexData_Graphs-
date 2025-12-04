"""
Visualización del grafo tras aplicar perturbación espectral (usando función real)
---------------------------------------------------------------
Carga el grafo original de grafo_vulnerable_demo.py, aplica la función perturbacion_espectral_conservando_metricas de metodos_proteccion_grafo.py y muestra el grafo resultante con los mismos nodos/letras.
"""

import matplotlib.pyplot as plt
from metodos_proteccion_grafo import G_original, perturbacion_espectral_conservando_metricas
import networkx as nx

# Aplicar perturbación espectral real
G_perturb = perturbacion_espectral_conservando_metricas(G_original, ruido=0.3)

# Visualización
plt.figure(figsize=(10,7))
pos = nx.spring_layout(G_perturb, seed=42)
nx.draw(G_perturb, pos, with_labels=True, node_color='lightblue', node_size=300, edge_color='gray', font_size=8)
plt.title('Grafo tras perturbación espectral (función real)')
plt.tight_layout()
plt.savefig('grafo_perturbacion_espectral_funcion.png', dpi=150)
plt.show()
print("Imagen guardada como 'grafo_perturbacion_espectral_funcion.png'")
