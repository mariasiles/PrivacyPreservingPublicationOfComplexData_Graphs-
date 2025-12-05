import networkx as nx
import matplotlib.pyplot as plt



"""
DEMO: Grafo con Múltiples Vulnerabilidades
-----------------------------------------
Genera un grafo artificial con vulnerabilidades de grado, vecindario, subgrafos, atributos y patrones reconocibles.
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import random

np.random.seed(42)
random.seed(42)


# Crear grafo base con nodos renombrados a letras
G = nx.Graph()
ciudades = ['Madrid', 'Barcelona', 'Valencia', 'Sevilla', 'Bilbao', 'Zaragoza']
profesiones = ['Ingeniero', 'Profesor', 'Médico', 'Abogado', 'Artista', 'Estudiante']
edades = [34, 41, 29, 52, 37, 23]

# Asignar letras a todos los nodos
letras = [chr(ord('A') + i) for i in range(30)]
letra_idx = 0
nombre_map = {}

# Generador de salario y situación
def generar_salario():
    return random.randint(18000, 90000)
def generar_situacion():
    return random.choice(['Casado', 'Soltero'])

# 1. Comunidad densa (clique) con grado único
clique = [letras[letra_idx + i] for i in range(6)]
for i, n in enumerate(clique):
    nombre_map[f'C{i}'] = n
    G.add_node(n, tipo='Persona', comunidad='Clique', ciudad=ciudades[i], profesion=profesiones[i], edad=edades[i], salario=generar_salario(), situacion=generar_situacion())
for u in clique:
    for v in clique:
        if u != v:
            G.add_edge(u, v)
letra_idx += 6

# 2. Nodo con grado muy alto (hub)
hub = letras[letra_idx]
nombre_map['Hub'] = hub
G.add_node(hub, tipo='Persona', comunidad='Hub', grado_vulnerable=True, ciudad='Madrid', profesion='Empresario', edad=45, salario=generar_salario(), situacion=generar_situacion())
for n in clique:
    G.add_edge(hub, n)
letra_idx += 1
extras = []
for i in range(5):
    extra = letras[letra_idx]
    nombre_map[f'Extra_{i}'] = extra
    G.add_node(extra, tipo='Persona', comunidad='Hub',
               ciudad=random.choice(ciudades),
               profesion=random.choice(profesiones),
               edad=random.randint(20, 60),
               salario=generar_salario(),
               situacion=generar_situacion())
    G.add_edge(hub, extra)
    extras.append(extra)
    letra_idx += 1

# 3. Nodo con grado bajo
leaf = letras[letra_idx]
nombre_map['Leaf'] = leaf
G.add_node(leaf, tipo='Persona', comunidad='Leaf', grado_vulnerable=True, ciudad='Valencia', profesion='Estudiante', edad=19, salario=generar_salario(), situacion=generar_situacion())
G.add_edge(leaf, clique[0])
letra_idx += 1

# 4. Estrella única
center = letras[letra_idx]
nombre_map['Estrella_Centro'] = center
G.add_node(center, comunidad='Estrella', subgrafo_vulnerable=True, ciudad='Bilbao', profesion='Técnico', edad=38, salario=generar_salario(), situacion=generar_situacion())
letra_idx += 1
satellites = []
for i in range(5):
    sat = letras[letra_idx]
    nombre_map[f'Estrella_Sat_{i}'] = sat
    G.add_node(sat, comunidad='Estrella', ciudad='-', profesion='-', edad=0, salario=generar_salario(), situacion=generar_situacion())
    G.add_edge(center, sat)
    satellites.append(sat)
    letra_idx += 1

# Asegurar que el centro solo esté conectado a los satélites y los satélites solo al centro
for n in list(G.neighbors(center)):
    if n not in satellites:
        G.remove_edge(center, n)
for sat in satellites:
    for n in list(G.neighbors(sat)):
        if n != center:
            G.remove_edge(sat, n)

# 5. Puente entre dos comunidades
bridge = letras[letra_idx]
nombre_map['Puente'] = bridge
G.add_node(bridge, tipo='Persona', comunidad='Puente', puente_vulnerable=True, ciudad='Sevilla', profesion='Consultor', edad=50, salario=generar_salario(), situacion=generar_situacion())
G.add_edge(bridge, clique[1])
G.add_edge(bridge, center)
letra_idx += 1

# 6. Cluster de atributo sensible (homofilia)
sensibles = []
sens_ciudades = ['Madrid', 'Barcelona', 'Valencia']
sens_profesiones = ['Médico', 'Profesor', 'Estudiante']
sens_edades = [56, 48, 22]
for i in range(3):
    n = letras[letra_idx]
    nombre_map[f'Sens_{i}'] = n
    G.add_node(n, tipo='Persona', comunidad='Sens', enfermedad='Diabetes', atributo_vulnerable=True,
               ciudad=sens_ciudades[i], profesion=sens_profesiones[i], edad=sens_edades[i], salario=generar_salario(), situacion=generar_situacion())
    sensibles.append(n)
    letra_idx += 1
for u in sensibles:
    for v in sensibles:
        if u != v:
            G.add_edge(u, v)

# 7. Vecindario único
vec_unico = letras[letra_idx]
nombre_map['Vecindario_Unico'] = vec_unico
G.add_node(vec_unico, tipo='Persona', comunidad='VecUnico', vecindario_vulnerable=True, ciudad='Zaragoza', profesion='Artista', edad=27, salario=generar_salario(), situacion=generar_situacion())
G.add_edge(vec_unico, clique[2])
G.add_edge(vec_unico, sensibles[0])
letra_idx += 1

# 8. Algunos enlaces aleatorios para ruido
# Restaurar estructura natural: clique y extras con conexiones originales
for extra in extras:
    for v in list(G.neighbors(extra)):
        if v != hub:
            G.remove_edge(extra, v)
    if not G.has_edge(extra, hub):
        G.add_edge(extra, hub)

# 3. El hub solo conectado al clique y extras
for v in list(G.neighbors(hub)):
    if v not in clique and v not in extras:
        G.remove_edge(hub, v)

# 4. El leaf solo conectado a C0
for v in list(G.neighbors(leaf)):
    if v != clique[0]:
        G.remove_edge(leaf, v)
if not G.has_edge(leaf, clique[0]):
    G.add_edge(leaf, clique[0])


# 5. El puente solo conectado a C1 y Estrella_Centro (ya garantizado)
# 6. El clique: para evitar que C1 y C2 sean puentes, añadimos una conexión extra entre el clique y el hub
for c in clique:
    if not G.has_edge(c, hub):
        G.add_edge(c, hub)

# 7. Vecindario_Unico: para evitar que sea puente, añadimos una conexión extra con Sens_1
if not G.has_edge(vec_unico, sensibles[1]):
    G.add_edge(vec_unico, sensibles[1])

# 6. Los sensibles solo conectados entre sí
for n in sensibles:
    for v in list(G.neighbors(n)):
        if v not in sensibles:
            G.remove_edge(n, v)
    for v in sensibles:
        if n != v and not G.has_edge(n, v):
            G.add_edge(n, v)

# 7. Vecindario_Unico solo conectado a C2 y Sens_0
for v in list(G.neighbors(vec_unico)):
    if v != clique[2] and v != sensibles[0]:
        G.remove_edge(vec_unico, v)
if not G.has_edge(vec_unico, clique[2]):
    G.add_edge(vec_unico, clique[2])
if not G.has_edge(vec_unico, sensibles[0]):
    G.add_edge(vec_unico, sensibles[0])

    
# Lista de nodos con atributo sensible
atributos_sensibles = ['U', 'V', 'W']

plt.figure(figsize=(10,7))
pos = nx.spring_layout(G, seed=42)
# Color rojo para los nodos sensibles, azul para los demás
colores = ['red' if n in atributos_sensibles else 'lightblue' for n in G.nodes()]
nx.draw(G, pos, with_labels=True, node_color=colores, node_size=400, edge_color='gray', font_size=10)
plt.title('Nodos con atributos sensibles (rojo)')
plt.tight_layout()
plt.savefig('ATRIBUTOS_SENSIBLES.png', dpi=150)
plt.show()
print("Imagen guardada como 'ATRIBUTOS_SENSIBLES.png'")