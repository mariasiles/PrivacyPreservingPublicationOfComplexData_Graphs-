"""
DEMO: Métodos de Protección de Privacidad en Grafos
--------------------------------------------------
Implementa y compara varios métodos de protección de privacidad en grafos:

- K-anonimato estructural (de grado): Modifica el grafo para que cada nodo tenga el mismo grado que al menos otros k-1 nodos.
- K-anonimato de vecindario: Ajusta el grafo para que el vecindario de cada nodo sea similar al de al menos otros k-1 nodos.
- Perturbación de aristas (Random Switch): Intercambia extremos de aristas para modificar las conexiones sin alterar el grado de los nodos.
- Clustering Preserving Randomization: Modifica la estructura interna de las comunidades detectadas en el grafo, preservando la modularidad y el agrupamiento.
- Agrupamiento con supernodos (Cluster-preserving con KMeans): Agrupa nodos similares en supernodos y publica estadísticas agregadas, ocultando detalles individuales.
- Perturbación espectral: Añade ruido controlado a la matriz de adyacencia para modificar el grafo, intentando conservar métricas espectrales y globales (coeficiente de clustering, betweenness, centralidad, distancias, etc.).
- Diferential Privacy para aristas (Randomized Response): Aplica un mecanismo de privacidad diferencial a nivel de arista, usando randomized response para garantizar protección formal edge-DP.

Calcula métricas de privacidad y utilidad para comparar los resultados de cada método, incluyendo reducción de nodos vulnerables, preservación de comunidades, centralidad, distancias, y métricas espectrales.
"""


import networkx as nx
import numpy as np
import random
from copy import deepcopy
import pandas as pd
np.random.seed(1)
random.seed(1)

np.random.seed(1)
random.seed(1)

# --- Cargar el grafo original ---
from grafo_vulnerable_demo import G as G_original

# --- MÉTODO 1: K-Anonimidad de Grado ---
def kanonimidad_grado(grafo, k=2):
    """Garantiza que cada nodo tenga el mismo grado que al menos otros k-1 nodos."""
    G = deepcopy(grafo)
    grados = dict(G.degree())
    grado_counter = {}
    for g in grados.values():
        grado_counter[g] = grado_counter.get(g, 0) + 1
    # Detectar grados únicos
    grados_unicos = [g for g, c in grado_counter.items() if c < k]
    for nodo, grado in grados.items():
        if grado in grados_unicos:
            # Buscar grado objetivo más cercano que tenga al menos k-1 nodos
            posibles_grados = [g for g, c in grado_counter.items() if c >= k-1 and g != grado]
            if not posibles_grados:
                continue
            target_grado = min(posibles_grados, key=lambda x: abs(x - grado))
            # Ajustar grado del nodo al target_grado
            while G.degree(nodo) < target_grado:
                posibles = [n for n in G.nodes() if n != nodo and not G.has_edge(nodo, n)]
                if posibles:
                    G.add_edge(nodo, random.choice(posibles))
                else:
                    break
            while G.degree(nodo) > target_grado:
                vecinos = list(G.neighbors(nodo))
                if vecinos:
                    G.remove_edge(nodo, random.choice(vecinos))
                else:
                    break
            # Actualizar contador de grados
            grados[nodo] = G.degree(nodo)
            grado_counter = {}
            for g in grados.values():
                grado_counter[g] = grado_counter.get(g, 0) + 1
            grados_unicos = [g for g, c in grado_counter.items() if c < k]
    return G

# --- MÉTODO 2: Perturbación de Aristas (Random Switch) ---
def perturbacion_aristas_random_switch(grafo, n_switch=10):
    """Intercambia extremos de aristas para mantener el grado pero modificar conexiones."""
    G = deepcopy(grafo)
    edges = list(G.edges())
    for _ in range(n_switch):
        if len(edges) < 2:
            break
        e1, e2 = random.sample(edges, 2)
        a, b = e1
        c, d = e2
        if len({a, b, c, d}) == 4:
            if not G.has_edge(a, d) and not G.has_edge(c, b):
                G.remove_edge(a, b)
                G.remove_edge(c, d)
                G.add_edge(a, d)
                G.add_edge(c, b)
                edges.remove(e1)
                edges.remove(e2)
                edges.append((a, d))
                edges.append((c, b))
    return G

# --- MÉTODO 3: Clustering Preserving Randomization ---
def clustering_preserving_randomization(grafo, p=0.3):
    """Modifica la estructura preservando comunidades usando randomización controlada."""
    G = deepcopy(grafo)
    # Detectar comunidades
    from networkx.algorithms.community import greedy_modularity_communities
    comunidades = list(greedy_modularity_communities(G))
    for comunidad in comunidades:
        miembros = list(comunidad)
        num_mod = max(1, int(len(miembros) * p))
        for _ in range(num_mod):
            u, v = random.sample(miembros, 2)
            if G.has_edge(u, v):
                G.remove_edge(u, v)
            else:
                G.add_edge(u, v)
    return G

# --- MÉTODO 4: Agrupamiento con Supernodos ---
def agrupamiento_supernodos(grafo, n_clusters=4):
    """Agrupa nodos similares y publica estadísticas agregadas del grupo."""
    G = deepcopy(grafo)
    from sklearn.cluster import KMeans
    A = nx.to_numpy_array(G)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(A)
    labels = kmeans.labels_
    clusters = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(labels):
        clusters[label].append(list(G.nodes())[idx])
    # Crear supernodo por cluster
    G_super = nx.Graph()
    for c, nodos in clusters.items():
        G_super.add_node(f'Supernodo_{c}', size=len(nodos))
    # Conectar supernodos si hay alguna arista entre sus miembros
    for i in range(n_clusters):
        for j in range(i+1, n_clusters):
            miembros_i = clusters[i]
            miembros_j = clusters[j]
            conectado = False
            for u in miembros_i:
                for v in miembros_j:
                    if G.has_edge(u, v):
                        conectado = True
                        break
                if conectado:
                    break
            if conectado:
                G_super.add_edge(f'Supernodo_{i}', f'Supernodo_{j}')
    return G_super
# --- MÉTODO 5: Protección a nivel de Arista ---

# --- MÉTODO 8: Diferential Privacy para aristas (Randomized Response) ---
def diferencial_privacidad_aristas(grafo, epsilon=1.5):
    """Aplica randomized response para cada arista, garantizando edge-DP."""
    G = deepcopy(grafo)
    nodes = list(G.nodes())
    n = len(nodes)
    # Probabilidades
    p = np.exp(epsilon) / (1 + np.exp(epsilon))
    q = 1 / (1 + np.exp(epsilon))
    # Crear nueva matriz de adyacencia
    A = nx.to_numpy_array(G)
    A_dp = np.zeros_like(A)
    for i in range(n):
        for j in range(i+1, n):
            if A[i, j] == 1:
                A_dp[i, j] = A_dp[j, i] = np.random.binomial(1, p)
            else:
                A_dp[i, j] = A_dp[j, i] = np.random.binomial(1, q)
    G_dp = nx.from_numpy_array(A_dp)
    mapping = dict(enumerate(nodes))
    G_dp = nx.relabel_nodes(G_dp, mapping)
    return G_dp

# --- MÉTODO 7: Perturbación Espectral ---
def perturbacion_espectral_conservando_metricas(grafo, ruido=0.1):
    """Modifica el grafo manteniendo valores propios y métricas globales."""
    G = deepcopy(grafo)
    A = nx.to_numpy_array(G)
    eigvals_orig = np.linalg.eigvalsh(A)
    # Reducir el ruido para preservar estructura local
    ruido_matrix = np.random.normal(0, 0.01, A.shape)
    A_perturb = np.clip(A + ruido_matrix, 0, 1)
    # Mantener simetría
    A_perturb = (A_perturb + A_perturb.T) / 2
    eigvals_perturb = np.linalg.eigvalsh(A_perturb)
    # Ajustar para conservar suma de valores propios
    factor = np.sum(eigvals_orig) / np.sum(eigvals_perturb)
    A_perturb *= factor
    # Solo modificar un pequeño porcentaje de aristas (las de menor peso)
    n = A.shape[0]
    num_mod = max(1, int(0.05 * np.count_nonzero(A)))  # 5% de las aristas
    A_bin = (A > 0.5).astype(int)
    # Encuentra los pares con mayor diferencia entre A_bin y A_perturb
    diffs = np.abs(A_perturb - A_bin)
    indices = np.dstack(np.unravel_index(np.argsort(diffs.ravel())[::-1], (n, n)))[0]
    count = 0
    for i, j in indices:
        if i >= j:
            continue
        if count >= num_mod:
            break
        # Modifica solo si hay diferencia significativa
        if diffs[i, j] > 0.3:
            A_bin[i, j] = A_bin[j, i] = 1 - A_bin[i, j]
            count += 1
    G_perturb = nx.from_numpy_array(A_bin)
    mapping = dict(enumerate(G.nodes()))
    G_perturb = nx.relabel_nodes(G_perturb, mapping)
    return G_perturb

# --- MÉTODO 6: K-Anonimidad de Vecindario ---
def kanonimidad_vecindario(grafo, k=2):
    """Hace que el vecindario de cada nodo sea similar al de al menos otros k-1 nodos."""
    G = deepcopy(grafo)
    vecinos_dict = {n: set(G.neighbors(n)) for n in G.nodes()}
    # Buscar nodos con vecindario único
    vecindarios = list(vecinos_dict.values())
    for nodo in G.nodes():
        v = vecinos_dict[nodo]
        similares = [n for n in G.nodes() if n != nodo and len(v & vecinos_dict[n]) >= max(1, int(0.7 * max(len(v), len(vecinos_dict[n]))))]
        if len(similares) < k-1:
            # Generalizar: añadir conexiones para igualar vecindarios
            candidatos = [n for n in G.nodes() if n != nodo and n not in v]
            for otro in candidatos:
                if len(similares) >= k-1:
                    break
                G.add_edge(nodo, otro)
                v.add(otro)
                similares = [n for n in G.nodes() if n != nodo and len(v & vecinos_dict[n]) >= max(1, int(0.7 * max(len(v), len(vecinos_dict[n]))))]
    return G


# --- MÉTRICAS DE PRIVACIDAD Y UTILIDAD ---
def metricas(g_original, g_mod):
    # Privacidad: reducción de nodos vulnerables
    from grafo_vulnerable_demo import analizar_vulnerabilidades
    vuln_orig = analizar_vulnerabilidades(g_original)
    vuln_mod = analizar_vulnerabilidades(g_mod)
    priv_metric = {}
    for k in vuln_orig:
        orig = len(vuln_orig[k])
        mod = len(vuln_mod[k])
        priv_metric[k] = {'original': orig, 'modificado': mod}

    # --- 1. Métricas de estructura local ---
    deg_orig = [d for n, d in g_original.degree()]
    deg_mod = [d for n, d in g_mod.degree()]
    clustering_orig = nx.average_clustering(g_original)
    clustering_mod = nx.average_clustering(g_mod)

    # Distribución de grado: histogramas normalizados
    from scipy.stats import entropy, wasserstein_distance
    from scipy.spatial.distance import jensenshannon
    bins = np.arange(0, max(max(deg_orig), max(deg_mod))+2)
    hist_orig, _ = np.histogram(deg_orig, bins=bins, density=True)
    hist_mod, _ = np.histogram(deg_mod, bins=bins, density=True)
    kl_div = entropy(hist_orig+1e-9, hist_mod+1e-9)  # +1e-9 para evitar log(0)
    js_div = jensenshannon(hist_orig, hist_mod)
    wass_dist = wasserstein_distance(deg_orig, deg_mod)

    # --- 2. Métricas de comunidad ---
    from networkx.algorithms.community import greedy_modularity_communities, modularity
    try:
        comm_orig = list(greedy_modularity_communities(g_original))
        comm_mod = list(greedy_modularity_communities(g_mod))
        modularidad_orig = modularity(g_original, comm_orig)
        modularidad_mod = modularity(g_mod, comm_mod)
    except:
        modularidad_orig = modularidad_mod = 'N/A'

    # --- 3. Métricas de distancias ---
    try:
        diam_orig = nx.diameter(g_original)
        diam_mod = nx.diameter(g_mod)
        spl_orig = nx.average_shortest_path_length(g_original)
        spl_mod = nx.average_shortest_path_length(g_mod)
    except:
        diam_orig = diam_mod = spl_orig = spl_mod = 'N/A'

    # --- 4. Métricas de centralidad ---
    try:
        deg_cent_orig = list(nx.degree_centrality(g_original).values())
        deg_cent_mod = list(nx.degree_centrality(g_mod).values())
        bet_cent_orig = list(nx.betweenness_centrality(g_original).values())
        bet_cent_mod = list(nx.betweenness_centrality(g_mod).values())
        close_cent_orig = list(nx.closeness_centrality(g_original).values())
        close_cent_mod = list(nx.closeness_centrality(g_mod).values())
        pr_orig = list(nx.pagerank(g_original).values())
        pr_mod = list(nx.pagerank(g_mod).values())
        from scipy.stats import spearmanr
        spearman_deg = spearmanr(deg_cent_orig, deg_cent_mod)[0]
        spearman_bet = spearmanr(bet_cent_orig, bet_cent_mod)[0]
        spearman_close = spearmanr(close_cent_orig, close_cent_mod)[0]
        spearman_pr = spearmanr(pr_orig, pr_mod)[0]
    except:
        spearman_deg = spearman_bet = spearman_close = spearman_pr = 'N/A'

    # --- 5. Métricas espectrales ---
    try:
        lap_orig = nx.laplacian_matrix(g_original).toarray()
        lap_mod = nx.laplacian_matrix(g_mod).toarray()
        frob_dist = np.linalg.norm(lap_orig - lap_mod)
        eig_orig = np.linalg.eigvalsh(lap_orig)
        eig_mod = np.linalg.eigvalsh(lap_mod)
        eig_diff = np.linalg.norm(eig_orig - eig_mod)
    except:
        frob_dist = eig_diff = 'N/A'

    # --- 6. Métricas de privacidad avanzadas ---
    # Proporción de nodos reidentificables (grado único, vecindario único)
    n_total = g_original.number_of_nodes()
    grado_unico_orig = priv_metric.get('grado', {}).get('original', 0)
    grado_unico_mod = priv_metric.get('grado', {}).get('modificado', 0)
    vec_unico_orig = priv_metric.get('vecindario', {}).get('original', 0)
    vec_unico_mod = priv_metric.get('vecindario', {}).get('modificado', 0)
    prop_grado_unico_orig = grado_unico_orig / n_total if n_total else 0
    prop_grado_unico_mod = grado_unico_mod / n_total if n_total else 0
    prop_vec_unico_orig = vec_unico_orig / n_total if n_total else 0
    prop_vec_unico_mod = vec_unico_mod / n_total if n_total else 0

    util_metric = {
        'clustering_diff': abs(clustering_orig - clustering_mod),
        'kl_div_grado': kl_div,
        'js_div_grado': js_div,
        'wasserstein_grado': wass_dist,
        'modularidad_diff': abs(modularidad_orig - modularidad_mod) if modularidad_orig != 'N/A' else 'N/A',
        'diametro_diff': abs(diam_orig - diam_mod) if diam_orig != 'N/A' else 'N/A',
        'spl_diff': abs(spl_orig - spl_mod) if spl_orig != 'N/A' else 'N/A',
        'spearman_deg': spearman_deg,
        'spearman_bet': spearman_bet,
        'spearman_close': spearman_close,
        'spearman_pr': spearman_pr,
        'frob_dist_laplacian': frob_dist,
        'eig_diff_laplacian': eig_diff,
        'densidad_diff': abs(nx.density(g_original) - nx.density(g_mod)),
        'componentes_orig': nx.number_connected_components(g_original),
        'componentes_mod': nx.number_connected_components(g_mod),
        'prop_grado_unico_orig': prop_grado_unico_orig,
        'prop_grado_unico_mod': prop_grado_unico_mod,
        'prop_vec_unico_orig': prop_vec_unico_orig,
        'prop_vec_unico_mod': prop_vec_unico_mod
    }
    resumen = {
        'privacidad_total_reduccion': sum([priv_metric[k]['original'] - priv_metric[k]['modificado'] for k in priv_metric]),
        'utilidad_total_cambio': sum([v for k, v in util_metric.items() if isinstance(v, (int, float))])
    }
    return priv_metric, util_metric, resumen

# --- EJECUCIÓN Y COMPARACIÓN ---


metodos = {
    'kanonimidad_grado': kanonimidad_grado(G_original, k=3),
    'kanonimidad_vecindario': kanonimidad_vecindario(G_original, k=2),
    'perturbacion_aristas_random_switch': perturbacion_aristas_random_switch(G_original, n_switch=15),
    'clustering_preserving_randomization': clustering_preserving_randomization(G_original, p=0.4),
    'perturbacion_espectral': perturbacion_espectral_conservando_metricas(G_original, ruido=0.3),
    'agrupamiento_supernodos': agrupamiento_supernodos(G_original, n_clusters=4),
    'diferencial_privacidad_aristas': diferencial_privacidad_aristas(G_original, epsilon=2.4)
    # 'proteccion_arista_ruido': proteccion_arista_ruido(G_original, p=0.15) # Si quieres añadirlo, define la función
}


# --- Mostrar resultados en tabla ---
resultados = []
for nombre, G_mod in metodos.items():
    if nombre == 'agrupamiento_supernodos':
        # No se calculan vulnerabilidades individuales
        resultados.append({'Método': nombre, 'Nota': 'Solo supernodos, sin métricas individuales'})
    else:
        priv, util, resumen = metricas(G_original, G_mod)
        fila = {'Método': nombre}
        # Métricas de privacidad
        for k, v in priv.items():
            fila[f'priv_{k}_orig'] = v['original']
            fila[f'priv_{k}_mod'] = v['modificado']
        # Métricas de utilidad
        for k, v in util.items():
            fila[k] = v
        # Resumen
        for k, v in resumen.items():
            fila[k] = v
        resultados.append(fila)


tabla = pd.DataFrame(resultados)
# Redondear valores numéricos para mejor legibilidad
for col in tabla.columns:
    if tabla[col].dtype == float:
        tabla[col] = tabla[col].round(4)

# Seleccionar y ordenar columnas clave para la comparación
columnas_clave = [
    'Método',
    'priv_grado_orig', 'priv_grado_mod', 'priv_vecindario_orig', 'priv_vecindario_mod',
    'priv_subgrafo_orig', 'priv_subgrafo_mod', 'priv_atributo_orig', 'priv_atributo_mod', 'priv_puente_orig', 'priv_puente_mod',
    'clustering_diff', 'kl_div_grado', 'js_div_grado', 'wasserstein_grado', 'modularidad_diff', 'diametro_diff', 'spl_diff',
    'spearman_deg', 'spearman_bet', 'spearman_close', 'spearman_pr',
    'frob_dist_laplacian', 'eig_diff_laplacian', 'densidad_diff',
    'componentes_orig', 'componentes_mod',
    'prop_grado_unico_orig', 'prop_grado_unico_mod', 'prop_vec_unico_orig', 'prop_vec_unico_mod',
    'privacidad_total_reduccion', 'utilidad_total_cambio', 'Nota'
]
columnas_finales = [c for c in columnas_clave if c in tabla.columns]
tabla_final = tabla[columnas_finales]


print("\n===== COMPARATIVA DE MÉTODOS DE PROTECCIÓN =====")
print(tabla_final.to_string(index=False))

# --- Interpretación automática de resultados ---
def interpretar_fila(fila):
    resumen = []
    # Comunidades
    def safe_float(val):
        try:
            return float(val)
        except:
            return None
    mod_diff = safe_float(fila.get('modularidad_diff'))
    if mod_diff is not None:
        if mod_diff < 0.05:
            resumen.append("Preserva bien las comunidades")
        elif mod_diff < 0.2:
            resumen.append("Modifica parcialmente las comunidades")
        else:
            resumen.append("Destruye la estructura de comunidades")
    # Clustering
    clust_diff = safe_float(fila.get('clustering_diff'))
    if clust_diff is not None:
        if clust_diff < 0.05:
            resumen.append("Preserva el coeficiente de clustering")
        elif clust_diff < 0.2:
            resumen.append("Varía moderadamente el clustering")
        else:
            resumen.append("Destruye el clustering local")
    # Centralidad
    spearman_deg = safe_float(fila.get('spearman_deg'))
    if spearman_deg is not None:
        if spearman_deg > 0.8:
            resumen.append("Preserva la importancia relativa de los nodos")
        elif spearman_deg > 0.5:
            resumen.append("Cambia parcialmente la importancia de los nodos")
        else:
            resumen.append("Destruye el ranking de nodos importantes")
    # Privacidad
    priv_red = safe_float(fila.get('privacidad_total_reduccion'))
    if priv_red is not None:
        if priv_red > 10:
            resumen.append("Alta protección de privacidad")
        elif priv_red > 3:
            resumen.append("Protección de privacidad moderada")
        else:
            resumen.append("Protección de privacidad baja")
    # Utilidad
    util_cambio = safe_float(fila.get('utilidad_total_cambio'))
    if util_cambio is not None:
        if util_cambio < 10:
            resumen.append("Alta preservación de utilidad")
        elif util_cambio < 30:
            resumen.append("Utilidad moderadamente preservada")
        else:
            resumen.append("Utilidad muy alterada")
    return "; ".join(resumen)

tabla_final['Interpretación'] = tabla_final.apply(interpretar_fila, axis=1)

print("\n===== RESUMEN INTERPRETATIVO DE MÉTODOS =====")
print(tabla_final[['Método', 'Interpretación']].to_string(index=False))

# Exportar a CSV para análisis externo si se desea
tabla_final.to_csv('comparativa_metodos_proteccion.csv', index=False)
print("\nTabla exportada como 'comparativa_metodos_proteccion.csv'")
