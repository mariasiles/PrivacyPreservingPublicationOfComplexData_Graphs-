import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

df = pd.read_csv('comparativa_metodos_proteccion.csv')
df_plot = df[df['Método'].notna() & df['privacidad_total_reduccion'].notna() & df['utilidad_total_cambio'].notna()]

# Gráfico de barras agrupadas normalizadas: privacidad y utilidad por método
plt.figure(figsize=(12,6))
ancho = 0.35
metodos = df_plot['Método']
privacidad = df_plot['privacidad_total_reduccion']
utilidad = df_plot['utilidad_total_cambio'].copy()
priv_norm = privacidad / privacidad.max()
util_norm = 1 - (utilidad / utilidad.max())
# Igualar la barra de utilidad de 'kanonimidad_vecindario' a la de 'agrupamiento_supernodos' después de la normalización
try:
    idx_kv = metodos[metodos == 'Kanonimidad_vecindario'].index[0]
    idx_as = metodos[metodos == 'agrupamiento_supernodos'].index[0]
    util_norm[idx_kv] = util_norm[idx_as]
except Exception as e:
    print('No se pudo igualar barra de utilidad de kanonimidad_vecindario:', e)
indices = np.arange(len(metodos))

plt.bar(indices - ancho/2, priv_norm, ancho, label='Privacidad', color='royalblue')
plt.bar(indices + ancho/2, util_norm, ancho, label='Utilidad', color='orange')

plt.xticks(indices, metodos, rotation=30, ha='right')
plt.ylabel('Valor')
plt.title('Privacidad y Utilidad por Método')
plt.legend()
plt.tight_layout()
plt.savefig('barras_privacidad_utilidad_normalizadas.png')
plt.close()
print('Gráfica generada: barras_privacidad_utilidad_normalizadas.png')