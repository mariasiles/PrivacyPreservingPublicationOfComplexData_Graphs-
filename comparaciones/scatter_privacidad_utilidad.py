import pandas as pd
import matplotlib.pyplot as plt

# Leer datos
csv = 'comparativa_metodos_proteccion.csv'
df = pd.read_csv(csv)

# Filtrar métodos válidos
df_plot = df[df['Método'].notna() & df['privacidad_total_reduccion'].notna() & df['utilidad_total_cambio'].notna()]

# Normalizar privacidad y utilidad (más alto = mejor)
priv = df_plot['privacidad_total_reduccion'] / df_plot['privacidad_total_reduccion'].max()
util = 1 - (df_plot['utilidad_total_cambio'] / df_plot['utilidad_total_cambio'].max())
metodos = df_plot['Método']

plt.figure(figsize=(10,7))
plt.scatter(priv, util, s=120, color='royalblue')

# Etiquetas de cada método
for x, y, label in zip(priv, util, metodos):
    plt.text(x+0.01, y+0.01, label, fontsize=10)

plt.xlabel('Privacidad')
plt.ylabel('Utilidad')
plt.title('Comparativa de Métodos: Privacidad vs Utilidad')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('scatter_privacidad_utilidad.png', dpi=150)
plt.show()
print('Gráfica generada: scatter_privacidad_utilidad.png')
