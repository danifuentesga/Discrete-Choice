# -*- coding: utf-8 -*-
"""
ELECCION DISCRETA

TAREA 2

Autor: Daniel Fuentes García
Profesor: Edwin Muñoz
Laboratorista: Adair Hernández
GitHub: https://github.com/danifuentesga
Fecha: 01 de febrero de 2026


...Código arcaico, pero Wingardium funciona....


"""
#%% PROBLEMA 1 A)

# Construcción de la base de datos.
# Como hicimos en clase (aunque a mí me faltó hacerlo),
# voy a crear un DataFrame con las variables que necesito.
# Es decir: individuo, ingreso, u_total, v_obs, etc.
# La idea es que las estimaciones que vaya haciendo se guarden
# en la base de datos y al final tener una base bien estructurada.

import pandas as pd
import numpy as np


# Crear lista de ingresos
ingresos = [500] * 50 + [4000] * 50 + [10000] * 50


# Crear una base con 150 individuos
base = pd.DataFrame(
    {
        "individuo": range(1, 151),
        "ingreso": ingresos,
    }
)


# Creo las variables que quiero en mi base, por ahora vacías

base["alternativa"] = np.nan
base["precio"] = np.nan
base["v_obs"] = np.nan
base["e_no_obs"] = np.nan
base["u_total"] = np.nan


# Cada fila se repite cuatro veces porque tengo 4 alternativas
base = base.loc[base.index.repeat(4)].reset_index(drop=True)


# Alternativa es j = {1, 2, 3, 4}
# Los precios son 100, 200, 300 y 400
base["alternativa"] = [1, 2, 3, 4] * 150
base["precio"] = [100, 200, 300, 400] * 150


# Checamos
print(base)

# Calculamos la utilidad observada con la fórmula que nos dan

base["v_obs"] = 3 + 2 * np.log(base["ingreso"] - base["precio"])


# Checamos
print(base)


# Revisamos cualquier individuo para verificar que estamos bien
# Ejemplo: individuo número 70

print(base[base["individuo"] == 70])

# Ahora hago el proceso de pseudo random number generation.
# Como vimos en clase, uso el comando moderno de NumPy.
# Si lo entiendo bien, esto sería solo un episodio.

rng = np.random.default_rng(12345)

# De acuerdo con las slides: loc = miu = 0 y scale = beta = 1
base["e_no_obs"] = rng.gumbel(loc=0, scale=1, size=len(base))

# Utilidad total = utilidad observada + utilidad no observada
base["u_total"] = base["v_obs"] + base["e_no_obs"]

# Checamos rápido (para no imprimir todo el DataFrame)
print(base.head(12))

# Guardamos la alternativa de mayor u_total para cada individuo
elecciones = base.loc[base.groupby("individuo")["u_total"].idxmax()]

print(elecciones.head(10))

# Calculo la demanda por alternativa: contar cuántos eligieron 1, 2, 3, 4
demanda = elecciones["alternativa"].value_counts().sort_index()
print(demanda)

# La suma de la demanda debe ser 150, pues solo hay 150 personas
print(demanda.sum())

# Entonces, para el episodio 1:
# Simulo las elecciones de cada consumidor generando una realización
# de la utilidad no observada, y la alternativa elegida es la de mayor
# utilidad total.


# Para 10,000 episodios hago lo mismo dentro de un ciclo for.

# Aquí guardaremos la demanda de cada alternativa en cada episodio
# (cada episodio produce 4 números: demanda de j = 1, 2, 3, 4)
resultados = []

# Número de episodios
E = 10000

for e in range(E):
    # Igual que antes, generamos la parte no observada de la utilidad
    base["e_no_obs"] = rng.gumbel(loc=0, scale=1, size=len(base))

    # Suma de la utilidad observada y no observada
    base["u_total"] = base["v_obs"] + base["e_no_obs"]

    # Elegimos la alternativa con mayor utilidad total por individuo
    elecciones = base.loc[base.groupby("individuo")["u_total"].idxmax()]

    # Contamos la demanda por alternativa
    demanda = elecciones["alternativa"].value_counts().sort_index()

    # Asegurar que siempre haya 4 alternativas (si falta alguna, poner 0)
    demanda = demanda.reindex([1, 2, 3, 4], fill_value=0)

    # Guardar como lista: [D1, D2, D3, D4]
    resultados.append(demanda.tolist())


# Convertir a DataFrame: cada fila es un episodio
demanda_episodios = pd.DataFrame(
    resultados,
    columns=["D1", "D2", "D3", "D4"],
)

print(demanda_episodios.head())


# Graficas de Distibución de Demanda

import matplotlib.pyplot as plt


# Histogramas por alternativa (uno por uno)

# Colores pastel suaves
pasteles = {
    "D1": (0.65, 0.80, 0.90),  # azul pastel
    "D2": (0.99, 0.80, 0.60),  # naranja pastel
    "D3": (0.70, 0.87, 0.70),  # verde pastel
    "D4": (0.98, 0.70, 0.70),  # rojo/rosa pastel
}

for col in ["D1", "D2", "D3", "D4"]:
    plt.figure()
    plt.hist(
        demanda_episodios[col],
        bins=30,
        color=pasteles[col],
        alpha=0.8,
        label=f"Alt {col[-1]}"
    )

    plt.title(f"Distribución de la demanda - Alternativa {col[-1]}", fontsize=16)
    plt.xlabel("Demanda", fontsize=16)
    plt.ylabel("Frecuencia", fontsize=16)
    plt.xlim(0, 100)

    plt.legend(fontsize=12, frameon=False)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    ax = plt.gca()
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(prune='lower'))
    ax.yaxis.set_major_locator(MaxNLocator(prune='lower'))

    plt.show()



# Todas las alternativas en una sola figura
plt.figure(figsize=(10, 6))

plt.hist(demanda_episodios["D1"], bins=30, alpha=0.5, label="Alt 1")
plt.hist(demanda_episodios["D2"], bins=30, alpha=0.5, label="Alt 2")
plt.hist(demanda_episodios["D3"], bins=30, alpha=0.5, label="Alt 3")
plt.hist(demanda_episodios["D4"], bins=30, alpha=0.5, label="Alt 4")

plt.title("Distribución de la demanda por alternativa (10,000 episodios)", fontsize=18)
plt.xlabel("Demanda", fontsize=16)
plt.ylabel("Frecuencia", fontsize=16)

plt.legend(fontsize=12, frameon=False)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.show()



#Promedio y DE de alernativa

promedio = demanda_episodios.mean()
desv_std = demanda_episodios.std()

print("Promedio")
print(promedio)

print("DE")
print(desv_std)

#%% PROBLEMA 1 B)

# Debo usar formula del logit condicional 
#    P(j|n) = exp(v_jn) / sum_k exp(v_kn)

#Creo las vairables nuevas que voy a necesitar

#Nuevas variables

base["exp_v"] = np.nan        # e**v_obs
base["p_logit"] = np.nan      # resultado de dividir exp_v entre suma_exp_v
base["suma_exp_v"] = np.nan   # suma de exp_v para un indivduo

print(base)

#con la v_obs que ya tenemos hacemos e**v_obs

base["exp_v"] = np.exp(base["v_obs"])

print(base)

# Ahora para cada inviduo calculamos la suma de sus exp_v

base["suma_exp_v"] = (
    base.groupby("individuo")["exp_v"].transform("sum")
)

print(base)

# Calcular la probabilidad logit
base["p_logit"] = base["exp_v"] / base["suma_exp_v"]

# Creo variable elecciones plogit donde se guarda cada eleccion

elecciones_plogit = (
    base
    .groupby("individuo")
    .sample(n=1, weights="p_logit", random_state=rng)
)

print(base)

#Calculo demandas

demanda_plogit = elecciones_plogit["alternativa"].value_counts().sort_index()

#Ahora para 10000 episodios

resultados_plogit = []

E = 10000

for e in range(E):

    # Elección logit por individuo
    elecciones_plogit = (
        base
        .groupby("individuo")
        .sample(n=1, weights="p_logit", random_state=rng)
    )

    # Demanda por alternativa
    demanda_plogit = (
        elecciones_plogit["alternativa"]
        .value_counts()
        .sort_index()
        .reindex([1, 2, 3, 4], fill_value=0)
    )

    # Guardar [D1, D2, D3, D4]
    resultados_plogit.append(demanda_plogit.tolist())


# Convertir a DataFrame: cada fila = un episodio
demanda_episodios_plogit = pd.DataFrame(
    resultados_plogit,
    columns=["D1", "D2", "D3", "D4"]
)

print(demanda_episodios_plogit)


#Graficos de demanda gumbel vs plogit

# Colores
color_gumbel = (0.65, 0.80, 0.90)  # azul pastel
color_logit = (0.99, 0.80, 0.60)   # naranja pastel

for j in [1, 2, 3, 4]:
    col = f"D{j}"

    plt.figure(figsize=(8, 5))

    # Histograma Gumbel
    plt.hist(
        demanda_episodios[col],
        bins=30,
        alpha=0.6,
        color=color_gumbel,
        label="Gumbel",
    )

    # Histograma Logit
    plt.hist(
        demanda_episodios_plogit[col],
        bins=30,
        alpha=0.6,
        color=color_logit,
        label="Logit",
    )

    plt.title(f"Comparación de demanda simulada - Alternativa {j}", fontsize=16)
    plt.xlabel("Demanda", fontsize=16)
    plt.ylabel("Frecuencia", fontsize=16)
    plt.xlim(0, 100)

    # Label box pequeño y sin marco
    plt.legend(fontsize=12, frameon=False)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    ax = plt.gca()
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(prune='lower'))
    ax.yaxis.set_major_locator(MaxNLocator(prune='lower'))


    plt.show()

    
# Grafica de todas juntas

# Colores por alternativa (pasteles suaves)
pasteles = {
    "D1": (0.65, 0.80, 0.90),  # azul
    "D2": (0.99, 0.80, 0.60),  # naranja
    "D3": (0.70, 0.87, 0.70),  # verde
    "D4": (0.98, 0.70, 0.70),  # rojo
}

plt.figure(figsize=(12, 7))

for col in ["D1", "D2", "D3", "D4"]:
    # Gumbel (relleno)
    plt.hist(
        demanda_episodios[col],
        bins=30,
        alpha=0.35,
        color=pasteles[col],
        label=f"Gumbel {col[-1]}",
    )

    # Logit (línea punteada)
    plt.hist(
        demanda_episodios_plogit[col],
        bins=30,
        histtype="step",
        linewidth=3,
        linestyle="--",
        color=pasteles[col],
        label=f"Logit {col[-1]}",
    )

# Leyenda GRANDE y sin marco (label box)
plt.legend(fontsize=12, ncol=2, frameon=False)

plt.xlabel("Demanda", fontsize=18)
plt.ylabel("Frecuencia", fontsize=18)
plt.title(
    "Distribución de la demanda: Gumbel vs Logit (todas las alternativas)",
    fontsize=20
)
plt.xlim(0, 100)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

ax = plt.gca()
from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(prune='lower'))
ax.yaxis.set_major_locator(MaxNLocator(prune='lower'))


plt.show()

#Promedio y DE de alernativa

promedio_plogit = demanda_episodios_plogit.mean()
desv_std_plogit = demanda_episodios_plogit.std()

print("Promedio")
print(promedio_plogit)

print("DE")
print(desv_std_plogit)

#%% PROBLEMA 2 A)

# p_logit ya está en la base (por individuo y alternativa)
demanda_esperada = base.groupby("alternativa")["p_logit"].sum()
print(demanda_esperada)













































