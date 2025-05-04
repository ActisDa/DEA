#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from scipy.stats import ttest_ind
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

def differential_expression(df, control_samples, problem_samples):
    """
    Realiza análisis de expresión diferencial entre dos grupos.

    Parámetros:
    - df: DataFrame con muestras en filas y genes en columnas
    - control_samples: lista de nombres de muestras control
    - problem_samples: lista de nombres de muestras a comparar vs control

    Retorna:
    - df_results: DataFrame con p-valores, FDR y log2FC por gen
    """

    # Validar que todas las muestras existan en el índice
    all_samples = control_samples + problem_samples
    missing = [s for s in all_samples if s not in df.index]
    if missing:
        raise ValueError(f"Las siguientes muestras no están en el DataFrame: {missing}")

    results = []

    for gene in df.columns:
        g1 = df.loc[control_samples, gene].dropna()
        g2 = df.loc[problem_samples, gene].dropna()

        if len(g1) < 2 or len(g2) < 2:
            continue  # Saltamos genes con datos insuficientes

        # Estadística t
        stat, pval = ttest_ind(g1, g2, equal_var=False)

        # Log2 Fold Change
        mean_control = g1.mean()
        mean_problem = g2.mean()

        # Prevenir división por cero
        if mean_control == 0:
            logfc = np.nan
        else:
            logfc = np.log2((mean_problem + 1e-8) / (mean_control + 1e-8))

        results.append((gene, pval, logfc))

    if not results:
        raise ValueError("No hay suficientes datos para realizar el análisis.")

    # Crear DataFrame
    df_results = pd.DataFrame(results, columns=["ID", "pval", "log2FC"]).set_index("ID")

    # Corrección por FDR (Benjamini-Hochberg)
    df_results["FDR"] = multipletests(df_results["pval"], method="fdr_bh")[1]

    return df_results.sort_values("pval")



def significative_genes(df, FDR, log2FC):
    """
    Detecta genes con expresión diferencial significativa.
    
    Parámetros:
    - df: DataFrame con nombre de los genes y valores de pval y log2FC.
    - pval: valor de pval crítico.
    - log2FC: valor de log2FC crítico.
    
    Retorna:
    - dataframe filtrada en base a los valores de pval y log2FC críticos.
    """
    return df[(df['FDR'] < FDR) & (abs(df['log2FC']) > log2FC)]

