#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from scipy.stats import ttest_ind
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

def differential_expression(df, control_samples, problem_samples):
    """
    Performs differential expression analysis between two groups.

    Parameters:
    - df: DataFrame with samples in rows and genes in columns
    - control_samples: list of control sample names
    - problem_samples: list of sample names to compare against controls

    Returns:
    - df_results: DataFrame with p-values, FDR, and log2FC per gene
    """

    # Validate that all specified samples exist in the DataFrame
    all_samples = control_samples + problem_samples
    missing = [s for s in all_samples if s not in df.index]
    if missing:
        raise ValueError(f"The following samples are missing from the DataFrame: {missing}")

    results = []

    for gene in df.columns:
        # Extract expression values for the current gene
        g1 = df.loc[control_samples, gene].dropna()
        g2 = df.loc[problem_samples, gene].dropna()

        # Skip genes with insufficient data
        if len(g1) < 2 or len(g2) < 2:
            continue

        # Perform Welch's t-test (unequal variances)
        stat, pval = ttest_ind(g1, g2, equal_var=False)

        # Compute mean expression for each group
        mean_control = g1.mean()
        mean_problem = g2.mean()

        # Compute log2 fold change, avoiding division by zero
        if mean_control == 0:
            logfc = np.nan
        else:
            logfc = np.log2((mean_problem + 1e-8) / (mean_control + 1e-8))

        results.append((gene, pval, logfc))

    if not results:
        raise ValueError("Not enough data to perform the analysis.")

    # Create result DataFrame
    df_results = pd.DataFrame(results, columns=["ID", "pval", "log2FC"]).set_index("ID")

    # Adjust p-values for multiple testing using Benjamini-Hochberg FDR
    df_results["FDR"] = multipletests(df_results["pval"], method="fdr_bh")[1]

    # Return sorted results by raw p-value
    return df_results.sort_values("pval")


def significative_genes(df, FDR, log2FC):
    """
    Identifies significantly differentially expressed genes.

    Parameters:
    - df: DataFrame with gene names and values for pval and log2FC
    - FDR: critical false discovery rate threshold
    - log2FC: critical absolute log2 fold change threshold

    Returns:
    - Filtered DataFrame with significant genes based on FDR and log2FC cutoffs
    """
    return df[(df['FDR'] < FDR) & (abs(df['log2FC']) > log2FC)]

