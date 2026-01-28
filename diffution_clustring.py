import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from clustring_functions import (find_optimal_k_and_cluster, _palette_from_dataset, run_kmeans_on_pca_data,
                                     _cmap_for_dataset,build_cluster_summary)
import os, random, numpy as np

import re



def main():
    SEED = 42

    os.environ["PYTHONHASHSEED"] = str(SEED)  # hashes (rarely matters)
    random.seed(SEED)  # Python’s RNG
    np.random.seed(SEED)  # NumPy RNG

    pattern = r"^DC\d+$"
    diff_map_ses1 = pd.read_csv("data/SCHAEFER_mat_cor/csv_out/diffusion_map_ses1_labeled.csv",index_col=0)
    comp_cols = [c for c in diff_map_ses1.columns if re.match(pattern, str(c))]
    diff_map_ses1= diff_map_ses1.loc[:, comp_cols[:12]]
    # --- קריאה לפונקציה ושמירת תוצאות ---
    results_diff_map_ses1 = find_optimal_k_and_cluster(pca_df=diff_map_ses1,random_state=SEED)

    # --- Create the combined plot ---
    plt.figure(figsize=(10, 7))

    # קווי הסילואט (ללא שינוי משמעותי)
    plt.plot(results_diff_map_ses1['k_values'], results_diff_map_ses1['silhouette_scores'], marker='o', color='blue')

    # # --- סימון k האופטימלי ---
    best_k_diff_map_score_ses1 = results_diff_map_ses1['silhouette_scores'][
        results_diff_map_ses1['k_values'].index(results_diff_map_ses1['best_k'])]


    star_diff_map_ses1, = plt.plot(results_diff_map_ses1['best_k'], best_k_diff_map_score_ses1 ,
                               marker='*', markersize=15, color='blue',
                               label=f'VCSF k: {results_diff_map_ses1["best_k"]}')

    plt.title('Combined Silhouette Scores for VCSF, VGMM, and VWM Datasets - session 1')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.xticks(range(2, 11))
    plt.grid(True, linestyle='--')

    # מקרא רק עם שלושת הכוכבים
    plt.legend(handles=[star_diff_map_ses1])
    plt.show()

    # ###############################################################
    #
    print("Running K-Means on VCSF PCA data...")
    diff_map_labels_ses1, diff_map_kmeans_ses1, diff_map_pca_clustered_ses1 = run_kmeans_on_pca_data(
        diff_map_ses1,
        k=results_diff_map_ses1["best_k"],
        title="diff_map_clustring_ses1",
        csv_path="data/SCHAEFER_mat_cor/csv_out/diff_map_clusters_ses1.csv",
        dataset_name="connectivity",random_state=SEED
    )

    diff_map_ses2 = pd.read_csv("data/SCHAEFER_mat_cor/csv_out/diffusion_map_ses2_labeled.csv",index_col=0)
    comp_cols = [c for c in diff_map_ses2.columns if re.match(pattern, str(c))]
    diff_map_ses2= diff_map_ses2.loc[:, comp_cols[:18]]
    # --- קריאה לפונקציה ושמירת תוצאות ---
    results_diff_map_ses2 = find_optimal_k_and_cluster(pca_df=diff_map_ses2,random_state=SEED)


    # --- Create the combined plot ---
    plt.figure(figsize=(10, 7))

    # קווי הסילואט (ללא שינוי משמעותי)
    plt.plot(results_diff_map_ses2['k_values'], results_diff_map_ses2['silhouette_scores'], marker='o', color='blue')

    # # --- סימון k האופטימלי ---
    best_k_diff_map_score_ses2 = results_diff_map_ses2['silhouette_scores'][
        results_diff_map_ses2['k_values'].index(results_diff_map_ses2['best_k'])]


    star_diff_map_ses2, = plt.plot(results_diff_map_ses2['best_k'], best_k_diff_map_score_ses2 ,
                               marker='*', markersize=15, color='blue',
                               label=f'VCSF k: {results_diff_map_ses2["best_k"]}')

    plt.title('Combined Silhouette Scores for VCSF, VGMM, and VWM Datasets - session 2')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.xticks(range(2, 11))
    plt.grid(True, linestyle='--')

    # מקרא רק עם שלושת הכוכבים
    plt.legend(handles=[star_diff_map_ses2])
    plt.show()

    # ###############################################################
    # # session 2
    #
    print("Running K-Means on VCSF PCA data...")
    diff_map_labels_ses2, diff_map_kmeans_ses2, diff_map_pca_clustered_ses2 = run_kmeans_on_pca_data(
        diff_map_ses2,
        k=results_diff_map_ses2["best_k"],
        csv_path="data/SCHAEFER_mat_cor/csv_out/diff_map_clusters_ses2.csv",
        title="diff_map_clustring_ses2",
        dataset_name="connectivity",random_state=SEED
    )


if __name__ == "__main__":
    main()
