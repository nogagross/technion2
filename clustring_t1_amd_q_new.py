import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# ==== your external helpers (unchanged) ====
from pca_functions import (
    find_optimal_pca_dimensions,  # (csv_path, fig_dir) -> (num90, num95, num100)
    perform_pca,                  # (df, n_components) -> (scores_df, pca_model)
    save_top_loadings             # (pca, feature_names, pc_index, out_csv, dataset_name)
)
from clustring_functions import (
    find_optimal_k_and_cluster,   # (pca_df) -> {"k_values":[], "silhouette_scores":[], "best_k":K}
    run_kmeans_on_pca_data        # (pca_df, k, title, csv_path, dataset_name, random_state)
)
from vizualizations_functions import plot_grouped_bars  # optional bar chart

# --------------------------
# CONFIG — EDIT THESE
# --------------------------
SEED = 17
np.random.seed(SEED)

# path to your big dataset
BIG_CSV = "ttest_yeo/fmri_T1_clinical_merged_updated.csv"

# name of subject column & eTIV column inside BIG_CSV
# subject col will be autodetected if None
SUBJECT_COL = "Subject_Code"     # e.g., "Subject_Code" or "subject_id" or None to auto
ETIV_COL    = "eTIV"     # e.g., "eTIV" or "TIV" or None to auto

# features you want to use (names as they appear in BIG_CSV)
# paste your list here; missing ones will be dropped & reported
FEATURES = [
    "bankssts_lh", "bankssts_rh", "caudalanteriorcingulate_lh",
    "caudalanteriorcingulate_rh", "caudalmiddlefrontal_lh", "caudalmiddlefrontal_rh",
    "cuneus_lh", "cuneus_rh", "entorhinal_lh", "entorhinal_rh",
    "frontalpole_lh", "frontalpole_rh", "fusiform_lh", "fusiform_rh",
    "inferiorparietal_lh", "inferiorparietal_rh", "inferiortemporal_lh",
    "inferiortemporal_rh", "insula_lh", "insula_rh",
    "isthmuscingulate_lh", "isthmuscingulate_rh",
    "lateraloccipital_lh", "lateraloccipital_rh",
    "lateralorbitofrontal_lh", "lateralorbitofrontal_rh",
    "lingual_lh", "lingual_rh",
    "medialorbitofrontal_lh", "medialorbitofrontal_rh",
    "middletemporal_lh", "middletemporal_rh",
    "paracentral_lh", "paracentral_rh",
    "parahippocampal_lh", "parahippocampal_rh",
    "parsopercularis_lh", "parsopercularis_rh",
    "parsorbitalis_lh", "parsorbitalis_rh",
    "parstriangularis_lh", "parstriangularis_rh",
    "pericalcarine_lh", "pericalcarine_rh",
    "postcentral_lh", "postcentral_rh",
    "posteriorcingulate_lh", "posteriorcingulate_rh",
    "precentral_lh", "precentral_rh",
    "precuneus_lh", "precuneus_rh",
    "rostralanteriorcingulate_lh", "rostralanteriorcingulate_rh",
    "rostralmiddlefrontal_lh", "rostralmiddlefrontal_rh",
    "superiorfrontal_lh", "superiorfrontal_rh",
    "superiorparietal_lh", "superiorparietal_rh",
    "superiortemporal_lh", "superiortemporal_rh",
    "supramarginal_lh", "supramarginal_rh",
    "temporalpole_lh", "temporalpole_rh",
    "transversetemporal_lh", "transversetemporal_rh"
]


# outputs
OUT_DIR          = Path("clusteres_output/single_dataset")
OUT_MATRIX_CSV   = OUT_DIR / "features_tivnorm_ses1.csv"       # subjects×features (normalized)
OUT_CLUSTERS_CSV = OUT_DIR / "clusters_ses1.csv"
OUT_LOADINGS_CSV = OUT_DIR / "pc1_loadings_ses1.csv"
FIG_DIR          = Path("figures/single_dataset")              # for explained variance figs

DATASET_NAME = "BIG"   # a short label for legends/titles/files

# --------------------------
# helpers
# --------------------------
def _auto_subject_col(df: pd.DataFrame) -> str:
    cands = ["Subject_Code", "subject_id", "subject", "id", "participant", "case"]
    low = {c.lower(): c for c in df.columns}
    for k in cands:
        if k.lower() in low:
            return low[k.lower()]
    # fallback: first non-numeric column
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            return c
    # if all numeric, force first col
    return df.columns[0]

def _auto_etiv_col(df: pd.DataFrame) -> str:
    cands = ["eTIV", "etiv", "TIV", "tiv", "ETIV"]
    low = {c.lower(): c for c in df.columns}
    for k in cands:
        if k.lower() in low:
            return low[k.lower()]
    raise ValueError("Could not find an eTIV/TIV column. Set ETIV_COL to the correct name.")

def _safe_feature_list(df: pd.DataFrame, wanted: list[str]) -> list[str]:
    """Return features that exist; print the ones that don't."""
    present = [f for f in wanted if f in df.columns]
    missing = [f for f in wanted if f not in df.columns]
    if missing:
        print(f"⚠️ Missing {len(missing)} columns (will ignore): {missing[:10]}{' …' if len(missing)>10 else ''}")
    if not present:
        raise ValueError("None of the requested FEATURES were found in the file.")
    return present

# --------------------------
# main
# --------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(BIG_CSV)
    subj_col = SUBJECT_COL or _auto_subject_col(df)
    etiv_col = ETIV_COL or _auto_etiv_col(df)

    print(f"Using subject column: {subj_col}")
    print(f"Using eTIV column   : {etiv_col}")

    # choose features
    feats = _safe_feature_list(df, FEATURES)

    # keep only subject + eTIV + features
    keep_cols = [subj_col, etiv_col] + feats
    df = df[keep_cols].copy()

    # drop rows without eTIV/TIV (or zero)
    df[etiv_col] = pd.to_numeric(df[etiv_col], errors="coerce")
    df = df.dropna(subset=[etiv_col])
    df = df.loc[df[etiv_col] != 0]

    # convert features to numeric and divide by eTIV
    for c in feats:
        df[c] = pd.to_numeric(df[c], errors="coerce") / df[etiv_col]

    # build matrix: subjects×features
    mat = df[[subj_col] + feats].dropna().drop_duplicates(subset=[subj_col])
    mat = mat.set_index(subj_col)
    mat.index.name = "Subject_Code"  # standardize for downstream utils
    mat.to_csv(OUT_MATRIX_CSV)
    print(f"✅ Saved normalized matrix: {OUT_MATRIX_CSV}  (n={len(mat)} subjects, d={len(feats)} features)")

    # 1) Find optimal PCA dimensionalities (90/95/100%)
    num90, num95, num100 = find_optimal_pca_dimensions(str(OUT_MATRIX_CSV), str(FIG_DIR))
    print(f"PCA dims for variance thresholds — 90%:{num90}, 95%:{num95}, 100%:{num100}")

    # 2) Run PCA with 90% (change if you want)
    scores_df, pca_model = perform_pca(mat, n_components=num90)
    print(f"PCA scores shape: {scores_df.shape}")

    # 3) Silhouette search for best K on the PCA scores
    results = find_optimal_k_and_cluster(pca_df=scores_df)
    print(f"Best K = {results['best_k']}  (searched {results['k_values']})")

    # silhouette curve (single dataset)
    plt.figure(figsize=(7,5))
    plt.plot(results['k_values'], results['silhouette_scores'], marker='o')
    best_idx = results['k_values'].index(results['best_k'])
    plt.plot(results['best_k'], results['silhouette_scores'][best_idx], marker='*', ms=14)
    plt.title(f"Silhouette vs K — {DATASET_NAME} (ses1)")
    plt.xlabel("K"); plt.ylabel("Silhouette score"); plt.grid(True, ls="--")
    plt.tight_layout(); plt.show()

    # 4) Final K-means with best K
    labels, kmodel, pca_clustered = run_kmeans_on_pca_data(
        scores_df,
        k=results['best_k'],
        title=f"K-Means on PCA ({DATASET_NAME}, ses1)",
        csv_path=str(OUT_CLUSTERS_CSV),
        dataset_name=DATASET_NAME,
        random_state=SEED,
    )
    print(f"✅ Saved clusters: {OUT_CLUSTERS_CSV}")

    # 5) PC1 loadings
    save_top_loadings(
        pca=pca_model,
        feature_names=list(mat.columns),
        pc_index=0,
        out_csv=str(OUT_LOADINGS_CSV),
        dataset_name=DATASET_NAME
    )
    print(f"✅ Saved PC1 loadings: {OUT_LOADINGS_CSV}")

    # 6) Optional: a small bar chart of chosen PCA dims (1 dataset only)
    plot_grouped_bars(
        groups=[DATASET_NAME],
        s1=[num90], s2=[num95], s3=[num100],
        s1_label="90% var", s2_label="95% var", s3_label="100% var",
        title="PCA dimensions — session 1",
        ylabel="# PCs"
    )

if __name__ == "__main__":
    main()
