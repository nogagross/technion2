import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import os


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# in save_top_loadings file (e.g. visualization_functions.py)
from clustring_functions import _cmap_for_dataset

def find_optimal_pca_dimensions(file_path: str, save_dir: str):
    """
    Loads a CSV file, performs PCA on numeric columns,
    and plots the cumulative explained variance.

    Args:
        file_path (str): Path to the input CSV file.
        save_dir (str): Directory where the PNG figure will be saved.

    Returns:
        (num90, num95, num100) or None on error.
    """
    # --- Load ---
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"⚠️ Error reading file {file_path}: {e}")
        return None

    # --- Feature selection/cleaning ---
    drop_like = {"subject_id", "subject", "subject_code", "euler", "tiv", "b_clusters", "group"}
    cols_to_drop = [c for c in df.columns if str(c).strip().lower() in drop_like]
    X = df.drop(columns=cols_to_drop, errors="ignore").select_dtypes(include="number")

    X = X.dropna(axis=1, how="all")
    if X.empty:
        print("⚠️ No numeric feature columns found after filtering.")
        return None

    X = X.fillna(X.mean(numeric_only=True))
    nunique = X.nunique(dropna=False)
    X = X.loc[:, nunique > 1]
    if X.shape[1] == 0:
        print("⚠️ All remaining columns have zero variance.")
        return None

    # --- PCA ---
    Z = StandardScaler().fit_transform(X)
    n_samples, n_features = Z.shape
    max_components = min(n_samples, n_features)
    if max_components < 1:
        print("⚠️ Not enough data for PCA.")
        return None

    pca = PCA(n_components=max_components, svd_solver='full').fit(Z)
    cumulative_var = pca.explained_variance_ratio_.cumsum() * 100

    # --- Prepare save path (fig name based on CSV name to avoid overwriting) ---
    os.makedirs(save_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(file_path))[0]
    out_path = os.path.join(save_dir, f"{stem}_pca_cumulative_variance.png")

    # --- Plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_components + 1), cumulative_var, marker='o')
    plt.axhline(y=90, linestyle='--', label='90% threshold')
    plt.axhline(y=95, linestyle='--', label='95% threshold')
    plt.title("Cumulative Explained Variance")
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Cumulative Explained Variance (%)")
    if max_components <= 20:
        plt.xticks(range(1, max_components + 1))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_path, dpi=150)
    plt.show()
    plt.close()

    num90 = next((i + 1 for i, v in enumerate(cumulative_var) if v >= 90), None)
    num95 = next((i + 1 for i, v in enumerate(cumulative_var) if v >= 95), None)
    num100 = max_components

    if num90:
        print(f"✅ 90% variance: {num90} components")
        print(f"✅ 95% variance: {num95} components")
    print(f"✅ 100% variance: {num100} components")
    print(f"🖼️ Saved figure: {out_path}")

    return num90, num95, num100




# Define the PCA function
def perform_pca(data_df, n_components):
    """
    Performs Principal Component Analysis on a DataFrame.

    Args:
        data_df (pd.DataFrame): The DataFrame to perform PCA on.
        n_components (int): The number of principal components to return.

    Returns:
        tuple: A tuple containing:
               - pd.DataFrame: The transformed data with principal components.
               - PCA: The fitted PCA model object.
    """
    # Exclude non-numeric columns (like 'group') from the analysis
    # Drop the first unnamed column that might be created when saving/loading the CSVs
    if 'group' in data_df.columns:
        numeric_data = data_df.drop('group', axis=1).select_dtypes(include=['number'])
    else:
        numeric_data = data_df.select_dtypes(include=['number'])

    # Standardize the data (important for PCA)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)

    # Perform PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)

    # Create a new DataFrame with the principal components
    column_names = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(data=principal_components, columns=column_names, index=data_df.index)

    # Add the 'group' column back for analysis if it existed
    if 'group' in data_df.columns:
        pca_df['group'] = data_df['group']

    return pca_df, pca

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def save_top_loadings(
    pca,
    feature_names,
    pc_index=0,
    top_n=10,
    out_csv=None,
    dataset_name=None,   # VCSF/VGMM/VWM (לקביעת פלטת צבעים)
    cmap=None,           # לעקוף cmap (למשל 'Blues')
    show_values=False,   # לרשום ערכים מעל העמודות
    save_dir=None,       # ← חדש: תיקייה לשמירת ה-PNG (כמו ב-find_optimal_pca_dimensions)
    source_path=None     # ← חדש: נתיב קובץ המקור להפקת שם חכם לתמונה
):
    """
    Plot top-N loadings for a given PC, save figure (if save_dir is given),
    and optionally save a CSV with the loadings.

    If save_dir is provided, the figure is saved to:
        <save_dir>/<stem>_pc<pc_index+1>_top<top_n>_loadings.png
    where <stem> is derived from source_path (if given), else out_csv, else dataset_name.
    """
    if pc_index >= pca.n_components_:
        print(f"Warning: PC index {pc_index+1} is out of bounds for a model with {pca.n_components_} components.")
        return

    # --- pick top-N by |loading| ---
    weights = pca.components_[pc_index, :]
    order = np.argsort(np.abs(weights))[::-1][:top_n]
    top_features = [(feature_names[i], float(weights[i])) for i in order]

    # --- colors (dataset-based) ---
    cmap_use = _cmap_for_dataset(dataset_name, cmap)
    pos_col = cmap_use(0.80)
    neg_col = cmap_use(0.45)
    bar_colors = [pos_col if weights[i] >= 0 else neg_col for i in order]

    # --- plot ---
    plt.figure(figsize=(10, 6))
    x = np.arange(len(order))
    vals = [weights[i] for i in order]
    bars = plt.bar(x, vals, color=bar_colors)
    plt.axhline(0, color='black', linewidth=1)
    plt.xticks(x, [feature_names[i] for i in order], rotation=75, ha='right')
    plt.ylabel("Loading weight")
    title_ds = f" {dataset_name}" if dataset_name else ""
    plt.title(f"Top {top_n} loadings for PC{pc_index+1}{title_ds}")
    plt.tight_layout()

    if show_values:
        for rect, v in zip(bars, vals):
            y = rect.get_height()
            offset = 0.01 if y >= 0 else -0.01
            plt.text(rect.get_x() + rect.get_width()/2, y + offset,
                     f"{v:+.3f}", ha='center', va='bottom' if y>=0 else 'top', fontsize=8)

    # --- save figure (same style as before) ---
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        # figure stem from source_path -> out_csv -> dataset_name -> generic
        if source_path:
            stem = os.path.splitext(os.path.basename(source_path))[0]
        elif out_csv:
            stem = os.path.splitext(os.path.basename(out_csv))[0]
        elif dataset_name:
            stem = str(dataset_name).lower()
        else:
            stem = "loadings"
        fig_path = os.path.join(save_dir, f"{stem}_pc{pc_index+1}_top{top_n}_loadings.png")
        plt.savefig(fig_path, dpi=150)
        print(f"🖼️ Saved figure: {fig_path}")

    # show on screen (like before)
    plt.show()
    plt.close()

    # --- optional CSV save ---
    if out_csv is not None:
        df = pd.DataFrame({
            "feature": [f for f, _ in top_features],
            f"PC{pc_index+1}_loading": [v for _, v in top_features]
        })
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        df.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"✅ Saved loadings to {out_csv}")

    # print to console
    print(f"\nTop {top_n} loadings for PC{pc_index+1}{title_ds}:")
    for feat, val in top_features:
        print(f"{feat:30s} {val:+.4f}")

    return top_features

