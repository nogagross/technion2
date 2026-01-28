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

def plot_pca_weights(pca, df, pc_index, prefix=None, top_n=None):
    """
    Plot feature weights for a given principal component.

    pca: fitted PCA object
    df: original dataframe before PCA (numeric columns only)
    pc_index: index of the PC (0 for PC1, 1 for PC2, etc.)
    prefix: optional filter for column names (e.g., "b")
    top_n: optionally show only top N absolute weights
    """
    # Filter relevant columns if prefix is given
    cols = df.columns
    if prefix:
        cols = [c for c in cols if c.startswith(prefix) and 'lec' not in c and c != 'Subject_Code']

    # Get weights
    weights = pca.components_[pc_index]

    # Pair with column names
    feature_weights = list(zip(cols, weights))

    # Sort by absolute importance
    feature_weights.sort(key=lambda x: abs(x[1]), reverse=True)

    if top_n:
        feature_weights = feature_weights[:top_n]

    # Plot
    features, vals = zip(*feature_weights)
    plt.figure(figsize=(10, 8))
    plt.bar(features, vals)
    plt.xticks(rotation=30)
    plt.ylabel("Weight")
    plt.title(f"PC{pc_index+1} Feature Weights")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_pca_weights_separate_and_table(
    pcas,
    dfs,
    prefixes,
    titles=None,
    pc_index=0,
    top_n=15,
    figsize=(10, 6),
    colors=None,
    fontsize=12,
    save_csv_path=None,   # למשל: "top_pca_loadings_pc0.csv"
    sort_by_abs=True      # ממויין לפי ערך מוחלט בירידה
):
    """
    מצייר גרף נפרד לכל דאטאסט של ה-top_n loadings עבור ה-PC הנבחר,
    עם סקאלה Y אחידה בין כל הגרפים (מחושבת על סמך ה-top_n בכל דאטאסט),
    ומחזיר DataFrame עם ה-top_n לכל דאטאסט. אופציונלית שומר ל-CSV.

    Parameters
    ----------
    pcas : list[sklearn.decomposition.PCA]
    dfs : list[pd.DataFrame]
    prefixes : list[str]
        prefix לזיהוי הפיצ'רים הרלוונטיים בכל df.
    titles : list[str] | None
        שם/כותרת לכל דאטאסט (לכותרת הגרף ולעמודת "dataset" בטבלה).
    pc_index : int
        איזה רכיב PCA להציג (0 = PC1).
    top_n : int
        כמה פיצ'רים משמעותיים להציג/להחזיר לכל דאטאסט.
    figsize : tuple
        גודל פיגר לכל גרף.
    colors : list[str] | None
        צבע לכל דאטאסט; אם None ישתמש ב-tab colors.
    fontsize : int
        גודל פונט.
    save_csv_path : str | None
        אם לא None – שומר את ה-DataFrame לקובץ CSV.
    sort_by_abs : bool
        אם True  בוחרים את ה-top_n לפי ערך מוחלט של ה-loading.
    """
    FEATURE_MAPPING = {
    "GAD7_1": "felt nervous, anxious, or tense- GAD7_1",
    "GAD7_2": "Not being able to stop worry or control worrying- GAD7_2",
    "GAD7_3": "Worrying too much about different things- GAD7_3",
    "GAD7_4": "Trouble relaxing -GAD7_4",
    "GAD7_5": "Being so restless that it's hard to sit still-GAD7_5",
    "GAD7_6": "Becoming easily annoyed or irritable-GAD7_6",
    "GAD7_7": "Feeling afraid as if something awful might happen-GAD7_7",
    "PHQ_1": "Little interest or pleasure in doing things-PHQ_1",
    "PHQ_2": "Feeling down, depressed, or hopeless-PHQ_2",
    "PHQ_3": "Trouble falling or staying asleep, or sleeping too much-PHQ_3",
    "PHQ_4": "Feeling tired or having little energy-PHQ_4",
    "PHQ_5": "Poor appetite or overeating-PHQ_5",
    "PHQ_6": "Feeling bad about yourself, failure, or letting family down-PHQ_6",
    "PHQ_7": "Difficulty focusing on things like reading/TV-PHQ_7",
    "PHQ_8": "Changes in activity: Slowed speech/movement or restlessness/agitation-PHQ_8",
    "PHQ_9": "Thoughts of self-harm or death-PHQ_9"}    
    n = len(pcas)
    if colors is None:
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red","tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan"]
    if titles is None:
        titles = [f"Dataset {i+1}" for i in range(n)]

    # --- שלב 1: אסוף את ה-top_n לכל דאטאסט + מצא גבולות Y אחידים ---
    per_dataset_top = []   # נשמור כאן (title, Series של top loadings)
    all_vals = []

    for pca, df, prefix, title in zip(pcas, dfs, prefixes, titles):
        # בחירת פיצ'רים לפי prefix והחרגות כמו קודם
        features = [c for c in df.columns
                    if c.startswith(prefix) and 'lec' not in c and c != 'Subject_Code']
        comp = pca.components_[pc_index]
        # התאמה לאורך במקרה שיש הבדל (ליתר ביטחון)
        if len(comp) != len(features):
            # ננסה לקחת רק את מספר הפיצ'רים התואם לאורך comp לפי סדר הופעת features
            # (בדרך כלל לא נדרש, אבל מגן מקריסות)
            features = features[:len(comp)]

        loadings = pd.Series(comp[:len(features)], index=features)

        # מיון ובחירת top_n
        if sort_by_abs:
            ordered = loadings.reindex(loadings.abs().sort_values(ascending=False).index)
        else:
            ordered = loadings.sort_values(ascending=False)
        top_loadings = ordered.iloc[:top_n]

        per_dataset_top.append((title, top_loadings))
        all_vals.extend(top_loadings.values.tolist())

    # גבולות אחידים לכל הגרפים (על סמך ה-top_n מכל דאטאסט)
    y_min, y_max = min(all_vals), max(all_vals)
    pad = 0.05 * (y_max - y_min) if (y_max - y_min) > 0 else 0.05
    y_min, y_max = y_min - pad, y_max + pad

    # --- שלב 2: ציור גרף נפרד לכל דאטאסט ---
    for i, (title, top_loadings) in enumerate(per_dataset_top):
        fig, ax = plt.subplots(figsize=figsize)

        # map feature names to full question text
        mapped_labels = [
            FEATURE_MAPPING.get(f.split("_", 1)[-1], f)
            for f in top_loadings.index
        ]

        ax.bar(mapped_labels, top_loadings.values, color=colors[i % len(colors)])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel("Weight", fontsize=fontsize)
        ax.set_title(f"{title} — PC{pc_index+1} top {top_n} loadings", fontsize=fontsize+2)
        ax.set_xticks(range(len(top_loadings.index)))
        # ax.set_xticklabels(top_loadings.index, rotation=45, ha="right", fontsize=fontsize-1)
        ax.tick_params(axis='y', labelsize=fontsize-1)
        ax.grid(alpha=0.25)

        plt.tight_layout()
        plt.show()

    # --- שלב 3: בניית טבלת התוצאות המאוחדת והחזרת DataFrame ---
    rows = []
    for title, top_loadings in per_dataset_top:
        # נוסיף גם דירוג בתוך הדאטאסט (לפי abs)
        order = np.argsort(-np.abs(top_loadings.values))  # אינדקסים לפי abs יורד
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)

        for j, (feat, val) in enumerate(top_loadings.items()):
            rows.append({
                "dataset": title,
                "pc": pc_index + 1,
                "rank_within_dataset": int(ranks[j]),
                "feature": feat,
                "loading": float(val),
                "abs_loading": float(abs(val))
            })

    result_df = pd.DataFrame(rows).sort_values(
        by=["dataset", "abs_loading"], ascending=[True, False]
    ).reset_index(drop=True)

    if save_csv_path is not None:
        result_df.to_csv(save_csv_path, index=False)

    return result_df
import matplotlib.pyplot as plt
import pandas as pd

def plot_pca_weights_two_cols_split(
    pcas,
    dfs,
    prefixes,
    titles=None,
    pc_index=0,
    top_n=10,
    figsize=(16, 12),
    colors=None,
    fontsize=14
):
    """
    מצייר barplots אנכיים של PCA loadings בגריד 3x2:
    טור שמאלי: Pre, T1, T2
    טור ימני: T3, Post
    (המשבצת האחרונה ריקה)
    עם סקאלה אחידה בין כל הגרפים.
    """
    if colors is None:
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    # --- חישוב גבולות Y אחידים ---
    all_vals = []
    for pca, df, prefix in zip(pcas, dfs, prefixes):
        features = [c for c in df.columns if c.startswith(prefix) and 'lec' not in c and c != 'Subject_Code']
        comp = pca.components_[pc_index]
        loadings = pd.Series(comp, index=features)
        top_loadings = loadings.reindex(loadings.abs().sort_values(ascending=False).index)[:top_n]
        all_vals.extend(top_loadings.values)
    y_min, y_max = min(all_vals), max(all_vals)
    pad = 0.05 * (y_max - y_min)
    y_min, y_max = y_min - pad, y_max + pad

    # --- מיפוי ל-grid ---
    layout_map = {
        0: (0, 0),  # Pre  → שורה 1 טור שמאלי
        1: (1, 0),  # T1   → שורה 2 טור שמאלי
        2: (2, 0),  # T2   → שורה 3 טור שמאלי
        3: (0, 1),  # T3   → שורה 1 טור ימני
        4: (1, 1),  # Post → שורה 2 טור ימני
    }

    fig, axes = plt.subplots(3, 2, figsize=figsize)

    for i, (pca, df, prefix) in enumerate(zip(pcas, dfs, prefixes)):
        row, col = layout_map[i]
        ax = axes[row, col]

        features = [c for c in df.columns if c.startswith(prefix) and 'lec' not in c and c != 'Subject_Code']
        comp = pca.components_[pc_index]
        loadings = pd.Series(comp, index=features)
        top_loadings = loadings.reindex(loadings.abs().sort_values(ascending=False).index)[:top_n]

        ax.bar(top_loadings.index, top_loadings.values, color=colors[i % len(colors)])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel("Weight", fontsize=fontsize)
        ax.set_xticks(range(len(top_loadings.index)))
        ax.set_xticklabels(top_loadings.index, rotation=45, ha="right", fontsize=fontsize-2)
        ax.tick_params(axis='y', labelsize=fontsize-2)

        if titles is not None:
            ax.set_title(titles[i], fontsize=fontsize+2)

    # למחוק את התא האחרון הריק (שורה 3 טור ימין)
    fig.delaxes(axes[2, 1])

    plt.tight_layout()
    plt.show()



