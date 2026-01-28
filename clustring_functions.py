# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
def relabel_clusters_by_pc1(data_pca, labels):
    """
    Relabel clusters so that:
    - cluster 0 = leftmost (smallest mean PC1)
    - cluster numbers increase from left to right
    """
    labels = np.asarray(labels)

    unique_labels = np.unique(labels)
    centroids_pc1 = {}

    for lab in unique_labels:
        centroids_pc1[lab] = data_pca[labels == lab, 0].mean()

    # sort clusters by PC1 centroid
    ordered = sorted(centroids_pc1, key=centroids_pc1.get)

    mapping = {old: new for new, old in enumerate(ordered)}
    new_labels = np.vectorize(mapping.get)(labels)

    return new_labels, mapping

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import os
# (ודא שכל שאר הייבואים שלך נמצאים בראש הקובץ)

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import os  # נדרש עבור בדיקת הנתיב ב-pc1_csv_path


from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import pairwise_distances


def relabel_clusters_by_pc1(data_pca, labels):
    """
    Relabel clusters so that:
    - cluster 0 = leftmost (smallest mean PC1)
    - cluster numbers increase from left to right
    """
    labels = np.asarray(labels)

    unique_labels = np.unique(labels)
    centroids_pc1 = {}

    for lab in unique_labels:
        centroids_pc1[lab] = data_pca[labels == lab, 0].mean()

    # sort clusters by PC1 centroid
    ordered = sorted(centroids_pc1, key=centroids_pc1.get)

    mapping = {old: new for new, old in enumerate(ordered)}
    new_labels = np.vectorize(mapping.get)(labels)

    return new_labels, mapping



def _cmap_for_dataset(dataset_name=None, cmap=None):
    """
    בוחר cmap לפי שם הדאטהסט, או לפי cmap שנמסר ידנית.
    """
    if cmap is None:
        name_to_cmap = {
            'VCSF': 'Blues',
            'VGMM': 'Reds',
            'VWM' : 'Greens'
        }
        cmap_name = name_to_cmap.get(str(dataset_name).upper() if dataset_name else None, 'tab10')
        return cm.get_cmap(cmap_name)
    # אם התקבל שם cmap כמחרוזת
    if isinstance(cmap, str):
        return cm.get_cmap(cmap)
    # אחרת מניחים שזה אובייקט cmap
    return cmap

def find_optimal_k_and_cluster(
    pca_df,
    k_range=range(2, 11),
    random_state=42
):
    """
    מבצע אשכולות K-Means ומחפש את ה-k האופטימלי באמצעות ציון הסילואט.

    Args:
        pca_df (pd.DataFrame): DataFrame המכיל את רכיבי ה-PCA.
        k_range (range): טווח ערכי k לבדיקה.
        random_state (int): זרע אקראיות לקבלת תוצאות חוזרות.

    Returns:
        dict: מילון עם k האופטימלי, רשימת k-ים שנבדקו, וציוני סילואט.
    """
    X = pca_df.select_dtypes(include=np.number).copy()
    if X.empty:
        raise ValueError("The DataFrame contains no numeric data.")

    ks = list(k_range)
    sil_scores = []

    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state, n_init='10')
        labels = km.fit_predict(X)
        sil_scores.append(silhouette_score(X, labels))

    best_k = ks[int(np.argmax(sil_scores))]

    return {
        "best_k": best_k,
        "k_values": ks,
        "silhouette_scores": sil_scores
    }



def _palette_from_dataset(k, dataset_name=None, cmap=None):
    """
    מחזיר רשימת צבעים באורך k. ברירת מחדל: מפות צבע לפי הדאטהסט.
    """
    if cmap is None:
        # בוחרים cmap לפי שם הדאטהסט
        name_to_cmap = {
            'VCSF': 'Blues',
            'VGMM': 'Reds',
            'VWM' : 'Greens'
        }
        cmap_name = name_to_cmap.get(str(dataset_name).upper() if dataset_name else None, 'tab10')
        cmap = cm.get_cmap(cmap_name)
    else:
        # אם הועבר מחרוזת
        if isinstance(cmap, str):
            cmap = cm.get_cmap(cmap)

    # נמנעים מצבעים בהירים מדי/כהים מדי
    return [cmap(x) for x in np.linspace(0.35, 0.95, k)]

def run_kmeans_on_pca_data(
    pca_df,
    k,
    plot=True,
    title="K-Means Clustering on PCA Data",
    csv_path=None,
    fix_left_to_right=True,   # למפות קלאסטרים משמאל לימין
    colors=None,              # רשימת צבעים ידנית (תגבר על cmap)
    dataset_name=None,        # קובע משפחת צבעים לפי הדאטהסט (VCSF/VGMM/VWM)
    cmap=None,
    random_state=None):


    # 1) Drop non-numeric columns and perform K-Means with a fixed seed
    data_for_clustering = pca_df.select_dtypes(include=np.number)
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=random_state)
    orig_labels = kmeans.fit_predict(data_for_clustering)

    labels = orig_labels.copy()

    # 2) Relabel clusters so that 0 is the leftmost (smallest x / first PCA column)
    if fix_left_to_right and data_for_clustering.shape[1] >= 1:
        order = np.argsort(kmeans.cluster_centers_[:, 0])  # מיון לפי ציר X
        old_to_new = {old: new for new, old in enumerate(order)}
        labels = np.vectorize(old_to_new.get)(labels)
        kmeans.labels_ = labels
        kmeans.cluster_centers_ = kmeans.cluster_centers_[order]

    # 3) Add cluster labels to the DataFrame (על עותק כדי לא לשנות מבחוץ)
    pca_df = pca_df.copy()
    pca_df['Cluster'] = labels

    # 4) Save results to a CSV file
    if csv_path:
        pca_df.index.name = "Subject_Code"
        pca_df.to_csv(csv_path, index=True)
        print(f"Cluster assignments saved to {csv_path}")

    # 5) Plot with dataset-based palette
    if plot and data_for_clustering.shape[1] >= 2:
        plt.figure(figsize=(8, 6))

        # אם לא נמסרה רשימת צבעים, יוצרים פלטה לפי הדאטהסט/ cmap
        if colors is None:
            colors = _palette_from_dataset(k, dataset_name=dataset_name, cmap=cmap)

        xcol, ycol = pca_df.columns[0], pca_df.columns[1]

        for cid in range(k):
            m = pca_df['Cluster'] == cid
            plt.scatter(
                pca_df.loc[m, xcol],
                pca_df.loc[m, ycol],
                s=50,
                color=colors[cid % len(colors)],
                label=f"Cluster {cid} (n={m.sum()})"
            )


        plt.xlabel(xcol)
        plt.ylabel(ycol)
        plt.title(title)
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        path_fig = f"data/SCHAEFER_mat_cor/csv_out/{title}"

    return labels, kmeans, pca_df


def build_cluster_summary(feature_files, subject_col="Subject", output_csv="cluster_summary.csv"):
    """
    feature_files: מילון מהצורה:
    {
      "VGMM": {"ses1": "/path/to/vgmm_clusters_ses1.csv",
               "ses2": "/path/to/vgmm_clusters_ses2.csv"},
      "VCSF": {"ses1": "/path/to/vcsf_clusters_ses1.csv",
               "ses2": "/path/to/vcsf_clusters_ses2.csv"},
      "VWM":  {"ses1": "/path/to/vwm_clusters_ses1.csv",
               "ses2": "/path/to/vwm_clusters_ses2.csv"},
    }
    subject_col: שם עמודת הנבדקת בכל קובץ (ברירת מחדל "Subject")
    output_csv: נתיב לקובץ ה-CSV המאוחד
    """
    summary = None
    desired_cols_order = [subject_col]  # נתחיל עם עמודת הנבדקת

    for feat, ses_paths in feature_files.items():
        for ses in ("ses1", "ses2"):
            if ses not in ses_paths:
                continue
            path = ses_paths[ses]
            if not Path(path).exists():
                print(f"[אזהרה] לא נמצא קובץ: {path}")
                continue

            df = pd.read_csv(path)

            # לוודא שיש עמודות Subject ו-Cluster; אם אין Subject ננסה לנחש/לחלץ
            if subject_col not in df.columns:
                if "Index" in df.columns:
                    df[subject_col] = df["Index"]
                else:
                    # fallback: ניקח את העמודה הראשונה
                    df[subject_col] = df.iloc[:, 0]

            if "Cluster" not in df.columns:
                raise ValueError(f"הקובץ {path} לא מכיל עמודה 'Cluster'.")

            # נשמור רק Subject ו-Cluster, נסיר כפילויות לפי Subject
            df = df[[subject_col, "Cluster"]].drop_duplicates(subset=[subject_col])

            # שם העמודה הסופית, למשל VGMM_ses1
            col_name = f"{feat}_ses1" if ses == "ses1" else f"{feat}_ses2"
            df = df.rename(columns={"Cluster": col_name}).set_index(subject_col)

            # הצטרפות חיצונית (outer) כדי לשמור נבדקות שאין להן נתון בפיצ'ר/סשן
            summary = df if summary is None else summary.join(df, how="outer")

            if col_name not in desired_cols_order:
                desired_cols_order.append(col_name)

    # החזרת subject לעמודת אינדקס רגילה ושמירה
    if summary is None:
        raise RuntimeError("לא נטענו שום קובץ. בדקי את הנתיבים ב-feature_files.")

    summary = summary.reset_index()

    # סדר עמודות: Subject ואז כל הפיצ'רים לפי הסדר שנכנסו
    existing = [c for c in desired_cols_order if c in summary.columns]
    other = [c for c in summary.columns if c not in existing]
    summary = summary[existing + other]

    summary.to_csv(output_csv, index=False)
    print(f"נשמר קובץ מאוחד ל-CSV: {output_csv}")
    return summary


def get_clusters_series_from_file(cluster_file, subj_series, expected_period):
    cl = pd.read_csv(cluster_file)
    # subject id col
    sid_col = next((c for c in cl.columns if str(c).lower() in
                   {"subject_id","subject","id","participant","participant_id"}), cl.columns[0])
    # <period>_clusters or any *_clusters
    target = f"{expected_period}_clusters".lower()
    cluster_col = next((c for c in cl.columns if str(c).lower() == target),
                       next((c for c in cl.columns if str(c).lower().endswith("_clusters")), None))
    if cluster_col is None:
        raise ValueError(f"No '*_clusters' column in {cluster_file}")

    left  = pd.DataFrame({"subject_id": subj_series.astype(str).str.strip(),
                          "__row__": np.arange(len(subj_series))})
    right = cl[[sid_col, cluster_col]].copy()
    right[sid_col] = right[sid_col].astype(str).str.strip()
    merged = left.merge(right, left_on="subject_id", right_on=sid_col, how="left").sort_values("__row__")
    return merged[cluster_col].reset_index(drop=True)



def pca_kmeans_minimal_outputs(
    df,
    prefix,
    n_components,
    k_range=range(2, 11),
    top_k_features=10,
    random_state=42,
    save_dir=None,
    subject_id_col='subject_id',
    save_subjects_csv = None,
    save_pca_csv = None  # <-- הוספנו את הפרמטר הזה
):
    """
    מפיק:
      1) גרף סילואט לכל k ובחירת k*.
      2) הדפסה של k*.
      3) CSV: subjects_per_cluster.csv (שיוך נבדקות לאשכול).
      4) לכל אשכול: הדפסה + גרף של טופ הפיצ'רים שתורמים (לפי |z|).

    מחזיר dict קטן עם:
      - best_k
      - assignments (DataFrame עם cluster + silhouette_sample)
      - top_features_by_cluster (dict: לכל אשכול Series של טופ-פיצ'רים לפי |z|)
      - paths (נתיבי קבצים אם save_dir לא None)
    """
    # 1) בחירת עמודות רלוונטיות
    if prefix:
        data_columns = [
            c for c in df.columns
            if c.startswith(prefix) and 'lec' not in c and c != 'Subject_Code'
        ]
    else:
        print("Prefix is empty, selecting 'PHQ_' and 'GAD7_' columns explicitly.")
        data_columns = [
            c for c in df.columns
            if (c.startswith('PHQ_') or c.startswith('GAD7_'))
        ]
    X = df[data_columns].copy()
    if X.empty:
        raise ValueError(f"No valid columns for prefix '{prefix}'")

# ניקוי
    X = X.apply(pd.to_numeric, errors='coerce')

    # --- התיקון כאן: מחיקת שורות במקום מילוי ---

    # שמירת מספר השורות המקורי לבדיקה
    original_row_count = X.shape[0]

    # 2. מחיקת כל שורה שיש בה לפחות ערך חסר (NaN) אחד
    X = X.dropna()

    # הדפסת סיכום על השורות שנמחקו
    dropped_row_count = original_row_count - X.shape[0]
    if dropped_row_count > 0:
        print(f"Dropped {dropped_row_count} rows (out of {original_row_count}) due to missing values (NaN).")
        print(f"Continuing analysis with {X.shape[0]} complete rows.")

    # בדיקה למקרה שכל השורות נמחקו
    if X.empty:
        raise ValueError("After dropping NaNs, the DataFrame is empty. Cannot proceed.")

    # --- סוף התיקון ---


    # 2) סטנדרטיזציה
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    print(Xs)
    # 3) PCA
    max_components = min(X.shape[0], X.shape[1])
    if n_components > max_components:
        n_components = max_components
    pca = PCA(n_components=n_components, random_state=random_state)
    Z = pca.fit_transform(Xs)  # (n_samples, n_components)

    if save_pca_csv is not None:
        try:
            # 1. יצירת שמות עמודות לרכיבי ה-PCA
            pca_col_names = [f'PC{i + 1}' for i in range(Z.shape[1])]

            # 2. יצירת DataFrame מהרכיבים
            pca_df = pd.DataFrame(Z, columns=pca_col_names)

            # 3. (חשוב!) שימוש באינדקס של X כדי לחבר לנתונים המקוריים
            #    האינדקס של X הוא האינדקס *אחרי* מחיקת שורות (dropna)
            pca_df.index = X.index

            # 4. בחירת עמודות המזהים (subject + timepoint) מה-df המקורי
            #    תוך שימוש באינדקס של X כדי לקבל רק את השורות הרלוונטיות
            id_cols_to_grab = [subject_id_col]
            if 'timepoint' in df.columns:
                id_cols_to_grab.append('timepoint')

            identifiers_df = df.loc[X.index, id_cols_to_grab]

            # 5. חיבור של המזהים ורכיבי ה-PCA יחד (זה לצד זה)
            pca_with_ids_df = pd.concat([identifiers_df, pca_df], axis=1)

            # 6. שמירה לקובץ CSV
            pca_with_ids_df.to_csv(save_pca_csv, index=False)
            print(f"\n✅ נתוני PCA נשמרו בהצלחה ב: {save_pca_csv}")

        except Exception as e:
            print(f"\n⚠️ אזהרה: לא ניתן היה לשמור את קובץ ה-PCA. שגיאה: {e}")
    # 4) בחירת k לפי סילואט + גרף
    ks = list(k_range)
    sil_scores = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        lbl = km.fit_predict(Z)
        sil_scores.append(silhouette_score(Z, lbl))

    # ציור גרף סילואט
    plt.figure(figsize=(8,5))
    plt.plot(ks, sil_scores, marker='o')
    plt.title(f"Silhouette vs k (PCA={n_components}, prefix='{prefix}')")
    plt.xlabel("k")
    plt.ylabel("Silhouette score")
    plt.xticks(ks)
    plt.grid(True)
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(Path(save_dir) / "silhouette_vs_k.png", dpi=150)
    plt.show()

    best_k = ks[int(np.argmax(sil_scores))]
    print(f"\n✅ מספר הקלאסטרים שנבחר לפי סילואט: {best_k}")
    # 5) אימון סופי + שיוכי נבדקות + סילואט פרטני
    kmeans = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(Z)
    sil_each = silhouette_samples(Z, labels)

        # pull subject identifiers
    if subject_id_col in df.columns:
        subj_ids = df.loc[X.index, subject_id_col]
    else:
        # fallback to index
        subj_ids = pd.Series(X.index, index=X.index, name='subject_id')

    assignments = pd.DataFrame({
        "subject_id": subj_ids,           # always named 'subject_id' in the output
        "cluster": labels,
        "silhouette_sample": sil_each
    }, index=X.index)
    print(assignments)

    # optional save CSV with just subject_id + cluster
    if save_subjects_csv is not None:
        print(assignments.loc[:, ["subject_id", "cluster"]])


    return {
        "best_k": best_k,
        "sil_scores" : pd.DataFrame(sil_scores),
        "assignments": assignments
    }


def run_kmeans_clustering(
        df,
        prefix,
        n_components,
        k,
        plot=True,
        title="k_means_plot",
        csv_path=None,
        pc1_csv_path=None,
        include_timepoints=False  # <--- פרמטר חדש שהוספנו
):
    """
    מבצע ניתוח PCA ו-K-Means על נתונים.
    הפונקציה אינה מבצעת התאמת עקביות (consistency) בין אשכולות.

    Args:
        df (pd.DataFrame): DataFrame הקלט.
        prefix (str): קידומת העמודות שיש לנתח.
        n_components (int): מספר רכיבי ה-PCA.
        k (int): מספר האשכולות ל-K-Means.
        plot (bool): האם להציג גרף פיזור של האשכולות.
        title (str): כותרת הגרף.
        csv_path (str, optional): נתיב לשמירת תוצאות האשכולות לקובץ CSV.
        pc1_csv_path (str, optional): נתיב לשמירת Subject_Code ו-PC1.
        include_timepoints (bool, optional): האם לכלול את עמודת 'timepoint' ב-CSV של האשכולות. (ברירת מחדל: False) # <--- תיאור

    Returns:
        tuple: תוויות האשכולות, נתוני PCA, אובייקט KMeans, אובייקט PCA.
    """
    # 1) סינון עמודות וטיפול בערכים חסרים
    data_columns = [c for c in df.columns if
                    c.startswith(prefix) and 'lec' not in c and c not in ['Subject_Code', 'timepoint']]
    df_sub = df[data_columns].copy()
    df_sub = df_sub.apply(pd.to_numeric, errors='coerce').fillna(df_sub.mean())

    if df_sub.empty:
        print("Warning: DataFrame is empty after column selection.")
        return None, None, None, None

    # 2) סטנדרטיזציה
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_sub)

    # 3) PCA
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data_scaled)

    # 4) KMeans
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=None)
    labels = kmeans.fit_predict(data_pca)

    labels, mapping = relabel_clusters_by_pc1(data_pca, labels)

    # also reorder cluster centers if you use them later
    kmeans.labels_ = labels
    kmeans.cluster_centers_ = kmeans.cluster_centers_[list(mapping.keys())]

    # 5) שמירת תוצאות ל-CSV (שיוך לאשכולות)
    if csv_path:
        if "Subject_Code" not in df.columns:
            print("Warning: 'Subject_Code' column not found, cannot save to CSV.")
        else:
            # ----------------------------------------------------
            #               השינוי המרכזי כאן 👇
            # ----------------------------------------------------
            output_data = {
                "Subject_Code": df.loc[df_sub.index, "Subject_Code"],
                "Cluster": labels
            }

            # בדיקה אם צריך לכלול את timepoint
            if include_timepoints and "timepoint" in df.columns:
                output_data["timepoint"] = df.loc[df_sub.index, "timepoint"]
            elif include_timepoints and "timepoint" not in df.columns:
                print("Warning: 'timepoint' column requested but not found in the original DataFrame.")

            out = pd.DataFrame(output_data, index=df_sub.index)
            # ----------------------------------------------------

            out.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"Cluster assignments saved to {csv_path}")

    # 5.5) Save ALL PCs to CSV
    if pc1_csv_path:
        if "Subject_Code" not in df.columns:
            print("Warning: 'Subject_Code' column not found, cannot save PCs to CSV.")
        else:
            n_pcs = data_pca.shape[1]   # how many PCs exist

            # Create PC column names dynamically
            pc_cols = {f"PC{i+1}": data_pca[:, i] for i in range(n_pcs)}

            # Add subject codes
            pc_out = pd.DataFrame({
                "Subject_Code": df.loc[df_sub.index, "Subject_Code"],
                **pc_cols
            }, index=df_sub.index)

            # Create directory if needed
            pc_dir = os.path.dirname(pc1_csv_path)
            if pc_dir and not os.path.exists(pc_dir):
                os.makedirs(pc_dir, exist_ok=True)

            pc_out.to_csv(pc1_csv_path, index=False, encoding='utf-8-sig')
            print(f"All PCs (PC1–PC{n_pcs}) saved to {pc1_csv_path}")

    # 6) הצגת גרף פיזור
    if plot and data_pca.shape[1] >= 2:
        plt.figure(figsize=(8, 6))
        for cid in np.unique(labels):
            m = labels == cid
            plt.scatter(data_pca[m, 0], data_pca[m, 1], s=50, label=f"Cluster {cid} (n={m.sum()})")

        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.title(title)
        plt.legend(title="Clusters (n)", loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return labels, data_pca, kmeans, pca

def align_clusters_to_previous(
        current_assignments_df,
        previous_assignments_df,
        current_cluster_col='cluster',
        previous_cluster_col='cluster',
        subject_id_col='subject_id'
):
    """
    Relabels clusters in current_assignments_df to align with the majority cluster
    from previous_assignments_df, based on overlapping subjects.

    Args:
        current_assignments_df (pd.DataFrame): DataFrame with current subject_id and cluster.
        previous_assignments_df (pd.DataFrame): DataFrame with previous subject_id and cluster.
        ... column names ...

    Returns:
        pd.Series: New cluster labels aligned to the previous period.
    """

    # 1. Merge the two assignment DataFrames on subject_id
    merged = current_assignments_df.merge(
        previous_assignments_df[[subject_id_col, previous_cluster_col]],
        on=subject_id_col,
        how='left',
        suffixes=('_current', '_prev')
    )

    # Handle subjects that were NOT in the previous session (assign NaN for prev cluster)
    merged['Cluster_Prev'] = merged[f'{previous_cluster_col}_prev'].fillna(-1).astype(int)

    # 2. Determine the best mapping from current_label -> previous_label
    current_labels = merged[f'{current_cluster_col}_current'].unique()

    # Store the mapping: current_label -> most frequent previous_label
    mapping = {}

    # Keep track of previous labels that have already been 'claimed' by a current label
    claimed_prev_labels = set()

    # The goal is to maximize the overlap without mapping two current clusters to the same previous cluster
    # We should iterate through current clusters and pick the *unclaimed* previous cluster they overlap with most

    # We'll calculate the overlap matrix (Current Cluster x Previous Cluster)
    overlap_matrix = merged.groupby(f'{current_cluster_col}_current')['Cluster_Prev'].value_counts().unstack(
        fill_value=0)

    # We only care about aligning to the actual cluster labels (0, 1, 2, ...) from the previous session
    if -1 in overlap_matrix.columns:
        overlap_matrix = overlap_matrix.drop(columns=[-1])  # Drop the 'not found' column

    # If the previous session had no clusters (e.g. for the 'b' time point), just return the current labels
    if overlap_matrix.empty:
        return current_assignments_df[current_cluster_col]

        # Greedy assignment: find the best match and remove the claim
    temp_matrix = overlap_matrix.copy()

    # Map current_label -> new_label (which is the previous_label)
    final_mapping = {old: old for old in current_labels}  # Default: no change

    # Iterate for as many clusters as the minimum of the two sets
    num_clusters_to_map = min(len(overlap_matrix.index), len(overlap_matrix.columns))

    for _ in range(num_clusters_to_map):
        # Find the max overlap cell (row=current_label, col=previous_label)
        max_val = temp_matrix.max().max()
        if max_val == 0:
            break

        # Get the indices (Current and Previous labels)
        max_idx = temp_matrix[temp_matrix == max_val].stack().index[0]
        current_label, previous_label = max_idx

        # Assign the mapping and record the claim
        final_mapping[current_label] = previous_label

        # Remove the row (current_label) and column (previous_label) so they can't be chosen again
        temp_matrix = temp_matrix.drop(index=current_label, errors='ignore')
        temp_matrix = temp_matrix.drop(columns=previous_label, errors='ignore')

    print(f"\nCluster Alignment Mapping: {final_mapping}")

    # 3. Apply the mapping to the current labels
    new_labels = merged[f'{current_cluster_col}_current'].map(final_mapping)

    return new_labels.rename(current_cluster_col)




def invert_binary_columns(input_path, output_path, column_names):
    """
    Load a file, invert 0↔1 values in the specified columns,
    and save to a new file.

    Args:
        input_path (str): Path to the input CSV/XLSX file.
        output_path (str): Path to save the modified file.
        column_names (list[str]): List of column names to invert.
    """
    # 1. Load file (auto-detect CSV or Excel)
    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    else:
        df = pd.read_excel(input_path)

    # 2. Validate all requested columns exist
    missing = [col for col in column_names if col not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in file: {missing}")

    # 3. Invert values for each requested column
    for col in column_names:
        df[col] = df[col].apply(lambda x: 1 if x == 0 else (0 if x == 1 else x))

    # 4. Save file
    if output_path.endswith(".csv"):
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        df.to_excel(output_path, index=False)

    print(f"✅ File saved to {output_path}")




def gap_statistic(X, n_refs=10, max_clusters=10, k_min=1, random_state=None):
    """
    Compute Gap Statistic for k in [k_min, max_clusters].
    Returns (results_df, optimal_k), where results_df has columns ['k','gap','sk'].
    """
    if max_clusters < k_min or k_min < 1:
        raise ValueError("Require 1 <= k_min <= max_clusters.")

    rng = np.random.default_rng(random_state)

    # Scale data
    X = StandardScaler().fit_transform(X)

    # Bounds for reference data (in scaled space)
    shape = X.shape
    min_vals, max_vals = X.min(axis=0), X.max(axis=0)

    ks, gaps, sks = [], [], []

    for k in range(k_min, max_clusters + 1):
        # Wk for real data
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X)
        Wk = np.log(np.min(pairwise_distances(X, km.cluster_centers_), axis=1).sum())

        # Wk for reference datasets
        Wkb = []
        for _ in range(n_refs):
            X_ref = rng.uniform(min_vals, max_vals, size=shape)
            km_ref = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            km_ref.fit(X_ref)
            Wkb.append(np.log(np.min(pairwise_distances(X_ref, km_ref.cluster_centers_), axis=1).sum()))
        Wkb = np.asarray(Wkb)

        ks.append(k)
        gaps.append(Wkb.mean() - Wk)
        # Tibshirani: sd * sqrt(1 + 1/B). Using sample sd (ddof=1) is common:
        sks.append(Wkb.std(ddof=1) * np.sqrt(1 + 1/n_refs))

    results_df = pd.DataFrame({"k": ks, "gap": gaps, "sk": sks})

    # Choose k: smallest k such that gap(k) >= gap(k+1) - s_{k+1}
    optimal_k = ks[-1]  # fallback
    for i in range(len(results_df) - 1):
        if results_df.loc[i, "gap"] >= results_df.loc[i + 1, "gap"] - results_df.loc[i + 1, "sk"]:
            optimal_k = int(results_df.loc[i, "k"])
            break

    return results_df, optimal_k

