# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples

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
        km = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
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
    random_state=17):


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
    save_dir=None,   # למשל: "/content/outputs"
    subject_id_col='subject_id',
    save_subjects_csv = None
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
    data_columns = [
        c for c in df.columns
        if c.startswith(prefix) and 'lec' not in c and c != 'Subject_Code'
    ]
    X = df[data_columns].copy()
    if X.empty:
        raise ValueError(f"No valid columns for prefix '{prefix}'")

    # ניקוי
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.mean())

    # הסרת שונות אפס
    var0 = X.var()
    drop_cols = var0[var0 == 0].index.tolist()
    if drop_cols:
        X = X.drop(columns=drop_cols)

    # 2) סטנדרטיזציה
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # 3) PCA
    max_components = min(X.shape[0], X.shape[1])
    if n_components > max_components:
        n_components = max_components
    pca = PCA(n_components=n_components, random_state=random_state)
    Z = pca.fit_transform(Xs)  # (n_samples, n_components)

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

# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

def run_kmeans_clustering(
    df,
    prefix,
    n_components,
    k,
    plot=True,
    title="k_means_plot",
    csv_path=None
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

    Returns:
        tuple: תוויות האשכולות, נתוני PCA, אובייקט KMeans, אובייקט PCA.
    """
    # 1) סינון עמודות וטיפול בערכים חסרים
    data_columns = [c for c in df.columns if c.startswith(prefix) and 'lec' not in c and c != 'Subject_Code']
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
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=None)
    labels = kmeans.fit_predict(data_pca)

    # 5) שמירת תוצאות ל-CSV
    if csv_path:
        if "Subject_Code" not in df.columns:
            print("Warning: 'Subject_Code' column not found, cannot save to CSV.")
        else:
            out = pd.DataFrame({
                "Subject_Code": df.loc[df_sub.index, "Subject_Code"],
                "Cluster": labels
            }, index=df_sub.index)
            out.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"Cluster assignments saved to {csv_path}")

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
