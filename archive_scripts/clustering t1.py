import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path
import os
import re

from pca_functions import (find_optimal_pca_dimensions, perform_pca, save_top_loadings)
from clustring_functions import (find_optimal_k_and_cluster, _palette_from_dataset, run_kmeans_on_pca_data,
                                     _cmap_for_dataset,build_cluster_summary)
from vizualizations_functions import (plot_grouped_bars, plot_clusters_with_group_overlay,
                                          subplot_clusters_by_group)
from preprocessing_functions import (_load_clustered_df, _ensure_subject_col, _ensure_subject_col, _load_mapping,
                                         get_feature_names)


def main():
    seed = 17
    np.random.seed(seed)
    # --- הגדרות ---
    ROOT = Path("data/cat12")
    SUBFOLDERS = ["NT", "CT"]  # התיקיות הצפויות בתוך cat12
    OUTPUT_CSV = ROOT / "sessions_summary.csv"

    # תבנית: תחילת שם קובץ = NT/CT + ספרות (מזהה נבדק), ולאחר מכן אופציונלית _1 או _2 = מספר סשן
    # דוגמאות חוקיות: NT028.xlsx, NT028_1.xlsx, CT123-2.csv, CT045 2.mat
    PATTERN = re.compile(r'^(?P<id>(?:NT|CT)\d+)(?:[ _-]?(?P<session>[12]))?\b', re.IGNORECASE)

    rows = []

    for sub in SUBFOLDERS:
        folder = (ROOT / sub).resolve()

        if not folder.is_dir():
            print(f"⚠️ לא נמצאה תיקייה (או שזה לא תיקייה): {folder}")
            continue

        for p in folder.iterdir():
            if not p.is_file():
                continue
            if p.name.startswith("~$"):
                # קבצי temp של אקסל
                continue
            if p.suffix.lower() not in {".xlsx", ".csv"}:
                continue

            name = p.stem
            m = PATTERN.match(name)
            if not m:
                print(f"⚠️ דילוג: שם קובץ לא מזוהה לפי התבנית -> {p.name}")
                continue

            subject_id = m.group("id").upper()
            session_str = m.group("session")
            session = int(session_str) if session_str else 1
            rows.append({"subject_id": subject_id, "session": session})

    sessions_df = pd.DataFrame(rows)
    if sessions_df.empty:
        print("לא נמצאו קבצים תואמים בתיקיות NT/CT.")
    else:
        sessions_df = sessions_df.drop_duplicates().sort_values(by=["subject_id", "session"]).reset_index(drop=True)
        sessions_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        print(f"✅ summery session files saved: {OUTPUT_CSV}")
        print(sessions_df.head(20))


    # --- יצירת רשימת הקבצים הרלוונטיים (Session 1 לכל הנבדקים) ---
    files_ses1 = []
    files_ses2 = []
    num_subjects_processed_1 = 0
    num_subjects_processed_2 = 0

    session_1_subjects = sessions_df[sessions_df['session'] == 1]['subject_id'].tolist()
    session_2_subjects = sessions_df[sessions_df['session'] == 2]['subject_id'].tolist()
    base_path = "data/cat12"

    for subject in session_1_subjects:
        group = subject[:2]
        file_path = os.path.join(base_path, group, f'{subject}_1.xlsx')

        if os.path.exists(file_path):
            files_ses1.append({'subject': subject, 'path': file_path})
        else:
            print(f"Warning: File not found for subject {subject} at {file_path}")

    # --- אתחול דאטה-פריימים ---
    vcsf_df = pd.DataFrame()
    vgmm_df = pd.DataFrame()
    vwm_df = pd.DataFrame()

    subjects_included = []  # נשמור מי נכלל בפועל (עם TIV תקין)

    # --- לולאה על קבצי סשן 1 ---
    for file_info in files_ses1:
        subject_id = file_info['subject']
        file_path = file_info['path']

        try:
            subject_data = pd.read_excel(
                file_path, header=0,
                names=['name', 'Vcsf', 'Vgm', 'Vwm', 'Euler', 'TIV']
            )
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue

        # ודא שיש עמודת TIV תקינה (לא NaN ולא 0)
        tiv_series = pd.to_numeric(subject_data['TIV'], errors='coerce').dropna()
        if tiv_series.empty:
            print(f"Warning: Missing TIV for subject {subject_id} in {file_path} — skipping this subject.")
            continue

        tiv_value = tiv_series.iloc[0]
        if tiv_value == 0:
            print(f"Warning: TIV=0 for subject {subject_id} in {file_path} — skipping this subject.")
            continue

        # חישוב הערכים אחרי חלוקה ב-TIV
        vcsf_norm = subject_data['Vcsf'] / tiv_value
        vgmm_norm = subject_data['Vgm'] / tiv_value
        vwm_norm = subject_data['Vwm'] / tiv_value

        # יצירת אינדקס שמות האזורים בפעם הראשונה
        if vcsf_df.empty:
            region_names = subject_data['name']
            vcsf_df['name'] = region_names
            vgmm_df['name'] = region_names
            vwm_df['name'] = region_names
            vcsf_df.set_index('name', inplace=True)
            vgmm_df.set_index('name', inplace=True)
            vwm_df.set_index('name', inplace=True)

        # הוספת העמודות המחולקות (Normalized by TIV)
        vcsf_df[subject_id] = vcsf_norm.values
        vgmm_df[subject_id] = vgmm_norm.values
        vwm_df[subject_id] = vwm_norm.values

        subjects_included.append(subject_id)

    # --- טרנספוזיציה: נבדקים כשורות ואזורי מוח כעמודות ---
    vcsf_final_df = vcsf_df.T
    vgmm_final_df = vgmm_df.T
    vwm_final_df = vwm_df.T

    # --- יצירת תיקיות יעד אם צריך ---
    out_base = "clusteres_output"
    os.makedirs(os.path.join(out_base, "vcsf"), exist_ok=True)
    os.makedirs(os.path.join(out_base, "vgmm"), exist_ok=True)
    os.makedirs(os.path.join(out_base, "vwm"), exist_ok=True)

    # --- שמירה: ערכים אחרי חלוקה ב-TIV ---
    vcsf_out = os.path.join(out_base, "vcsf", "all_subjects_vcsf_ses1.csv")
    vgmm_out = os.path.join(out_base, "vgmm", "all_subjects_vgmm_ses1.csv")
    vwm_out = os.path.join(out_base, "vwm", "all_subjects_vwm_ses1.csv")
    # אחרי vcsf_final_df = vcsf_df.T ... וכו'
    for df in (vcsf_final_df, vgmm_final_df, vwm_final_df):
        df.index.name = "Subject_Code"  # ← כאן

    vcsf_final_df.to_csv(vcsf_out)
    vgmm_final_df.to_csv(vgmm_out)
    vwm_final_df.to_csv(vwm_out)

    # --- הדפסה לסיכום ---
    num_subjects_processed_1 = len(subjects_included)
    print("\n" + "-" * 50)
    print("Saved (values divided by each subject's TIV):")
    print(f"• {vcsf_out}")
    print(f"• {vgmm_out}")
    print(f"• {vwm_out}")
    print(f"\nA total of {num_subjects_processed_1} subjects (Session 1) were included after TIV checks.")
    excluded = set([f['subject'] for f in files_ses1]) - set(subjects_included)
    if excluded:
        print(f"Excluded subjects due to missing/zero TIV: {sorted(list(excluded))}")


    # --- יצירת רשימת הקבצים הרלוונטיים (Session 2 לכל הנבדקים) ---
    for subject in session_2_subjects:
        group = subject[:2]
        file_path = os.path.join(base_path, group, f'{subject}_2.xlsx')
        if os.path.exists(file_path):
            files_ses2.append({'subject': subject, 'path': file_path})
        else:
            print(f"Warning: File not found for subject {subject} at {file_path}")

    # --- אתחול דאטה-פריימים ---
    vcsf_df = pd.DataFrame()
    vgmm_df = pd.DataFrame()
    vwm_df = pd.DataFrame()

    subjects_included = []

    # --- לולאה על קבצי סשן 2 ---
    for file_info in files_ses2:
        subject_id = file_info['subject']
        file_path = file_info['path']

        try:
            subject_data = pd.read_excel(
                file_path, header=0,
                names=['name', 'Vcsf', 'Vgm', 'Vwm', 'Euler', 'TIV']
            )
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue

        # בדיקת TIV תקין (לא חסר/לא 0)
        tiv_series = pd.to_numeric(subject_data['TIV'], errors='coerce').dropna()
        if tiv_series.empty:
            print(f"Warning: Missing TIV for subject {subject_id} in {file_path} — skipping.")
            continue
        tiv_value = tiv_series.iloc[0]
        if tiv_value == 0:
            print(f"Warning: TIV=0 for subject {subject_id} in {file_path} — skipping.")
            continue

        # חלוקה ב-TIV
        vcsf_norm = subject_data['Vcsf'] / tiv_value
        vgmm_norm = subject_data['Vgm'] / tiv_value
        vwm_norm = subject_data['Vwm'] / tiv_value

        # יצירת אינדקס שמות אזורים בפעם הראשונה
        if vcsf_df.empty:
            region_names = subject_data['name']
            for df in (vcsf_df, vgmm_df, vwm_df):
                df['name'] = region_names
                df.set_index('name', inplace=True)

        # הוספת העמודות המחולקות (Normalized by TIV)
        vcsf_df[subject_id] = vcsf_norm.values
        vgmm_df[subject_id] = vgmm_norm.values
        vwm_df[subject_id] = vwm_norm.values

        subjects_included.append(subject_id)

    # --- טרנספוזיציה: נבדקים כשורות ואזורי מוח כעמודות ---
    vcsf_final_df = vcsf_df.T
    vgmm_final_df = vgmm_df.T
    vwm_final_df = vwm_df.T

    # --- הוספת עמודת group (לפי 2 התווים הראשונים של מזהה הנבדק) ---
    vcsf_final_df['group'] = vcsf_final_df.index.str[:2]
    vgmm_final_df['group'] = vgmm_final_df.index.str[:2]
    vwm_final_df['group'] = vwm_final_df.index.str[:2]

    # --- שמירה תחת אותם שמות מקוריים (ימחוק/יבריש אם כבר קיים) ---
    vcsf_out = os.path.join(out_base, "vcsf", "all_subjects_vcsf_ses2.csv")
    vgmm_out = os.path.join(out_base, "vgmm", "all_subjects_vgmm_ses2.csv")
    vwm_out = os.path.join(out_base, "vwm", "all_subjects_vwm_ses2.csv")

    vcsf_final_df.to_csv(vcsf_out)
    vgmm_final_df.to_csv(vgmm_out)
    vwm_final_df.to_csv(vwm_out)

    # --- סיכום ---
    num_subjects_processed_2 = len(subjects_included)
    print("\n" + "-" * 50)
    print("Saved (values divided by each subject's TIV) for Session 2 under the original filenames:")
    print(f"• {vcsf_out}")
    print(f"• {vgmm_out}")
    print(f"• {vwm_out}")
    print(f"\nA total of {num_subjects_processed_2} subjects (Session 2) were included after TIV checks.")
    excluded = set([f['subject'] for f in files_ses2]) - set(subjects_included)
    if excluded:
        print(f"Excluded subjects due to missing/zero TIV: {sorted(list(excluded))}")

    # הגדר את נתיבי הקבצים שלך
    file_paths = {
        "VCSF": "clusteres_output/vcsf/all_subjects_vcsf_ses1.csv",
        "VGMM": "clusteres_output/vgmm/all_subjects_vgmm_ses1.csv",
        "VWM": "clusteres_output/vwm/all_subjects_vwm_ses1.csv"
    }

    # חלץ את שמות הפיצ'רים מכל דאטה-סט
    vcsf_features = get_feature_names(file_paths["VCSF"])
    vgmm_features = get_feature_names(file_paths["VGMM"])
    vwm_features = get_feature_names(file_paths["VWM"])

    # הדפס את התוצאות
    print("--- שמות הפיצ'רים עבור כל דאטה-סט ---")

    if vcsf_features:
        print(f"\n✅ VCSF features ({len(vcsf_features)}):")
        print(vcsf_features)

    if vgmm_features:
        print(f"\n✅ VGMM features ({len(vgmm_features)}):")
        print(vgmm_features)

    if vwm_features:
        print(f"\n✅ VWM features ({len(vwm_features)}):")
        print(vwm_features)

    # Example usage
    print("\nExplained Variance session 1 Vgm\n")

    vgm_ses1 = "clusteres_output/vgmm/all_subjects_vgmm_ses1.csv"
    num90_vgmm_ses1, num95_vgmm_ses1, num100_vgmm_ses1 = find_optimal_pca_dimensions(vgm_ses1,'figures/gm')

    print("Components for 90%:", num90_vgmm_ses1)
    print("Components for 95%:", num95_vgmm_ses1)
    print("Components for 100%:", num100_vgmm_ses1)

    print("\nExplained Variance session 1 CSF \n")

    # Example usage
    csf_ses1 = "clusteres_output/vcsf/all_subjects_vcsf_ses1.csv"
    num90_csf_ses1, num95_csf_ses1, num100_csf_ses1 = find_optimal_pca_dimensions(csf_ses1,'figures/csf')

    print("Components for 90%:", num90_csf_ses1)
    print("Components for 95%:", num95_csf_ses1)
    print("Components for 100%:", num100_csf_ses1)

    print("\nExplained Variance session 1 Vwm\n")

    # Example usage
    vwm_ses1 = "clusteres_output/vwm/all_subjects_vwm_ses1.csv"
    num90_vwm_ses1, num95_vwm_ses1, num100_vwm_ses1 = find_optimal_pca_dimensions(vwm_ses1,'figures/wm')

    print("Components for 90%:", num90_vwm_ses1)
    print("Components for 95%:", num95_vwm_ses1)
    print("Components for 100%:", num100_vwm_ses1)

    ########################################################################################

    print("\nExplained Variance session 2 Vgm\n")

    # Example usage
    vgmm_ses2 = "clusteres_output/vgmm/all_subjects_vgmm_ses2.csv"
    num90_vgmm_ses2, num95_vgmm_ses2, num100_vgmm_ses2 = find_optimal_pca_dimensions(vgmm_ses2,'figures/gm')

    print("Components for 90%:", num90_vgmm_ses2)
    print("Components for 95%:", num95_vgmm_ses2)
    print("Components for 100%:", num100_vgmm_ses2)

    print("\nExplained Variance session 2 CSF\n")

    # Example usage
    csf_ses2 = "clusteres_output/vcsf/all_subjects_vcsf_ses2.csv"
    num90_csf_ses2, num95_csf_ses2, num100_csf_ses2 = find_optimal_pca_dimensions(csf_ses2,'figures/csf')

    print("Components for 90%:", num90_csf_ses2)
    print("Components for 95%:", num95_csf_ses2)
    print("Components for 100%:", num100_csf_ses2)

    print("\nExplained Variance session 2 Vwm\n")

    # Example usage
    vwm_ses2 = "clusteres_output/vwm/all_subjects_vwm_ses2.csv"
    num90_vwm_ses2, num95_vwm_ses2, num100_vwm_ses2 = find_optimal_pca_dimensions(vwm_ses2,'figures/wm')

    print("Components for 90%:", num90_vwm_ses2)
    print("Components for 95%:", num95_vwm_ses2)
    print("Components for 100%:", num100_vwm_ses2)


    # --- Perform PCA on VCSF data with 3 dimensions ---
    vcsf_df_ses1 = pd.read_csv(csf_ses1,index_col = 0)
    print("\n" + "=" * 50)
    print(f"PCA on VCSF Data with {num90_csf_ses1} dimensions:")
    vcsf_pca_df_ses1, vcsf_pca_model_ses1 = perform_pca(vcsf_df_ses1, num90_csf_ses1)
    print(f"Original Data Shape: {vcsf_df_ses1.shape}")
    print(f"Transformed Data Shape: {vcsf_pca_df_ses1.shape}")
    print("\nExplained variance ratio:")
    for i, var_ratio_ses1 in enumerate(vcsf_pca_model_ses1.explained_variance_ratio_):
        print(f" - PC{i + 1}: {var_ratio_ses1:.4f}")
    print("\nFirst 5 rows of the PCA-transformed data:")
    print(vcsf_pca_df_ses1.head())
    print("=" * 50)

    # --- Perform PCA on VGMM data with 5 dimensions ---
    vgmm_df_ses1 = pd.read_csv(vgm_ses1,index_col = 0)
    print("\n" + "=" * 50)
    print(f"PCA on VGMM Data with {num90_vgmm_ses1} dimensions:")
    vgmm_pca_df_ses1, vgmm_pca_model_ses1 = perform_pca(vgmm_df_ses1, num90_vgmm_ses1)
    print(f"Original Data Shape: {vgmm_df_ses1.shape}")
    print(f"Transformed Data Shape: {vgmm_pca_df_ses1.shape}")
    print("\nExplained variance ratio:")
    for i, var_ratio_ses1 in enumerate(vgmm_pca_model_ses1.explained_variance_ratio_):
        print(f" - PC{i + 1}: {var_ratio_ses1:.4f}")
    print("\nFirst 5 rows of the PCA-transformed data:")
    print(vgmm_pca_df_ses1.head())
    print("=" * 50)

    # --- Perform PCA on VWM data with 2 dimensions ---
    vwm_df_ses1 = pd.read_csv(vwm_ses1,index_col = 0)
    print("\n" + "=" * 50)
    print(f"PCA on VWM Data with {num90_vwm_ses1} dimensions:")
    vwm_pca_df_ses1, vwm_pca_model_ses1 = perform_pca(vwm_df_ses1, num90_vwm_ses1)
    print(f"Original Data Shape: {vwm_df_ses1.shape}")
    print(f"Transformed Data Shape: {vwm_pca_df_ses1.shape}")
    print("\nExplained variance ratio:")
    for i, var_ratio_ses1 in enumerate(vwm_pca_model_ses1.explained_variance_ratio_):
        print(f" PC{i + 1}: {var_ratio_ses1:.4f}")
    print("\nFirst 5 rows of the PCA-transformed data:")
    print(vwm_pca_df_ses1.head())
    print("=" * 50)



    # --- Perform PCA on VCSF data with 3 dimensions ---
    vcsf_df_ses2 = pd.read_csv(csf_ses2,index_col = 0)
    print("\n" + "=" * 50)
    print(f"PCA on VCSF Data with {num90_csf_ses2} dimensions:")
    vcsf_pca_df_ses2, vcsf_pca_model_ses2 = perform_pca(vcsf_df_ses2, num90_csf_ses2)
    print(f"Original Data Shape: {vcsf_df_ses2.shape}")
    print(f"Transformed Data Shape: {vcsf_pca_df_ses2.shape}")
    print("\nExplained variance ratio:")
    for i, var_ratio_ses2 in enumerate(vcsf_pca_model_ses2.explained_variance_ratio_):
        print(f" - PC{i + 1}: {var_ratio_ses2:.4f}")
    print("\nFirst 5 rows of the PCA-transformed data:")
    print(vcsf_pca_df_ses2.head())
    print("=" * 50)

    # --- Perform PCA on VGMM data with 5 dimensions ---
    vgmm_df_ses2 = pd.read_csv(vgmm_ses2,index_col=0)
    print("\n" + "=" * 50)
    print(f"PCA on VGMM Data with {num90_vgmm_ses2} dimensions:")
    vgmm_pca_df_ses2, vgmm_pca_model_ses2 = perform_pca(vgmm_df_ses2, num90_vgmm_ses2)
    print(f"Original Data Shape: {vgmm_df_ses2.shape}")
    print(f"Transformed Data Shape: {vgmm_pca_df_ses2.shape}")
    print("\nExplained variance ratio:")
    for i, var_ratio_ses2 in enumerate(vgmm_pca_model_ses2.explained_variance_ratio_):
        print(f" - PC{i + 1}: {var_ratio_ses2:.4f}")
    print("\nFirst 5 rows of the PCA-transformed data:")
    print(vgmm_pca_df_ses2.head())
    print("=" * 50)

    # --- Perform PCA on VWM data with 2 dimensions ---
    vwm_df_ses2 = pd.read_csv(vwm_ses2,index_col =0)
    print("\n" + "=" * 50)
    print(f"PCA on VWM Data with {num90_vwm_ses2} dimensions:")
    vwm_pca_df_ses2, vwm_pca_model_ses2 = perform_pca(vwm_df_ses2, num90_vwm_ses2)
    print(f"Original Data Shape: {vwm_df_ses2.shape}")
    print(f"Transformed Data Shape: {vwm_pca_df_ses2.shape}")
    print("\nExplained variance ratio:")
    for i, var_ratio_ses2 in enumerate(vwm_pca_model_ses2.explained_variance_ratio_):
        print(f" - PC{i + 1}: {var_ratio_ses2:.4f}")
    print("\nFirst 5 rows of the PCA-transformed data:")
    print(vwm_pca_df_ses2.head())
    print("=" * 50)

    # ===== Example usage (replace with your real data) =====
    groups = ['Vgm', 'CSF', 'Vwm']


    plot_grouped_bars(groups,[num90_vgmm_ses1, num90_csf_ses1, num90_vwm_ses1],
                      [num95_vgmm_ses1, num95_csf_ses1, num95_vwm_ses1],
                      [num100_vgmm_ses1, num100_csf_ses1, num100_vwm_ses1],
                      s1_label="90% variability",
                      s2_label="95% variability",
                      s3_label="100% variability",
                      title="Number of Dimentions for each Time period -session 1 ",
                      ylabel="# of dimentions")

    # ===== Example usage (replace with your real data) =====
    plot_grouped_bars(groups,[num90_vgmm_ses2, num90_csf_ses2, num90_vwm_ses2],
                      [num95_vgmm_ses2, num95_csf_ses2, num95_vwm_ses2],
                      [num100_vgmm_ses2, num100_csf_ses2, num100_vwm_ses2],
                      s1_label="90% variability",
                      s2_label="95% variability",
                      s3_label="100% variability",
                      title="Number of Dimentions for each Time period -session 2 ",
                      ylabel="# of dimentions")

    # --- קריאה לפונקציה ושמירת תוצאות ---
    results_vcsf_ses1 = find_optimal_k_and_cluster(pca_df=vcsf_pca_df_ses1)
    results_vgmm_ses1 = find_optimal_k_and_cluster(pca_df=vgmm_pca_df_ses1)
    results_vwm_ses1 = find_optimal_k_and_cluster(pca_df=vwm_pca_df_ses1)

    # --- Create the combined plot ---
    plt.figure(figsize=(10, 7))

    # קווי הסילואט (ללא שינוי משמעותי)
    plt.plot(results_vcsf_ses1['k_values'], results_vcsf_ses1['silhouette_scores'], marker='o', color='blue')
    plt.plot(results_vgmm_ses1['k_values'], results_vgmm_ses1['silhouette_scores'], marker='o', color='red')
    plt.plot(results_vwm_ses1['k_values'], results_vwm_ses1['silhouette_scores'], marker='o', color='green')

    # --- סימון k האופטימלי ---
    best_k_vcsf_score_ses1 = results_vcsf_ses1['silhouette_scores'][
        results_vcsf_ses1['k_values'].index(results_vcsf_ses1['best_k'])]
    best_k_vgmm_score_ses1 = results_vgmm_ses1['silhouette_scores'][
        results_vgmm_ses1['k_values'].index(results_vgmm_ses1['best_k'])]
    best_k_vwm_score_ses1 = results_vwm_ses1['silhouette_scores'][
        results_vwm_ses1['k_values'].index(results_vwm_ses1['best_k'])]

    star_vcsf_ses1, = plt.plot(results_vcsf_ses1['best_k'], best_k_vcsf_score_ses1,
                               marker='*', markersize=15, color='blue',
                               label=f'VCSF k: {results_vcsf_ses1["best_k"]}')
    star_vgmm_ses1, = plt.plot(results_vgmm_ses1['best_k'], best_k_vgmm_score_ses1,
                               marker='*', markersize=15, color='red',
                               label=f'VGMM k: {results_vgmm_ses1["best_k"]}')
    star_vwm_ses1, = plt.plot(results_vwm_ses1['best_k'], best_k_vwm_score_ses1,
                              marker='*', markersize=15, color='green',
                              label=f'VWM k: {results_vwm_ses1["best_k"]}')

    plt.title('Combined Silhouette Scores for VCSF, VGMM, and VWM Datasets - session 1')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.xticks(range(2, 11))
    plt.grid(True, linestyle='--')

    # מקרא רק עם שלושת הכוכבים
    plt.legend(handles=[star_vcsf_ses1, star_vgmm_ses1, star_vwm_ses1])
    plt.show()

    ###############################################################
    # session 2

    # --- קריאה לפונקציה ושמירת תוצאות ---
    results_vcsf_ses2 = find_optimal_k_and_cluster(pca_df=vcsf_pca_df_ses2)
    results_vgmm_ses2 = find_optimal_k_and_cluster(pca_df=vgmm_pca_df_ses2)
    results_vwm_ses2 = find_optimal_k_and_cluster(pca_df=vwm_pca_df_ses2)

    # --- Create the combined plot ---
    plt.figure(figsize=(10, 7))

    plt.plot(results_vcsf_ses2['k_values'], results_vcsf_ses2['silhouette_scores'], marker='o', color='blue')
    plt.plot(results_vgmm_ses2['k_values'], results_vgmm_ses2['silhouette_scores'], marker='o', color='red')
    plt.plot(results_vwm_ses2['k_values'], results_vwm_ses2['silhouette_scores'], marker='o', color='green')

    # --- סימון k האופטימלי ---
    best_k_vcsf_score_ses2 = results_vcsf_ses2['silhouette_scores'][
        results_vcsf_ses2['k_values'].index(results_vcsf_ses2['best_k'])]
    best_k_vgmm_score_ses2 = results_vgmm_ses2['silhouette_scores'][
        results_vgmm_ses2['k_values'].index(results_vgmm_ses2['best_k'])]
    best_k_vwm_score_ses2 = results_vwm_ses2['silhouette_scores'][
        results_vwm_ses2['k_values'].index(results_vwm_ses2['best_k'])]

    star_vcsf_ses2, = plt.plot(results_vcsf_ses2['best_k'], best_k_vcsf_score_ses2,
                               marker='*', markersize=15, color='blue',
                               label=f'VCSF k: {results_vcsf_ses2["best_k"]}')
    star_vgmm_ses2, = plt.plot(results_vgmm_ses2['best_k'], best_k_vgmm_score_ses2,
                               marker='*', markersize=15, color='red',
                               label=f'VGMM k: {results_vgmm_ses2["best_k"]}')
    star_vwm_ses2, = plt.plot(results_vwm_ses2['best_k'], best_k_vwm_score_ses2,
                              marker='*', markersize=15, color='green',
                              label=f'VWM k: {results_vwm_ses2["best_k"]}')

    plt.title('Combined Silhouette Scores for VCSF, VGMM, and VWM Datasets - session 2')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.xticks(range(2, 11))
    plt.grid(True, linestyle='--')

    # מקרא רק עם שלושת הכוכבים
    plt.legend(handles=[star_vcsf_ses2, star_vgmm_ses2, star_vwm_ses2])
    plt.show()

    # --- Example Usage with your PCA DataFrames ---

    # Assuming you have already run the previous PCA code and have these DataFrames:
    # vcsf_pca_df, vgmm_pca_df, vwm_pca_df

    # Run K-Means on VCSF data (example: k=4)
    print("Running K-Means on VCSF PCA data...")
    vcsf_labels_ses1, vcsf_kmeans_ses1, vcsf_pca_clustered_ses1 = run_kmeans_on_pca_data(
        vcsf_pca_df_ses1,
        k=results_vcsf_ses1["best_k"],
        title="K-Means Clusters on VCSF PCA Data (Session 1)",
        csv_path="clusteres_output/vcsf/vcsf_clusters_ses1.csv",
        dataset_name="VCSF" ,random_state=seed,
    )

    print("\nRunning K-Means on VGMM PCA data...")
    vgmm_labels_ses1, vgmm_kmeans_ses1, vgmm_pca_clustered_ses1 = run_kmeans_on_pca_data(
        vgmm_pca_df_ses1,
        k=results_vgmm_ses1["best_k"],
        title="K-Means Clusters on VGMM PCA Data (Session 1)",
        csv_path="clusteres_output/vgmm/vgmm_clusters_ses1.csv",
        dataset_name="VGMM",random_state=seed,
    )

    print("\nRunning K-Means on VWM PCA data...")
    vwm_labels_ses1, vwm_kmeans_ses1, vwm_pca_clustered_ses1 = run_kmeans_on_pca_data(
        vwm_pca_df_ses1,
        k=results_vwm_ses1["best_k"],
        title="K-Means Clusters on VWM PCA Data (Session 1)",
        csv_path="clusteres_output/vwm/vwm_clusters_ses1.csv",
        dataset_name="VWM",random_state=seed,
    )

    #####################################################################################
    print("Running K-Means on VCSF PCA data...")
    vcsf_labels_ses2, vcsf_kmeans_ses2, vcsf_pca_clustered_ses2 = run_kmeans_on_pca_data(
        vcsf_pca_df_ses2,
        k=results_vcsf_ses2["best_k"],
        title="K-Means Clusters on VCSF PCA Data (Session 2)",
        csv_path="clusteres_output/vcsf/vcsf_clusters_ses2.csv",
        dataset_name="VCSF",random_state=seed,
    )

    print("\nRunning K-Means on VGMM PCA data...")
    vgmm_labels_ses2, vgmm_kmeans_ses2, vgmm_pca_clustered_ses2 = run_kmeans_on_pca_data(
        vgmm_pca_df_ses2,
        k=results_vgmm_ses2["best_k"],
        title="K-Means Clusters on VGMM PCA Data (Session 2)",
        csv_path="clusteres_output/vgmm/vgmm_clusters_ses2.csv",
        dataset_name="VGMM",random_state=seed
    )

    print("\nRunning K-Means on VWM PCA data...")
    vwm_labels_ses2, vwm_kmeans_ses2, vwm_pca_clustered_ses2 = run_kmeans_on_pca_data(
        vwm_pca_df_ses2,
        k=results_vwm_ses2["best_k"],
        title="K-Means Clusters on VWM PCA Data (Session 2)",
        csv_path="clusteres_output/vwm/vwm_clusters_ses2.csv",
        dataset_name="VWM",random_state=seed,
    )

    print("--- VCSF Loadings ---")
    save_top_loadings(
        pca=vcsf_pca_model_ses1,
        feature_names=vcsf_features,
        pc_index=0,
        out_csv="clusteres_output/vcsf/vcsf_pc1_loadings_ses1.csv",
        dataset_name="VCSF")

    print("\n--- VGMM Loadings ---")
    save_top_loadings(
        pca=vgmm_pca_model_ses1,
        feature_names=vgmm_features,
        pc_index=0,
        out_csv="clusteres_output/vgmm/vgmm_pc1_loadings_ses1.csv",
        dataset_name="VGMM"
    )

    print("\n--- VWM Loadings ---")
    save_top_loadings(
        pca=vwm_pca_model_ses1,
        feature_names=vwm_features,
        pc_index=0,
        out_csv="clusteres_output/vwm/vwm_pc1_loadings_ses1.csv",
        dataset_name="VWM")

    #############################################################################################################

    # --- הפעלת הפונקציה על כל דאטה-סט ---

    print("--- VCSF Loadings ---")
    save_top_loadings(
        pca=vcsf_pca_model_ses2,
        feature_names=vcsf_features,
        pc_index=0,
        out_csv="clusteres_output/vcsf/vcsf_pc1_loadings_ses2.csv",
        dataset_name="VCSF")

    print("\n--- VGMM Loadings ---")
    save_top_loadings(
        pca=vgmm_pca_model_ses2,
        feature_names=vgmm_features,
        pc_index=0,
        out_csv="clusteres_output/vgmm/vgmm_pc1_loadings_ses2.csv",
        dataset_name="VGMM"
    )

    print("\n--- VWM Loadings ---")
    save_top_loadings(
        pca=vwm_pca_model_ses2,
        feature_names=vwm_features,
        pc_index=0,
        out_csv="clusteres_output/vwm/vwm_pc1_loadings_ses2.csv",
        dataset_name="VWM")
    feature_files_2ses = {
        "VGMM": {
            "ses1": "clusteres_output/vgmm/vgmm_clusters_ses1.csv",
            "ses2": "clusteres_output/vgmm/vgmm_clusters_ses2.csv",
        },
        "VCSF": {
            "ses1": "clusteres_output/vcsf/vcsf_clusters_ses1.csv",
            "ses2": "clusteres_output/vcsf/vcsf_clusters_ses2.csv",
        },
        "VWM": {
            "ses1": "clusteres_output/vwm/vwm_clusters_ses1.csv",  # זה היה בלי "_ses1"
            "ses2": "clusteres_output/vwm/vwm_clusters_ses2.csv",
        },
    }

    feature_files_1ses = {
        "VGMM": {
            "ses1": "clusteres_output/vgmm/vgmm_clusters_ses1.csv",
        },
        "VCSF": {
            "ses1": "clusteres_output/vcsf/vcsf_clusters_ses1.csv",
        },
        "VWM": {
            "ses1": "clusteres_output/vwm/vwm_clusters_ses1.csv",  # זה היה בלי "_ses1"
        },
    }

    summary_df_2ses = build_cluster_summary(
        feature_files_2ses,
        subject_col="Subject",
        output_csv="clusteres_output/cluster_summary_session1_2.csv"
    )

    summary_df_1ses = build_cluster_summary(
        feature_files_1ses,
        subject_col="Subject",
        output_csv="clusteres_output/cluster_summary_only_session1.csv"
    )

    MAP =  'groups_trajectory_from_q.csv'
    datasets_ses1 = [
        ("VCSF", "clusteres_output/vcsf/vcsf_clusters_ses1.csv"),
        ("VGMM", "clusteres_output/vgmm/vgmm_clusters_ses1.csv"),
        ("VWM", "clusteres_output/vwm/vwm_clusters_ses1.csv"),
    ]

    groups = ["all good", "all bad", "worsning", "fluctuating", "improving"]  # בדיוק כפי שכתבת

    marked_vgmm_s1 = plot_clusters_with_group_overlay(
        "clusteres_output/vgmm/vgmm_clusters_ses1.csv",
        mapping_path=MAP,
        mapping_subject_col="Subject_Code",
        mapping_group_col="trajectory_group",
        wanted_groups=["all good", "all bad", 'worsning', 'fluctuating', 'improving'],
        subject_col=0,  # <-- first column BY POSITION even if it has no header
        dataset_name="VGMM",
        title="VGMM S1 — Original Clusters with Group Overlay"
    )

    marked_vcsf_s1 = plot_clusters_with_group_overlay(
        "clusteres_output/vcsf/vcsf_clusters_ses1.csv",
        mapping_path=MAP,
        mapping_subject_col="Subject_Code",
        mapping_group_col="trajectory_group",
        wanted_groups=["all good", "all bad", 'worsning', 'fluctuating', 'improving'],
        subject_col=0,  # <-- first column BY POSITION even if it has no header
        dataset_name="VCSF",
        title="VCSF S1 — Original Clusters with Group Overlay"
    )

    marked_vwm_s1 = plot_clusters_with_group_overlay(
        "clusteres_output/vwm/vwm_clusters_ses1.csv",
        mapping_path=MAP,
        mapping_subject_col="Subject_Code",
        mapping_group_col="trajectory_group",
        wanted_groups=["all good", "all bad", 'worsning', 'fluctuating', 'improving'],
        subject_col=0,  # <-- first column BY POSITION even if it has no header
        dataset_name="VWM",
        title="VGWM S1 — Original Clusters with Group Overlay"
    )
    subplot_clusters_by_group(
        datasets_ses1,
        groups,
        mapping_path=MAP,
        mapping_subject_col="Subject_Code",
        mapping_group_col="trajectory_group",
        subject_col="Subject_Code",  # העמודה הראשונה ללא כותרת
        alpha_all=0.9,
        s_all=20,
        s_marked=100,
        figsize=(18, 10)
    )


if __name__ == "__main__":
    main()
