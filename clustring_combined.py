import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path
import os
import re
import warnings
import os


# -----------------------------
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

    # --- הגדרות כלליות ---
    ROOT = Path("data/cat12")
    SUBFOLDERS = ["NT", "CT"]
    OUTPUT_CSV = ROOT / "sessions_summary.csv"

    PATTERN = re.compile(r'^(?P<id>(?:NT|CT)\d+)(?:[ _-]?(?P<session>[12]))?\b', re.IGNORECASE)

    rows = []

    # --- איסוף נתוני סשנים מקבצים ויצירת sessions_df ---
    for sub in SUBFOLDERS:
        folder = (ROOT / sub).resolve()
        if not folder.is_dir():
            print(f"⚠️ לא נמצאה תיקייה (או שזה לא תיקייה): {folder}")
            continue

        for p in folder.iterdir():
            if not p.is_file() or p.name.startswith("~$") or p.suffix.lower() not in {".xlsx", ".csv"}:
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
        return
    else:
        sessions_df = sessions_df.drop_duplicates().sort_values(by=["subject_id", "session"]).reset_index(drop=True)
        sessions_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        print(f"✅ summery session files saved: {OUTPUT_CSV}")
        print(sessions_df.head(20))

    # --- הגדרות נתיב ---
    base_path = "data/cat12/"
    out_base = "clusteres_output/combined"
    os.makedirs(out_base, exist_ok=True)

    session_1_subjects = sessions_df[sessions_df['session'] == 1]['subject_id'].tolist()
    session_2_subjects = sessions_df[sessions_df['session'] == 2]['subject_id'].tolist()

    # ====================================================================
    # --- חלק 1: עיבוד סשן 1 ---
    # ====================================================================
    files_ses1 = []

    for subject in session_1_subjects:
        group = subject[:2]
        file_path = os.path.join(base_path, group, f'{subject}_1.xlsx')

        if os.path.exists(file_path):
            files_ses1.append({'subject': subject, 'path': file_path})

    normalized_dfs_list_ses1 = []
    subjects_included_ses1 = []

    # --- לולאה על קבצי סשן 1 ---
    for file_info in files_ses1:
        subject_id = file_info['subject']
        file_path = file_info['path']

        try:
            subject_data = pd.read_excel(
                file_path, header=0,
                names=['name', 'Vcsf', 'Vgm', 'Vwm', 'Euler', 'TIV']
            )
        except Exception:
            continue

        # TIV checks
        tiv_series = pd.to_numeric(subject_data['TIV'], errors='coerce').dropna()
        if tiv_series.empty or tiv_series.iloc[0] == 0:
            continue

        tiv_value = tiv_series.iloc[0]

        # חישוב הערכים אחרי חלוקה ב-TIV (נרמול)
        vcsf_norm = subject_data['Vcsf'] / tiv_value
        vgmm_norm = subject_data['Vgm'] / tiv_value
        vwm_norm = subject_data['Vwm'] / tiv_value
        region_names = subject_data['name']

        # --- יצירת שורת הנתונים המאוחדת לנבדק הנוכחי (תיקון לוגי למניעת אזהרות) ---
        subject_data_dict = {}

        # בניית כל העמודות במילון לפני יצירת ה-DataFrame
        for i, region in enumerate(region_names):
            subject_data_dict[f'{region}_Vcsf'] = vcsf_norm.iloc[i]
            subject_data_dict[f'{region}_Vgmm'] = vgmm_norm.iloc[i]
            subject_data_dict[f'{region}_Vwm'] = vwm_norm.iloc[i]

        # יצירת ה-DataFrame בשלמותו בצעד אחד
        subject_row = pd.DataFrame({
            col: [value] for col, value in subject_data_dict.items()
        }, index=[subject_id])

        normalized_dfs_list_ses1.append(subject_row)
        subjects_included_ses1.append(subject_id)

    # --- שרשור סשן 1 ושמירה ---
    if normalized_dfs_list_ses1:
        combined_final_df_ses1 = pd.concat(normalized_dfs_list_ses1, axis=0)
        combined_final_df_ses1.index.name = 'SubjectID'

        combined_out_ses1 = os.path.join(out_base, "all_subjects_combined_ses1.csv")
        combined_final_df_ses1.to_csv(combined_out_ses1)

        num_subjects_processed_1 = len(subjects_included_ses1)
        print("\n" + "=" * 60)
        print("Combined Data Saved Successfully (Normalized by TIV):")
        print(f"• File: {combined_out_ses1}")
        print(f"• Shape (Subjects x Features): {combined_final_df_ses1.shape}")
        print("-" * 60)
        print(f"\nA total of {num_subjects_processed_1} subjects (Session 1) were included after TIV checks.")
    else:
        print("\nNo subjects were processed or included for Session 1 after TIV checks.")
        num_subjects_processed_1 = 0

    # ====================================================================
    # --- חלק 2: עיבוד סשן 2 ---
    # ====================================================================
    files_ses2 = []  # רשימה נפרדת לקבצי סשן 2

    for subject in session_2_subjects:
        group = subject[:2]
        # נתיב הקובץ הוא עבור סשן 2 (_2.xlsx)
        file_path = os.path.join(base_path, group, f'{subject}_2.xlsx')

        if os.path.exists(file_path):
            files_ses2.append({'subject': subject, 'path': file_path})

    normalized_dfs_list_ses2 = []
    subjects_included_ses2 = []

    # --- לולאה על קבצי סשן 2 ---
    for file_info in files_ses2:
        subject_id = file_info['subject']
        file_path = file_info['path']

        try:
            subject_data = pd.read_excel(
                file_path, header=0,
                names=['name', 'Vcsf', 'Vgm', 'Vwm', 'Euler', 'TIV']
            )
        except Exception:
            continue

        # TIV checks (אותה לוגיקה)
        tiv_series = pd.to_numeric(subject_data['TIV'], errors='coerce').dropna()
        if tiv_series.empty or tiv_series.iloc[0] == 0:
            continue

        tiv_value = tiv_series.iloc[0]

        # חישוב הערכים אחרי חלוקה ב-TIV (נרמול)
        vcsf_norm = subject_data['Vcsf'] / tiv_value
        vgmm_norm = subject_data['Vgm'] / tiv_value
        vwm_norm = subject_data['Vwm'] / tiv_value
        region_names = subject_data['name']

        # --- יצירת שורת הנתונים המאוחדת לנבדק הנוכחי (תיקון לוגי למניעת אזהרות) ---
        subject_data_dict = {}

        # בניית כל העמודות במילון לפני יצירת ה-DataFrame
        for i, region in enumerate(region_names):
            subject_data_dict[f'{region}_Vcsf'] = vcsf_norm.iloc[i]
            subject_data_dict[f'{region}_Vgmm'] = vgmm_norm.iloc[i]
            subject_data_dict[f'{region}_Vwm'] = vwm_norm.iloc[i]

        # יצירת ה-DataFrame בשלמותו בצעד אחד
        subject_row = pd.DataFrame({
            col: [value] for col, value in subject_data_dict.items()
        }, index=[subject_id])

        normalized_dfs_list_ses2.append(subject_row)
        subjects_included_ses2.append(subject_id)

    # --- שרשור סשן 2 ושמירה ---
    if normalized_dfs_list_ses2:
        combined_final_df_ses2 = pd.concat(normalized_dfs_list_ses2, axis=0)
        combined_final_df_ses2.index.name = 'SubjectID'

        combined_out_ses2 = os.path.join(out_base, "all_subjects_combined_ses2.csv")
        combined_final_df_ses2.to_csv(combined_out_ses2)

        num_subjects_processed_2 = len(subjects_included_ses2)
        print("\n" + "=" * 60)
        print("Combined Data Saved Successfully (Normalized by TIV):")
        print(f"• File: {combined_out_ses2}")
        print(f"• Shape (Subjects x Features): {combined_final_df_ses2.shape}")
        print("-" * 60)
        print(f"\nA total of {num_subjects_processed_2} subjects (Session 2) were included after TIV checks.")
    else:
        print("\nNo subjects were processed or included for Session 2 after TIV checks.")
        num_subjects_processed_2 = 0
    # הגדר את נתיבי הקבצים שלך
    file_paths = {
        "combined": "clusteres_output/combined/all_subjects_combined_ses1.csv",

    }

    # חלץ את שמות הפיצ'רים מכל דאטה-סט
    combined_features = get_feature_names(file_paths["combined"])


    # הדפס את התוצאות
    print("--- שמות הפיצ'רים עבור כל דאטה-סט ---")

    print(f"\n✅ VCSF features ({len(combined_features)}):")
    print(combined_features)


    # Example usage
    print("\nExplained Variance session 1 combined\n")

    combined_ses1 = "clusteres_output/combined/all_subjects_combined_ses1.csv"
    num90_combined_ses1, num95_combined_ses1, num100_combined_ses1 = find_optimal_pca_dimensions(combined_ses1,'figures/combined')

    print("Components for 90%:", num90_combined_ses1)
    print("Components for 95%:", num95_combined_ses1)
    print("Components for 100%:", num100_combined_ses1)

    print("\nExplained Variance session 2 combined\n")

    combined_ses2 = "clusteres_output/combined/all_subjects_combined_ses2.csv"
    num90_combined_ses2, num95_combined_ses2, num100_combined_ses2 = find_optimal_pca_dimensions(combined_ses2,'figures/combined')

    print("Components for 90%:", num90_combined_ses2)
    print("Components for 95%:", num95_combined_ses2)
    print("Components for 100%:", num100_combined_ses2)


    # --- Perform PCA on VCSF data with 3 dimensions ---
    combined_df_ses1 = pd.read_csv(combined_ses1,index_col = 0)
    print("\n" + "=" * 50)
    print(f"PCA on VCSF Data with {num90_combined_ses1} dimensions:")
    combined_pca_df_ses1, combined_pca_model_ses1 = perform_pca(combined_df_ses1, num90_combined_ses1)
    print(f"Original Data Shape: {combined_df_ses1.shape}")
    print(f"Transformed Data Shape: {combined_pca_df_ses1.shape}")
    print("\nExplained variance ratio:")
    for i, var_ratio_ses1 in enumerate(combined_pca_model_ses1.explained_variance_ratio_):
        print(f" - PC{i + 1}: {var_ratio_ses1:.4f}")
    print("\nFirst 5 rows of the PCA-transformed data:")
    print(combined_pca_df_ses1.head())
    print("=" * 50)

    # --- Perform PCA on VCSF data with 3 dimensions ---
    combined_df_ses2 = pd.read_csv(combined_ses2,index_col = 0)
    print("\n" + "=" * 50)
    print(f"PCA on VCSF Data with {num90_combined_ses2} dimensions:")
    combined_pca_df_ses2, combined_pca_model_ses2 = perform_pca(combined_df_ses2, num90_combined_ses2)
    print(f"Original Data Shape: {combined_df_ses2.shape}")
    print(f"Transformed Data Shape: {combined_pca_df_ses2.shape}")
    print("\nExplained variance ratio:")
    for i, var_ratio_ses2 in enumerate(combined_pca_model_ses1.explained_variance_ratio_):
        print(f" - PC{i + 1}: {var_ratio_ses2:.4f}")
    print("\nFirst 5 rows of the PCA-transformed data:")
    print(combined_pca_df_ses2.head())
    print("=" * 50)

    # ===== Example usage (replace with your real data) =====
    groups = ['pre', 'post']


    plot_grouped_bars(groups,[num90_combined_ses1, num90_combined_ses2],
                      [num95_combined_ses1, num95_combined_ses2],
                      [num100_combined_ses1,num100_combined_ses2],
                      s1_label="90% variability",
                      s2_label="95% variability",
                      s3_label="100% variability",
                      title="Number of Dimentions for each Time period -session 1 ",
                      ylabel="# of dimentions")



    # --- קריאה לפונקציה ושמירת תוצאות ---
    results_combined_ses1 = find_optimal_k_and_cluster(pca_df=combined_pca_df_ses1)
    results_combined_ses2 = find_optimal_k_and_cluster(pca_df=combined_pca_df_ses2)


    # --- Create the combined plot ---
    plt.figure(figsize=(10, 7))

    # קווי הסילואט (ללא שינוי משמעותי)
    plt.plot(results_combined_ses1['k_values'], results_combined_ses1['silhouette_scores'], marker='o', color='blue')
    plt.plot(results_combined_ses2['k_values'], results_combined_ses2['silhouette_scores'], marker='o', color='red')


    # --- סימון k האופטימלי ---
    best_k_combined_score_ses1 = results_combined_ses1['silhouette_scores'][
        results_combined_ses1['k_values'].index(results_combined_ses1['best_k'])]
    best_k_combined_score_ses2 = results_combined_ses2['silhouette_scores'][
        results_combined_ses2['k_values'].index(results_combined_ses2['best_k'])]

    star_combined_ses1, = plt.plot(results_combined_ses1['best_k'], best_k_combined_score_ses1,
                               marker='*', markersize=15, color='blue',
                               label=f'pre 1 k: {results_combined_ses1["best_k"]}')
    star_combined_ses2, = plt.plot(results_combined_ses2['best_k'], best_k_combined_score_ses2,
                               marker='*', markersize=15, color='red',
                               label=f'post k: {results_combined_ses2["best_k"]}')
    plt.title('Combined Silhouette Scores for VCSF, VGMM, and VWM Datasets ')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.xticks(range(2, 11))
    plt.grid(True, linestyle='--')

    # מקרא רק עם שלושת הכוכבים
    plt.legend(handles=[star_combined_ses1, star_combined_ses2])
    plt.show()

    ###
    # --- Example Usage with your PCA DataFrames ---

    # Assuming you have already run the previous PCA code and have these DataFrames:
    # vcsf_pca_df, vgmm_pca_df, vwm_pca_df

    # Run K-Means on VCSF data (example: k=4)
    print("Running K-Means on combined PCA data...")
    combined_labels_ses1, combined_kmeans_ses1, combined_pca_clustered_ses1 = run_kmeans_on_pca_data(
        combined_pca_df_ses1,
        k=results_combined_ses1["best_k"],
        title="K-Means Clusters on combined PCA Data (Session 1)",
        csv_path="clusteres_output/combined/combined_clusters_ses1.csv",
        dataset_name="combined" ,random_state=seed,
    )

    # Run K-Means on VCSF data (example: k=4)
    print("Running K-Means on combined PCA data...")
    combined_labels_ses2, combined_kmeans_ses3, combined_pca_clustered_ses2 = run_kmeans_on_pca_data(
        combined_pca_df_ses2,
        k=results_combined_ses2["best_k"],
        title="K-Means Clusters on combined PCA Data (Session 2)",
        csv_path="clusteres_output/combined/combined_clusters_ses2.csv",
        dataset_name="combined" ,random_state=seed,
    )


    print("--- combined session 1 Loadings ---")
    save_top_loadings(
        pca=combined_pca_model_ses1,
        feature_names=combined_features,
        pc_index=0,
        out_csv="clusteres_output/combined/combined_pc1_loadings_ses1.csv",
        dataset_name="combined")






    #############################################################################################################

    # --- הפעלת הפונקציה על כל דאטה-סט ---

    print("--- combined session 2 Loadings ---")
    save_top_loadings(
        pca=combined_pca_model_ses2,
        feature_names=combined_features,
        pc_index=0,
        out_csv="clusteres_output/combined/combined_pc1_loadings_ses2.csv",
        dataset_name="combined")

    feature_files_2ses = {
        "combined": {
            "ses1": "clusteres_output/combined/combined_clusters_ses1.csv",
            "ses2": "clusteres_output/combined/combined_clusters_ses2.csv"}}

    feature_files_1ses = {
        "VGMM": {
            "ses1": "clusteres_output/combined/combined_clusters_ses1.csv",
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
        ("combined", "clusteres_output/combined/combined_clusters_ses1.csv"),
    ]

    groups = ["all good", "all bad", "worsning", "fluctuating", "improving"]  # בדיוק כפי שכתבת

    marked_vgmm_s1 = plot_clusters_with_group_overlay(
        "clusteres_output/combined/combined_clusters_ses1.csv",
        mapping_path=MAP,
        mapping_subject_col="Subject_Code",
        mapping_group_col="trajectory_group",
        wanted_groups=["all good", "all bad", 'worsning', 'fluctuating', 'improving'],
        subject_col=0,  # <-- first column BY POSITION even if it has no header
        dataset_name="combined",
        title="combinedS1 — Original Clusters with Group Overlay"
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
        s_marked=1---00,
        figsize=(30,5)
    )


if __name__ == "__main__":
    main()
