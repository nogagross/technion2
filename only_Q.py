import os, random, numpy as np
import pandas as pd
from pathlib import Path
import shutil
from questionnaire_functions import (export_questionnaire,analyze_questionnaire)
from pca_functions import (find_optimal_pca_dimensions)
from clustring_functions import(pca_kmeans_minimal_outputs,run_kmeans_clustering)
from vizualizations_functions import (plot_grouped_bars,plot_multi_dataset_scatters_colored)
import matplotlib.pyplot as plt


def main():
    seed = 17
    np.random.seed(seed)

    # -*- coding: utf-8 -*-

    # נתיב לקובץ הנתונים שלך (קובץ אקסל)
    # עדכן את הנתיב לשם הקובץ האמיתי שלך
    file_path = 'data/q_data/Study_Questionnaire_Responses_June.xlsx'

    # רשימת העמודות שברצונך לבחור
    # ערוך את הרשימה הזו כדי להתאים לצרכים שלך
    columns_to_select = ["Subject_Code", "b_questionnaire_completion", "after_questionnaire_completion", "t1_MAAS_date",
                         "t2_PHQ_date", "t3_MAAS_date"]

    df = pd.read_excel(file_path)
    columns = df.columns.tolist()
    print("the columns in the dataframe:")
    for col in columns:
        print("-", col)

    # סינון העמודות ובחירת הרצויות בלבד
    selected_df = df.loc[:, columns_to_select]

    print("DataFrame מקורי:")
    print(f"שורות: {selected_df .shape[0]}, עמודות: {selected_df .shape[1]}")

    print("\nDataFrame חדש עם העמודות שנבחרו:")
    print(f"שורות: {selected_df.shape[0]}, עמודות: {selected_df.shape[1]}")



    # הצגת ה-DataFrame החדש
    print("\nהצגת ה-DataFrame החדש (5 השורות הראשונות):")
    print(selected_df.head())

    # see the columns in the dataframe
    columns = selected_df.columns.tolist()
    print("the columns in the dataframe:")
    for col in columns:
        print("-", col)\

    # remove the unnecessery columns (columns related to war ,metadata, queshtions from an interview)
    columns_to_drop = [
        'sasrq_date',
        'sasrq_1', 'sasrq_2', 'sasrq_3', 'sasrq_4', 'sasrq_5', 'sasrq_6', 'sasrq_7', 'sasrq_8', 'sasrq_9', 'sasrq_10',
        'sasrq_11', 'sasrq_12', 'sasrq_13', 'sasrq_14', 'sasrq_15', 'sasrq_16', 'sasrq_17', 'sasrq_18', 'sasrq_19',
        'sasrq_20',
        'sasrq_21', 'sasrq_22', 'sasrq_23', 'sasrq_24', 'sasrq_25', 'sasrq_26', 'sasrq_27', 'sasrq_28', 'sasrq_29',
        'sasrq_30',
        'sasrq_31', 'sasrq_total',
        'PROMOTE_date', 'war_pregnancy_status',
        'PROMOTE_1', 'PROMOTE_2', 'PROMOTE_3', 'PROMOTE_4', 'PROMOTE_5', 'PROMOTE_6', 'PROMOTE_7', 'PROMOTE_8',
        'PROMOTE_9', 'PROMOTE_10', 'PROMOTE_11', 'PROMOTE_12', 'PROMOTE_13', 'PROMOTE_14', 'PROMOTE_15',
        'WEQ_date',
        'WEQ_1', 'WEQ_2', 'WEQ_3', 'WEQ_4', 'WEQ_5', 'WEQ_6', 'WEQ_7', 'WEQ_8', 'WEQ_9', 'WEQ_10',
        'WEQ_11', 'WEQ_12', 'WEQ_13', 'WEQ_14', 'WEQ_15', 'WEQ_16', 'WEQ_17', 'WEQ_18', 'WEQ_19',
        'Posttraumatic _Growth_Inventory_date', 'Posttraumatic _Growth_1', 'Posttraumatic _Growth_2',
        'Posttraumatic _Growth_3',
        'Posttraumatic _Growth_4', 'Posttraumatic _Growth_5', 'Posttraumatic _Growth_6', 'Posttraumatic _Growth_7',
        'Posttraumatic _Growth_8', 'Posttraumatic _Growth_9', 'Posttraumatic _Growth_10', 'Posttraumatic _Growth_total',
        'Posttraumatic _Growth_RO', 'Posttraumatic _Growth_NP', 'Posttraumatic _Growth_PS',
        'Posttraumatic _Growth_SC', 'Posttraumatic _Growth_AL',
        'war_pcl5_date',
        'war_pcl_1', 'war_pcl_2', 'war_pcl_3', 'war_pcl_4', 'war_pcl_5', 'war_pcl_6', 'war_pcl_7', 'war_pcl_8',
        'war_pcl_9',
        'war_pcl_10', 'war_pcl_11', 'war_pcl_12', 'war_pcl_13', 'war_pcl_14', 'war_pcl_15', 'war_pcl_16', 'war_pcl_17',
        'war_pcl_18', 'war_pcl_19', 'war_pcl_20', 'war_pcl_total', 'war_pcl_cutoff', 'war_pcl_dsm',
        'war_phq_date',
        'war_phq_1', 'war_phq_2', 'war_phq_3', 'war_phq_4', 'war_phq_5', 'war_phq_6', 'war_phq_7', 'war_phq_8',
        'war_phq_9',
        'war_phq_10', 'war_phq_total',
        'war_gad7_date',
        'war_gad_1', 'war_gad_2', 'war_gad_3', 'war_gad_4', 'war_gad_5', 'war_gad_6', 'war_gad_7', 'war_gad_total',
        'country_of_birth',
        'Country_of_Birth_(Israel/Other)',
        'country_of_birth_mom',
        'country_of_birth_dad',
        'year_of_aliyah',
        'family_status',
        'Years_Marriage',
        'education_years',
        'education_years_code',
        'education_years_partner',
        'education_years_partner_code',
        'profession',
        'profession_partner',
        'religion',
        'religion_other',
        'income',
        'b_questionnaire_completion',
        'after_questionnaire_completion',
        'first_fmri_scan_date',
        'second_fmri_scan_date',
        'third_fmri_scan_date',
        'b_questionnaire_and_fmri_days_difference',
        'pregnancy_start_date',
        'b_fmri_and_pregnancy_days_difference',
        'newborn_birth_date',
        'Days_from_Birth_to_Questionnaire_Completion',
        'Demographics_Date',
        'date_of_birth',
        'diamond_interview_date',
        'b_diamond_anxiety_phobias_past',
        'b_diamond_Anxiety_phobias_present',
        'b_diamond_ocd_past',
        'b_diamond_ocd_present',
        'b_diamond_adhd_past',
        'b_diamond_adhd_present',
        'b_diamond_depression_past',
        'b_diamond_depression_present',
        'b_diamond_adjustment_past',
        'b_diamond_adjustment_present',
        'b_diamond_ptsd_past',
        'b_diamond_ptsd_present',
        'b_diamond_eating_disorder_past',
        'b_diamond_eating_disorder_present',
        'b_diamond_PMS_past',
        'b_diamond_PMS_present',
        'b_diamond_other_past',
        'b_diamond_other_present',
        'b_diamond_past',
        'b_diamond_present',
        't1_Fertility_treatments',
        'Conception_method','second_fmri_questionnaire_date','newborn_birth_date.2',
        '2FMRI_period_since_birth','2FMRI_last_period_date','2FMRI_breastfeeding',
        '2FMRI_average_sleep_hours',
        '2FMRI_birth_control_pills_usage',
        '2FMRI_additional_notes',
        'war_pregnancy_status',
        'after_bits_date',
        'after_PHQ_date',
        'after_GAD7_date',
        'after_MPAS_date',
        'after_DERS_date',
        'after_LHQ_date',
        'b_ctq_Date',
        'b_lec_date',
        'b_PCL5_date',
        'b_strength_date',
        'b_PHQ9_date'
        'b_GAD7_date',
        'b_PBI_date',
        'b_DES_date',
        't1_DES_date',
        't1_PHQ_date',
        't2_PHQ_date',
        't3_PHQ_date',
        't1_GAD7_date',
        't1_MAAS_date',
        't2_MAAS_date',
        't3_MAAS_date',
        't3_GAD7_date',
        't2_GAD7_date',
        'b_PHQ9_date',
        'b_GAD7_date',
        'b_social_support_date',
        'b_DERS_date',
        'b_LHQ_date',
        'b_IRI_date',
        'birth_week',
        'birth_type',
        'after_DES_date',
        'after_CTQ_date',
        'b_lec_1a',
        "b_lec_2a",
        "b_lec_3a",
        "b_lec_4a",
        "b_lec_5a",
        "b_lec_6a",
        "b_lec_7a",
        "b_lec_8a",
        "b_lec_9a",
        "b_lec_10a",
        "b_lec_11a",
        "b_lec_12a",
        "b_lec_13a",
        "b_lec_14a",
        "b_lec_15a",
        "b_lec_16a",
        "b_lec_17a",
        "b_ctq_total",
        "b_ctq_NEGLECT",
        "b_ctq_ABUSE  ",
        "b_ctq_sexual_abuse",
        "b_ctq_physical_abuse",
        "b_ctq_emotional_abuse","b_ctq_physical_neglect",
        "b_ctq_emotional_neglect","b_ctq_sexual_abuse_cutoff",
        "b_ctq_physical_abuse_cutoff",
        "b_ctq_emotional_abuse_cutoff","b_ctq_physical_neglect_cutoff","b_ctq_emotional_neglect_cutoff",
        "b_ctq_denial _score",
        "b_lec_0_to_16_total",
        "b_lec_interpersonal_events",
        "b_lec_non_interpersonal_events",
        "b_pcl_total",
        "b_pcl_cutoff",
        "b_pcl_dsm",
        "b_strength_average",
        "b_PHQ_total",
        "b_GAD7_total",
        "b_social_support_total", "b_PBI_mom_care", "b_PBI_dad_care","b_PBI_mom_overprotection", "b_PBI_dad_overprotection",
        "b_DERS_total",
        "b_DERS_Nonacceptance_Emotional_Responses",
        "b_DERS_Goal_Directed_Behavior",
        "b_DERS_Impulse_Control",
        "b_DERS_Lack_Emotional_Awareness",
        "b_DERS_Emotion_Regulation_Strategies",
        "b_DERS_Lack_Emotional_Clarity",
        "b_DES_average",
        "b_DES_Absorption ",
        "b_DES_Amnesia",
        "b_DES_Depersonalization ",
        "b_LHQ_total",
        "b_IRI_Perspective_Taking",
        "b_IRI_Empathic_Concern",
        "b_IRI_Personal_Distress",
        "b_IRI_Fantasy",
        't1_DES_total',
        't1_DES_Absorption ',
        't1_DES_Amnesia',
        't1_DES_Depersonalization',
        "T1_PHQ_total",
        'T2_PHQ_total',
        'T3_PHQ_total',
        'T1_GAD7_total',
        'T2_GAD7_total',
        'T3_GAD7_total',
        't1_MAAS_Attachment ',
        't1_MAAS_Preoccupation ',
        't2_MAAS_total',
        't2_MAAS_Attachment ',
        't2_MAAS_Preoccupation ',
        't3_MAAS_total',
        't3_MAAS_Attachment ',
        't3_MAAS_Preoccupation ',
        'after_bits_PTSD_total',
        'after_bits_birth_symptoms',
        'after_bits_General_symptoms',
        'after_bits_Dissociatie_symptoms',
        'after_bits_PTSD_criterion',
        'after_bits_Re_experiencing ',
        'after_bits_Avoidance ',
        'after_bits_Negative_Cognitions ',
        'after_bits_Hyperarousal',
        'after_PHQ_total',
        'after_GAD7_total',
        'after_MPAS_total',
        'after_MPAS_proximity',
        'after_MPAS_Acceptance',
        'after_MPAS_Tolerance',
        'after_MPAS_Competence',
        'after_MPAS_Competence',
        'after_MPAS_Attachment',
        'after_MPAS_Hostility',
        'after_MPAS_Interaction',
        'after_DES_total','after_DES_Absorption ',
        'after_DES_Amnesia',
        'after_DES_Depersonalization ',
        'after_CTQ_total',
        'after_CTQ_cutoff',
        'after_CTQ_NEGLECT',
        'after_CTQ_ABUSE  ',
        'after_CTQ_sexual_abuse',
        'after_CTQ_physical_abuse',
        'after_CTQ_emotional_abuse',
        'after_CTQ_physical_neglect',
        'after_CTQ_emotional_neglect',
        'after_CTQ_sexual_abuse_cutoff','after_CTQ_physical_abuse_cutoff',
        'after_CTQ_emotional_abuse_cutoff',
        'after_CTQ_physical_neglect_cutoff',
        'after_CTQ_emotional_neglect_cutoff',
        'after_CTQ_denial _score', 'after_DERS_total',
        'after_DERS_Nonacceptance_Emotional_Responses',
        'after_DERS_Goal_Directed_Behavior',
        'after_DERS_Impulse_Control',
        'after_DERS_Lack_Emotional_Awareness',
        'after_DERS_Emotion_Regulation_Strategies',
        'after_DERS_Lack_Emotional_Clarity','after_LHQ_total',
        't1_MAAS_total',
        'b_ctq_cutoff',
        'b_PBI_mom_overprotection',
        'b_PBI_dad_care','t1_DES_Depersonalization '
    ]
    df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')
    print(df_cleaned.columns.tolist())

    # Remove the subjects who were dropped out of the study
    df_filtered = df_cleaned[df_cleaned['Dropped_Out_of_Study'] != 1]

    # calculating the number of nan in each column
    counts = df_filtered.notna().sum()

    # Calculating the percentage of respondents who answered the specific questionnaire out of all respondents
    total = len(df_filtered)
    percentages = (counts / total * 100)

    # creating a new table of number of students and the precentages of the subjects
    summary_df = pd.DataFrame([counts, percentages], index=['number of subject', '% of subjects'])


    # שמירת הטבלה כקובץ Excel
    summary_df.to_excel("data/q_data/summary_table.xlsx")

    # הרצה עבור כל סט
    export_questionnaire(df_cleaned, 'b', 'data/q_data/time_points/b_questionnaire')
    export_questionnaire(df_cleaned, 't1', 'data/q_data/time_points/t1_questionnaire')
    export_questionnaire(df_cleaned, 't2', 'data/q_data/time_points/t2_questionnaire')
    export_questionnaire(df_cleaned, 't3', 'data/q_data/time_points/t3_questionnaire')
    export_questionnaire(df_cleaned, 'after', 'data/q_data/time_points/after_questionnaire')

    q_before = 'data/q_data/time_points/b_questionnaire.csv'
    q_t1  = 'data/q_data/time_points/t1_questionnaire.csv'
    q_t2 = 'data/q_data/time_points/t2_questionnaire.csv'
    q_t3 = 'data/q_data/time_points/t3_questionnaire.csv'
    q_after = 'data/q_data/time_points/after_questionnaire.csv'

    analyze_questionnaire('b',q_before)
    analyze_questionnaire( 't1', q_t1)
    analyze_questionnaire( 't2', q_t2)
    analyze_questionnaire('t3', q_t3)
    analyze_questionnaire( 'after', q_after)


    n_dims_b_90, n_dims_b_95, n_dims_b_100 = find_optimal_pca_dimensions(q_before,save_dir="data/only_Q_outputs/ses1_figures")
    print(f"\nRecommended number of dimensions for 90%: {n_dims_b_90}")

    n_dims_t1_90, n_dims_t1_95, n_dims_t1_100 = find_optimal_pca_dimensions(q_t1,save_dir="data/only_Q_outputs/ses1_figures")
    print(f"\nRecommended number of dimensions for 90%: {n_dims_t1_90}")

    n_dims_t2_90, n_dims_t2_95, n_dims_t2_100 = find_optimal_pca_dimensions(q_t2,save_dir="data/only_Q_outputs/ses1_figures")
    print(f"\nRecommended number of dimensions for 90%: {n_dims_t2_90}")

    n_dims_t3_90, n_dims_t3_95, n_dims_t3_100 = find_optimal_pca_dimensions(q_t3,save_dir="data/only_Q_outputs/ses1_figures")
    print(f"\nRecommended number of dimensions for 90%: {n_dims_t3_90}")

    n_dims_after_90, n_dims_after_95, n_dims_after_100 = find_optimal_pca_dimensions(q_after,save_dir="data/only_Q_outputs/ses1_figures")
    print(f"\nRecommended number of dimensions for 90%: {n_dims_after_90}")

    # ===== Example usage (replace with your real data) =====
    groups = ["pre", "trimester 1", "trimester 2", "trimester 3", "after"]
    s1 = [n_dims_b_90, n_dims_t1_90, n_dims_t2_90, n_dims_t3_90, n_dims_after_90]  # values for column 1 in each group
    s2 = [n_dims_b_95, n_dims_t1_95, n_dims_t2_95, n_dims_t3_95, n_dims_after_95]  # values for column 2
    s3 = [n_dims_b_100, n_dims_t1_100, n_dims_t2_100, n_dims_t3_100, n_dims_after_100]  # values for column 2

    plot_grouped_bars(groups, s1, s2, s3,
                      s1_label="Explain 90% variability",
                      s2_label="Explain 95% variability",
                      s3_label="Explain 100% variability",
                      title="Number of Dimentions for each Time period",
                      ylabel="# of dimentions")

    df_b = pd.read_csv(q_before)
    df_t1 = pd.read_csv(q_t1)
    df_t2 = pd.read_csv(q_t2)
    df_t3 = pd.read_csv(q_t3)
    df_after =pd.read_csv(q_after)

    res_b = pca_kmeans_minimal_outputs(
        df_b,
        prefix="b",
        n_components=n_dims_b_90,
        k_range=range(2, 15),
        top_k_features=20,
        save_dir="data/only_Q_outputs/ses1_figures"
    )



    res_after = pca_kmeans_minimal_outputs(
        df_after,
        prefix="after",
        n_components=n_dims_after_90,
        k_range=range(2, 15),
        top_k_features=20,
        save_dir="data/only_Q_outputs"
    )


    # You already found that e.g., 22 PCA components explain 95% of variance
    res_t1 = pca_kmeans_minimal_outputs(
        df_t1,
        prefix="t1",
        n_components=n_dims_t1_90,
        k_range=range(2, 15),
        top_k_features=20,
        save_dir="data/only_Q_outputs"
    )


    # Load your Excel file

    # You already found that e.g., 22 PCA components explain 95% of variance
    res_t2 = pca_kmeans_minimal_outputs(
        df_t2,
        prefix="t2",
        n_components=n_dims_t2_90,
        k_range=range(2, 15),
        top_k_features=20,
        save_dir="data/only_Q_outputs")


    # You already found that e.g., 22 PCA components explain 95% of variance
    res_t3 = pca_kmeans_minimal_outputs(
        df_t3,
        prefix="t3",
        n_components=n_dims_t3_90,
        k_range=range(2, 15),
        top_k_features=20,
        save_dir="data/only_Q_outputs",)


    # You already found that e.g., 22 PCA components explain 95% of variance
    res_after = pca_kmeans_minimal_outputs(
        df_after,
        prefix="after",
        n_components=n_dims_after_90,
        k_range=range(2, 15),
        top_k_features=20,
        save_dir= "data/only_Q_outputs")



    sil_scores_b = res_b["sil_scores"]
    print("k* =", res_b["best_k"])

    sil_scores_t1 = res_t1["sil_scores"]
    print("k* =", res_t1["best_k"])

    sil_scores_t2 = res_t2["sil_scores"]
    print("k* =", res_t2["best_k"])

    sil_scores_t3 = res_t3["sil_scores"]
    print("k* =", res_t3["best_k"])

    sil_scores_after = res_after["sil_scores"]
    print("k* =", res_after["best_k"])

    ks = list(range(2, 15))  # same k_range you used in the runs

    sil_b = res_b["sil_scores"].iloc[:, 0].values
    sil_t1 = res_t1["sil_scores"].iloc[:, 0].values
    sil_t2 = res_t2["sil_scores"].iloc[:, 0].values
    sil_t3 = res_t3["sil_scores"].iloc[:, 0].values
    sil_after = res_after["sil_scores"].iloc[:, 0].values

    best_k_b_sil = res_b["best_k"]
    best_k_t1_sil = res_t1["best_k"]
    best_k_t2_sil = res_t2["best_k"]
    best_k_t3_sil = res_t3["best_k"]
    best_k_after_sil = res_after["best_k"]

    fig, ax = plt.subplots(figsize=(9, 6))

    datasets = [
        ("pre", sil_b, best_k_b_sil, 'o'),
        ("trimester 1", sil_t1, best_k_t1_sil, 's'),
        ("trimester 2", sil_t2, best_k_t2_sil, 'D'),
        ("trimester 3", sil_t3, best_k_t3_sil, '^'),
        ("after", sil_after, best_k_after_sil, 'v'),
    ]

    for name, sil, best_k, m in datasets:
        ax.plot(ks, sil, marker=m, label=f"{name}")
        # highlight the chosen k: star + vertical line + label
        i = ks.index(best_k)
        y = sil[i]
        ax.scatter([best_k], [y], s=220, marker='*', edgecolors='black', zorder=5)
        ax.axvline(best_k, linestyle='--', alpha=0.35)
        ax.annotate(f"k={best_k}", xy=(best_k, y),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, weight="bold")

    ax.set_title("Silhouette vs k (all datasets)")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette score")
    ax.set_xticks(ks)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.legend(title="Dataset", ncol=2)
    plt.tight_layout()
    plt.show()

    labels_b_sil, data_pca_b_sil, model_b_sil, pca_b_sil = run_kmeans_clustering(
        df=df_b,
        prefix="b",  # your prefix
        n_components=n_dims_b_90,
        k=best_k_b_sil,  # from your gap statistic
        plot=True,
        title="Kmeans clusting silluate - before pregnancy",
        csv_path="data/only_Q_outputs/clusters_before.csv")

    labels_t1_sil, data_pca_t1_sil, model_t1_sil, pca_t1 = run_kmeans_clustering(
        df=df_t1,
        prefix="t1",  # your prefix
        n_components=n_dims_t1_90,
        k=best_k_t1_sil,  # from your gap statistic
        plot=True,
        title="Kmeans clusting silluate - first trimester",
        csv_path="data/only_Q_outputs/clusters_t1.csv"
    )

    labels_t2_sil, data_pca_t2_sil, model_t2_sil, pca_t2 = run_kmeans_clustering(
        df=df_t2,
        prefix="t2",  # your prefix
        n_components=n_dims_t2_90,
        k=best_k_t2_sil,  # from your gap statistic
        plot=True,
        title="Kmeans clusting silluate - second trimester",
        csv_path="data/only_Q_outputs/clusters_t2.csv"
    )

    labels_t3_sil, data_pca_t3_sil, model_t3_sil, pca_t3 = run_kmeans_clustering(
        df=df_t3,
        prefix="t3",  # your prefix
        n_components=n_dims_t3_90,
        k=best_k_t3_sil,  # from your gap statistic
        plot=True,
        title="Kmeans clusting silluate - second trimester",
        csv_path="data/only_Q_outputs/clusters_t3.csv"
    )

    labels_after_sil, data_pca_after_sil, model_after_sil, pca_after = run_kmeans_clustering(
        df=df_after,
        prefix="after",  # your prefix
        n_components=n_dims_after_90,
        k=best_k_after_sil,  # from your gap statistic
        plot=True,
        title="Kmeans clusting silluate - after pregnancy",
        csv_path="data/only_Q_outputs/clusters_after.csv"
    )

    plot_multi_dataset_scatters_colored(
        datasets=[data_pca_b_sil, data_pca_t1_sil, data_pca_t2_sil, data_pca_t3_sil, data_pca_after_sil],
        titles=["Pre pregnancy", "1st trimester", "2nd trimester", "3rd trimester", "Post pregnancy"],
        labels_list=[labels_b_sil, labels_t1_sil, labels_t2_sil, labels_t3_sil, labels_after_sil],
        x_idx=0, y_idx=1,
        base_colors=["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"],
        show_counts=True,  # <-- on
        show_percent=False  # <-- optional %
    )


if __name__ == "__main__":
    main()
