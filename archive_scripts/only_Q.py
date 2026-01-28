import os, random, numpy as np
import pandas as pd
from pathlib import Path
import shutil
from questionnaire_functions import (export_questionnaire,analyze_questionnaire)
from pca_functions import (find_optimal_pca_dimensions,plot_pca_weights_two_cols_split,plot_pca_weights_separate_and_table)
from clustring_functions import(pca_kmeans_minimal_outputs,run_kmeans_clustering,invert_binary_columns)
from vizualizations_functions import (plot_grouped_bars,plot_multi_dataset_scatters_colored,plot_one_period_with_labels,get_labels_from_file,plot_one_period_with_labels_and_ttest,stats_by_group,safe)
from preprocessing_functions import (load_one,transition_for_pair)
from functools import reduce
import textwrap
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ttest_rel
def main():
    SEED = 17
    os.environ["PYTHONHASHSEED"] = str(SEED)

    # לכבות רנדומליות בספריות BLAS/OMP
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    random.seed(SEED)
    np.random.seed(SEED)

    print("Seed fixed:", SEED)

    # -*- coding: utf-8 -*-

    # נתיב לקובץ הנתונים שלך (קובץ אקסל)
    # עדכן את הנתיב לשם הקובץ האמיתי שלך
    file_path = 'data/q_data/Study_Questionnaire_Responses_October.xlsx'

    # רשימת העמודות שברצונך לבחור
    # ערוך את הרשימה הזו כדי להתאים לצרכים שלך
    columns_to_select = ["Subject_Code", "b_questionnaire_completion", "after_questionnaire_completion", "t1_MAAS_date",
                         "t2_PHQ_date", "t3_MAAS_date"]

    df = pd.read_excel(file_path)
    columns = df.columns.tolist()
    print("the columns in the dataframe:")
    for col in columns:
        print(col)

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
        print(col)
    phq_columns = ['b_PHQ_1','b_PHQ_2','b_PHQ_3','b_PHQ_4','b_PHQ_5','b_PHQ_6','b_PHQ_7','b_PHQ_8','b_PHQ_9','b_PHQ_10',
                   't1_PHQ_1','t1_PHQ_2','t1_PHQ_3','t1_PHQ_4','t1_PHQ_5','t1_PHQ_6','t1_PHQ_7','t1_PHQ_8','t1_PHQ_9','t1_PHQ_10',
                   't2_PHQ_1','t2_PHQ_2','t2_PHQ_3','t2_PHQ_4','t2_PHQ_5','t2_PHQ_6','t2_PHQ_7','t2_PHQ_8','t2_PHQ_9','t2_PHQ_10',
                   't3_PHQ_1','t3_PHQ_2','t3_PHQ_3','t3_PHQ_4','t3_PHQ_5','t3_PHQ_6','t3_PHQ_7','t3_PHQ_8','t3_PHQ_9','t3_PHQ_10',
                   'after_PHQ_1','after_PHQ_2','after_PHQ_3','after_PHQ_4','after_PHQ_5','after_PHQ_6','after_PHQ_7','after_PHQ_8','after_PHQ_9','after_PHQ_10']
    gad_columns = ['b_GAD7_1','b_GAD7_2','b_GAD7_3','b_GAD7_4','b_GAD7_5','b_GAD7_6','b_GAD7_7',
                   't1_GAD7_1','t1_GAD7_2','t1_GAD7_3','t1_GAD7_4','t1_GAD7_5','t1_GAD7_6','t1_GAD7_7',
                   't2_GAD7_1', 't2_GAD7_2', 't2_GAD7_3', 't2_GAD7_4', 't2_GAD7_5', 't2_GAD7_6', 't2_GAD7_7',
                   't3_GAD7_1', 't3_GAD7_2', 't3_GAD7_3', 't3_GAD7_4', 't3_GAD7_5', 't3_GAD7_6', 't3_GAD7_7',
                   'after_GAD7_1', 'after_GAD7_2', 'after_GAD7_3', 'after_GAD7_4', 'after_GAD7_5', 'after_GAD7_6', 'after_GAD7_7'
                   ]
    # remove the unnecessery columns (columns related to war ,metadata, queshtions from an interview)
    # columns_to_drop = [
    #     'sasrq_date',
    #     'sasrq_1', 'sasrq_2', 'sasrq_3', 'sasrq_4', 'sasrq_5', 'sasrq_6', 'sasrq_7', 'sasrq_8', 'sasrq_9', 'sasrq_10',
    #     'sasrq_11', 'sasrq_12', 'sasrq_13', 'sasrq_14', 'sasrq_15', 'sasrq_16', 'sasrq_17', 'sasrq_18', 'sasrq_19','sasrq_20',
    #     'sasrq_21', 'sasrq_22', 'sasrq_23', 'sasrq_24', 'sasrq_25', 'sasrq_26', 'sasrq_27', 'sasrq_28', 'sasrq_29','sasrq_31', 'sasrq_total',
    #     'PROMOTE_date', 'war_pregnancy_status','PROMOTE_1', 'PROMOTE_2', 'PROMOTE_3', 'PROMOTE_4', 'PROMOTE_5', 'PROMOTE_6', 'PROMOTE_7', 'PROMOTE_8','PROMOTE_9', 'PROMOTE_10', 'PROMOTE_11', 'PROMOTE_12', 'PROMOTE_13', 'PROMOTE_14', 'PROMOTE_15',
    #     'WEQ_date', 'WEQ_1', 'WEQ_2', 'WEQ_3', 'WEQ_4', 'WEQ_5', 'WEQ_6', 'WEQ_7', 'WEQ_8', 'WEQ_9', 'WEQ_10','WEQ_11', 'WEQ_12', 'WEQ_13', 'WEQ_14', 'WEQ_15', 'WEQ_16', 'WEQ_17', 'WEQ_18', 'WEQ_19',
    #     'Posttraumatic _Growth_Inventory_date', 'Posttraumatic _Growth_1', 'Posttraumatic _Growth_2','Posttraumatic _Growth_3',
    #     'Posttraumatic _Growth_4', 'Posttraumatic _Growth_5', 'Posttraumatic _Growth_6', 'Posttraumatic _Growth_7',
    #     'Posttraumatic _Growth_8', 'Posttraumatic _Growth_9', 'Posttraumatic _Growth_10', 'Posttraumatic _Growth_total',
    #     'Posttraumatic _Growth_RO', 'Posttraumatic _Growth_NP', 'Posttraumatic _Growth_PS',
    #     'Posttraumatic _Growth_SC', 'Posttraumatic _Growth_AL','war_pcl5_date',
    #     'war_pcl_1', 'war_pcl_2', 'war_pcl_3', 'war_pcl_4', 'war_pcl_5', 'war_pcl_6', 'war_pcl_7', 'war_pcl_8','war_pcl_9',
    #     'war_pcl_10', 'war_pcl_11', 'war_pcl_12', 'war_pcl_13', 'war_pcl_14', 'war_pcl_15', 'war_pcl_16', 'war_pcl_17',
    #     'war_pcl_18', 'war_pcl_19', 'war_pcl_20', 'war_pcl_total', 'war_pcl_cutoff', 'war_pcl_dsm',
    #     'war_phq_date',
    #     'war_phq_1', 'war_phq_2', 'war_phq_3', 'war_phq_4', 'war_phq_5', 'war_phq_6', 'war_phq_7', 'war_phq_8','war_phq_9',
    #     'war_phq_10', 'war_phq_total','war_gad7_date','war_gad_1', 'war_gad_2', 'war_gad_3', 'war_gad_4', 'war_gad_5', 'war_gad_6', 'war_gad_7', 'war_gad_total',
    #     'country_of_birth','Country_of_Birth_(Israel/Other)','country_of_birth_mom','country_of_birth_dad','year_of_aliyah',
    #     'family_status','Years_Marriage','education_years','education_years_code','education_years_partner','education_years_partner_code','profession',
    #     'profession_partner','religion','religion_other','income','b_questionnaire_completion','after_questionnaire_completion','first_fmri_scan_date','second_fmri_scan_date','third_fmri_scan_date',
    #     'b_questionnaire_and_fmri_days_difference',
    #     'pregnancy_start_date','b_fmri_and_pregnancy_days_difference','newborn_birth_date','Days_from_Birth_to_Questionnaire_Completion','Demographics_Date',
    #     'date_of_birth','diamond_interview_date','b_diamond_anxiety_phobias_past','b_diamond_Anxiety_phobias_present','b_diamond_ocd_past',
    #     'b_diamond_ocd_present','b_diamond_adhd_past','b_diamond_adhd_present','b_diamond_depression_past','b_diamond_depression_present',
    #     'b_diamond_adjustment_past',
    #     'b_diamond_adjustment_present',
    #     'b_diamond_ptsd_past',
    #     'b_diamond_ptsd_present',
    #     'b_diamond_eating_disorder_past',
    #     'b_diamond_eating_disorder_present',
    #     'b_diamond_PMS_past',
    #     'b_diamond_PMS_present',
    #     'b_diamond_other_past',
    #     'b_diamond_other_present',
    #     'b_diamond_past',
    #     'b_diamond_present',
    #     't1_Fertility_treatments',
    #     'Conception_method','second_fmri_questionnaire_date','newborn_birth_date.2',
    #     '2FMRI_period_since_birth','2FMRI_last_period_date','2FMRI_breastfeeding',
    #     '2FMRI_average_sleep_hours',
    #     '2FMRI_birth_control_pills_usage',
    #     '2FMRI_additional_notes',
    #     'war_pregnancy_status',
    #     'after_bits_date',
    #     'after_PHQ_date',
    #     'after_GAD7_date',
    #     'after_MPAS_date',
    #     'after_DERS_date',
    #     'after_LHQ_date',
    #     'b_ctq_Date',
    #     'b_lec_date',
    #     'b_PCL5_date',
    #     'b_strength_date',
    #     'b_PHQ9_date'
    #     'b_GAD7_date',
    #     'b_PBI_date',
    #     'b_DES_date',
    #     't1_DES_date',
    #     't1_PHQ_date',
    #     't2_PHQ_date',
    #     't3_PHQ_date',
    #     't1_GAD7_date',
    #     't1_MAAS_date',
    #     't2_MAAS_date',
    #     't3_MAAS_date',
    #     't3_GAD7_date',
    #     't2_GAD7_date',
    #     'b_PHQ9_date',
    #     'b_GAD7_date',
    #     'b_social_support_date',
    #     'b_DERS_date',
    #     'b_LHQ_date',
    #     'b_IRI_date',
    #     'birth_week',
    #     'birth_type',
    #     'after_DES_date',
    #     'after_CTQ_date',
    #     'b_lec_1a',
    #     "b_lec_2a",
    #     "b_lec_3a",
    #     "b_lec_4a",
    #     "b_lec_5a",
    #     "b_lec_6a",
    #     "b_lec_7a",
    #     "b_lec_8a",
    #     "b_lec_9a",
    #     "b_lec_10a",
    #     "b_lec_11a",
    #     "b_lec_12a",
    #     "b_lec_13a",
    #     "b_lec_14a",
    #     "b_lec_15a",
    #     "b_lec_16a",
    #     "b_lec_17a",
    #     "b_ctq_total",
    #     "b_ctq_NEGLECT",
    #     "b_ctq_ABUSE  ",
    #     "b_ctq_sexual_abuse",
    #     "b_ctq_physical_abuse",
    #     "b_ctq_emotional_abuse","b_ctq_physical_neglect",
    #     "b_ctq_emotional_neglect","b_ctq_sexual_abuse_cutoff",
    #     "b_ctq_physical_abuse_cutoff",
    #     "b_ctq_emotional_abuse_cutoff","b_ctq_physical_neglect_cutoff","b_ctq_emotional_neglect_cutoff",
    #     "b_ctq_denial _score",
    #     "b_lec_0_to_16_total",
    #     "b_lec_interpersonal_events",
    #     "b_lec_non_interpersonal_events",
    #     "b_pcl_total",
    #     "b_pcl_cutoff",
    #     "b_pcl_dsm",
    #     "b_strength_average",
    #     "b_PHQ_total",
    #     "b_GAD7_total",
    #     "b_social_support_total", "b_PBI_mom_care", "b_PBI_dad_care","b_PBI_mom_overprotection", "b_PBI_dad_overprotection",
    #     "b_DERS_total",
    #     "b_DERS_Nonacceptance_Emotional_Responses",
    #     "b_DERS_Goal_Directed_Behavior",
    #     "b_DERS_Impulse_Control",
    #     "b_DERS_Lack_Emotional_Awareness",
    #     "b_DERS_Emotion_Regulation_Strategies",
    #     "b_DERS_Lack_Emotional_Clarity",
    #     "b_DES_average",
    #     "b_DES_Absorption ",
    #     "b_DES_Amnesia",
    #     "b_DES_Depersonalization ",
    #     "b_LHQ_total",
    #     "b_IRI_Perspective_Taking",
    #     "b_IRI_Empathic_Concern",
    #     "b_IRI_Personal_Distress",
    #     "b_IRI_Fantasy",
    #     't1_DES_total',
    #     't1_DES_Absorption ',
    #     't1_DES_Amnesia',
    #     't1_DES_Depersonalization',
    #     "T1_PHQ_total",
    #     'T2_PHQ_total',
    #     'T3_PHQ_total',
    #     'T1_GAD7_total',
    #     'T2_GAD7_total',
    #     'T3_GAD7_total',
    #     't1_MAAS_Attachment ',
    #     't1_MAAS_Preoccupation ',
    #     't2_MAAS_total',
    #     't2_MAAS_Attachment ',
    #     't2_MAAS_Preoccupation ',
    #     't3_MAAS_total',
    #     't3_MAAS_Attachment ',
    #     't3_MAAS_Preoccupation ',
    #     'after_bits_PTSD_total',
    #     'after_bits_birth_symptoms',
    #     'after_bits_General_symptoms',
    #     'after_bits_Dissociatie_symptoms',
    #     'after_bits_PTSD_criterion',
    #     'after_bits_Re_experiencing ',
    #     'after_bits_Avoidance ',
    #     'after_bits_Negative_Cognitions ',
    #     'after_bits_Hyperarousal',
    #     'after_PHQ_total',
    #     'after_GAD7_total',
    #     'after_MPAS_total',
    #     'after_MPAS_proximity',
    #     'after_MPAS_Acceptance',
    #     'after_MPAS_Tolerance',
    #     'after_MPAS_Competence',
    #     'after_MPAS_Competence',
    #     'after_MPAS_Attachment',
    #     'after_MPAS_Hostility',
    #     'after_MPAS_Interaction',
    #     'after_DES_total','after_DES_Absorption ',
    #     'after_DES_Amnesia',
    #     'after_DES_Depersonalization ',
    #     'after_CTQ_total',
    #     'after_CTQ_cutoff',
    #     'after_CTQ_NEGLECT',
    #     'after_CTQ_ABUSE  ',
    #     'after_CTQ_sexual_abuse',
    #     'after_CTQ_physical_abuse',
    #     'after_CTQ_emotional_abuse',
    #     'after_CTQ_physical_neglect',
    #     'after_CTQ_emotional_neglect',
    #     'after_CTQ_sexual_abuse_cutoff','after_CTQ_physical_abuse_cutoff',
    #     'after_CTQ_emotional_abuse_cutoff',
    #     'after_CTQ_physical_neglect_cutoff',
    #     'after_CTQ_emotional_neglect_cutoff',
    #     'after_CTQ_denial _score', 'after_DERS_total',
    #     'after_DERS_Nonacceptance_Emotional_Responses','after_DERS_Goal_Directed_Behavior','after_DERS_Impulse_Control','after_DERS_Lack_Emotional_Awareness','after_DERS_Emotion_Regulation_Strategies',
    #     'after_DERS_Lack_Emotional_Clarity','after_LHQ_total','t1_MAAS_total','b_ctq_cutoff','b_PBI_mom_overprotection','b_PBI_dad_care','t1_DES_Depersonalization '
    # ]
    # df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')
    df_cleaned = df[['Subject_Code','Dropped_Out_of_Study']+phq_columns+gad_columns]
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
        save_dir="data/only_Q_outputs/ses1_figures",

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

    # PC1 (pc_index=0), top 15:
    df_pc1 = plot_pca_weights_separate_and_table(
        pcas=[pca_b_sil, pca_t1, pca_t2, pca_t3, pca_after],
        dfs=[df_b, df_t1, df_t2, df_t3, df_after],
        prefixes=["b", "t1", "t2", "t3", "after"],
        titles=["Pre pregnancy", "1st trimester", "2nd trimester", "3rd trimester", "Post pregnancy"],
        pc_index=0,
        top_n=15,
        figsize=(10, 6),
        fontsize=15,
        save_csv_path="data/only_Q_outputs/top_loadings_PC1.csv"  # או "top_loadings_PC1.csv"
    )

    # הצצה לטבלה:
    print(df_pc1.head(10))

    plot_pca_weights_two_cols_split(
        pcas=[pca_b_sil, pca_t1, pca_t2, pca_t3, pca_after],
        dfs=[df_b, df_t1, df_t2, df_t3, df_after],
        prefixes=["b", "t1", "t2", "t3", "after"],
        titles=["Pre pregnancy", "1st trimester", "2nd trimester", "3rd trimester", "Post pregnancy"],
        pc_index=0,
        top_n=15,
        figsize=(16, 9),
        fontsize=10
    )

    # Map each file to the column name you want in the merged table
    files = {
        "data/only_Q_outputs/clusters_before.csv": "before",
        "data/only_Q_outputs/clusters_t1.csv": "t1",
        "data/only_Q_outputs/clusters_t2.csv": "t2",
        "data/only_Q_outputs//clusters_t3.csv": "t3",
        "data/only_Q_outputs/clusters_after.csv": "after",
    }


    # Load and prepare all dataframes
    dfs = [load_one(path, col) for path, col in files.items()]

    # Outer-join on Subject_Code
    merged = reduce(lambda left, right: pd.merge(left, right, on="Subject_Code", how="outer"), dfs)

    # Replace NaN with None (optional)
    merged = merged.where(pd.notna(merged), None)

    # Save
    merged.to_csv('data/only_Q_outputs/clusters_merged_by_subject.csv', index=False)
    print("Saved: clusters_merged_by_subject.csv")
    print(merged.head())



    invert_binary_columns(
        input_path="data/only_Q_outputs/clusters_merged_by_subject.csv",
        output_path='data/only_Q_outputs/merged_file_inverted_2.csv',
        column_names=["t1", "t3","after"]  # any set of binary columns
    )

    # --- Top/Bottom by PC1 (BEFORE) ---
    # PC1 scores (first PCA component)
    pc1_b = pd.Series(data_pca_b_sil[:, 0], index=df_b.index, name="PC1")

    # Pair with subject identifiers (fallback to index if Subject_Code missing)
    if "Subject_Code" in df_b.columns:
        subj = df_b.loc[pc1_b.index, "Subject_Code"]
    else:
        subj = pc1_b.index.astype(str)

    pc1_table_b = pd.DataFrame({"Subject": subj, "PC1": pc1_b.values})

    # Top 3 and bottom 3
    top3_b = pc1_table_b.nlargest(3, "PC1")
    bottom3_b = pc1_table_b.nsmallest(3, "PC1")

    print("\nTop 3 subjects by PC1 (before):")
    print(top3_b.to_string(index=False))

    print("\nBottom 3 subjects by PC1 (before):")
    print(bottom3_b.to_string(index=False))

    # Optional: save to disk
    pc1_table_b.sort_values("PC1").to_csv("data/only_Q_outputs/pc1_scores_before.csv", index=False,
                                          encoding="utf-8-sig")

    # הגדרת נתיבי הקבצים
    # שימו לב שהנתיבים מצביעים כעת לקבצים ב-Google Drive
    questionnaire_path = 'data/q_data/Study_Questionnaire_Responses_October.xlsx'
    clusters_path = 'data/only_Q_outputs/merged_file_inverted_2.csv'

    # הגדרת העמודות שאתה רוצה מקובץ השאלונים
    # עליך למלא את הרשימה עם שמות העמודות המדויקים שתרצה לכלול.
    # columns_to_select = ['b_LHQ_total', 'b_DES_average', 't1_MAAS_total','t1_DES_total','t2_MAAS_total','t3_MAAS_total','after_MPAS_total','after_DES_total','after_LHQ_total']
    columns_to_select = ['b_PHQ_total','T1_PHQ_total','T2_PHQ_total','T3_PHQ_total','after_PHQ_total','b_GAD7_total','T1_GAD7_total','T2_GAD7_total','T3_GAD7_total','after_GAD7_total']

    # הגדרת עמודות הקלאסטרים
    cluster_columns = ['before', 't1', 't2', 't3', 'after']

    # טעינת הנתונים
    try:
        # קריאת קובץ האקסל באמצעות pd.read_excel
        # אם הקובץ מכיל יותר מגיליון אחד, הוסף את הפרמטר: sheet_name='שם_הגיליון'
        df_questionnaire = pd.read_excel(questionnaire_path)

        print(df_questionnaire.columns.tolist())
        # קריאת קובץ ה-CSV באמצעות pd.read_csv
        df_clusters = pd.read_csv(clusters_path)

    except FileNotFoundError as e:
        print(f"שגיאה: הקובץ לא נמצא. אנא ודא שהנתיבים נכונים ושה-Google Drive מחובר.")
        exit()

    # בחירת העמודות הרצויות מקובץ השאלונים והקלאסטרים
    df_q_subset = df_questionnaire[['Subject_Code'] + columns_to_select]
    df_c_subset = df_clusters[['Subject_Code'] + cluster_columns]

    # איחוד הקבצים על בסיס Subject_Code, תוך שמירה על נבדקים משותפים בלבד (inner join)
    df_merged = pd.merge(df_q_subset, df_c_subset, on='Subject_Code', how='inner')

    # שמירת הקובץ הסופי עם קידוד utf-8-sig
    output_filename = 'data/only_Q_outputs/filtered_merged_data.csv'
    df_merged.to_csv(output_filename, index=False, encoding='utf-8-sig')

    print(f"\nהקובץ הסופי נשמר בשם '{output_filename}'")
    print("\nהנה דוגמה לנתונים המאוחדים:")
    print(df_merged.head())



    # מיפוי העמודות לכל תקופה
    periods = {
        'b': ["b_PHQ_total","b_GAD7_total"],
        't1': ["T1_PHQ_total","T1_GAD7_total"],
        't2': ["T2_PHQ_total","T2_GAD7_total"],
        't3': ["T3_PHQ_total","T3_GAD7_total"],
        'after':["after_PHQ_total","after_GAD7_total"]
    }

    # === שימוש: גריד אחד לכל התקופות לפי התוויות מהקובץ ===
    file_path = 'data/only_Q_outputs/filtered_merged_data.csv'
    df = pd.read_csv(file_path)

    # קחי את שם העמודה של התוויות כפי שהיא בקובץ (למשל 'before')
    labels = get_labels_from_file(df, label_col='before')

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, period_name in enumerate(periods.keys()):
        plot_one_period_with_labels(df, labels, period_name, axes[i],periods)

    # מחיקת סאב-פלוט מיותר (המשבצת השישית)
    for j in range(len(periods), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle('All periods — clusters from file column "before"', fontsize=16)
    plt.tight_layout()
    plt.show()

    # -*- coding: utf-8 -*-


    # =========================================================================
    # This code performs a two-sided t-test, with the option for FDR correction.
    # =========================================================================

    # --- Settings ---
    # Set this variable to True to apply FDR correction to all t-tests.
    # If False, the tests will be performed without correction.
    APPLY_FDR_CORRECTION = False

    # Image-compatible colors (Cluster 0 dark, Cluster 1 light)
    PALETTES = {
        'b': ['#2c6b7d', '#9bd3e0'],  # Dark/light turquoise
        't1': ['#7a4b1f', '#f4c9a5'],  # Dark/light brown
        't2': ['#215b2a', '#a3e3ac'],  # Dark/light green
        't3': ['#6b1e1e', '#f6b0b0'],  # Dark/light reddish
        'after': ['#3b2464', '#c7b6ee'],  # Dark/light purple
    }

    # If there are cluster label columns for each period in the file
    USE_FILE_LABELS = True
    LABEL_COL_PER_PERIOD = {
        'b': 'before',
        't1': 't1',
        't2': 't2',
        't3': 't3',
        'after': 'after',
    }




    # ========== Data Reading ==========
    file_path = 'data/only_Q_outputs/filtered_merged_data.csv'
    df = pd.read_csv(file_path)

    period_list = list(periods.keys())
    n_periods = len(period_list)

    # Prepare labels for each clustering row
    labels_per_period = {}
    for key in period_list:
        col = LABEL_COL_PER_PERIOD.get(key)
        if USE_FILE_LABELS and (col in df.columns):
            labels_per_period[key] = get_labels_from_file(df, col)
        else:
            raise ValueError(f"No label column found for period '{key}'. "
                             f"Update LABEL_COL_PER_PERIOD or disable USE_FILE_LABELS.")

    # ============ Performing t-test and collecting data for tables ============
    all_stats = []
    all_t_test_results = []
    all_p_values = []
    p_val_map = []  # Map of indexes to link p-values to original results

    for row_idx, cluster_period in enumerate(period_list):
        labels = labels_per_period[cluster_period]

        for col_idx, period_name in enumerate(period_list):
            cols = periods[period_name]
            block = df.loc[labels.index, cols].join(labels).dropna(subset=cols)
            if block.empty:
                continue

            counts = block['cluster'].value_counts().sort_index()
            num_clusters = len(counts)

            # Create a statistics table
            means = block.groupby('cluster')[cols].mean().sort_index()
            stds = block.groupby('cluster')[cols].std().sort_index()
            stats_df_result = pd.concat([means.stack().rename('mean'), stds.stack().rename('std')], axis=1)
            stats_df_result.index.names = ['cluster', 'variable']
            stats_df_result['N_in_cluster'] = stats_df_result.index.get_level_values('cluster').map(counts)
            stats_df_result['period_shown'] = period_name
            stats_df_result['clusters_from'] = cluster_period
            all_stats.append(stats_df_result)

            # Perform t-test and collect data
            if num_clusters >= 2:
                cluster_names = counts.index.tolist()
                for var in cols:
                    group1 = block[block['cluster'] == cluster_names[0]][var].dropna()
                    group2 = block[block['cluster'] == cluster_names[1]][var].dropna()

                    if len(group1) > 1 and len(group2) > 1:
                        try:
                            t_stat, p_val = stats.ttest_ind(group1, group2, nan_policy='omit', equal_var=False)

                            result = {
                                'clusters_from_period': cluster_period,
                                'period_shown': period_name,
                                'variable': var,
                                't_statistic': t_stat,
                                'p_value': p_val,
                            }
                            all_t_test_results.append(result)
                            all_p_values.append(p_val)
                        except Exception as e:
                            print(f"An error occurred while performing the t-test for {var}: {e}")

    t_test_df = pd.DataFrame(all_t_test_results)
    if APPLY_FDR_CORRECTION and not t_test_df.empty:
        reject, pvals_corrected = fdrcorrection(t_test_df['p_value'])
        t_test_df['p_value_corrected'] = pvals_corrected
    else:
        t_test_df['p_value_corrected'] = pd.NA

    # ========== Drawing the grid and printing the statistics ==========
    fig, axes = plt.subplots(n_periods, n_periods, figsize=(4 * n_periods, 3 * n_periods))
    if n_periods == 1:
        axes = [[axes]]

    for row_idx, cluster_period in enumerate(period_list):
        labels = labels_per_period[cluster_period]
        row_palette = PALETTES.get(cluster_period, ['#444444', '#bbbbbb'])

        for col_idx, period_name in enumerate(period_list):
            ax = axes[row_idx][col_idx]
            plot_one_period_with_labels_and_ttest(
                df, labels, period_name, ax,
                palette=row_palette,
                clusters_from_period=cluster_period,
                t_test_results_df=t_test_df,periods =periods)

        # Row label
        axes[row_idx][0].set_ylabel(f'Clusters from {cluster_period}', fontsize=11)

    # === Shorten X labels in all graphs ===
    for row in axes:
        for ax in row:
            ticks = ax.get_xticks()
            orig_labels = [t.get_text() for t in ax.get_xticklabels()]
            if not orig_labels:
                continue

            new_labels = []
            for lab in orig_labels:
                txt = lab.replace('\n', ' ').strip()
                tokens = [tok for tok in txt.split('_') if tok]
                if len(tokens) >= 3:
                    core = " ".join(tokens[1:-1])
                elif len(tokens) == 2:
                    core = tokens[1]
                else:
                    parts = txt.split()
                    core = parts[len(parts) // 2] if parts else ""
                new_labels.append(core)

            ax.set_xticks(ticks)
            ax.set_xticklabels(new_labels, rotation=0, ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()

    # ============ Merging and printing the tables ============
    if all_stats:
        stats_df = pd.concat(all_stats, axis=0)
        stats_df = stats_df[['period_shown', 'clusters_from', 'N_in_cluster', 'mean', 'std']]
        print("\n=== Averages and standard deviations for each cell in the grid ===")
        print(stats_df)

        stats_df.to_csv('data/only_Q_outputs/grid_cluster_stats.csv', index=True, encoding='utf-8-sig')

    # ============ Merging and printing t-test results ============
    if not t_test_df.empty:
        print(
            f"\n=== Two-sided t-test results {'with FDR correction' if APPLY_FDR_CORRECTION else 'without correction'} ===")
        print(t_test_df)

        t_test_df.to_csv('data/only_Q_outputs/t_test_results_without_FDR.csv', index=False, encoding='utf-8-sig')


    # ========= INPUTS =========
    # A note on file paths: The script assumes your working directory is the same as the
    # one where the script runs. If this is not the case, make sure the paths are absolute.
    file_path = "data/q_data/5_timepoints/filtered_merged_data_with _trajectory.csv"
    out_dir = Path("data/only_Q_outputs/charts_configurable")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ========= READ & BASIC CLEAN =========
    base = pd.read_csv(file_path)
    base.columns = [c.strip() for c in base.columns]
    print(base.columns.tolist())
    # (1) drop rows missing value in column #2 (second column in the file)
    # If you meant a different column, change index 1 accordingly.
    second_col = base.columns[1]
    base = base[base[second_col].notna()].copy()

    # (2) last column = label
    label_col = base.columns[-1]

    # normalize labels
    labels_norm = (
        base[label_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )

    # map variants -> canonical names
    to_canonical = {
        "remain clinical": "remain clinical",
        "remain healthy": "remain healthy",
        "worsening": "worsening", # merge spelling
        "improving": "improving",
        "fluctuating": "fluctuating",
        "nan": None,              # literal "nan" strings
    }

    # The original line had a typo `to_ccanonical`, which has been fixed.
    canon = labels_norm.map(lambda x: to_canonical.get(x, x))
    base = base.assign(_group=canon)

    # drop rows without a valid group (None/NaN)
    base = base.dropna(subset=["_group"]).copy()

    # (safety) also drop any literal "nan" that slipped through
    base = base[base["_group"] != "nan"].copy()

    # ========= SPLIT TO GROUPS =========
    groups = {g: df.drop(columns=["_group"]).copy() for g, df in base.groupby("_group")}
    # counts per group (number of subjects/rows)
    group_counts = pd.Series({g: len(df) for g, df in groups.items()},
                             name="n_subjects").sort_index()
    print("\nSubjects per group:")
    print(group_counts)

    # optional: save to CSV next to the charts
    group_counts.to_csv(out_dir / "group_counts.csv", header=True)

    # Expect these canonical groups; missing ones will just be absent
    print("Groups:", list(groups.keys()))



    # paired metrics with explicit before/after column names
    paired_pairs = {
        # special DES case: before uses 'average', after uses 'total'
        "PHQ": ("b_PHQ_total", "after_PHQ_total"),
        "GAD7 total": ("b_GAD7_total", "after_GAD7_total"),

    }


    # ========= PAIRED (BEFORE/AFTER) CHARTS WITH ERROR BARS + T-TEST =========
    ttest_results = []
    all_paired_stats = []

    for label, (b_col, a_col) in paired_pairs.items():
        if b_col not in base.columns or a_col not in base.columns:
            print(f"[warning] missing columns for '{label}' (expected: {b_col}, {a_col})")
            continue

        b_stats = stats_by_group(groups, b_col)
        a_stats = stats_by_group(groups, a_col)

        all_idx = sorted(set(b_stats.index) | set(a_stats.index))
        b_stats = b_stats.reindex(all_idx)
        a_stats = a_stats.reindex(all_idx)

        mask = ~(b_stats["mean"].isna() & a_stats["mean"].isna())
        b_stats = b_stats[mask]
        a_stats = a_stats[mask]

        if b_stats.empty and a_stats.empty:
            print(f"[info] no data to plot for '{label}'")
            continue

        x = np.arange(len(all_idx))
        width = 0.38

        plt.figure()
        plt.bar(x - width/2, b_stats["mean"].values, width,
                yerr=b_stats["std"].values, capsize=4, label="before")
        plt.bar(x + width/2, a_stats["mean"].values, width,
                yerr=a_stats["std"].values, capsize=4, label="after", color="purple")

        plt.title(f"{label} — Mean (±SD) by Group (before vs after)")
        plt.xlabel("Group")
        plt.ylabel(f"Mean {label}")
        xtick_labels = [f"{g}\n(n={group_counts.get(g, 0)})" for g in b_stats.index]
        plt.xticks(x, xtick_labels, rotation=0)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{safe(label)}_before_vs_after_by_group.png", dpi=150)
        plt.show()

        # ---- run paired t-test per group and collect all stats ----
        for gname, gdf in groups.items():
            if b_col in gdf.columns and a_col in gdf.columns:
                paired = gdf[[b_col, a_col]].dropna()
                if not paired.empty:
                    stat, pval = ttest_rel(paired[b_col], paired[a_col])

                    # Get means and stds for the current group
                    mean_b = paired[b_col].mean()
                    mean_a = paired[a_col].mean()
                    std_b = paired[b_col].std(ddof=1)
                    std_a = paired[a_col].std(ddof=1)
                    diff = mean_b - mean_a

                    # Add to paired stats list
                    all_paired_stats.append({
                        "metric": label,
                        "group": gname,
                        "mean_before": mean_b,
                        "mean_after": mean_a,
                        "std_before": std_b,
                        "std_after": std_a,
                        "diff_(before-after)": diff,
                        "t_value": stat,
                        "p_value": pval,
                        "n": len(paired)
                    })

    # ========= SINGLE-METRIC CHARTS WITH ERROR BARS =========
    all_single_stats = []



    # ========= SAVE FINAL STATS TO CSV =========
    print(f"Charts saved to: {out_dir}")

    # Paired stats
    if all_paired_stats:
        paired_df = pd.DataFrame(all_paired_stats)
        paired_df['t_value'] = paired_df['t_value'].round(3)
        paired_df['p_value'] = paired_df['p_value'].round(4)
        print("\nPaired t-test results (before vs after):")
        print(paired_df)

        paired_df.to_csv(out_dir / "paired_stats_report.csv", index=False)

    out_before_after = transition_for_pair(
        csv_path="data/only_Q_outputs/clusters_merged_by_subject.csv",
        from_period="before",   # <-- you type this
        to_period="after",  # <-- and this
        out_dir="out",
        save_prefix=None,
        annotate=True
    )

    out_t1_after = transition_for_pair(
        csv_path="data/only_Q_outputs/clusters_merged_by_subject.csv",
        from_period="t1",  # <-- you type this
        to_period="after",  # <-- and this
        out_dir="out",
        save_prefix=None,
        annotate=True
    )

    out_t2_after = transition_for_pair(
        csv_path="data/only_Q_outputs/clusters_merged_by_subject.csv",
        from_period="t2",  # <-- you type this
        to_period="after",  # <-- and this
        out_dir="out",
        save_prefix=None,
        annotate=True
    )

    out_t3_after = transition_for_pair(
        csv_path="data/only_Q_outputs/clusters_merged_by_subject.csv",
        from_period="t3",  # <-- you type this
        to_period="after",  # <-- and this
        out_dir="out",
        save_prefix=None,
        annotate=True
    )


    def plot_subject_trajectories(
            csv_path: str,
            subjects: list[str],
            timepoints: list[str] = ["before", "t1", "t2", "t3", "after"],
            subject_col: str = "Subject_Code",
            jitter: float = 0.03,  # vertical jitter to separate identical paths
            annotate_last: bool = False,
            figsize=(9, 5),
    ):
        df = pd.read_csv(csv_path)
        df[subject_col] = df[subject_col].astype(str)

        # keep only selected subjects & needed columns
        cols = [subject_col] + timepoints
        df = df.loc[df[subject_col].isin([str(s) for s in subjects]), cols].copy()

        # long format
        long = df.melt(id_vars=subject_col, value_vars=timepoints,
                       var_name="time", value_name="state")
        long = long.dropna(subset=["state"])

        # enforce order on x
        tp_to_x = {tp: i for i, tp in enumerate(timepoints)}
        long["x"] = long["time"].map(tp_to_x).astype(int)

        # ensure numeric (0/1)
        long["state"] = pd.to_numeric(long["state"], errors="coerce")
        long = long.dropna(subset=["state"])
        long["state"] = long["state"].astype(float)

        # jitter by subject so parallel lines are visible
        subj_to_offset = {s: (i - (len(subjects) - 1) / 2) * jitter for i, s in enumerate(subjects)}
        long["y"] = long["state"] + long[subject_col].map(subj_to_offset)

        # plot
        fig, ax = plt.subplots(figsize=figsize)
        for s in subjects:
            d = long[long[subject_col] == str(s)].sort_values("x")
            if d.empty:
                continue
            ax.plot(d["x"], d["y"], marker="o", linewidth=2, label=str(s))
            if annotate_last:
                ax.text(d["x"].iloc[-1] + 0.05, d["y"].iloc[-1], str(s),
                        va="center", fontsize=10)

        ax.set_xticks(range(len(timepoints)))
        ax.set_xticklabels(timepoints)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["0", "1"])
        ax.set_ylim(-0.4, 1.4)
        ax.set_xlabel("Time")
        ax.set_ylabel("Cluster")
        ax.set_title("Subject trajectories across time")
        ax.grid(True, axis="x", alpha=0.3)
        ax.legend(title="Subject", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
        plt.tight_layout()
        plt.show()

    # Example: pick your 6 subjects
    subjects = ["CT030", "NT007", "NT065", "NT017", "NT054", "NT102"]
    plot_subject_trajectories("data/only_Q_outputs/clusters_merged_by_subject.csv", subjects)


if __name__ == "__main__":
    main()
