from pathlib import Path
from statistics import correlation
from typing import Dict, List, Optional, Tuple, Union
import re
import numpy as np
from sympy import false
import torch
DEVICE = torch.device("cpu")

# ודא שהקובץ RDiff_map_for_arseney_group.py נמצא באותה תיקייה
from RDiff_map_for_arseney_group import get_diffusion_embedding ,affinity_mat
from correlation_calculation_NOGA import corrcoef_safe
import csv
from io import StringIO

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from diffution_map_functions import (plot_session1_scree,plot_session2_scree,elbow_max_distance,elbow_flat_slope,_regularize_matrix_spd,build_stacked_array_and_index,report_nan_inf_in_subjects,
                                     save_subject_map_to_csv,report_missing_row_names_before_load,build_missing_rows_aggregation,write_missing_rows_csv,
                                     load_tsv_folder_as_subject_dict,save_diffusion_output_with_labels)
# =========================
# CONFIG – EDIT THESE
# =========================
FOLDER = r"data/SCHAEFER_mat_cor"  # e.g. r"C:\data\tsv" or "/home/user/data"
PATTERN = "*.tsv"  # which files to include
DTYPE: Union[type, str] = float  # float | int | 'U' (strings) | etc.
ENCODING = "utf-8"
HAS_HEADER: Optional[bool] = None  # True / False / None for auto-detect

# ignore first row/column after loading
IGNORE_FIRST_ROW = True
IGNORE_FIRST_COL = True
# --- NEW: CSV with row names to remove (first column = row_name) ---
MISSING_ROWS_CSV = Path("data/SCHAEFER_mat_cor/csv_out/missing_rows_by_subject.csv")
STACK_ORDER = "key"
Key = Tuple[str, int, int]  # ('NT', 137, 1) לדוגמה
OUT_SUBFOLDER = "csv_out"


def main():
    # לפני טעינה ל-NumPy: דיווח שמות השורות שהן all-NaN בכל קובץ
    pre_missing_rows = report_missing_row_names_before_load(FOLDER, PATTERN, ENCODING)
    # אגרגציה: עבור כל שם שורה – באילו נבדקים היא all-NaN
    agg = build_missing_rows_aggregation(FOLDER, PATTERN, ENCODING)

    # כתיבת קובץ סיכום ל-CSV
    report_csv = Path(FOLDER) / OUT_SUBFOLDER / "missing_rows_by_subject.csv"
    write_missing_rows_csv(agg, report_csv)
    print(f"\nWrote per-row missing report to: {report_csv.resolve()}")

    subject_arrays = load_tsv_folder_as_subject_dict(
        FOLDER, PATTERN, DTYPE, ENCODING, HAS_HEADER,
        drop_first_row=IGNORE_FIRST_ROW,
        drop_first_col=IGNORE_FIRST_COL,MISSING_ROWS_CSV = MISSING_ROWS_CSV)


    print(f"\nLoaded {len(subject_arrays)} (group, subject, session) entries from TSV files.\n")

    # -----------------------------------------------------------
    # ✨ שלב 0: סינון לפי סשן (SESSION = 1)
    # -----------------------------------------------------------
    TARGET_SESSION = 1

    filtered_subject_arrays: Dict[Key, Union[np.ndarray, List[np.ndarray]]] = {}

    for key, val in subject_arrays.items():
        _, _, session = key
        if session == TARGET_SESSION:
            filtered_subject_arrays[key] = val

    print(f"Filtered to include only Session {TARGET_SESSION}. Retained {len(filtered_subject_arrays)} entries.")
    subject_arrays = filtered_subject_arrays  # החלפה במילון המסונן

    # וידוא שהמטריצות עדיין ריבועיות ושוות בגודל
    shapes = {k: v.shape for k, v in subject_arrays.items() if not isinstance(v, list)}
    unique_shapes = set(shapes.values())
    print("Unique shapes after filtering and removal:", unique_shapes)

    # הדפסה קצרה
    count = 0
    for (grp, num, ses), val in subject_arrays.items():
        if isinstance(val, list):
            shp = [v.shape for v in val]
            dt = [v.dtype for v in val]
            print(f"  - ({grp}, {num}, ses={ses}) -> {len(val)} arrays, shapes={shp}, dtypes={dt}")
        else:
            print(f"  - ({grp}, {num}, ses={ses}) -> shape={val.shape} dtype={val.dtype}")
        count += 1
        if count >= 10:
            print("  ... (showing first 10)")
            break

    # Save each subject array to CSVs (like before)
    out_dir = Path(FOLDER) / OUT_SUBFOLDER
    out_dir = save_subject_map_to_csv(subject_arrays, out_dir, ENCODING)
    print(f"\nSaved subject CSVs to: {out_dir.resolve()}\n")

    # ===== NEW: בנייה של המערך המאוחד + מילון אינדקסים =====
    all_arrays, index_by_key, keys_for_index = build_stacked_array_and_index(
        subject_arrays, order=STACK_ORDER
    )

    print(f"\nStacked array shape: {all_arrays.shape}, dtype={all_arrays.dtype}")
    print(f"Index map entries: {len(index_by_key)} (keys) / {len(keys_for_index)} (indices)")
    # דוגמה: מציאת האינדקס של נבדק מסוים ושליפה
    example_key = ('NT', 137, 1)
    if example_key in index_by_key:
        idx = index_by_key[example_key]
        if isinstance(idx, list):
            idx = idx[0]
        print(f"Example {example_key} -> index {idx}, slice shape={all_arrays[idx].shape}")
    else:
        print(f"Example key {example_key} not found in index map (likely filtered out).")

    window_length = all_arrays.shape[-1]

    # -----------------------------------------------------------
    # ✨ שלב 1: רגולריזציית SPD (Positive Definite)
    # -----------------------------------------------------------

    safe_correlations = np.empty_like(all_arrays)

    print("\nApplying SPD regularization to all correlation matrices using helper function...")

    for i in range(all_arrays.shape[0]):
        C_orig = all_arrays[i]
        safe_correlations[i] = C_orig

    # 3. החלפת המערך הגולמי במערך המרוגלר
    correlations = safe_correlations
    array_len_ses1 = len(correlations)
    # -----------------------------------------------------------

    # Scan for NaNs/Infs and report
    nan_inf_report = report_nan_inf_in_subjects(subject_arrays)

    # The result P_affinity is the row-normalized affinity matrix (110x110 Tensor)
    # ready for use in subsequent spectral analysis (like the diffusion map).
    # # ===== שלב 2: ניתוח הדיפוזיה - (UNCOMMENTED) =====
    # הפונקציה get_diffusion_embedding חייבת להיות זמינה (מהקובץ RDiff_map_for_arseney_group)
    diffusion_map, distances ,eigen_values_ses1 = get_diffusion_embedding(
        correlations, array_len_ses1, scale_k=(0.1*array_len_ses1), signal=None, subsampling=0, mode='riemannian'
    )
    print(eigen_values_ses1)
    # ===== שלב 3: שמירת פלט הדיפוזיה עם תוויות (Labels) - (UNCOMMENTED) =====
    out_dir = Path(FOLDER) / OUT_SUBFOLDER

    # 1. שמירת מפת הדיפוזיה (הוספת DC1, DC2,... ככותרות)
    dm_csv_path = out_dir / f"diffusion_map_ses{TARGET_SESSION}_labeled.csv"
    dm_csv_path = save_diffusion_output_with_labels(
        diffusion_map,
        keys_for_index,
        is_distance_matrix=False,
        base_path=dm_csv_path,
        encoding=ENCODING
    )

    # 2. שמירת מטריצת המרחקים (הוספת שמות נבדקים ככותרות שורה ועמודה)
    dist_csv_path = out_dir / f"diffusion_distances_ses{TARGET_SESSION}_labeled.csv"
    dist_csv_path = save_diffusion_output_with_labels(
        distances,
        keys_for_index,
        is_distance_matrix=True,
        base_path=dist_csv_path,
        encoding=ENCODING
    )

    print("\nSaved Labeled Diffusion Map CSV to:")
    print("  -", dm_csv_path.resolve())
    print("Saved Labeled Diffusion Distances CSV to:")
    print("  -", dist_csv_path.resolve())
    plot_session1_scree(eigen_values_ses1,k=50, save_path="figures/session1_scree.png")

    # -----------------------------------------------------------
    # ✨ שלב 0: סינון לפי סשן (SESSION = 1)
    # -----------------------------------------------------------
    pre_missing_rows = report_missing_row_names_before_load(FOLDER, PATTERN, ENCODING)
    # אגרגציה: עבור כל שם שורה – באילו נבדקים היא all-NaN
    agg = build_missing_rows_aggregation(FOLDER, PATTERN, ENCODING)

    # כתיבת קובץ סיכום ל-CSV
    report_csv = Path(FOLDER) / OUT_SUBFOLDER / "missing_rows_by_subject.csv"
    write_missing_rows_csv(agg, report_csv)
    print(f"\nWrote per-row missing report to: {report_csv.resolve()}")

    subject_arrays = load_tsv_folder_as_subject_dict(
        FOLDER, PATTERN, DTYPE, ENCODING, HAS_HEADER,
        drop_first_row=IGNORE_FIRST_ROW,
        drop_first_col=IGNORE_FIRST_COL,MISSING_ROWS_CSV = MISSING_ROWS_CSV
    )

    print(f"\nLoaded {len(subject_arrays)} (group, subject, session) entries from TSV files.\n")

    TARGET_SESSION = 2
    filtered_subject_arrays: Dict[Key, Union[np.ndarray, List[np.ndarray]]] = {}

    for key, val in subject_arrays.items():
        _, _, session = key
        if session == TARGET_SESSION:
            filtered_subject_arrays[key] = val

    print(f"Filtered to include only Session {TARGET_SESSION}. Retained {len(filtered_subject_arrays)} entries.")
    subject_arrays = filtered_subject_arrays  # החלפה במילון המסונן

    # וידוא שהמטריצות עדיין ריבועיות ושוות בגודל
    shapes = {k: v.shape for k, v in subject_arrays.items() if not isinstance(v, list)}
    unique_shapes = set(shapes.values())
    print("Unique shapes after filtering and removal:", unique_shapes)

    # הדפסה קצרה
    count = 0
    for (grp, num, ses), val in subject_arrays.items():
        if isinstance(val, list):
            shp = [v.shape for v in val]
            dt = [v.dtype for v in val]
            print(f"  - ({grp}, {num}, ses={ses}) -> {len(val)} arrays, shapes={shp}, dtypes={dt}")
        else:
            print(f"  - ({grp}, {num}, ses={ses}) -> shape={val.shape} dtype={val.dtype}")
        count += 1
        if count >= 10:
            print("  ... (showing first 10)")
            break

    # Save each subject array to CSVs (like before)
    out_dir = Path(FOLDER) / OUT_SUBFOLDER
    out_dir = save_subject_map_to_csv(subject_arrays, out_dir, ENCODING)
    print(f"\nSaved subject CSVs to: {out_dir.resolve()}\n")

    # ===== NEW: בנייה של המערך המאוחד + מילון אינדקסים =====
    all_arrays, index_by_key, keys_for_index = build_stacked_array_and_index(
        subject_arrays, order=STACK_ORDER
    )

    print(f"\nStacked array shape: {all_arrays.shape}, dtype={all_arrays.dtype}")
    print(f"Index map entries: {len(index_by_key)} (keys) / {len(keys_for_index)} (indices)")
    # דוגמה: מציאת האינדקס של נבדק מסוים ושליפה


    # -----------------------------------------------------------
    # ✨ שלב 1: רגולריזציית SPD (Positive Definite)
    # -----------------------------------------------------------

    safe_correlations = np.empty_like(all_arrays)

    print("\nApplying SPD regularization to all correlation matrices using helper function...")

    for i in range(all_arrays.shape[0]):
        C_orig = all_arrays[i]
        safe_correlations[i] = C_orig

    # 3. החלפת המערך הגולמי במערך המרוגלר
    correlations_ses2 = safe_correlations
    array_len_ses2  = len(correlations)
    # -----------------------------------------------------------

    # Scan for NaNs/Infs and report
    nan_inf_report = report_nan_inf_in_subjects(subject_arrays)

    # # ===== שלב 2: ניתוח הדיפוזיה - (UNCOMMENTED) =====
    # הפונקציה get_diffusion_embedding חייבת להיות זמינה (מהקובץ RDiff_map_for_arseney_group)
    diffusion_map_ses2, distances_ses2, eigen_values_ses2 = get_diffusion_embedding(
        correlations_ses2, array_len_ses2, scale_k=3, signal=None, subsampling=0, mode='riemannian'
    )

    # ===== שלב 3: שמירת פלט הדיפוזיה עם תוויות (Labels) - (UNCOMMENTED) =====
    out_dir = Path(FOLDER) / OUT_SUBFOLDER

    # 1. שמירת מפת הדיפוזיה (הוספת DC1, DC2,... ככותרות)
    dm_csv_path = out_dir / f"diffusion_map_ses{TARGET_SESSION}_labeled.csv"
    dm_csv_path = save_diffusion_output_with_labels(
        diffusion_map_ses2,
        keys_for_index,
        is_distance_matrix=False,
        base_path=dm_csv_path,
        encoding=ENCODING
    )

    # 2. שמירת מטריצת המרחקים (הוספת שמות נבדקים ככותרות שורה ועמודה)
    dist_csv_path = out_dir / f"diffusion_distances_ses{TARGET_SESSION}_labeled.csv"
    dist_csv_path = save_diffusion_output_with_labels(
        distances_ses2,
        keys_for_index,
        is_distance_matrix=True,
        base_path=dist_csv_path,
        encoding=ENCODING
    )

    print(eigen_values_ses2)

    print("\nSaved Labeled Diffusion Map CSV to:")
    print("  -", dm_csv_path.resolve())
    print("Saved Labeled Diffusion Distances CSV to:")
    print("  -", dist_csv_path.resolve())

    plot_session2_scree(eigen_values_ses2, save_path="figures/session2_scree.png")

    k1_md, val1_md = elbow_max_distance(eigen_values_ses1)
    k1_fs, val1_fs = elbow_flat_slope(eigen_values_ses1, window=5, rel_slope=0.10)

    k2_md, val2_md = elbow_max_distance(eigen_values_ses2)
    k2_fs, val2_fs = elbow_flat_slope(eigen_values_ses2, window=2, rel_slope=0.10)

    print(f"Session 1 — max-distance: k={k1_md}, eig≈{val1_md:.6g}; flat-slope: k={k1_fs}, eig≈{val1_fs:.6g}")
    print(f"Session 2 — max-distance: k={k2_md}, eig≈{val2_md:.6g}; flat-slope: k={k2_fs}, eig≈{val2_fs:.6g}")

#for session one we are going to take 14 dimention
#ofr session 2 we are going to take 11 dimentions
if __name__ == "__main__":
    main()