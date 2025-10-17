import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np
import re
from typing import Optional, List
from matplotlib import cm
from preprocessing_functions import (normalize_subject_id,read_first_column,column_name_from_file,
                                     first_added_file_column,find_subject_file,find_case_insensitive_column,
                                     sanitize_filename_piece,read_euler_tiv,_session_suffixes
                                     ,read_subject_table_without_euler_tiv,flatten_df_with_names
                                     ,prepare_features_and_subjects)
from pca_functions import (find_optimal_pca_dimensions, perform_pca)
from vizualizations_functions import (save_top_loadings,plot_top_loadings,plot_pc12_colored,plot_grouped_bars)
from clustring_functions import (get_clusters_series_from_file)
def main():
    # --- הגדרות ---
    ROOT = Path("data/cat12")
    SUBFOLDERS = ["NT", "CT"]  # התיקיות הצפויות בתוך cat12
    OUTPUT_CSV_Flatten_ses1 = "output_q_and_t1/sessions_summary.csv"

    # תבנית: תחילת שם קובץ = NT/CT + ספרות (מזהה נבדק), ולאחר מכן אופציונלית _1 או _2 = מספר סשן
    # דוגמאות חוקיות: NT028.xlsx, NT028_1.xlsx, CT123-2.csv, CT045 2.mat
    PATTERN = re.compile(r'^(?P<id>(?:NT|CT)\d+)(?:[ _-]?(?P<session>[12]))?\b', re.IGNORECASE)

    rows = []

    for sub in SUBFOLDERS:
        folder = ROOT / sub
        if not folder.exists():
            print(f"⚠️ התיקייה לא נמצאה: {folder}")
            continue

        for entry in os.listdir(folder):
            p = folder / entry
            if not p.is_file():
                continue

            name = p.stem  # שם הקובץ בלי הסיומת
            m = PATTERN.match(name)
            if not m:
                # אם יש קבצים שלא עומדים בפורמט, נדלג ונעדכן בקונסול
                print(f"⚠️ דילוג: שם קובץ לא מזוהה לפי התבנית -> {p.name}")
                continue

            subject_id = m.group("id").upper()  # לדוגמה: NT028
            session_str = m.group("session")
            session = int(session_str) if session_str else 1  # ברירת מחדל: 1

            rows.append({"subject_id": subject_id, "session": session})

    # נבנה DataFrame, נסיר כפילויות אם יש (למשל אם אותו קובץ מופיע פעמיים), נמיין, ונשמור
    df_ses1 = pd.DataFrame(rows)
    if df_ses1.empty:
        print("לא נמצאו קבצים תואמים בתיקיות NT/CT.")
    else:
        df_ses1 = df_ses1.drop_duplicates().sort_values(by=["subject_id", "session"]).reset_index(drop=True)
        df_ses1.to_csv(OUTPUT_CSV_Flatten_ses1, index=False, encoding="utf-8")
        print(f"✅ נשמר קובץ הסיכום: {OUTPUT_CSV_Flatten_ses1}")
        print(df_ses1.head(20))

    # === Config ===
    DRIVE_ROOT = Path("output_q_and_t1")
    BASE_CSV = DRIVE_ROOT / "sessions_summary.csv"  # created earlier
    OUTPUT_CSV_Flatten_ses1 = DRIVE_ROOT / "sessions_summary_with_files.csv"

    # List your 5 files here. Use dicts so you can optionally pass sheet_name for Excel files.
    FILES = [
        {"path": Path("data/q_data/time_points/b_questionnaire.xlsx"), "sheet_name": "Data"},
        {"path": Path("data/q_data/time_points/t1_questionnaire.xlsx"), "sheet_name": "Data"},
        {"path": Path("data/q_data/time_points/t2_questionnaire.xlsx"), "sheet_name": "Data"},
        {"path": Path("data/q_data/time_points/t3_questionnaire.xlsx"), "sheet_name": "Data"},
        {"path": Path("data/q_data/time_points/after_questionnaire.xlsx"), "sheet_name": "Data"},
    ]

    # === Helpers ===
    ID_PAT_RE = re.compile(r'(?P<prefix>[A-Za-z]{1,4})[-_ ]*0*(?P<num>\d+)$')

    # === Load base CSV and prepare key ===
    df_ses1 = pd.read_csv(BASE_CSV)
    if "subject_id" not in df_ses1.columns:
        raise ValueError("Column 'subject_id' not found in base CSV.")
    df_ses1["norm_id"] = df_ses1["subject_id"].apply(normalize_subject_id)

    # === Build 0/1 columns per file ===
    for item in FILES:
        fpath = Path(item["path"])
        sheet = item.get("sheet_name", None)

        if not fpath.exists():
            print(f"⚠️ Skipping: file not found -> {fpath}")
            continue

        try:
            col_series = read_first_column(fpath, sheet_name=sheet)
        except Exception as e:
            print(f"⚠️ Skipping {fpath.name}: {e}")
            continue

        temp = pd.DataFrame({"raw_subject": col_series})
        temp["norm_id"] = temp["raw_subject"].apply(normalize_subject_id)
        present = set(temp["norm_id"].dropna().unique())

        col_name = column_name_from_file(fpath)
        df_ses1[col_name] = df_ses1["norm_id"].apply(lambda x: 1 if x in present else 0)

    # === Save ===
    out = df_ses1.drop(columns=["norm_id"])
    out.to_csv(OUTPUT_CSV_Flatten_ses1, index=False, encoding="utf-8")
    print(f"✅ Saved: {OUTPUT_CSV_Flatten_ses1}")
    print(out.head(20))

  ####################################################session1
    OUT_BASENAME_ses1 = "output_q_and_t1/selected_subjects_euler_tiv"  # we'll append __<cluster_col>.csv to this

    FILE_COLUMN = None  # None = autodetect first added file column
    SESSION_NUMBER = 1


    GROUP_FOLDERS = {
        "NT": Path("data/cat12/NT"),
        "CT": Path("data/cat12/CT"),
    }

    # clusters source file (the one that contains multiple candidate columns)
    CLUSTERS_FILE = Path("data/q_data/filtered_merged_data (9).csv")

    # which columns (in CLUSTERS_FILE) to export as b_clusters (case-insensitive names)
    CLUSTER_COLS: List[str] = [
        "before", "t1", "t2", "t3", "after"
        # add more here, e.g.: "after", "during", "b", "groupA", ...
    ]

    print("Using group folders:", {k: v.as_posix() for k, v in GROUP_FOLDERS.items()})

    df_ses1 = pd.read_csv("output_q_and_t1/sessions_summary_with_files.csv")
    if "subject_id" not in df_ses1.columns or "session" not in df_ses1.columns:
        raise ValueError("Expected 'subject_id' and 'session' in base CSV.")

    file_col = FILE_COLUMN if FILE_COLUMN else first_added_file_column(df_ses1)
    mask = (df_ses1["session"] == SESSION_NUMBER)
    selected = df_ses1.loc[mask, ["subject_id"]].drop_duplicates().reset_index(drop=True)

    rows = []
    for idx, sid in enumerate(selected["subject_id"], start=1):
        xlsx = find_subject_file(sid, session=SESSION_NUMBER, GROUP_FOLDERS=GROUP_FOLDERS)
        print(f"({idx}/{len(selected)}) looking for {sid} -> {xlsx}")
        if xlsx is None or xlsx.is_dir():
            print(f"({idx}/{len(selected)}) ⚠️ Missing/dir for {sid}")
            continue
        try:
            euler, tiv = read_euler_tiv(xlsx)
        except Exception as e:
            print(f"({idx}/{len(selected)}) ⚠️ Error reading {xlsx}: {e}")
            continue
        rows.append({"subject_id": sid, "Euler": euler, "TIV": tiv})
        print(f"({idx}/{len(selected)}) ✅ Added {sid} (Euler={euler}, TIV={tiv})")

    base_df = pd.DataFrame(rows, columns=["subject_id", "Euler", "TIV"]).drop_duplicates()

    # Load clusters file once
    cl = pd.read_csv(CLUSTERS_FILE)

    # choose subject id column in clusters file
    possible_sid_cols = [c for c in cl.columns if str(c).strip().lower() in
                         ["subject_id", "subject", "id", "participant", "participant_id"]]
    cl_sid_col = possible_sid_cols[0] if possible_sid_cols else cl.columns[0]
    # =========================
    # EXPORT one file per requested cluster column
    # =========================
    for requested_col in CLUSTER_COLS:
        actual_col = find_case_insensitive_column(cl, requested_col)
        if actual_col is None:
            print(f"⚠️ Skipping: column '{requested_col}' not found in {CLUSTERS_FILE.name}. "
                  f"Available: {list(cl.columns)}")
            continue

        # ניצור טבלה שמכילה את כל נבדקות ה-MRI (base_df),
        # ונצרף אליה (אם יש) את הקבוצה מהשאלונים ב-left merge – כדי לא לאבד נבדקות בלי שאלונים
        merged_ses1 = base_df.merge(
            cl[[cl_sid_col, actual_col]],
            how="left",
            left_on="subject_id",
            right_on=cl_sid_col
        )

        # שם עמודת הקבוצה שנוסיף
        new_col_name = f"{requested_col}_clusters"
        merged_ses1 = merged_ses1.rename(columns={actual_col: new_col_name})

        # אפשר למחוק רק ערכים חסרים בעמודות ה-MRI (אם חשוב לך לנקות),
        # אבל לא למחוק בגלל שאין קבוצה (new_col_name יכול להישאר NaN)
        merged_ses1 = merged_ses1.dropna(subset=["Euler", "TIV"])

        # נשמור את הכל – כולל מי שאין לה קבוצת שאלונים
        suffix = sanitize_filename_piece(requested_col)
        out_path = f"{OUT_BASENAME_ses1}__{suffix}_ses1.csv"
        merged_ses1[["subject_id", "Euler", "TIV", new_col_name]].to_csv(out_path, index=False, encoding="utf-8")

        print(f"✅ Saved: {out_path}  (kept all MRI subjects; questionnaire group optional in '{new_col_name}')")


        SUBJECTS_CSV_SES1= "output_q_and_t1/selected_subjects_euler_tiv__before_ses1.csv"  # קובץ הנבדקים האחרון שיצרת
        OUTPUT_CSV_Flatten_ses1 = "output_q_and_t1/subjects_flattened_session1.csv"

        SESSION_NUMBER = 1  # רק סשן 1

        # אם יש גיליון מועדף בקבצי הנבדקים (למשל "data") – שימי כאן את שמו;
        # אם None נכפה sheet_name=0 (הגיליון הראשון) כדי לא לקבל dict מ-read_excel
        PREFERRED_SHEET_NAME = None

        # המרה למספרים כשאפשר (מחרוזות לא-מספריות יהפכו NaN)
        COERCE_TO_NUMERIC = True


        subjects_df_ses1 = pd.read_csv(SUBJECTS_CSV_SES1)
        if "subject_id" not in subjects_df_ses1.columns:
            raise ValueError("Expected 'subject_id' column in the subjects file.")

        subjects = subjects_df_ses1["subject_id"].dropna().astype(str).unique().tolist()
        print(f"Subjects to process (session {SESSION_NUMBER} only): {len(subjects)}")

        feature_order: List[str] = []
        feature_set: set = set()
        rows_dicts: List[Dict] = []

        added = 0
        skipped_not_found = 0
        skipped_errors = 0

        for idx, sid in enumerate(subjects, start=1):
            xlsx = find_subject_file(sid, session=SESSION_NUMBER,GROUP_FOLDERS = GROUP_FOLDERS)
            if xlsx is None:
                print(f"({idx}/{len(subjects)}) ⚠️ Missing session-{SESSION_NUMBER} file for {sid}. Skipping.")
                skipped_not_found += 1
                continue

            try:
                df_subj = read_subject_table_without_euler_tiv(xlsx, sheet_name=PREFERRED_SHEET_NAME)
                values, names = flatten_df_with_names(df_subj)
            except Exception as e:
                print(f"({idx}/{len(subjects)}) ⚠️ Error reading/flattening {xlsx.name} for {sid}: {e}. Skipping.")
                skipped_errors += 1
                continue

            for n in names:
                if n not in feature_set:
                    feature_set.add(n)
                    feature_order.append(n)

            row_map = {"subject_id": sid}
            for n, v in zip(names, values):
                row_map[n] = v
            rows_dicts.append(row_map)

            added += 1
            print(f"({idx}/{len(subjects)}) ✅ Flattened {sid}: {len(values)} values, {len(names)} names")

        # Build DataFrame with all features
        if added == 0:
            out_df = pd.DataFrame(columns=["subject_id"])
        else:
            col_names = ["subject_id"] + feature_order
            out_df = pd.DataFrame(rows_dicts)
            out_df = out_df.reindex(columns=col_names)

        out_df.to_csv(OUTPUT_CSV_Flatten_ses1, index=False, encoding="utf-8")
        print("\n=========================")
        print(f"Saved flattened matrix session1 : {OUTPUT_CSV_Flatten_ses1}")
        print(f"Subjects added: {added}")
        print(f"Subjects skipped (file not found): {skipped_not_found}")
        print(f"Subjects skipped (errors): {skipped_errors}")
        print(f"Total features: {len(feature_order)}")
        print("=========================")

        print(out_df.head(5))




    # === Paths ===
    path_ses1_normalized = "output_q_and_t1/subjects_flattened_session1_tivnorm.csv"

    # === Load ===
    flat = pd.read_csv("output_q_and_t1/subjects_flattened_session1.csv")
    tiv = pd.read_csv("output_q_and_t1/selected_subjects_euler_tiv__before_ses1.csv")

    # Merge on subject_id
    merged_ses1 = flat.merge(tiv[["subject_id", "TIV"]], on="subject_id", how="left")

    # Identify feature columns (all except subject_id and TIV)
    exclude_cols = {"subject_id", "Euler", "TIV"}
    feature_cols = [c for c in merged_ses1.columns if c not in exclude_cols]

    # Divide features by each subject's TIV
    for c in feature_cols:
        merged_ses1[c] = merged_ses1[c] / merged_ses1["TIV"]

    # Save result (without duplicating TIV if you don’t want it)
    merged_ses1.to_csv(path_ses1_normalized, index=False, encoding="utf-8")

    print(f"✅ Saved TIV-normalized matrix for session 1 : {path_ses1_normalized}")
    print(f"Total subjects: {merged_ses1.shape[0]}, features normalized: {len(feature_cols)}")

    print(merged_ses1.head(5))

    num90_1, num95_1, num100_1 = find_optimal_pca_dimensions(path_ses1_normalized, "output_q_and_t1/ses1_figures")

    print("Components for 90% variability session1 :", num90_1)
    print("Components for 95% variability session1:", num95_1)
    print("Components for 100% variability session1:", num100_1)

    # --- 0) pick the features file and #components ---
    # --- 0) pick the features file and #components ---
    # use the SAME file you used to pick num90_1 for consistency

    # --- 1) prepare features -> subj_series, Z and names ---
    feature_names, Z, subj_series = prepare_features_and_subjects(path_ses1_normalized)

    # convert ndarray -> DataFrame for perform_pca (keeps subject order)
    X_df_ses1 = pd.DataFrame(Z, columns=feature_names, index=subj_series)

    # --- 2) PCA using your existing perform_pca signature ---
    pca_scores_df, pca = perform_pca(X_df_ses1, n_components=num90_1)

    # get what downstream code expects
    scores = pca_scores_df.iloc[:, :num90_1].to_numpy()  # (n_samples, n_components)
    explained = pca.explained_variance_ratio_

    # (optional) common axis limits for consistent subplots
    x, y = scores[:, 0], scores[:, 1]
    pad = 0.06
    xlim = (x.min() - (x.max() - x.min()) * pad or x.min() - 1, x.max() + (x.max() - x.min()) * pad or x.max() + 1)
    ylim = (y.min() - (y.max() - y.min()) * pad or y.min() - 1, y.max() + (y.max() - y.min()) * pad or y.max() + 1)

    # Build the clusters_series list [(series, period), ...] using your existing helper:
    cluster_files = [
        ("output_q_and_t1/selected_subjects_euler_tiv__before_ses1.csv", "before"),
        ("output_q_and_t1/selected_subjects_euler_tiv__t1_ses1.csv", "t1"),
        ("output_q_and_t1/selected_subjects_euler_tiv__t2_ses1.csv", "t2"),
        ("output_q_and_t1/selected_subjects_euler_tiv__t3_ses1.csv", "t3"),
        ("output_q_and_t1/selected_subjects_euler_tiv__after_ses1.csv", "after"),
    ]

    # ---------------- Make a 2×3 subplot figure ----------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()

    for i, (path, period) in enumerate(cluster_files):
        s = get_clusters_series_from_file(path, subj_series, expected_period=period)
        ax = axes[i]
        plot_pc12_colored(scores, explained, s, period_label=period, ax=ax)
        ax.set_xlim(xlim);
        ax.set_ylim(ylim)

        # כותרת לפי התקופה
        TITLES = {"before": "Pre pregnancy", "t1": "1st trimester", "t2": "2nd trimester",
                  "t3": "3rd trimester", "after": "Post pregnancy"}
        ax.set_title(TITLES.get(period.lower(), period))

        # NEW: axis labels
        ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% var)")
        ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% var)")

    # hide any unused panels
    for j in range(len(cluster_files), len(axes)):
        axes[j].axis("off")

    # **total subjects in title**
    fig.suptitle(f"PCA PC1 vs PC2 across periods  —  total subjects n={len(subj_series)} session 1 ",
                 fontsize=14, y=1.02)

    plt.tight_layout()
    plt.show()

    # ---------------- Plot loadings for PC1 & PC2 ----------------
    plot_top_loadings(pca, feature_names, pc_index=0, top_n=10)
    plot_top_loadings(pca, feature_names, pc_index=1, top_n=10)

    out_dir = Path("output_q_and_t1/ses1_figures")
    out_dir.mkdir(exist_ok=True)

    save_top_loadings(pca, feature_names, pc_index=0, top_n=10,
                      out_csv=out_dir / "PC1_top20_loadings_session1.csv")
    save_top_loadings(pca, feature_names, pc_index=1, top_n=10,
                      out_csv=out_dir / "PC2_top20_loadings_session1.csv")


    ####################################################################session2

    OUT_BASENAME_ses2 = "output_q_and_t1/selected_subjects_euler_tiv"  # we'll append __<cluster_col>.csv to this

    SESSION_NUMBER = 2


    df_ses2 = pd.read_csv("output_q_and_t1/sessions_summary_with_files.csv")
    if "subject_id" not in df_ses1.columns or "session" not in df_ses2.columns:
        raise ValueError("Expected 'subject_id' and 'session' in base CSV.")

    file_col = FILE_COLUMN if FILE_COLUMN else first_added_file_column(df_ses2)
    mask = (df_ses2["session"] == SESSION_NUMBER)
    selected = df_ses2.loc[mask, ["subject_id"]].drop_duplicates().reset_index(drop=True)

    rows = []
    for idx, sid in enumerate(selected["subject_id"], start=1):
        xlsx = find_subject_file(sid, session=SESSION_NUMBER, GROUP_FOLDERS=GROUP_FOLDERS)
        print(f"({idx}/{len(selected)}) looking for {sid} -> {xlsx}")
        if xlsx is None or xlsx.is_dir():
            print(f"({idx}/{len(selected)}) ⚠️ Missing/dir for {sid}")
            continue
        try:
            euler, tiv = read_euler_tiv(xlsx)
        except Exception as e:
            print(f"({idx}/{len(selected)}) ⚠️ Error reading {xlsx}: {e}")
            continue
        rows.append({"subject_id": sid, "Euler": euler, "TIV": tiv})
        print(f"({idx}/{len(selected)}) ✅ Added {sid} (Euler={euler}, TIV={tiv})")

    base_df = pd.DataFrame(rows, columns=["subject_id", "Euler", "TIV"]).drop_duplicates()

    # Load clusters file once
    cl = pd.read_csv(CLUSTERS_FILE)

    # choose subject id column in clusters file
    possible_sid_cols = [c for c in cl.columns if str(c).strip().lower() in
                         ["subject_id", "subject", "id", "participant", "participant_id"]]
    cl_sid_col = possible_sid_cols[0] if possible_sid_cols else cl.columns[0]
    # =========================
    # EXPORT one file per requested cluster column
    # =========================
    for requested_col in CLUSTER_COLS:
        actual_col = find_case_insensitive_column(cl, requested_col)
        if actual_col is None:
            print(f"⚠️ Skipping: column '{requested_col}' not found in {CLUSTERS_FILE.name}. "
                  f"Available: {list(cl.columns)}")
            continue

        # ניצור טבלה שמכילה את כל נבדקות ה-MRI (base_df),
        # ונצרף אליה (אם יש) את הקבוצה מהשאלונים ב-left merge – כדי לא לאבד נבדקות בלי שאלונים
        merged_ses2 = base_df.merge(
            cl[[cl_sid_col, actual_col]],
            how="left",
            left_on="subject_id",
            right_on=cl_sid_col
        )

        # שם עמודת הקבוצה שנוסיף
        new_col_name = f"{requested_col}_clusters"
        merged_ses2 = merged_ses2.rename(columns={actual_col: new_col_name})

        # אפשר למחוק רק ערכים חסרים בעמודות ה-MRI (אם חשוב לך לנקות),
        # אבל לא למחוק בגלל שאין קבוצה (new_col_name יכול להישאר NaN)
        merged_ses2 = merged_ses2.dropna(subset=["Euler", "TIV"])

        # נשמור את הכל – כולל מי שאין לה קבוצת שאלונים
        suffix = sanitize_filename_piece(requested_col)
        out_path = f"{OUT_BASENAME_ses2}__{suffix}_ses2.csv"
        merged_ses2[["subject_id", "Euler", "TIV", new_col_name]].to_csv(out_path, index=False, encoding="utf-8")

        print(f"✅ Saved: {out_path}  (kept all MRI subjects; questionnaire group optional in '{new_col_name}')")

        SUBJECTS_CSV_SES2 = "output_q_and_t1/selected_subjects_euler_tiv__before_ses2.csv"  # קובץ הנבדקים האחרון שיצרת
        OUTPUT_CSV_Flatten_ses2 = "output_q_and_t1/subjects_flattened_session2.csv"


        subjects_df_ses2 = pd.read_csv(SUBJECTS_CSV_SES2)
        if "subject_id" not in subjects_df_ses2.columns:
            raise ValueError("Expected 'subject_id' column in the subjects file.")

        subjects = subjects_df_ses2["subject_id"].dropna().astype(str).unique().tolist()
        print(f"Subjects to process (session {SESSION_NUMBER} only): {len(subjects)}")

        feature_order: List[str] = []
        feature_set: set = set()
        rows_dicts: List[Dict] = []

        added = 0
        skipped_not_found = 0
        skipped_errors = 0

        for idx, sid in enumerate(subjects, start=1):
            xlsx = find_subject_file(sid, session=SESSION_NUMBER, GROUP_FOLDERS=GROUP_FOLDERS)
            if xlsx is None:
                print(f"({idx}/{len(subjects)}) ⚠️ Missing session-{SESSION_NUMBER} file for {sid}. Skipping.")
                skipped_not_found += 1
                continue

            try:
                df_subj = read_subject_table_without_euler_tiv(xlsx, sheet_name=PREFERRED_SHEET_NAME)
                values, names = flatten_df_with_names(df_subj)
            except Exception as e:
                print(f"({idx}/{len(subjects)}) ⚠️ Error reading/flattening {xlsx.name} for {sid}: {e}. Skipping.")
                skipped_errors += 1
                continue

            for n in names:
                if n not in feature_set:
                    feature_set.add(n)
                    feature_order.append(n)

            row_map = {"subject_id": sid}
            for n, v in zip(names, values):
                row_map[n] = v
            rows_dicts.append(row_map)

            added += 1
            print(f"({idx}/{len(subjects)}) ✅ Flattened {sid}: {len(values)} values, {len(names)} names")

        # Build DataFrame with all features
        if added == 0:
            out_df = pd.DataFrame(columns=["subject_id"])
        else:
            col_names = ["subject_id"] + feature_order
            out_df = pd.DataFrame(rows_dicts)
            out_df = out_df.reindex(columns=col_names)

        out_df.to_csv(OUTPUT_CSV_Flatten_ses2, index=False, encoding="utf-8")
        print("\n=========================")
        print(f"Saved flattened matrix session2 : {OUTPUT_CSV_Flatten_ses2}")
        print(f"Subjects added: {added}")
        print(f"Subjects skipped (file not found): {skipped_not_found}")
        print(f"Subjects skipped (errors): {skipped_errors}")
        print(f"Total features: {len(feature_order)}")
        print("=========================")

        print(out_df.head(5))

    # === Paths ===
    path_ses2_normalized = "output_q_and_t1/subjects_flattened_session2_tivnorm.csv"

    # === Load ===
    flat = pd.read_csv("output_q_and_t1/subjects_flattened_session2.csv")
    tiv = pd.read_csv("output_q_and_t1/selected_subjects_euler_tiv__before_ses2.csv")

    # Merge on subject_id
    merged_ses2 = flat.merge(tiv[["subject_id", "TIV"]], on="subject_id", how="left")

    # Identify feature columns (all except subject_id and TIV)
    exclude_cols = {"subject_id", "Euler", "TIV"}
    feature_cols = [c for c in merged_ses2.columns if c not in exclude_cols]

    # Divide features by each subject's TIV
    for c in feature_cols:
        merged_ses2[c] = merged_ses2[c] / merged_ses2["TIV"]

    # Save result (without duplicating TIV if you don’t want it)
    merged_ses2.to_csv(path_ses2_normalized, index=False, encoding="utf-8")

    print(f"✅ Saved TIV-normalized matrix for session 2 : {path_ses2_normalized}")
    print(f"Total subjects: {merged_ses2.shape[0]}, features normalized: {len(feature_cols)}")

    print(merged_ses2.head(5))

    num90_2, num95_2, num100_2 = find_optimal_pca_dimensions(path_ses2_normalized, "output_q_and_t1/ses2_figures")

    print("Components for 90% variability session2 :", num90_2)
    print("Components for 95% variability session2:", num95_2)
    print("Components for 100% variability session2:", num100_2)

    # --- 0) pick the features file and #components ---
    # --- 0) pick the features file and #components ---
    # use the SAME file you used to pick num90_1 for consistency

    # --- 1) prepare features -> subj_series, Z and names ---
    feature_names, Z, subj_series = prepare_features_and_subjects(path_ses2_normalized)

    # convert ndarray -> DataFrame for perform_pca (keeps subject order)
    X_df_ses2 = pd.DataFrame(Z, columns=feature_names, index=subj_series)

    # --- 2) PCA using your existing perform_pca signature ---
    pca_scores_df, pca = perform_pca(X_df_ses2, n_components=num90_2)

    # get what downstream code expects
    scores = pca_scores_df.iloc[:, :num90_2].to_numpy()  # (n_samples, n_components)
    explained = pca.explained_variance_ratio_

    # (optional) common axis limits for consistent subplots
    x, y = scores[:, 0], scores[:, 1]
    pad = 0.06
    xlim = (x.min() - (x.max() - x.min()) * pad or x.min() - 1, x.max() + (x.max() - x.min()) * pad or x.max() + 1)
    ylim = (y.min() - (y.max() - y.min()) * pad or y.min() - 1, y.max() + (y.max() - y.min()) * pad or y.max() + 1)

    # Build the clusters_series list [(series, period), ...] using your existing helper:
    cluster_files = [
        ("output_q_and_t1/selected_subjects_euler_tiv__before_ses2.csv", "before"),
        ("output_q_and_t1/selected_subjects_euler_tiv__t1_ses2.csv", "t1"),
        ("output_q_and_t1/selected_subjects_euler_tiv__t2_ses2.csv", "t2"),
        ("output_q_and_t1/selected_subjects_euler_tiv__t3_ses2.csv", "t3"),
        ("output_q_and_t1/selected_subjects_euler_tiv__after_ses2.csv", "after"),
    ]

    # ---------------- Make a 2×3 subplot figure ----------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()

    for i, (path, period) in enumerate(cluster_files):
        s = get_clusters_series_from_file(path, subj_series, expected_period=period)
        ax = axes[i]
        plot_pc12_colored(scores, explained, s, period_label=period, ax=ax)
        ax.set_xlim(xlim);
        ax.set_ylim(ylim)

        # כותרת לפי התקופה
        TITLES = {"before": "Pre pregnancy", "t1": "1st trimester", "t2": "2nd trimester",
                  "t3": "3rd trimester", "after": "Post pregnancy"}
        ax.set_title(TITLES.get(period.lower(), period))

        # NEW: axis labels
        ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% var)")
        ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% var)")

    # hide any unused panels
    for j in range(len(cluster_files), len(axes)):
        axes[j].axis("off")

    # **total subjects in title**
    fig.suptitle(f"PCA PC1 vs PC2 across periods  —  total subjects n={len(subj_series)} session 1 ",
                 fontsize=14, y=1.02)

    plt.tight_layout()
    plt.show()

    # ---------------- Plot loadings for PC1 & PC2 ----------------
    plot_top_loadings(pca, feature_names, pc_index=0, top_n=10)
    plot_top_loadings(pca, feature_names, pc_index=1, top_n=10)

    out_dir = Path("output_q_and_t1/ses2_figures")
    out_dir.mkdir(exist_ok=True)

    save_top_loadings(pca, feature_names, pc_index=0, top_n=10,
                      out_csv=out_dir / "PC1_top20_loadings_session2.csv")
    save_top_loadings(pca, feature_names, pc_index=1, top_n=10,
                      out_csv=out_dir / "PC2_top20_loadings_session2.csv")

    # ===== Example usage (replace with your real data) =====
    groups = ["Pre", "Post"]
    s1 = [num90_1, num90_2]  # values for column 1 in each group
    s2 = [num95_1, num95_2]  # values for column 2
    s3 = [num100_1, num100_2]  # values for column 2

    plot_grouped_bars(groups, s1, s2, s3,
                      s1_label="Explain 90% variability",
                      s2_label="Explain 95% variability",
                      s3_label="Explain 100% variability",
                      title="Number of Dimentions for each Time period",
                      ylabel="# of dimentions")




if __name__ == "__main__":
    main()

