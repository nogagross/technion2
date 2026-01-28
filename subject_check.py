import os
import pandas as pd
from typing import List, Union, Iterable
from pathlib import Path
from typing import Union
import glob


def load_columns(
    path: str,
    columns: Union[str, Iterable[str]],
    *,
    sheet_name: Union[int, str] = 0,       # only used for Excel
    delimiter: str = None,                  # only used for CSVs; e.g., "," or "\t"
    case_insensitive: bool = True,
    drop_missing: bool = False              # if False, raises on missing cols
) -> pd.DataFrame:
    """
    Load a file and return a DataFrame with only the requested columns.

    Args:
        path: Path to a .csv, .xlsx, or .xls file.
        columns: Columns to select. Can be a list/tuple/set or a comma-separated string.
        sheet_name: Excel sheet index or name (only used for Excel files).
        delimiter: CSV delimiter (defaults to pandas' inference if None).
        case_insensitive: If True, matches requested columns ignoring case/extra spaces.
        drop_missing: If True, silently drops columns that aren't found; if False, raises.

    Returns:
        pandas.DataFrame with the selected columns (in the order requested).
    """
    if isinstance(columns, str):
        # allow "col1, col2, col3"
        columns = [c.strip() for c in columns.split(",") if c.strip()]
    else:
        columns = [str(c).strip() for c in columns]

    if not columns:
        raise ValueError("You must provide at least one column name.")

    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name=sheet_name)
    elif ext == ".csv":
        df = pd.read_csv(path, delimiter=delimiter) if delimiter else pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: '{ext}'. Use .csv, .xlsx, or .xls")

    # Build a mapping for case-insensitive, whitespace-normalized matching
    if case_insensitive:
        def norm(s): return str(s).strip().lower()
        actual_cols = {norm(c): c for c in df.columns}
        matched = []
        missing = []
        for want in columns:
            key = norm(want)
            if key in actual_cols:
                matched.append(actual_cols[key])
            else:
                missing.append(want)
    else:
        matched = [c for c in columns if c in df.columns]
        missing = [c for c in columns if c not in df.columns]

    if missing and not drop_missing:
        raise KeyError(
            "The following columns were not found in the file:\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\n\nAvailable columns are:\n"
            + ", ".join(map(str, df.columns))
        )

    if not matched:
        raise KeyError("None of the requested columns were found.")

    # Preserve the order requested by the user
    # Create a lookup from normalized actual names to actual names
    ordered = []
    seen = set()
    if case_insensitive:
        norm_map = {str(c).strip().lower(): c for c in df.columns}
        for want in columns:
            key = str(want).strip().lower()
            if key in norm_map:
                colname = norm_map[key]
                if colname not in seen:
                    ordered.append(colname)
                    seen.add(colname)
    else:
        for want in columns:
            if want in df.columns and want not in seen:
                ordered.append(want)
                seen.add(want)

    return df[ordered]


def _normalize_flag(s: pd.Series) -> pd.Series:
    """
    Convert mixed boolean/str/numeric flags to 0/1 integers.
    Treats {1, True, '1', 'true', 'yes', 'y', 't'} as 1; everything else -> 0.
    """
    truthy = {"1", "true", "yes", "y", "t"}
    def to01(v):
        try:
            # numeric?
            return 1 if float(v) == 1 else 0
        except Exception:
            return 1 if str(v).strip().lower() in truthy else 0
    return s.apply(to01).astype("int64")


def build_subject_sessions(
    path: Union[str, Path],
    *,
    subject_col: str = "subject_code",
    completed_first_col: str = "completed_first_fmri",
    completed_second_col: str = "completed_second_fmri",
    output_csv: Union[str, Path] = "subject_sessions.csv",
    sheet_name: Union[int, str] = 0,  # only used if the input is Excel
) -> pd.DataFrame:
    """
    Create a dataframe with columns [subject_code, session] according to:
      - If completed_second_fmri == 1 → two rows: ses-1 and ses-2
      - Else if completed_first_fmri == 1 → one row: ses-1
      - Else (both 0) → exclude
    Saves the result to `output_csv` and returns it.

    Args:
        path: Input CSV or Excel file.
        subject_col: Column with subject identifier.
        completed_first_col: Column indicating completion of first fMRI (0/1).
        completed_second_col: Column indicating completion of second fMRI (0/1).
        output_csv: Where to save the result CSV.
        sheet_name: Excel sheet index/name if reading an .xlsx/.xls file.

    Returns:
        A pandas DataFrame with columns ["subject_code", "session"].
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name=sheet_name)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .csv, .xlsx, or .xls.")

    # Helpful error if any required column is missing
    required = [subject_col, completed_first_col, completed_second_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Normalize flags to 0/1
    f1 = _normalize_flag(df[completed_first_col])
    f2 = _normalize_flag(df[completed_second_col])

    # Masks
    mask_second = f2.eq(1)
    mask_first_only = f1.eq(1) & f2.eq(0)

    # Build result (vectorized, no per-row loops)
    sub = df[[subject_col]].copy()

    ses_2_a = sub.loc[mask_second].assign(session="ses-1")
    ses_2_b = sub.loc[mask_second].assign(session="ses-2")
    ses_1   = sub.loc[mask_first_only].assign(session="ses-1")

    result = pd.concat([ses_2_a, ses_2_b, ses_1], ignore_index=True)

    # Standardize output column name to subject_code
    if subject_col != "subject_code":
        result = result.rename(columns={subject_col: "subject_code"})

    # Save
    result.to_csv(output_csv, index=False)
    print(f"Saved {len(result):,} rows to {output_csv}")

    return result
def _norm_subject(sub: str) -> str:
    """
    Normalize subject_code to match filename pattern 'sub-<code>'.
    Accepts '023', 'sub-023', 'SUB-023', etc., returns just the code part '023'.
    """
    s = str(sub).strip()
    s = s.replace("\\", "/")
    s = s.split("/")[-1]  # in case a path sneaks in
    s = s.lower()
    if s.startswith("sub-"):
        s = s[4:]
    return s
from pathlib import Path
import os, glob
from typing import Union

def _norm_session(session: str) -> str:
    """Return 'ses-<n>' for inputs like '1', 'ses-1', 'SES1'."""
    s = str(session).strip().lower().replace("_", "-")
    s = s.replace("ses", "ses-") if not s.startswith("ses-") else s
    # collapse double hyphens if any
    s = s.replace("ses--", "ses-")
    # if it's just a number, prefix
    if s == "" or s == "ses-" or s[-1].isdigit() and s.startswith("ses") and "-" not in s[3:]:
        # e.g., 'ses1' -> 'ses-1'
        s = s.replace("ses", "ses-")
    if not s.startswith("ses-"):
        # e.g., '1' -> 'ses-1'
        s = f"ses-{s}"
    return s

def _padded_sub_tag(subject_code: Union[str, int]) -> str:
    """
    Always return a 3-digit padded 'sub-XXX' tag:
      '23'      -> 'sub-023'
      'sub-23'  -> 'sub-023'
      '023'     -> 'sub-023'
      'sub-023' -> 'sub-023'
    """
    s = str(subject_code).strip().lower()
    if s.startswith("sub-"):
        s = s[4:]
    # keep only digits for safety (handles weird inputs like '023 ' or '23a')
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits == "":
        # fallback: use original (won't match unless your files use it)
        return "sub-" + s
    return f"sub-{digits.zfill(3)}"

def _exists_for(
    base_dir: Union[str, Path],
    subject_code: Union[str, int],
    session: str,
    atlas_keyword: str,
    *,
    verbose: bool = False
) -> int:
    """
    Look recursively in base_dir for files that contain:
      <sub-XXX>_<ses-Y>*seg-<atlas_keyword>
    where sub-XXX is always 3-digit padded.
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        if verbose:
            print(f"[WARN] Directory not found: {base_dir}")
        return 0

    sub_tag = _padded_sub_tag(subject_code)     # e.g., 'sub-023'
    ses_tag = _norm_session(session)            # e.g., 'ses-1'
    atlas_l = atlas_keyword.lower()             # 'yeo' or 'schaefer'

    # Example target: *sub-023_ses-1*seg-YEO*
    pattern = f"**/*{sub_tag}_{ses_tag}*seg-*"
    hits = glob.glob(str(base_dir / pattern), recursive=True)
    hits = [h for h in hits if atlas_l in os.path.basename(h).lower()]

    if verbose:
        if hits:
            print(f"[MATCH] {atlas_keyword}: {sub_tag} {ses_tag} -> {len(hits)} file(s). Example: {hits[0]}")
        else:
            print(f"[MISS ] {atlas_keyword}: {sub_tag} {ses_tag}")

    return 1 if hits else 0


def add_atlas_flags(
    sessions_csv: Union[str, Path],
    schaefer_dir: Union[str, Path],
    yeo_dir: Union[str, Path],
    output_csv: Union[str, Path] = None,
    *,
    subject_col: str = "subject_code",
    session_col: str = "session"
) -> pd.DataFrame:
    """
    Read subject-session CSV, add SCHAEFER and YEO columns based on presence of files
    in the given directories, and save to CSV.

    Args:
        sessions_csv: Path to CSV with at least [subject_col, session_col].
        schaefer_dir: Directory holding SCHAEFER files.
        yeo_dir: Directory holding YEO files.
        output_csv: Path to save the annotated CSV. Defaults to '<input>_with_atlases.csv'.
        subject_col: Column name for subject code.
        session_col: Column name for session label, e.g., 'ses-1', 'ses-2'.

    Returns:
        The annotated pandas DataFrame.
    """
    sessions_csv = Path(sessions_csv)
    df = pd.read_csv(sessions_csv)

    # Helpful check
    required = [subject_col, session_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing column(s): {missing}. Available columns: {list(df.columns)}"
        )

    # Compute flags
    df["SCHAEFER"] = [
        _exists_for(schaefer_dir, sub, ses, "SCHAEFER2018")
        for sub, ses in zip(df[subject_col], df[session_col])
    ]
    df["YEO"] = [
        _exists_for(yeo_dir, sub, ses, "YEO2011")
        for sub, ses in zip(df[subject_col], df[session_col])
    ]

    # Save
    if output_csv is None:
        output_csv = sessions_csv.with_name(sessions_csv.stem + "_with_atlases.csv")
    df.to_csv(output_csv, index=False)
    print(f"Saved annotated CSV to: {output_csv}")

    return df
def _cols_with_prefixes(df: pd.DataFrame, prefixes: List[str]) -> List[str]:
    """Case-insensitive column selection by any of the given prefixes."""
    low = {c.lower(): c for c in df.columns}
    out = []
    for lc, orig in low.items():
        if any(lc.startswith(p.lower()) for p in prefixes):
            out.append(orig)
    return out

def _period_flag(series_df: pd.DataFrame) -> pd.Series:
    """
    Return 0 if any NaN in the period's columns for the row, else 1.
    If period has no columns, return NaN for all rows.
    """
    if series_df.shape[1] == 0:
        return pd.Series(np.nan, index=series_df.index, dtype="float")
    has_nan = series_df.isna().any(axis=1)
    return (~has_nan).astype("int64")  # True->1 (complete), False->0 (has NaN)

def add_period_nan_flags_to_subject_list(
    questionnaire_xlsx: str,
    subject_list_csv: str,
    *,
    subject_col_questionnaire: str = "Subject_Code",
    subject_col_subjectlist: str = "Subject_Code",
    output_csv: str = "subject_mri_list_with_period_flags.csv",
) -> pd.DataFrame:
    """
    Compute per-period completeness flags (1=no NaNs, 0=has NaNs, NaN=no columns)
    from the questionnaire Excel and merge into subject_mri_list.csv.
    """
    # 1) Load questionnaire and subject list
    qdf = pd.read_excel(questionnaire_xlsx)
    sdf = pd.read_csv(subject_list_csv)

    leq_col  =[        'sasrq_date',
        'sasrq_1', 'sasrq_2', 'sasrq_3', 'sasrq_4', 'sasrq_5', 'sasrq_6', 'sasrq_7', 'sasrq_8', 'sasrq_9', 'sasrq_10',
        'sasrq_11', 'sasrq_12', 'sasrq_13', 'sasrq_14', 'sasrq_15', 'sasrq_16', 'sasrq_17', 'sasrq_18', 'sasrq_19',
        'sasrq_20',
        'sasrq_21', 'sasrq_22', 'sasrq_23', 'sasrq_24', 'sasrq_25', 'sasrq_26', 'sasrq_27', 'sasrq_28', 'sasrq_29',
        'sasrq_30',
        'sasrq_31', 'sasrq_total',
        'PROMOTE_date',
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
        "b_lec_0_to_16_total",'Became_Pregnant', 'b_ctq_Date','b_lec_date', 'b_lec_1', 'b_lec_2', 'b_lec_3', 'b_lec_4', 'b_lec_5', 'b_lec_6', 'b_lec_7', 'b_lec_8', 'b_lec_9', 'b_lec_10', 'b_lec_11', 'b_lec_12', 'b_lec_13', 'b_lec_14', 'b_lec_15', 'b_lec_16', 'b_lec_17', 'b_lec_interpersonal_events', 'b_lec_non_interpersonal_events','b_PCL5_date','b_strength_date','b_PBI_date','b_GAD7_date',
                       'birth_week', 'birth_type','b_DERS_date','b_LHQ_date','b_IRI_date']
    qdf= qdf.drop(columns= leq_col)
    # Ensure subject columns exist
    if subject_col_questionnaire not in qdf.columns:
        raise KeyError(f"'{subject_col_questionnaire}' not found in {questionnaire_xlsx}")
    if subject_col_subjectlist not in sdf.columns:
        raise KeyError(f"'{subject_col_subjectlist}' not found in {subject_list_csv}")

    # 2) Identify period columns (case-insensitive prefixes)
    # Your code previously used 'b' for "before"
    before_cols = _cols_with_prefixes(qdf, ["b"])
    print(before_cols)# before
    t1_cols     = _cols_with_prefixes(qdf, ["t1"])
    t2_cols     = _cols_with_prefixes(qdf, ["t2"])
    t3_cols     = _cols_with_prefixes(qdf, ["t3"])
    after_cols  = _cols_with_prefixes(qdf, ["after"])

    # 3) Build per-row flags in the questionnaire
    flags_df = qdf[[subject_col_questionnaire]].copy()
    flags_df["flag_before"] = _period_flag(qdf[before_cols])
    flags_df["flag_t1"]     = _period_flag(qdf[t1_cols])
    flags_df["flag_t2"]     = _period_flag(qdf[t2_cols])
    flags_df["flag_t3"]     = _period_flag(qdf[t3_cols])
    flags_df["flag_after"]  = _period_flag(qdf[after_cols])

    # If a subject appears multiple times, aggregate conservatively:
    # - If any row for the subject has NaN in a period → overall 0
    # - Else if all available rows are complete → 1
    # - If period has no columns at all → NaN
    def agg_period(col: pd.Series) -> float:
        # drop NaN flags (e.g., when period has no columns)
        non_nan = col.dropna()
        if non_nan.empty:
            return np.nan
        # if any 0 present → 0, else 1
        return 0.0 if (non_nan == 0).any() else 1.0

    agg = (
        flags_df
        .groupby(subject_col_questionnaire, dropna=False)
        .agg({
            "flag_before": agg_period,
            "flag_t1": agg_period,
            "flag_t2": agg_period,
            "flag_t3": agg_period,
            "flag_after": agg_period,
        })
        .reset_index()
    )

    # 4) Merge into your subject_mri_list.csv
    merged = sdf.merge(
        agg.rename(columns={subject_col_questionnaire: subject_col_subjectlist}),
        on=subject_col_subjectlist,
        how="left"
    )

    # 5) Save and return
    merged.to_csv(output_csv, index=False)
    print(f"Saved updated subject list with flags to: {output_csv}")
    return merged

def main():
    # Example 1: Excel (default first sheet)
    df = load_columns(
        "data/q_data/Study_Questionnaire_Responses_October.xlsx",
        ["Subject_Code", "Completed_First_fMRI", "Completed_Second_fMRI"]
    )
    print(df.head())

    df.to_csv("subject_mri_list.csv", index=False)

    df = pd.read_excel("data/q_data/Study_Questionnaire_Responses_October.xlsx")

    # 1️⃣ Identify all columns starting with "before" (case-insensitive)
    before_cols = [c for c in df.columns if str(c).lower().startswith("b")]
    t1_cols = [c for c in df.columns if str(c).lower().startswith("t1")]
    t2_cols = [c for c in df.columns if str(c).lower().startswith("t2")]
    t3_cols = [c for c in df.columns if str(c).lower().startswith("t3")]
    after_cols = [c for c in df.columns if str(c).lower().startswith("after")]

    print(f"Found {len(before_cols)} 'before' columns:")
    print(before_cols)

    # 2️⃣ Find rows where *any* of those columns have NaN
    mask_nan_before = df[before_cols].isna().any(axis=1)

    # 3️⃣ Get the subjects who have NaNs in those columns
    # (adjust 'Subject_Code' if your column is named differently)
    subjects_with_nan = df.loc[mask_nan_before, "Subject_Code"]

    print("\nSubjects with at least one NaN in 'before' columns:")
    print(subjects_with_nan.tolist())


    # 2️⃣ Find rows where *any* of those columns have NaN
    mask_nan_before = df[before_cols].isna().any(axis=1)

    # 3️⃣ Get the subjects who have NaNs in those columns
    # (adjust 'Subject_Code' if your column is named differently)
    subjects_with_nan = df.loc[mask_nan_before, "Subject_Code"]

    print("\nSubjects with at least one NaN in 'before' columns:")
    print(subjects_with_nan.tolist())


    # 2️⃣ Find rows where *any* of those columns have NaN
    mask_nan_before = df[before_cols].isna().any(axis=1)

    # 3️⃣ Get the subjects who have NaNs in those columns
    # (adjust 'Subject_Code' if your column is named differently)
    subjects_with_nan = df.loc[mask_nan_before, "Subject_Code"]

    print("\nSubjects with at least one NaN in 'before' columns:")
    print(subjects_with_nan.tolist())


    # 2️⃣ Find rows where *any* of those columns have NaN
    mask_nan_b = df[before_cols].isna().any(axis=1)
    mask_nan_t1 = df[t1_cols].isna().any(axis=1)
    mask_nan_t2 = df[t2_cols].isna().any(axis=1)
    mask_nan_t3 = df[t3_cols].isna().any(axis=1)
    mask_nan_after = df[after_cols].isna().any(axis=1)

    # 3️⃣ Get the subjects who have NaNs in those columns
    # (adjust 'Subject_Code' if your column is named differently)
    subjects_with_nan_b = df.loc[mask_nan_b, "Subject_Code"]
    subjects_with_nan_t1 = df.loc[mask_nan_t1, "Subject_Code"]
    subjects_with_nan_t2 = df.loc[mask_nan_t2, "Subject_Code"]
    subjects_with_nan_t3 = df.loc[mask_nan_t3, "Subject_Code"]
    subjects_with_nan_after = df.loc[mask_nan_after, "Subject_Code"]

    print("\nSubjects with at least one NaN in 'before' columns:")
    print(subjects_with_nan_b.tolist())

    print("\nSubjects with at least one NaN in 't1' columns:")
    print(subjects_with_nan_t1.tolist())

    print("\nSubjects with at least one NaN in 't2' columns:")
    print(subjects_with_nan_t2.tolist())

    print("\nSubjects with at least one NaN in 't3' columns:")
    print(subjects_with_nan_t3.tolist())

    print("\nSubjects with at least one NaN in 'after' columns:")
    print(subjects_with_nan_after.tolist())




    build_subject_sessions(
            path="subject_mri_list.csv",
            subject_col="Subject_Code",
            completed_first_col="Completed_First_fMRI",
            completed_second_col="Completed_Second_fMRI",
            output_csv="subject_sessions_mri.csv",
        )
    add_atlas_flags(
        sessions_csv=r"subject_sessions_mri.csv",  # your subject-session file
        schaefer_dir=r"data/SCHAEFER_mat_cor",
        yeo_dir="data/YEO_mat_cor",
        output_csv=r"subject_sessions_mri_with_atlases.csv",
        subject_col="subject_code",  # or "Subject_Code" if that's your header
        session_col="session",  # e.g., values like 'ses-1', 'ses-2'
    )

    add_period_nan_flags_to_subject_list(
        questionnaire_xlsx="data/q_data/Study_Questionnaire_Responses_October.xlsx",
        subject_list_csv="subject_mri_list.csv",
        subject_col_questionnaire="Subject_Code",
        subject_col_subjectlist="Subject_Code",
        output_csv="subject_mri_list_with_period_flags.csv",
    )


if __name__ == '__main__':
    main()


