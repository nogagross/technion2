import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import cm
import re
from typing import Optional, List,Iterable
from sklearn.preprocessing import StandardScaler

ID_PAT_RE = re.compile(r'(?P<prefix>[A-Za-z]{1,4})[-_ ]*0*(?P<num>\d+)$')
def _load_table_any(file_path, sheet_name='Data'):
    p = Path(file_path)
    ext = p.suffix.lower()
    if ext in {".csv", ".tsv"}:
        # sniff delimiter; fallback to comma
        try:
            return pd.read_csv(file_path, sep=None, engine="python")
        except Exception:
            return pd.read_csv(file_path)
    elif ext in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
        return pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
    elif ext == ".xls":
        # legacy excel; requires xlrd installed
        return pd.read_excel(file_path, sheet_name=sheet_name, engine="xlrd")
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _load_clustered_df(clustered_path):
    p = Path(clustered_path)
    if p.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    return pd.read_csv(p)

def _ensure_subject_col(df, subject_col):
    """Return df copy with a guaranteed '__SUBJECT__' column created from name or position."""
    out = df.copy()
    if isinstance(subject_col, int):
        if subject_col < 0 or subject_col >= out.shape[1]:
            raise IndexError(f"subject_col index {subject_col} out of range for columns: {list(out.columns)}")
        out["__SUBJECT__"] = out.iloc[:, subject_col].astype(str).str.strip()
    elif isinstance(subject_col, str):
        if subject_col in out.columns:
            out["__SUBJECT__"] = out[subject_col].astype(str).str.strip()
        else:
            out = out.reset_index(drop=False)
            if subject_col in out.columns:
                out["__SUBJECT__"] = out[subject_col].astype(str).str.strip()
            else:
                raise KeyError(f"Subject column '{subject_col}' not found (even after reset_index).")
    else:
        # autodetect (rarely used here)
        candidates = ["Subject", "subject", "Subject_ID", "Subject_Code", "id", "participant"]
        found = next((c for c in candidates if c in out.columns), None)
        if found:
            out["__SUBJECT__"] = out[found].astype(str).str.strip()
        else:
            unnamed = next((c for c in out.columns if isinstance(c, str) and c.lower().startswith("unnamed")), None)
            if unnamed:
                out["__SUBJECT__"] = out[unnamed].astype(str).str.strip()
            else:
                out["__SUBJECT__"] = out.iloc[:, 0].astype(str).str.strip()
    return out

def _ensure_pc_axes(df):
    """Return xcol,ycol names for plotting."""
    if {"PC1","PC2"}.issubset(df.columns):
        return "PC1", "PC2"
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) < 2:
        raise ValueError("Could not find two numeric columns for plotting.")
    return num_cols[0], num_cols[1]

def _load_mapping(mapping_path, mapping_subject_col, mapping_group_col):
    p = Path(mapping_path)
    mapping = pd.read_excel(p) if p.suffix.lower() in {".xlsx",".xls"} else pd.read_csv(p)
    mapping = mapping.rename(columns={mapping_subject_col: "MAP_SUBJ", mapping_group_col: "MAP_GROUP"})
    if "MAP_SUBJ" not in mapping or "MAP_GROUP" not in mapping:
        raise ValueError(f"Check mapping_subject_col/mapping_group_col. Got: {list(mapping.columns)}")
    mapping["MAP_SUBJ"] = mapping["MAP_SUBJ"].astype(str).str.strip()
    mapping["MAP_GROUP"] = mapping["MAP_GROUP"].astype(str).str.strip()
    return mapping



# רשימת העמודות שאינן פיצ'רים ויש להשמיט
# אתה יכול להוסיף או לשנות עמודות לפי הצרכים שלך
NON_FEATURE_COLS = ['group', 'subject_id', 'Unnamed: 0']

def get_feature_names(file_path):
    """
    טוען קובץ CSV, מסיר עמודות שאינן פיצ'רים, ומחזיר את רשימת שמות הפיצ'רים.

    Args:
        file_path (str): הנתיב לקובץ ה-CSV.

    Returns:
        list: רשימה של שמות הפיצ'רים.
    """
    try:
        # קורא את הקובץ
        df = pd.read_csv(file_path)

        # מסנן עמודות שאינן פיצ'רים
        feature_df = df.drop(columns=NON_FEATURE_COLS, errors='ignore')

        # מחזיר את שמות העמודות הנותרות
        return feature_df.columns.tolist()
    except FileNotFoundError:
        print(f"⚠️ שגיאה: הקובץ לא נמצא בנתיב: {file_path}")
        return None
    except Exception as e:
        print(f"⚠️ שגיאה בקריאת הקובץ {file_path}: {e}")
        return None


def _ensure_pc_axes(df):
    """Return xcol,ycol names for plotting."""
    if {"PC1","PC2"}.issubset(df.columns):
        return "PC1", "PC2"
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) < 2:
        raise ValueError("Could not find two numeric columns for plotting.")
    return num_cols[0], num_cols[1]

def normalize_subject_id(x: str, pattern: re.Pattern | None = None) -> str | None:
    """
    מאחד מזהי נבדקים לפורמט אחיד: PREFIX + מספר עם אפסים מובילים.
    אם pattern לא סופק, נשתמש ב-ID_PAT_RE הגלובלי.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    pat = pattern or ID_PAT_RE
    m = pat.search(s)
    if not m:
        return None
    prefix = m.group('prefix').upper()
    num    = m.group('num').lstrip("0") or "0"
    width = 3 if len(num) <= 3 else len(num)  # שמירת לפחות 3 ספרות
    return f"{prefix}{num.zfill(width)}"

def read_first_column(file_path: Path, sheet_name=None) -> pd.Series:
    """Return the first column as a Series, handling Excel/CSV appropriately."""
    ext = file_path.suffix.lower()
    if ext in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
        # Excel (OpenXML) -> openpyxl
        try:
            tmp = pd.read_excel(file_path, sheet_name=sheet_name, header=0, usecols=[0], engine="openpyxl")
        except Exception:
            tmp = pd.read_excel(file_path, sheet_name=sheet_name, header=None, usecols=[0], engine="openpyxl")
    elif ext == ".xls":
        # Legacy Excel (if you have xlrd; if not installed, pip install xlrd or convert to .xlsx)
        try:
            tmp = pd.read_excel(file_path, sheet_name=sheet_name, header=0, usecols=[0], engine="xlrd")
        except Exception:
            tmp = pd.read_excel(file_path, sheet_name=sheet_name, header=None, usecols=[0], engine="xlrd")
    elif ext in {".csv", ".tsv"}:
        sep = "," if ext == ".csv" else "\t"
        try:
            tmp = pd.read_csv(file_path, header=0, usecols=[0], sep=sep)
        except Exception:
            tmp = pd.read_csv(file_path, header=None, usecols=[0], sep=sep)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    first_col = tmp.columns[0]
    return tmp[first_col]

def column_name_from_file(path: Path) -> str:
    """Use the file stem up to the first space/_/-/( as the column name."""
    stem = path.stem
    print(stem)
    return re.split(r"[ _\-\(]", stem, maxsplit=1)[0]

def first_added_file_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c not in ("subject_id", "session"):
            return c
    raise ValueError("Could not find any added file columns")



def sanitize_name(s: str) -> str:
    SANITIZE_FEATURE_NAMES = True
    if not SANITIZE_FEATURE_NAMES:
        return str(s)
    s = str(s)
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^\w\-\.]+", "", s)  # שמירה רק על אותיות/ספרות/_/-
    return s


def read_euler_tiv(xlsx_path: Path):
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    lower_map = {str(c).lower(): c for c in df.columns}

    def pick_column(target: str) -> Optional[str]:
        tl = target.lower()
        if tl in lower_map:
            return lower_map[tl]
        for lc, orig in lower_map.items():
            if tl in lc:
                return orig
        return None

    euler_col = pick_column("euler")
    tiv_col   = pick_column("tiv")
    if euler_col is None: raise ValueError(f"No Euler col in {xlsx_path.name}")
    if tiv_col is None:   raise ValueError(f"No TIV col in {xlsx_path.name}")

    euler_val = df[euler_col].dropna().iloc[0] if not df[euler_col].dropna().empty else None
    tiv_val   = df[tiv_col].dropna().iloc[0] if not df[tiv_col].dropna().empty else None
    return euler_val, tiv_val

def find_case_insensitive_column(df: pd.DataFrame, target_name: str) -> Optional[str]:
    target = str(target_name).strip().lower()
    for c in df.columns:
        if str(c).strip().lower() == target:
            return c
    return None

def sanitize_filename_piece(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w\.-]+", "", s)
    return s

def _session_suffixes(session: int) -> Iterable[str]:
    """כל הווריאציות הרלוונטיות להופעת מספר הסשן בשם הקובץ."""
    if session is None or session == 1:
        # בסשן 1 לעתים אין סיומת בכלל
        return ["", "_1", "-1", " 1"]
    s = str(session)
    return [f"_{s}", f"-{s}", f" {s}"]

# preprocessing_functions.py

def group_from_subject(subject_id: str) -> str:
    m = re.match(r'^([A-Za-z]{2})\d+', str(subject_id).strip())
    if not m:
        raise ValueError(f"Bad subject_id format: {subject_id}")
    return m.group(1).upper()


def find_subject_file(subject_id: str, session: int = 1, GROUP_FOLDERS=None) -> Optional[Path]:
    if GROUP_FOLDERS is None:
        raise ValueError("GROUP_FOLDERS is required")

    grp = group_from_subject(subject_id)
    folder = GROUP_FOLDERS.get(grp)
    if folder is None:
        raise ValueError(f"No folder defined for group '{grp}'.")
    folder = Path(folder)

    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder does not exist or not a directory: {folder}")

    subject_id = str(subject_id).upper().strip()

    # סדר עדיפויות לסיומות
    exts = [".xlsx", ".csv", ".xls", ".mat", ".tsv"]
    # וריאציות של סשן בשם הקובץ
    suffixes = _session_suffixes(session)

    # 1) חיפוש “צר” – בדיוק <subject_id><suffix>.<ext>
    candidates: list[Path] = []
    for suf in suffixes:
        for ext in exts:
            candidates += list(folder.glob(f"{subject_id}{suf}{ext}"))

    # 2) אם לא נמצא – חיפוש “רחב” יותר: <subject_id><suffix>*.<ext>
    if not candidates:
        for suf in suffixes:
            for ext in exts:
                candidates += list(folder.glob(f"{subject_id}{suf}*{ext}"))

    # החזר מועמד תקין ראשון (קובץ בלבד)
    for p in candidates:
        if p.is_file():
            return p

    return None

def flatten_df_with_names(df: pd.DataFrame) -> (List, List):
    values, names = [], []
    row_labels = [sanitize_name(str(idx)) for idx in df.index.tolist()]
    col_labels = [sanitize_name(c) for c in df.columns.tolist()]

    for r_i, row in enumerate(df.itertuples(index=False), start=0):
        row_label = row_labels[r_i]
        for c_i, val in enumerate(row):
            col_label = col_labels[c_i]
            names.append(f"{row_label}_{col_label}")
            values.append(val)
    return values, names

def pick_col(df: pd.DataFrame, target: str) -> Optional[str]:
    lower_map = {str(c).lower(): c for c in df.columns}
    tl = target.lower()
    if tl in lower_map:
        return lower_map[tl]
    for lc, orig in lower_map.items():
        if tl in lc:
            return orig
    return None

def _ensure_dataframe(x):
    if isinstance(x, dict):
        for key in x.keys():
            if str(key).strip().lower() == "data":
                return x[key]
        first_key = next(iter(x.keys()))
        return x[first_key]
    return x
def read_subject_table_without_euler_tiv(xlsx_path: Path, sheet_name=None) -> pd.DataFrame:
    COERCE_TO_NUMERIC = True
    sheet_arg = sheet_name if sheet_name is not None else 0
    raw = pd.read_excel(xlsx_path, engine="openpyxl", sheet_name=sheet_arg)
    df = _ensure_dataframe(raw)

    # Identify Euler/TIV columns
    euler_col = pick_col(df, "euler")
    tiv_col   = pick_col(df, "tiv")
    Vcsf_col = pick_col(df, "Vcsf")
    Vwm_col = pick_col(df,"Vwm")
    drop_cols = [c for c in [euler_col, tiv_col,Vcsf_col,Vwm_col] if c is not None]

    # --- NEW: set the first column as row labels if it's not purely numeric ---
    first_col = df.columns[0]
    if df[first_col].dtype == object or not pd.api.types.is_numeric_dtype(df[first_col]):
        df = df.set_index(first_col)

    df2 = df.drop(columns=drop_cols) if drop_cols else df

    if COERCE_TO_NUMERIC:
        df2 = df2.apply(pd.to_numeric, errors="coerce")

    df2 = df2.dropna(axis=1, how="all")
    return df2
def prepare_features_and_subjects(features_path, sheet_name_features="Data"):
    df = _load_table_any(features_path, sheet_name=sheet_name_features)
    subj = df["subject_id"].astype(str).str.strip() if "subject_id" in df.columns else None

    drop_like = {"subject_id","subject","subject_code","euler","tiv","b_clusters","group"}
    X = df.drop(columns=[c for c in df.columns if str(c).strip().lower() in drop_like], errors="ignore")
    X = X.select_dtypes(include="number").dropna(axis=0, how="all")
    if X.empty:
        raise ValueError("No numeric features found.")
    X = X.fillna(X.mean(numeric_only=True))

    nunique = X.nunique(dropna=False)
    X = X.loc[:, nunique > 1]
    if X.shape[1] == 0:
        raise ValueError("All remaining columns have zero variance.")

    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    return X.columns.tolist(), Z, subj

