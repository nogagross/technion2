from pathlib import Path
from statistics import correlation
from typing import Dict, List, Optional, Tuple, Union
import re
import numpy as np
from RDiff_map_for_arseney_group import get_diffusion_embedding
# =========================
# CONFIG – EDIT THESE
# =========================
FOLDER = r"data/SCHAEFER_mat_cor"       # e.g. r"C:\data\tsv" or "/home/user/data"
PATTERN = "*.tsv"                        # which files to include
DTYPE: Union[type, str] = float          # float | int | 'U' (strings) | etc.
ENCODING = "utf-8"
HAS_HEADER: Optional[bool] = None        # True / False / None for auto-detect

# ignore first row/column after loading
IGNORE_FIRST_ROW = True
IGNORE_FIRST_COL = True
# --- NEW: CSV with row names to remove (first column = row_name) ---
MISSING_ROWS_CSV = Path("data/SCHAEFER_mat_cor/csv_out/missing_rows_by_subject.csv")

# מה עושים אם יש כמה קבצים לאותו נבדק?
# 'error'  -> זריקת שגיאה
# 'latest' -> לשמור את האחרון (דורס)
# 'collect'-> לאסוף לרשימה (Dict[Key, List[np.ndarray]])
ON_DUPLICATE = "latest"

# איך לקבוע סדר נבדקים בתוך המערך המאוחד:
# 'key'     -> ממוין לפי המפתח (קבוצה, מספר, סשן)
# 'insertion' -> לפי סדר ההכנסה מהמילון (הוא לפי סדר הקבצים שנמצאו)
STACK_ORDER = "key"
# =========================

Key = Tuple[str, int, int]  # ('NT', 137, 1) לדוגמה


def _read_tsv(path: Path,
              dtype: Union[type, str],
              encoding: str,
              has_header: Optional[bool]) -> np.ndarray:
    """Read a single TSV into np.ndarray, with optional header handling."""
    def _try(skip_header: int) -> np.ndarray:
        return np.genfromtxt(
            path,
            delimiter="\t",
            dtype=dtype,
            encoding=encoding,
            autostrip=True,
            comments=None,
            missing_values=("", "NA", "NaN", "null", "NULL"),
            filling_values=np.nan,
            skip_header=skip_header
        )

    if has_header is True:
        return _try(1)
    if has_header is False:
        return _try(0)

    # Auto-detect: try without header, fall back to with header
    try:
        arr = _try(0)
        if arr.size == 0:
            return _try(1)
        return arr
    except Exception:
        return _try(1)


def _drop_firsts(arr: np.ndarray,
                 drop_row: bool,
                 drop_col: bool) -> np.ndarray:
    """Drop first row and/or first column safely for 1D/2D+ arrays."""
    if arr.size == 0:
        return arr
    if drop_row and arr.shape[0] > 0:
        arr = arr[1:]
    if drop_col and arr.ndim > 1 and arr.shape[1] > 0:
        arr = arr[:, 1:]
    return arr


def parse_subject_key3(filename: str) -> Key:
    """ חילוץ (group, subject_number, session_number) משם קובץ. """
    stem = Path(filename).stem
    m = re.search(r'(?P<grp>[A-Za-z]+)_sub-(?P<num>\d+).*?ses-(?P<ses>\d+)', stem, flags=re.IGNORECASE)
    if not m:
        patterns = [
            r'group-(?P<grp>[A-Za-z]+).*?sub-(?P<num>\d+).*?ses-(?P<ses>\d+)',
            r'(?P<grp>[A-Za-z]+)[-_]?sub[-_]?(\s*)?(?P<num>\d+).*?ses[-_]?(\s*)?(?P<ses>\d+)'
        ]
        for p in patterns:
            m = re.search(p, stem, flags=re.IGNORECASE)
            if m:
                break
    if not m:
        raise ValueError(f"Cannot parse group/subject/session from filename: {filename}")
    grp = m.group('grp').upper()
    num = int(m.group('num'))
    ses = int(m.group('ses'))
    return grp, num, ses


def load_tsv_folder_as_subject_dict(folder: Union[str, Path],
                                    pattern: str,
                                    dtype: Union[type, str],
                                    encoding: str,
                                    has_header: Optional[bool],
                                    drop_first_row: bool,
                                    drop_first_col: bool,
                                    on_duplicate: str = "latest"
                                    ) -> Union[Dict[Key, np.ndarray], Dict[Key, List[np.ndarray]]]:
    """ מחזיר Dict[(group, subject, session), ndarray] או רשימות אם 'collect'. """
    folder = Path(folder)
    files: List[Path] = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {folder}")

    if on_duplicate not in {"error", "latest", "collect"}:
        raise ValueError("on_duplicate must be one of: 'error', 'latest', 'collect'")

    if on_duplicate == "collect":
        subject_map: Dict[Key, List[np.ndarray]] = {}
    else:
        subject_map: Dict[Key, np.ndarray] = {}

    for f in files:
        key = parse_subject_key3(f.name)
        arr = _read_tsv(f, dtype=dtype, encoding=encoding, has_header=has_header)
        arr = _drop_firsts(arr, drop_first_row, drop_first_col)

        if on_duplicate == "collect":
            subject_map.setdefault(key, []).append(arr)
        else:
            if key in subject_map and on_duplicate == "error":
                raise ValueError(f"Duplicate subject for key {key} at file {f.name}")
            subject_map[key] = arr

    return subject_map


# ===== NEW: בניית מערך מאוחד + מילון אינדקסים =====
def build_stacked_array_and_index(
    subject_map: Union[Dict[Key, np.ndarray], Dict[Key, List[np.ndarray]]],
    order: str = "key"
) -> Tuple[np.ndarray, Dict[Key, Union[int, List[int]]], List[Key]]:
    """
    יוצר מערך מאוחד בכל הצירוף של המטריצות + מילון מיפוי key->index.
    אם הערכים הם רשימות (כש-ON_DUPLICATE='collect'), המיפוי הוא key->List[int].
    מחזיר:
      - all_arrays: np.ndarray בצורת (N, R, C)
      - index_by_key: Dict[Key, int | List[int]]
      - keys_for_index: רשימת המפתח עבור כל אינדקס (אורך N)
    """
    # קיבוץ כל המטריצות לרשימה אחת + שמירת מיפוי
    entries: List[Tuple[Key, np.ndarray]] = []

    # סדר מפתחות
    keys = list(subject_map.keys())
    if order == "key":
        keys = sorted(keys)
    elif order == "insertion":
        pass
    else:
        raise ValueError("order must be 'key' or 'insertion'")

    for key in keys:
        val = subject_map[key]
        if isinstance(val, list):
            for arr in val:
                entries.append((key, np.asarray(arr)))
        else:
            entries.append((key, np.asarray(val)))

    if not entries:
        raise ValueError("No arrays to stack.")

    # בדיקת אחידות צורה
    shapes = [a.shape for _, a in entries]
    first_shape = shapes[0]
    bad = [(k, s) for (k, _), s in zip(entries, shapes) if s != first_shape]
    if bad:
        preview = ", ".join([f"{k}:{s}" for k, s in bad[:5]])
        raise ValueError(f"All arrays must share the same shape. First shape={first_shape}. Mismatches: {preview}")

    # טיפוס משותף (שומר על דיוק – למשל float64)
    common_dtype = np.result_type(*[a.dtype for _, a in entries])

    # בנייה
    arrays_list = [a.astype(common_dtype, copy=False) for _, a in entries]
    all_arrays = np.stack(arrays_list, axis=0)  # (N, R, C)

    # מיפוי key -> index / [indices] + רשימת reverse
    index_by_key: Dict[Key, Union[int, List[int]]] = {}
    keys_for_index: List[Key] = []

    for idx, (key, _) in enumerate(entries):
        keys_for_index.append(key)
        if key in index_by_key:
            # כבר יש—הופך/מוסיף לרשימה
            if isinstance(index_by_key[key], list):
                index_by_key[key].append(idx)
            else:
                index_by_key[key] = [index_by_key[key], idx]
        else:
            index_by_key[key] = idx

    return all_arrays, index_by_key, keys_for_index
# ===== END NEW =====
def report_nan_inf_in_subjects(
    subject_map: Union[Dict[Key, np.ndarray], Dict[Key, List[np.ndarray]]]
) -> Dict[Key, Union[Tuple[int, int, int], List[Tuple[int, int, int]]]]:
    """
    For each key, count (#NaN, #Inf, total elements) in its array(s).
    Returns a dict:
      key -> (n_nan, n_inf, total)      if value is a single ndarray
      key -> [(n_nan, n_inf, total), …] if value is a list of arrays
    Prints a readable report and a summary.
    """
    any_problem = False
    results: Dict[Key, Union[Tuple[int, int, int], List[Tuple[int, int, int]]]] = {}

    def _count(arr: np.ndarray) -> Tuple[int, int, int]:
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        total = int(arr.size)
        return n_nan, n_inf, total

    print("\n=== NaN/Inf scan over subject arrays ===")
    for key, val in subject_map.items():
        if isinstance(val, list):
            counts_list: List[Tuple[int, int, int]] = []
            for arr in val:
                c = _count(arr)
                counts_list.append(c)
            results[key] = counts_list
            # Aggregate for display
            n_nan_tot = sum(c[0] for c in counts_list)
            n_inf_tot = sum(c[1] for c in counts_list)
            total_tot = sum(c[2] for c in counts_list)
            if n_nan_tot > 0 or n_inf_tot > 0:
                any_problem = True
                pct_nan = 100.0 * n_nan_tot / max(1, total_tot)
                pct_inf = 100.0 * n_inf_tot / max(1, total_tot)
                print(f"  - {key}: NaN={n_nan_tot} ({pct_nan:.4f}%), Inf={n_inf_tot} ({pct_inf:.4f}%), arrays={len(counts_list)}")
        else:
            n_nan, n_inf, total = _count(val)
            results[key] = (n_nan, n_inf, total)
            if n_nan > 0 or n_inf > 0:
                any_problem = True
                pct_nan = 100.0 * n_nan / max(1, total)
                pct_inf = 100.0 * n_inf / max(1, total)
                print(f"  - {key}: NaN={n_nan} ({pct_nan:.4f}%), Inf={n_inf} ({pct_inf:.4f}%), shape={val.shape}")

    if not any_problem:
        print("No NaNs or Infs found in any subject arrays.")
    else:
        print(">>> One or more subjects contain NaNs and/or Infs (see above).")

    return results

# where CSVs will be written under FOLDER
OUT_SUBFOLDER = "csv_out"

def _fmt_for_dtype(dt: np.dtype) -> str:
    """Choose a reasonable np.savetxt fmt based on dtype."""
    if np.issubdtype(dt, np.integer):
        return "%d"
    if np.issubdtype(dt, np.floating):
        return "%.15g"
    return "%s"

def _build_csv_name_from_key(key: Key, dup_idx: Optional[int] = None) -> str:
    """
    Build a stable CSV filename from the composite key (group, subject, session).
    If dup_idx is not None, append it for collect-mode (e.g., _rep1).
    """
    grp, num, ses = key
    base = f"{grp}_sub-{num:03d}_ses-{ses}"
    if dup_idx is not None:
        base += f"_rep{dup_idx}"
    return base + ".csv"

def save_subject_map_to_csv(
    subject_map: Union[Dict[Key, np.ndarray], Dict[Key, List[np.ndarray]]],
    out_dir: Union[str, Path],
    encoding: str = "utf-8"
) -> Path:
    """
    Save each subject's array(s) to CSV in out_dir.
    - Single ndarray per key -> one CSV per key.
    - List per key (collect mode) -> one CSV per item with suffix _rep{i}.
    Returns the output directory path.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for key, val in subject_map.items():
        if isinstance(val, list):
            for i, arr in enumerate(val, start=1):
                name = _build_csv_name_from_key(key, dup_idx=i)
                csv_path = out_path / name
                fmt = _fmt_for_dtype(arr.dtype)
                with open(csv_path, "w", encoding=encoding, newline="") as f:
                    np.savetxt(f, arr, delimiter=",", fmt=fmt)
        else:
            name = _build_csv_name_from_key(key)
            csv_path = out_path / name
            fmt = _fmt_for_dtype(val.dtype)
            with open(csv_path, "w", encoding=encoding, newline="") as f:
                np.savetxt(f, val, delimiter=",", fmt=fmt)

    return out_path

from io import StringIO

def get_fully_missing_row_names_from_tsv(path: Path, encoding: str) -> List[str]:
    """
    קורא את הקובץ כטקסט, לוקח את שמות השורות מהעמודה הראשונה,
    ובודק על בלוק הנתונים (ללא השורה/עמודה הראשונות) אילו שורות הן all-NaN.
    מחזיר רשימת שמות השורות החסרות בשלמותן.
    """
    with open(path, "r", encoding=encoding) as f:
        text = f.read()

    lines = text.splitlines()
    if not lines:
        return []

    # כמה עמודות יש בשורה הראשונה (כדי לעבוד גם כשיש טאב בתוך שם עמודה)
    first_line = lines[0].split("\t")
    n_cols = len(first_line)

    # שמות השורות = התא הראשון בכל שורה אחרי הכותרת
    # maxsplit כדי לשמר תאים גם אם יש טאב בשם שדה מאוחר יותר
    row_names = []
    for ln in lines[1:]:
        parts = ln.split("\t", maxsplit=n_cols - 1)
        row_names.append(parts[0] if parts else "")

    # בלוק נומרי: מהשורה השנייה, מהעמודה השנייה ועד הסוף
    usecols = tuple(range(1, n_cols))
    data = np.genfromtxt(
        StringIO(text),
        delimiter="\t",
        dtype=float,
        autostrip=True,
        comments=None,
        missing_values=("", "NA", "NaN", "null", "NULL"),
        filling_values=np.nan,
        skip_header=1,
        usecols=usecols
    )

    # אם יש רק שורה אחת, genfromtxt מחזיר 1D—ניישר ל-2D
    if data.ndim == 1:
        data = data.reshape(1, -1)

    # שורות שהכול בהן NaN
    mask_full_nan = np.all(np.isnan(data), axis=1)
    idx = np.flatnonzero(mask_full_nan)

    return [row_names[i] for i in idx]
def report_missing_row_names_before_load(folder: Union[str, Path],
                                         pattern: str,
                                         encoding: str) -> Dict[str, List[str]]:
    """
    מדפיס ומחזיר מיפוי filename -> רשימת שמות השורות שהן all-NaN,
    לפני הטעינה למערכים.
    """
    folder = Path(folder)
    files = sorted(folder.glob(pattern))
    results: Dict[str, List[str]] = {}

    print("\n=== Fully-missing row NAMES per file (pre-load) ===")
    for f in files:
        try:
            names = get_fully_missing_row_names_from_tsv(f, encoding)
            results[f.name] = names
            if names:
                try:
                    key = parse_subject_key3(f.name)
                    print(f"  - {key} from {f.name}: {len(names)} rows -> {names}")
                except Exception:
                    print(f"  - {f.name}: {len(names)} rows -> {names}")
        except Exception as e:
            print(f"  ! {f.name}: error while scanning ({e})")

    if not any(results.values()):
        print("No fully-missing row names found in any file.")
    return results
import csv

def _key_to_str(key: Key) -> str:
    """ייצוג נחמד למפתח: CT_sub-003_ses-1"""
    grp, num, ses = key
    return f"{grp}_sub-{num:03d}_ses-{ses}"

def build_missing_rows_aggregation(
    folder: Union[str, Path],
    pattern: str,
    encoding: str
) -> Dict[str, List[Key]]:
    """
    עובר על כל הקבצים בתיקייה, מזהה שמות שורות שהן all-NaN בקובץ,
    ומחזיר מיפוי: row_name -> [keys של נבדקים שחסרה בהם השורה הזו].
    """
    folder = Path(folder)
    files = sorted(folder.glob(pattern))
    agg: Dict[str, List[Key]] = {}

    for f in files:
        # מי זה הנבדק?
        try:
            key = parse_subject_key3(f.name)  # ('CT', 3, 1) למשל
        except Exception:
            # אם לא מצליחים לפרש — מדלגים/אפשר גם להרים שגיאה
            continue

        # אילו שמות שורות חסרות בשלמותן בקובץ הזה?
        try:
            missing_names = get_fully_missing_row_names_from_tsv(f, encoding)
        except Exception:
            continue

        for name in missing_names:
            agg.setdefault(name, []).append(key)

    return agg

def write_missing_rows_csv(
    agg: Dict[str, List[Key]],
    out_path: Union[str, Path]
) -> Path:
    """
    כותב CSV עם עמודות:
      row_name, missing_count, subjects
    כש־subjects הוא רשימה מופרדת בפסיקים של CT_sub-003_ses-1 וכד'.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_name", "missing_count", "subjects"])
        for row_name in sorted(agg.keys()):
            keys = agg[row_name]
            # מיון לפי (group, subject, session)
            keys_sorted = sorted(keys)
            subjects_str = ",".join(_key_to_str(k) for k in keys_sorted)
            w.writerow([row_name, len(keys_sorted), subjects_str])

    return out_path
# --- NEW ---
def _load_names_to_remove(csv_path: Union[str, Path]) -> set[str]:
    """
    קורא את ה-CSV (עמודה ראשונה = row_name) ומחזיר סט של שמות למחיקה.
    מצופה כותרת שורה ראשונה: row_name, missing_count, subjects (אבל משתמשים רק בעמודה 1).
    """
    csv_path = Path(csv_path)
    names: set[str] = set()
    if not csv_path.exists():
        return names
    with open(csv_path, "r", encoding="utf-8") as f:
        # דילוג על הכותרת אם יש
        header = f.readline()
        for line in f:
            if not line.strip():
                continue
            row_name = line.split(",", 1)[0].strip()
            if row_name and row_name.lower() != "row_name":
                names.add(row_name)
    return names


def _read_tsv_numeric_and_row_names(
    path: Path,
    dtype: Union[type, str],
    encoding: str,
) -> tuple[np.ndarray, list[str]]:
    """
    קורא קובץ TSV ומחזיר:
      - data: בלוק נומרי אחרי דילוג על *השורה הראשונה* ועל *העמודה הראשונה*
      - row_names: שמות השורות שנלקחו מהעמודה הראשונה (לפי הסדר), תואמים ל- axis=0 של data.
    הערה: זה עוקף את _read_tsv + _drop_firsts, כדי להבטיח יישור תוויות<->נתונים.
    """
    from io import StringIO
    text = Path(path).read_text(encoding=encoding)
    lines = text.splitlines()
    if not lines:
        return np.empty((0, 0), dtype=float if dtype is float else dtype), []

    # כמה עמודות בשורה הראשונה
    first_line = lines[0].split("\t")
    n_cols = len(first_line)

    # שמות השורות (עמודה ראשונה בכל שורה אחרי הכותרת)
    row_names: list[str] = []
    for ln in lines[1:]:
        parts = ln.split("\t", maxsplit=n_cols - 1)
        row_names.append(parts[0] if parts else "")

    # בלוק נומרי: מהעמודה השנייה עד הסוף, דילוג על שורת הכותרת
    usecols = tuple(range(1, n_cols))
    data = np.genfromtxt(
        StringIO(text),
        delimiter="\t",
        dtype=dtype,
        autostrip=True,
        comments=None,
        missing_values=("", "NA", "NaN", "null", "NULL"),
        filling_values=np.nan,
        skip_header=1,
        usecols=usecols,
    )

    if data.ndim == 1:
        data = data.reshape(1, -1)

    return data, row_names
def load_tsv_folder_as_subject_dict(folder: Union[str, Path],
                                    pattern: str,
                                    dtype: Union[type, str],
                                    encoding: str,
                                    has_header: Optional[bool],
                                    drop_first_row: bool,
                                    drop_first_col: bool,
                                    on_duplicate: str = "latest"
                                    ) -> Union[Dict[Key, np.ndarray], Dict[Key, List[np.ndarray]]]:
    """
    מחזיר Dict[(group, subject, session), ndarray] או רשימות אם 'collect'.
    כעת כולל מחיקה של שורות ועמודות לפי שמות שמופיעים ב-CSV (עמודה ראשונה).
    """
    # *** חובה: שניהם True כדי ששמות השורות יתיישרו נכון לנתונים ***
    if not (drop_first_row and drop_first_col):
        raise ValueError("When removing rows/cols by names, please set IGNORE_FIRST_ROW=True and IGNORE_FIRST_COL=True.")

    names_to_remove = _load_names_to_remove(MISSING_ROWS_CSV)

    folder = Path(folder)
    files: List[Path] = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {folder}")

    if on_duplicate not in {"error", "latest", "collect"}:
        raise ValueError("on_duplicate must be one of: 'error', 'latest', 'collect'")

    if on_duplicate == "collect":
        subject_map: Dict[Key, List[np.ndarray]] = {}
    else:
        subject_map: Dict[Key, np.ndarray] = {}

    for f in files:
        key = parse_subject_key3(f.name)

        # קורא בלוק נומרי ושמות שורות מיושרים (כבר דילגנו על שורה/עמודה ראשונות)
        data, row_names = _read_tsv_numeric_and_row_names(f, dtype=dtype, encoding=encoding)

        # מחיקה לפי שמות (שני הצירים, כדי לשמור מטריצה ריבועית)
        if names_to_remove:
            idx_to_remove = [i for i, nm in enumerate(row_names) if nm in names_to_remove]
            if idx_to_remove:
                data = np.delete(data, idx_to_remove, axis=0)
                data = np.delete(data, idx_to_remove, axis=1)

        # הזרימה הרגילה של on_duplicate
        if on_duplicate == "collect":
            subject_map.setdefault(key, []).append(data)
        else:
            if key in subject_map and on_duplicate == "error":
                raise ValueError(f"Duplicate subject for key {key} at file {f.name}")
            subject_map[key] = data

    return subject_map

def save_array_csv(arr: np.ndarray, base_path: Union[str, Path], encoding: str = "utf-8") -> List[Path]:
    """
    Save arr to CSV(s).
    - 1D/2D -> one CSV at base_path
    - 3D    -> one CSV per slice along axis=0: base_path.stem + f"_{i}.csv"
    Returns list of written paths.
    """
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    if arr.ndim <= 2:
        with open(base_path, "w", encoding=encoding, newline="") as f:
            np.savetxt(f, arr, delimiter=",", fmt=_fmt_for_dtype(arr.dtype))
        written.append(base_path)
    elif arr.ndim == 3:
        for i in range(arr.shape[0]):
            p = base_path.with_name(base_path.stem + f"_{i}" + base_path.suffix)
            with open(p, "w", encoding=encoding, newline="") as f:
                np.savetxt(f, arr[i], delimiter=",", fmt=_fmt_for_dtype(arr.dtype))
            written.append(p)
    else:
        # Fallback: flatten everything to 2D (first dim kept)
        flat = arr.reshape(arr.shape[0], -1)
        with open(base_path, "w", encoding=encoding, newline="") as f:
            np.savetxt(f, flat, delimiter=",", fmt=_fmt_for_dtype(arr.dtype))
        written.append(base_path)
    return written




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
        drop_first_col=IGNORE_FIRST_COL,
        on_duplicate=ON_DUPLICATE
    )

    print(f"\nLoaded {len(subject_arrays)} (group, subject, session) entries from TSV files.\n")
    # וידוא שהמטריצות עדיין ריבועיות ושוות בגודל
    shapes = {k: v.shape for k, v in subject_arrays.items() if not isinstance(v, list)}
    unique_shapes = set(shapes.values())
    print("Unique shapes after removal:", unique_shapes)

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
        print(f"Example key {example_key} not found in index map.")

    correlations = all_arrays
    window_length = 140
    # Scan for NaNs/Infs and report
    nan_inf_report = report_nan_inf_in_subjects(subject_arrays)

    diffusion_map, distances = get_diffusion_embedding(
        correlations, window_length, scale_k=140, signal=None, subsampling=0, mode='riemannian'
    )

    # Save diffusion outputs
    out_dir = Path(FOLDER) / OUT_SUBFOLDER
    dm_csv_paths = save_array_csv(diffusion_map, out_dir / "diffusion_map.csv", ENCODING)
    dist_csv_paths = save_array_csv(distances, out_dir / "diffusion_distances.csv", ENCODING)

    print("Saved diffusion_map CSV to:")
    for p in dm_csv_paths:
        print("  -", p.resolve())
    print("Saved diffusion distances CSV to:")
    for p in dist_csv_paths:
        print("  -", p.resolve())


if __name__ == "__main__":
    main()
