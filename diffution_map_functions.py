from pathlib import Path
from statistics import correlation
from typing import Dict, List, Optional, Tuple, Union
import re
import numpy as np
from sympy import false
import torch
from correlation_calculation_NOGA import corrcoef_safe
import csv
from io import StringIO
import matplotlib.pyplot as plt
Key = Tuple[str, int, int]  # ('NT', 137, 1) לדוגמה

def plot_eigs_scree(
    eigs,
    title="Session",
    marker="o",
    k=None,
    figsize=(7, 4.5),
    save_path=None,
):
    """Plot ONLY the eigenvalues (scree plot) for a single session."""
    def _prep(vals):
        arr = np.asarray(vals).ravel()
        arr = np.real(arr)
        arr = arr[np.isfinite(arr)]
        arr = arr[arr >= 0]
        arr = np.sort(arr)[::-1]
        return arr

    e = _prep(eigs)
    if e.size == 0:
        raise ValueError("No valid eigenvalues to plot after cleaning.")


    y =  e
    if k is None:
        idx = np.arange(1, len(e) + 1)
    else:
        idx = np.arange(1, k + 1)


    plt.figure(figsize=figsize)
    plt.plot(idx, y[:k], marker=marker)
    plt.xlabel("Eigenvalue index")
    plt.ylabel( "Eigenvalue")
    plt.title(f"{title}: Scree")
    plt.grid(True, alpha=0.3)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.show()

def plot_session1_scree(eigs, **kwargs):
    plot_eigs_scree(eigs, title="Session 1", **kwargs)

def plot_session2_scree(eigs, **kwargs):
    plot_eigs_scree(eigs, title="Session 2", **kwargs)



def _prep_eigs(eigs):
    e = np.asarray(eigs).ravel().astype(float)
    e = np.real(e)
    e = e[np.isfinite(e)]
    e = e[e >= 0]
    return np.sort(e)[::-1]

def elbow_max_distance(eigs):
    e = _prep_eigs(eigs)
    x = np.arange(1, len(e) + 1).astype(float)
    # distance of each point to the line from (x1,e1) to (xn,en)
    x1, y1 = x[0], e[0]
    x2, y2 = x[-1], e[-1]
    denom = np.hypot(x2 - x1, y2 - y1)
    if denom == 0:
        return 1, e[0]
    # vectorized perpendicular distance
    dist = np.abs((y2 - y1)*x - (x2 - x1)*e + x2*y1 - y2*x1) / denom
    k = int(np.argmax(dist)) + 1
    return k, e[k-1]

def elbow_flat_slope(eigs, window=5, rel_slope=0.10):
    """
    window: MA window for smoothing the drop
    rel_slope: declare 'flat' when MA drop < rel_slope * initial MA drop
    """
    e = _prep_eigs(eigs)
    if len(e) < 2:
        return 1, e[0] if len(e) else (1, np.nan)
    drops = -np.diff(e)  # positive decreases
    if len(drops) < window:
        window = max(1, len(drops))
    ma = np.convolve(drops, np.ones(window)/window, mode='valid')
    init = ma[0]
    thresh = rel_slope * init if init > 0 else 0
    below = np.where(ma < thresh)[0]
    if len(below) == 0:
        k = len(e)  # never got flat; keep all
    else:
        k = int(below[0] + window)  # convert MA index -> eigen index
    return k, e[k-1]


def _regularize_matrix_spd(C: np.ndarray) -> np.ndarray:
    """
    פונקציית עזר לביצוע רגולריזציית SPD טהורה על מטריצת קורלציה קיימת.
    """
    # 1. ודא סימטריה (למניעת שגיאות נומריות)
    C_sym = (C + C.T) / 2

    # 2. רגולריזציית SPD: דחיפת הערכים העצמיים למינימום חיובי
    w, V = np.linalg.eigh(C_sym)
    w = np.maximum(w, 1e-6)  # floor the eigenvalues (epsilon=1e-6)
    C_safe = (V * w) @ V.T  # reconstruct

    return C_safe


# --- סוף פונקציות ה-SPD שהושתלו ---

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


def build_stacked_array_and_index(
        subject_map: Union[Dict[Key, np.ndarray], Dict[Key, List[np.ndarray]]],
        order: str = "key"
) -> Tuple[np.ndarray, Dict[Key, Union[int, List[int]]], List[Key]]:
    """
    יוצר מערך מאוחד בכל הצירוף של המטריצות + מילון מיפוי key->index.
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


def report_nan_inf_in_subjects(
        subject_map: Union[Dict[Key, np.ndarray], Dict[Key, List[np.ndarray]]]
) -> Dict[Key, Union[Tuple[int, int, int], List[Tuple[int, int, int]]]]:
    """
    For each key, count (#NaN, #Inf, total elements) in its array(s).
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
                print(
                    f"  - {key}: NaN={n_nan_tot} ({pct_nan:.4f}%), Inf={n_inf_tot} ({pct_inf:.4f}%), arrays={len(counts_list)}")
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


def get_fully_missing_row_names_from_tsv(path: Path, encoding: str) -> List[str]:
    """
    קורא את הקובץ כטקסט, לוקח את שמות השורות מהעמודה הראשונה,
    ובודק על בלוק הנתונים (ללא השורה/עמודה הראשונות) אילו שורות הן all-NaN.
    """
    with open(path, "r", encoding=encoding) as f:
        text = f.read()

    lines = text.splitlines()
    if not lines:
        return []

    # כמה עמודות יש בשורה הראשונה
    first_line = lines[0].split("\t")
    n_cols = len(first_line)

    # שמות השורות
    row_names = []
    for ln in lines[1:]:
        parts = ln.split("\t", maxsplit=n_cols - 1)
        row_names.append(parts[0] if parts else "")

    # בלוק נומרי
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

    # יישור ל-2D
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
    מדפיס ומחזיר מיפוי filename -> רשימת שמות השורות שהן all-NaN.
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
    יוצר מיפוי: row_name -> [keys של נבדקים שחסרה בהם השורה הזו].
    """
    folder = Path(folder)
    files = sorted(folder.glob(pattern))
    agg: Dict[str, List[Key]] = {}

    for f in files:
        # מי זה הנבדק?
        try:
            key = parse_subject_key3(f.name)
        except Exception:
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
    כותב CSV עם עמודות: row_name, missing_count, subjects
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
    קורא את ה-CSV ומחזיר סט של שמות למחיקה.
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
      - row_names: שמות השורות שנלקחו מהעמודה הראשונה
    """
    text = Path(path).read_text(encoding=encoding)
    lines = text.splitlines()
    if not lines:
        return np.empty((0, 0), dtype=float if dtype is float else dtype), []

    # כמה עמודות בשורה הראשונה
    first_line = lines[0].split("\t")
    n_cols = len(first_line)

    # שמות השורות
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


def load_tsv_folder_as_subject_dict(
        folder: Union[str, Path],
        pattern: str,
        dtype: Union[type, str],
        encoding: str,
        has_header: Optional[bool],
        drop_first_row: bool,
        drop_first_col: bool,
        on_duplicate: str = "latest",
        MISSING_ROWS_CSV: Optional[Path] = None
    ) -> Union[Dict[Key, np.ndarray], Dict[Key, List[np.ndarray]]]:
    """
    Load all TSV matrices and remove globally-missing rows/columns
    based on names appearing in MISSING_ROWS_CSV.
    """

    # Required for consistent alignment
    if not (drop_first_row and drop_first_col):
        raise ValueError(
            "When removing rows/cols by names, please set "
            "IGNORE_FIRST_ROW=True and IGNORE_FIRST_COL=True."
        )

    # === NEW: remove any row missing in ANY subject ===
    names_to_remove = set()

    if MISSING_ROWS_CSV is not None:
        import pandas as pd
        df = pd.read_csv(MISSING_ROWS_CSV)

        # CSV columns MUST include: RowName, MissingInSubjects
        for row, subjects in zip(df["row_name"], df["subjects"]):
            # If this row is missing in at least one subject → remove it globally
            if isinstance(subjects, str) and subjects.strip() != "":
                names_to_remove.add(row)

    print(f"Removing {len(names_to_remove)} rows/columns missing in at least one subject.")
    print(f"Removing {len(names_to_remove)} globally missing rows/columns.")

    folder = Path(folder)
    files: List[Path] = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {folder}")

    # Validate on_duplicate
    if on_duplicate not in {"error", "latest", "collect"}:
        raise ValueError("on_duplicate must be 'error', 'latest', or 'collect'")

    # Initialize map
    subject_map = {} if on_duplicate != "collect" else {}

    for f in files:
        key = parse_subject_key3(f.name)

        # Load numeric array and row-names (aligned)
        data, row_names = _read_tsv_numeric_and_row_names(
            f, dtype=dtype, encoding=encoding
        )

        # ----- GLOBAL row/column removal -----
        if names_to_remove:
            idx_to_remove = [
                i for i, nm in enumerate(row_names) if nm in names_to_remove
            ]
            if idx_to_remove:
                data = np.delete(data, idx_to_remove, axis=0)
                data = np.delete(data, idx_to_remove, axis=1)

        # Handle duplicates
        if on_duplicate == "collect":
            subject_map.setdefault(key, []).append(data)

        else:
            if key in subject_map and on_duplicate == "error":
                raise ValueError(f"Duplicate subject key {key}: file {f.name}")

            subject_map[key] = data

    return subject_map



def save_array_csv(arr: np.ndarray, base_path: Union[str, Path], encoding: str = "utf-8") -> List[Path]:
    """
    Save arr to CSV(s).
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


def _key_to_diffusion_name(key: Key) -> str:
    """
    ממיר מפתח נבדק (group, number, session) לפורמט השם הנדרש.
    """
    grp, num, ses = key
    return f"{grp}{num:03d}_ses{ses}"


def save_diffusion_output_with_labels(
        arr: np.ndarray,
        keys_for_index: List[Key],
        is_distance_matrix: bool,
        base_path: Union[str, Path],
        encoding: str = "utf-8"
) -> Path:
    """
    מוסיף תוויות שורה/עמודה למערך הפלט ושומר ל-CSV.
    """
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. יישור צורה ל-2D
    if arr.ndim == 3 and arr.shape[0] == 1:
        data = arr[0]
    elif arr.ndim == 2:
        data = arr
    else:
        raise ValueError(f"Array shape {arr.shape} not supported for labeling.")

    # 2. יצירת תוויות נבדקים
    subject_labels = [_key_to_diffusion_name(k) for k in keys_for_index]

    if is_distance_matrix:
        # מטריצת מרחק (K x K): כותרות העמודות ואינדקס השורות הם שמות הנבדקים
        data_to_save = data
        col_labels = subject_labels
    else:
        # מפת דיפוזיה: מעבר מ-(D, K) ל-(K, D) [דגימות x פיצ'רים]
        if data.shape[0] < data.shape[1]:
            data_to_save = data.T  # הופך ל-K x D (דגימות x פיצ'רים)
        else:
            data_to_save = data  # מניח שזה כבר K x D

        # יצירת תוויות עמודות: DC1, DC2, ...
        n_features = data_to_save.shape[1]
        col_labels = [f"DC{i + 1}" for i in range(n_features)]

    # 3. שמירה ל-CSV

    # שורת הכותרת
    header = [""] + col_labels

    with open(base_path, "w", encoding=encoding, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)  # כתיבת כותרות

        # כתיבת כל שורה: מתחילה עם שם הנבדק, ואחריה הנתונים המספריים
        for i, row in enumerate(data_to_save):
            row_to_write = [subject_labels[i]] + row.tolist()
            writer.writerow(row_to_write)

    return base_path