def extract_and_normalize_t1_data(
    columns_to_extract,
    output_filename,
    t1_file_path=r"T1\stats_t1_fs\ses1\ses1_T1_data.xlsx",
    transitions_file_path=r"only_Q_outputs/combined/timepoints_file_inverted_2_with_meta_and_transitions.csv",
    out_dir=r"only_Q_outputs/combined",
    t1_subject_col="subject_code",
    transitions_subject_col="Subject_Code",
    after_col="after",
    etiv_col="etiv",
    verbose=True
):
    """
    Extract T1 data from Excel, merge with transitions data, normalize by eTIV, and save.
    
    Parameters:
    -----------
    columns_to_extract : list of tuples
        List of (column_name, sheet_name) tuples to extract from Excel
    output_filename : str
        Name of the output CSV file (will be saved in out_dir)
    t1_file_path : str, optional
        Path to the T1 Excel file (default: r"T1\stats_t1_fs\ses1\ses1_T1_data.xlsx")
    transitions_file_path : str, optional
        Path to the transitions CSV file (default: r"only_Q_outputs/combined/timepoints_file_inverted_2_with_meta_and_transitions.csv")
    out_dir : str, optional
        Output directory for the final CSV file (default: r"only_Q_outputs/combined")
    t1_subject_col : str, optional
        Subject identifier column name in T1 data (default: "subject_code")
    transitions_subject_col : str, optional
        Subject identifier column name in transitions data (default: "Subject_Code")
    after_col : str, optional
        Name of the 'after' column in transitions data (default: "after")
    etiv_col : str, optional
        Name of the eTIV column for normalization (default: "etiv")
    verbose : bool, optional
        Whether to print progress messages (default: True)
    
    Returns:
    --------
    pd.DataFrame
        The final normalized dataframe with no NaN values
    """
    import pandas as pd
    import os
    import numpy as np
    from collections import defaultdict
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    output_file_path = os.path.join(out_dir, output_filename)
    
    # ----------------
    # Step 1: Load transitions file (to get 'after')
    # ----------------
    if verbose:
        print(f"Loading transitions file: {transitions_file_path}")
    transitions_df = pd.read_csv(transitions_file_path)
    
    if transitions_subject_col not in transitions_df.columns:
        raise ValueError(f"'{transitions_subject_col}' not found. Columns: {list(transitions_df.columns)}")
    if after_col not in transitions_df.columns:
        raise ValueError(f"'{after_col}' not found. Columns: {list(transitions_df.columns)}")
    
    transitions_df[transitions_subject_col] = transitions_df[transitions_subject_col].astype(str)
    transitions_df[after_col] = pd.to_numeric(transitions_df[after_col], errors="coerce")
    
    after_column_data = transitions_df[[transitions_subject_col, after_col]].copy()
    if verbose:
        print(f"Transitions subjects: {after_column_data[transitions_subject_col].nunique()}")
    
    # ----------------
    # Step 2: Group columns by sheet and extract from Excel
    # ----------------
    if verbose:
        print(f"\nProcessing {len(columns_to_extract)} column(s) to extract...")
    columns_by_sheet = defaultdict(list)
    for col_name, sheet_name in columns_to_extract:
        columns_by_sheet[sheet_name].append(col_name)
    
    if verbose:
        print(f"Columns grouped into {len(columns_by_sheet)} sheet(s)")
    
    all_data = {}  # {column_name: Series indexed by subject_code}
    
    for sheet_name, col_names in columns_by_sheet.items():
        if verbose:
            print(f"\n{'='*70}")
            print(f"Processing sheet: {sheet_name}")
            print(f"Columns requested: {col_names}")
            print(f"{'='*70}")
        
        try:
            t1_df = pd.read_excel(t1_file_path, sheet_name=sheet_name)
        except Exception as e:
            if verbose:
                print(f"Error loading sheet '{sheet_name}': {e}")
            continue
        
        if t1_subject_col not in t1_df.columns:
            if verbose:
                print(f"Warning: '{t1_subject_col}' not found in sheet '{sheet_name}'. Skipping.")
                print(f"Available columns sample: {list(t1_df.columns)[:25]}")
            continue
        
        t1_df[t1_subject_col] = t1_df[t1_subject_col].astype(str)
        
        existing_cols = [c for c in col_names if c in t1_df.columns]
        missing_cols = [c for c in col_names if c not in t1_df.columns]
        
        if missing_cols and verbose:
            print(f"Warning: Missing columns in '{sheet_name}': {missing_cols}")
        if not existing_cols:
            if verbose:
                print(f"Error: None of requested columns found in '{sheet_name}'. Skipping.")
            continue
        
        if verbose:
            print(f"Found {len(t1_df)} rows in this sheet")
        
        for col in existing_cols:
            t1_df[col] = pd.to_numeric(t1_df[col], errors="coerce")
            all_data[col] = t1_df.set_index(t1_subject_col)[col]
            if verbose:
                print(f"  Extracted '{col}': {all_data[col].notna().sum()} non-null values")
    
    if not all_data:
        raise RuntimeError("No data extracted from Excel. Check sheet/column names.")
    
    # ----------------
    # Step 3: Build merged dataframe in-memory
    # ----------------
    if verbose:
        print(f"\n{'='*70}")
        print("Creating merged dataframe (in-memory)...")
        print(f"{'='*70}")
    
    all_subjects = set()
    for s in all_data.values():
        all_subjects.update(s.index.dropna())
    all_subjects = sorted(all_subjects)
    
    df = pd.DataFrame({t1_subject_col: all_subjects})
    if verbose:
        print(f"Unique subjects in extracted T1 data: {len(df)}")
    
    for col_name, col_series in all_data.items():
        col_df = col_series.reset_index()
        col_df.columns = [t1_subject_col, col_name]
        df = df.merge(col_df, on=t1_subject_col, how="left")
    
    df = df.merge(
        after_column_data,
        left_on=t1_subject_col,
        right_on=transitions_subject_col,
        how="left"
    )
    
    if transitions_subject_col in df.columns and transitions_subject_col != t1_subject_col:
        df = df.drop(columns=[transitions_subject_col])
    
    if verbose:
        print(f"After merge: rows={len(df)}, cols={len(df.columns)}")
        print("Columns:", list(df.columns))
    
    # ----------------
    # Step 4: Normalize by eTIV and then REMOVE all rows with any NaN
    # ----------------
    if verbose:
        print(f"\n{'='*70}")
        print("Normalizing by eTIV and removing subjects with NaN...")
        print(f"{'='*70}")
    
    if etiv_col not in df.columns:
        raise ValueError(f"'{etiv_col}' missing after merge. Cannot normalize.")
    
    df[etiv_col] = pd.to_numeric(df[etiv_col], errors="coerce")
    
    # Columns to normalize = everything except subject + after + etiv
    exclude = {t1_subject_col, after_col, etiv_col}
    cols_to_norm = [c for c in df.columns if c not in exclude]
    
    # ensure numeric
    for c in cols_to_norm:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    
    # normalize
    for c in cols_to_norm:
        df[c] = df[c] / df[etiv_col]
    
    # drop etiv
    df = df.drop(columns=[etiv_col])
    
    # reorder columns
    if after_col in df.columns:
        other_cols = [c for c in df.columns if c not in [t1_subject_col, after_col]]
        df = df[[t1_subject_col, after_col] + other_cols]
    
    # replace inf with NaN (in case etiv was 0)
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # DROP all subjects that have ANY NaN in ANY column
    n_before = len(df)
    df = df.dropna()
    n_after = len(df)
    
    if verbose:
        print(f"Subjects before dropna: {n_before}")
        print(f"Subjects after dropna:  {n_after}")
        print(f"Removed subjects:       {n_before - n_after} ({(n_before - n_after)/max(n_before,1)*100:.1f}%)")
        print("Final columns:", list(df.columns))
        print(df.head(10))
    
    # ----------------
    # Step 5: Save the final file
    # ----------------
    df.to_csv(output_file_path, index=False)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Final normalized NO-NaN file saved: {output_file_path}")
        print(f"Total subjects: {len(df)}")
        print(f"Total columns: {len(df.columns)}")
        print(f"\nSummary:")
        print(df.describe(include="all"))
    
    return df
##############################################
def add_columns_from_excel(
    csv_file_path,
    excel_file_path,
    columns_to_add,
    subject_col_csv="subject_code",
    subject_col_excel=None,
    output_suffix="_with_extra_columns",
    verbose=True
):
    """
    Add columns from an Excel file to an existing CSV file.
    
    Parameters:
    -----------
    csv_file_path : str
        Path to the existing CSV file to update
    excel_file_path : str
        Path to the Excel file containing columns to add
    columns_to_add : list of str
        List of column names to add from the Excel file
    subject_col_csv : str, optional
        Subject identifier column name in CSV file (default: "subject_code")
    subject_col_excel : str, optional
        Subject identifier column name in Excel file. If None, will try to auto-detect (default: None)
    output_suffix : str, optional
        Suffix to add to output filename (default: "_with_extra_columns")
    verbose : bool, optional
        Whether to print progress messages (default: True)
    
    Returns:
    --------
    pd.DataFrame
        The merged dataframe
    """
    import pandas as pd
    import os
    
    # ----------------
    # Load CSV file
    # ----------------
    if verbose:
        print(f"\n{'='*70}")
        print(f"Processing: {os.path.basename(csv_file_path)}")
        print(f"{'='*70}")
    
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
    
    df_csv = pd.read_csv(csv_file_path)
    if verbose:
        print(f"Loaded CSV: {len(df_csv)} rows, {len(df_csv.columns)} columns")
        print(f"CSV columns: {list(df_csv.columns)}")
    
    if subject_col_csv not in df_csv.columns:
        raise ValueError(f"Subject column '{subject_col_csv}' not found in CSV. Available: {list(df_csv.columns)}")
    
    # ----------------
    # Load Excel file
    # ----------------
    if not os.path.exists(excel_file_path):
        raise FileNotFoundError(f"Excel file not found: {excel_file_path}")
    
    df_excel = pd.read_excel(excel_file_path, sheet_name=0)
    if verbose:
        print(f"\nLoaded Excel: {len(df_excel)} rows, {len(df_excel.columns)} columns")
        print(f"Excel columns (first 30): {list(df_excel.columns)[:30]}")
    
    # ----------------
    # Auto-detect subject column in Excel if not provided
    # ----------------
    if subject_col_excel is None:
        # Try common subject column names
        possible_names = ['Subject_Code', 'subject_code', 'Subject_Code', 'subject_id', 'Subject_ID', 
                         'subject', 'Subject', 'ID', 'id']
        for name in possible_names:
            if name in df_excel.columns:
                subject_col_excel = name
                if verbose:
                    print(f"Auto-detected subject column in Excel: '{subject_col_excel}'")
                break
        
        if subject_col_excel is None:
            raise ValueError(f"Could not auto-detect subject column. Available columns: {list(df_excel.columns)[:20]}")
    else:
        if subject_col_excel not in df_excel.columns:
            raise ValueError(f"Subject column '{subject_col_excel}' not found in Excel. Available: {list(df_excel.columns)[:20]}")
    
    # ----------------
    # Check which columns exist in Excel
    # ----------------
    existing_columns = [col for col in columns_to_add if col in df_excel.columns]
    missing_columns = [col for col in columns_to_add if col not in df_excel.columns]
    
    if missing_columns and verbose:
        print(f"\nWarning: These columns not found in Excel: {missing_columns}")
    
    if not existing_columns:
        raise ValueError(f"None of the requested columns found in Excel. Available columns: {list(df_excel.columns)[:30]}")
    
    if verbose:
        print(f"\nColumns to add: {existing_columns}")
    
    # ----------------
    # Prepare Excel data for merging
    # ----------------
    # Convert subject columns to string for matching
    df_excel[subject_col_excel] = df_excel[subject_col_excel].astype(str)
    df_csv[subject_col_csv] = df_csv[subject_col_csv].astype(str)
    
    # Select only subject column and columns to add
    excel_subset = df_excel[[subject_col_excel] + existing_columns].copy()
    
    # ----------------
    # Merge
    # ----------------
    n_before = len(df_csv)
    df_merged = df_csv.merge(
        excel_subset,
        left_on=subject_col_csv,
        right_on=subject_col_excel,
        how='left'
    )
    
    # Drop the duplicate subject column from Excel if it has a different name
    if subject_col_excel != subject_col_csv and subject_col_excel in df_merged.columns:
        df_merged = df_merged.drop(columns=[subject_col_excel])
    
    n_after = len(df_merged)
    
    if verbose:
        print(f"\nMerge results:")
        print(f"  Rows before merge: {n_before}")
        print(f"  Rows after merge: {n_after}")
        print(f"  Columns before: {len(df_csv.columns)}")
        print(f"  Columns after: {len(df_merged.columns)}")
        
        # Check how many subjects got matched
        matched = df_merged[existing_columns[0]].notna().sum() if existing_columns else 0
        print(f"  Subjects with matched data: {matched} / {n_before} ({100*matched/n_before:.1f}%)")
    
    # ----------------
    # Remove subjects with NaN in the newly added columns
    # ----------------
    n_before_dropna = len(df_merged)
    df_merged = df_merged.dropna(subset=existing_columns)
    n_after_dropna = len(df_merged)
    n_removed = n_before_dropna - n_after_dropna
    
    if verbose:
        print(f"\nRemoving subjects with NaN in added columns:")
        print(f"  Subjects before dropna: {n_before_dropna}")
        print(f"  Subjects after dropna:  {n_after_dropna}")
        print(f"  Removed subjects:       {n_removed} ({100*n_removed/max(n_before_dropna,1):.1f}%)")
    
    # ----------------
    # Save updated file
    # ----------------
    base_name = os.path.splitext(csv_file_path)[0]
    extension = os.path.splitext(csv_file_path)[1]
    output_path = f"{base_name}{output_suffix}{extension}"
    
    df_merged.to_csv(output_path, index=False)
    if verbose:
        print(f"\nSaved updated file: {output_path}")
    
    return df_merged
#############################################

def add_columns_from_excel(
    csv_file_path,
    excel_file_path,
    columns_to_add,
    subject_col_csv="subject_code",
    subject_col_excel=None,
    output_suffix="_with_extra_columns",
    verbose=True
):
    """
    Add columns from an Excel file to an existing CSV file.
    
    Parameters:
    -----------
    csv_file_path : str
        Path to the existing CSV file to update
    excel_file_path : str
        Path to the Excel file containing columns to add
    columns_to_add : list of str
        List of column names to add from the Excel file
    subject_col_csv : str, optional
        Subject identifier column name in CSV file (default: "subject_code")
    subject_col_excel : str, optional
        Subject identifier column name in Excel file. If None, will try to auto-detect (default: None)
    output_suffix : str, optional
        Suffix to add to output filename (default: "_with_extra_columns")
    verbose : bool, optional
        Whether to print progress messages (default: True)
    
    Returns:
    --------
    pd.DataFrame
        The merged dataframe
    """
    import pandas as pd
    import os
    
    # ----------------
    # Load CSV file
    # ----------------
    if verbose:
        print(f"\n{'='*70}")
        print(f"Processing: {os.path.basename(csv_file_path)}")
        print(f"{'='*70}")
    
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
    
    df_csv = pd.read_csv(csv_file_path)
    if verbose:
        print(f"Loaded CSV: {len(df_csv)} rows, {len(df_csv.columns)} columns")
        print(f"CSV columns: {list(df_csv.columns)}")
    
    if subject_col_csv not in df_csv.columns:
        raise ValueError(f"Subject column '{subject_col_csv}' not found in CSV. Available: {list(df_csv.columns)}")
    
    # ----------------
    # Load Excel file
    # ----------------
    if not os.path.exists(excel_file_path):
        raise FileNotFoundError(f"Excel file not found: {excel_file_path}")
    
    df_excel = pd.read_excel(excel_file_path, sheet_name=0)
    if verbose:
        print(f"\nLoaded Excel: {len(df_excel)} rows, {len(df_excel.columns)} columns")
        print(f"Excel columns (first 30): {list(df_excel.columns)[:30]}")
    
    # ----------------
    # Auto-detect subject column in Excel if not provided
    # ----------------
    if subject_col_excel is None:
        # Try common subject column names
        possible_names = ['Subject_Code', 'subject_code', 'Subject_Code', 'subject_id', 'Subject_ID', 
                         'subject', 'Subject', 'ID', 'id']
        for name in possible_names:
            if name in df_excel.columns:
                subject_col_excel = name
                if verbose:
                    print(f"Auto-detected subject column in Excel: '{subject_col_excel}'")
                break
        
        if subject_col_excel is None:
            raise ValueError(f"Could not auto-detect subject column. Available columns: {list(df_excel.columns)[:20]}")
    else:
        if subject_col_excel not in df_excel.columns:
            raise ValueError(f"Subject column '{subject_col_excel}' not found in Excel. Available: {list(df_excel.columns)[:20]}")
    
    # ----------------
    # Check which columns exist in Excel
    # ----------------
    existing_columns = [col for col in columns_to_add if col in df_excel.columns]
    missing_columns = [col for col in columns_to_add if col not in df_excel.columns]
    
    if missing_columns and verbose:
        print(f"\nWarning: These columns not found in Excel: {missing_columns}")
    
    if not existing_columns:
        raise ValueError(f"None of the requested columns found in Excel. Available columns: {list(df_excel.columns)[:30]}")
    
    if verbose:
        print(f"\nColumns to add: {existing_columns}")
    
    # ----------------
    # Prepare Excel data for merging
    # ----------------
    # Convert subject columns to string for matching
    df_excel[subject_col_excel] = df_excel[subject_col_excel].astype(str)
    df_csv[subject_col_csv] = df_csv[subject_col_csv].astype(str)
    
    # Select only subject column and columns to add
    excel_subset = df_excel[[subject_col_excel] + existing_columns].copy()
    
    # ----------------
    # Merge
    # ----------------
    n_before = len(df_csv)
    df_merged = df_csv.merge(
        excel_subset,
        left_on=subject_col_csv,
        right_on=subject_col_excel,
        how='left'
    )
    
    # Drop the duplicate subject column from Excel if it has a different name
    if subject_col_excel != subject_col_csv and subject_col_excel in df_merged.columns:
        df_merged = df_merged.drop(columns=[subject_col_excel])
    
    n_after = len(df_merged)
    
    if verbose:
        print(f"\nMerge results:")
        print(f"  Rows before merge: {n_before}")
        print(f"  Rows after merge: {n_after}")
        print(f"  Columns before: {len(df_csv.columns)}")
        print(f"  Columns after: {len(df_merged.columns)}")
        
        # Check how many subjects got matched
        matched = df_merged[existing_columns[0]].notna().sum() if existing_columns else 0
        print(f"  Subjects with matched data: {matched} / {n_before} ({100*matched/n_before:.1f}%)")
    
    # ----------------
    # Remove subjects with NaN in the newly added columns
    # ----------------
    n_before_dropna = len(df_merged)
    df_merged = df_merged.dropna(subset=existing_columns)
    n_after_dropna = len(df_merged)
    n_removed = n_before_dropna - n_after_dropna
    
    if verbose:
        print(f"\nRemoving subjects with NaN in added columns:")
        print(f"  Subjects before dropna: {n_before_dropna}")
        print(f"  Subjects after dropna:  {n_after_dropna}")
        print(f"  Removed subjects:       {n_removed} ({100*n_removed/max(n_before_dropna,1):.1f}%)")
    
    # ----------------
    # Save updated file
    # ----------------
    base_name = os.path.splitext(csv_file_path)[0]
    extension = os.path.splitext(csv_file_path)[1]
    output_path = f"{base_name}{output_suffix}{extension}"
    
    df_merged.to_csv(output_path, index=False)
    if verbose:
        print(f"\nSaved updated file: {output_path}")
    
    return df_merged

######################################################
def logistic_regression_cv(
    input_file_path,
    subject_col="subject_code",
    target_col="after",
    n_folds=5,
    cv_random_state=42,
    results_save_path=None,
    verbose=True,
    top_n_features=None,
    use_upsampling=True
):
    """
    Perform logistic regression with cross-validation on a dataset using upsampling (SMOTE).
    
    Parameters:
    -----------
    input_file_path : str
        Path to the input CSV file (normalized dataframe)
    subject_col : str, optional
        Subject identifier column name (will be excluded from features, default: "subject_code")
    target_col : str, optional
        Target column for binary classification (default: "after")
    n_folds : int, optional
        Number of CV folds (default: 5)
    cv_random_state : int, optional
        Random state for cross-validation (default: 42)
    results_save_path : str, optional
        Path to save the results CSV. If None, uses default path (default: None)
    verbose : bool, optional
        Whether to print progress messages (default: True)
    top_n_features : int, optional
        Number of top features to show in the visualization. If None, shows all features (default: None)
    use_upsampling : bool, optional
        Whether to apply SMOTE upsampling to training data. If False, uses original training data without upsampling (default: True)
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'results_df': DataFrame with results for each fold
        - 'display_df': DataFrame with fold and all metric columns (r2, accuracy, precision, recall, f1, roc_auc) plus mean row
        - 'mean_r2': Mean R² value
        - 'std_r2': Standard deviation of R²
        - 'mean_accuracy': Mean accuracy across folds
        - 'std_accuracy': Standard deviation of accuracy
        - 'mean_precision': Mean precision across folds
        - 'std_precision': Standard deviation of precision
        - 'mean_recall': Mean recall across folds
        - 'std_recall': Standard deviation of recall
        - 'mean_f1': Mean F1 score across folds
        - 'std_f1': Standard deviation of F1 score
        - 'mean_roc_auc': Mean ROC-AUC across folds
        - 'std_roc_auc': Standard deviation of ROC-AUC
        - 'scaler': The fitted StandardScaler object
        - 'model': The trained logistic regression model (from last fold)
        - 'feature_importance_df': DataFrame with feature coefficients and importance
        - 'coefficients_mean': Array of mean coefficients across folds
        - 'coefficients_std': Array of std of coefficients across folds
        - 'feature_names': List of feature names
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from imblearn.over_sampling import SMOTE
    import warnings
    warnings.filterwarnings('ignore')
    
    # ----------------
    # Load the dataframe
    # ----------------
    if verbose:
        print(f"Loading file: {input_file_path}")
    df = pd.read_csv(input_file_path)
    
    if verbose:
        print(f"Loaded {len(df)} subjects")
        print(f"Total columns: {len(df.columns)}")
        print(f"Columns: {list(df.columns)}")
    
    # ----------------
    # Check required columns
    # ----------------
    if subject_col not in df.columns:
        raise ValueError(f"'{subject_col}' column not found. Available columns: {list(df.columns)}")
    
    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' column not found. Available columns: {list(df.columns)}")
    
    # Convert target to numeric
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    
    # Check target distribution
    target_counts = df[target_col].value_counts().sort_index()
    if verbose:
        print(f"\nTarget distribution ({target_col}):")
        for val, count in target_counts.items():
            print(f"  {target_col} = {val}: {count} subjects ({100*count/len(df):.1f}%)")
    
    # ----------------
    # Prepare features and target
    # ----------------
    if verbose:
        print(f"\n{'='*70}")
        print("Preparing features and target...")
        print(f"{'='*70}")
    
    # Get feature columns (all columns except subject_code and target)
    feature_columns = [col for col in df.columns if col not in [subject_col, target_col]]
    
    if len(feature_columns) == 0:
        raise ValueError("No feature columns found! Check your column names.")
    
    if verbose:
        print(f"Feature columns ({len(feature_columns)}): {feature_columns}")
    
    # Extract features (X) and target (y)
    X = df[feature_columns].copy()
    y = df[target_col].copy()
    
    # Convert features to numeric
    for col in feature_columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    
    # Remove rows with any missing values
    valid_mask = X.notna().all(axis=1) & y.notna()
    X_clean = X[valid_mask].copy()
    y_clean = y[valid_mask].copy()
    
    if verbose:
        print(f"\nValid subjects (after removing missing): {len(X_clean)}")
        print(f"Removed {len(X) - len(X_clean)} subjects with missing values")
    
    # Check if we have both classes
    unique_classes = y_clean.unique()
    if len(unique_classes) < 2:
        raise ValueError(f"Target has only {len(unique_classes)} unique class(es). Need at least 2 for binary classification.")
    
    if verbose:
        print(f"Target classes: {sorted(unique_classes)}")
    
    # ----------------  
    # Prepare data (DO NOT SCALE YET - scaling will be done inside CV loop)
    # ----------------
    if verbose:
        print(f"\n{'='*70}")
        print("Preparing data for cross-validation...")
        print(f"{'='*70}")
    
    X_values = X_clean.values
    y_values = y_clean.values
    
    if verbose:
        print(f"Data prepared: {X_values.shape}")
    
    # ----------------  
    # Logistic Regression with Cross-Validation (optionally with Upsampling)
    # ----------------
    if verbose:
        print(f"\n{'='*70}")
        if use_upsampling:
            print("LOGISTIC REGRESSION: Binary Classification with SMOTE Upsampling")
        else:
            print("LOGISTIC REGRESSION: Binary Classification (no upsampling)")
        print(f"Target: {target_col}")
        print(f"Features: {len(feature_columns)} columns")
        print(f"Cross-validation: {n_folds}-fold stratified")
        print(f"Upsampling: {'Enabled (SMOTE)' if use_upsampling else 'Disabled'}")
        print(f"{'='*70}")
    
    # Use StratifiedKFold to ensure balanced class distribution in each fold
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cv_random_state)
    
    # Store results for each fold
    fold_results = []
    
    # Store coefficients from each fold for feature analysis
    all_coefficients = []
    
    # Store scaler from last fold for return value
    scaler = None
    
    # Manual cross-validation to calculate R² for each fold
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_values, y_values), 1):
        # Split data (BEFORE scaling to avoid data leakage)
        X_train_raw, X_test_raw = X_values[train_idx], X_values[test_idx]
        y_train, y_test = y_values[train_idx], y_values[test_idx]
        
        # Scale features - fit scaler ONLY on training data, then transform both
        scaler_fold = StandardScaler()
        X_train = scaler_fold.fit_transform(X_train_raw)
        X_test = scaler_fold.transform(X_test_raw)
        
        # Store scaler from last fold
        scaler = scaler_fold
        
        # Initialize logistic regression model for this fold
        # Use class_weight='balanced' if not using upsampling, otherwise no class_weight needed
        if use_upsampling:
            logistic_model = LogisticRegression(
                max_iter=1000,
                random_state=cv_random_state,
                solver='lbfgs'
            )
            # Initialize SMOTE for upsampling
            smote = SMOTE(random_state=cv_random_state)
        else:
            logistic_model = LogisticRegression(
                max_iter=1000,
                random_state=cv_random_state,
                solver='lbfgs',
                class_weight='balanced'  # Use class balancing instead of upsampling
            )
        
        # Apply upsampling (SMOTE) to training data only if requested
        if use_upsampling:
            X_train_final, y_train_final = smote.fit_resample(X_train, y_train)
            if verbose:
                print(f"\n--- Fold {fold_idx}/{n_folds} ---")
                print(f"  Training data before upsampling: {len(X_train)} samples")
                print(f"  Training data after upsampling: {len(X_train_final)} samples")
                unique, counts = np.unique(y_train_final, return_counts=True)
                class_dist = dict(zip(unique, counts))
                print(f"  Class distribution after upsampling: {class_dist}")
        else:
            X_train_final, y_train_final = X_train, y_train
            if verbose:
                print(f"\n--- Fold {fold_idx}/{n_folds} ---")
                print(f"  Training data: {len(X_train)} samples")
                unique, counts = np.unique(y_train, return_counts=True)
                class_dist = dict(zip(unique, counts))
                print(f"  Class distribution: {class_dist}")
        
        # Train model
        logistic_model.fit(X_train_final, y_train_final)
        
        # Store coefficients for this fold
        all_coefficients.append(logistic_model.coef_[0])
        
        # Predict probabilities and classes
        y_pred_proba = logistic_model.predict_proba(X_test)[:, 1]  # Probability of class 1
        y_pred = logistic_model.predict(X_test)
        
        # Calculate R² (pseudo R² for logistic regression)
        # Simple R² calculation: correlation-based pseudo R²
        # R² = (correlation between predicted probabilities and actual values)²
        if len(np.unique(y_test)) > 1:
            # Calculate pseudo R² using correlation
            correlation = np.corrcoef(y_pred_proba, y_test)[0, 1]
            r2_fold = correlation ** 2 if not np.isnan(correlation) else 0
        else:
            r2_fold = 0
        
        # Set negative R² to 0
        if r2_fold < 0:
            r2_fold = 0
        
        # Calculate other metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        
        # ROC-AUC (if both classes are present)
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except:
            roc_auc = np.nan
        
        fold_results.append({
            'fold': fold_idx,
            'r2': r2_fold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'n_train': len(y_train),
            'n_train_final': len(X_train_final),
            'n_test': len(y_test)
        })
    
    # ----------------
    # Calculate average metrics
    # ----------------
    if verbose:
        print(f"\n{'='*70}")
        print("CROSS-VALIDATION RESULTS SUMMARY")
        print(f"{'='*70}")
    
    results_df = pd.DataFrame(fold_results)
    
    # Calculate mean and std for all metrics
    mean_r2 = results_df['r2'].mean()
    std_r2 = results_df['r2'].std()
    mean_accuracy = results_df['accuracy'].mean()
    std_accuracy = results_df['accuracy'].std()
    mean_precision = results_df['precision'].mean()
    std_precision = results_df['precision'].std()
    mean_recall = results_df['recall'].mean()
    std_recall = results_df['recall'].std()
    mean_f1 = results_df['f1'].mean()
    std_f1 = results_df['f1'].std()
    mean_roc_auc = results_df['roc_auc'].mean()
    std_roc_auc = results_df['roc_auc'].std()
    
    if verbose:
        print(f"\nALL PARAMETERS EVALUATION (Mean across {n_folds} folds):")
        print(f"  R² (pseudo R²):           {mean_r2:.4f} (+/- {std_r2:.4f})")
        print(f"  Accuracy:                  {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")
        print(f"  Precision:                 {mean_precision:.4f} (+/- {std_precision:.4f})")
        print(f"  Recall:                    {mean_recall:.4f} (+/- {std_recall:.4f})")
        print(f"  F1 Score:                  {mean_f1:.4f} (+/- {std_f1:.4f})")
        print(f"  ROC-AUC:                   {mean_roc_auc:.4f} (+/- {std_roc_auc:.4f})")
    
    # ----------------
    # Display detailed results table
    # ----------------
    if verbose:
        print(f"\n{'='*70}")
        print("Detailed Results by Fold:")
        print(f"{'='*70}")
    
    # Create a copy of results_df with all metric columns, and add mean row
    metric_columns = ['fold', 'r2', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    display_df = results_df[metric_columns].copy()
    
    # Create mean row with all metrics
    mean_row = {
        'fold': 'Mean',
        'r2': mean_r2,
        'accuracy': mean_accuracy,
        'precision': mean_precision,
        'recall': mean_recall,
        'f1': mean_f1,
        'roc_auc': mean_roc_auc
    }
    
    # Append mean row to display dataframe
    display_df = pd.concat([display_df, pd.DataFrame([mean_row])], ignore_index=True)
    
    if verbose:
        print(display_df.to_string(index=False))
    
    # ----------------
    # Feature Importance Analysis
    # ----------------
    # Calculate average coefficients across all folds
    coefficients_array = np.array(all_coefficients)
    mean_coefficients = np.mean(coefficients_array, axis=0)
    std_coefficients = np.std(coefficients_array, axis=0)
    
    # Create feature importance dataframe
    feature_importance_df = pd.DataFrame({
        'feature': feature_columns,
        'coefficient_mean': mean_coefficients,
        'coefficient_std': std_coefficients,
        'abs_coefficient': np.abs(mean_coefficients)
    })
    
    # Sort by absolute coefficient value (most important first)
    feature_importance_df = feature_importance_df.sort_values('abs_coefficient', ascending=False).reset_index(drop=True)
    
    if verbose:
        print(f"\n{'='*70}")
        print("FEATURE IMPORTANCE (sorted by absolute coefficient)")
        print(f"{'='*70}")
        print(feature_importance_df.to_string(index=False))
    
    # ----------------
    # Create Feature Importance Visualization
    # ----------------
    # Select top N features if specified
    plot_df = feature_importance_df.copy()
    if top_n_features is not None and top_n_features > 0:
        plot_df = plot_df.head(top_n_features)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, max(6, len(plot_df) * 0.4)))
    
    # Color bars based on positive/negative coefficients
    colors = ['#2E86AB' if x >= 0 else '#A23B72' for x in plot_df['coefficient_mean']]
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(plot_df)), plot_df['coefficient_mean'], color=colors)
    
    # Add a vertical line at zero
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')
    
    # Set y-axis labels to feature names
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['feature'], fontsize=16)
    
    # Labels and title
    ax.set_xlabel('Coefficient Value', fontsize=18, fontweight='bold')
    ax.set_ylabel('Features', fontsize=18, fontweight='bold')
    ax.set_title(f'Feature Importance - Logistic Regression\nTarget: {target_col} (Top {len(plot_df)} features)', 
                 fontsize=20, fontweight='bold', pad=20)
    
    # Add grid for better readability
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        value = row['coefficient_mean']
        x_pos = value + (0.02 * max(abs(plot_df['coefficient_mean']))) if value >= 0 else value - (0.02 * max(abs(plot_df['coefficient_mean'])))
        ax.text(x_pos, i, f'{value:.4f}', va='center', 
                ha='left' if value >= 0 else 'right', fontsize=14)
    
    # Set x-axis limits with padding on both sides for better visibility
    x_min = plot_df['coefficient_mean'].min()
    x_max = plot_df['coefficient_mean'].max()
    x_range = x_max - x_min
    padding = max(0.15 * abs(x_range), 0.15 * max(abs(x_min), abs(x_max)))  # 15% padding or 15% of max absolute value
    ax.set_xlim(x_min - padding, x_max + padding)
    
    # Invert y-axis so highest importance is at top
    ax.invert_yaxis()

    
    plt.tight_layout()
    
    # Display the figure in the notebook
    plt.show()
    
    # ----------------
    # Save results to CSV
    # ----------------
    if results_save_path is None:
        results_save_path = r"only_Q_outputs/combined/logistic_regression_cv_results.csv"
    
    results_df.to_csv(results_save_path, index=False)
    if verbose:
        print(f"\n{'='*70}")
        print(f"Results saved to: {results_save_path}")
        print(f"{'='*70}")
    
    # Save feature importance to CSV
    if results_save_path:
        feature_importance_path = results_save_path.replace('.csv', '_feature_importance.csv')
        feature_importance_df.to_csv(feature_importance_path, index=False)
        if verbose:
            print(f"Feature importance saved to: {feature_importance_path}")
    
    return {
        'results_df': results_df,
        'display_df': display_df,
        'mean_r2': mean_r2,
        'std_r2': std_r2,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'mean_precision': mean_precision,
        'std_precision': std_precision,
        'mean_recall': mean_recall,
        'std_recall': std_recall,
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'mean_roc_auc': mean_roc_auc,
        'std_roc_auc': std_roc_auc,
        'scaler': scaler,
        'model': logistic_model,
        'feature_importance_df': feature_importance_df,
        'coefficients_mean': mean_coefficients,
        'coefficients_std': std_coefficients,
        'feature_names': feature_columns
    }


###############################################


def lasso_linear_regression_cv(
    input_file_path,
    subject_col="subject_code",
    target_col="after",
    n_folds=5,
    cv_random_state=42,
    results_save_path=None,
    verbose=True,
    top_n_features=None,
    exclude_columns=None,
    cv_alphas=100,
    n_jobs=None
):
    """
    Perform Lasso (L1-regularized) linear regression with cross-validation on a dataset.
    
    Parameters:
    -----------
    input_file_path : str
        Path to the input CSV file (normalized dataframe)
    subject_col : str, optional
        Subject identifier column name (will be excluded from features, default: "subject_code")
    target_col : str, optional
        Target column for regression (default: "after")
    n_folds : int, optional
        Number of CV folds (default: 5)
    cv_random_state : int, optional
        Random state for cross-validation (default: 42)
    results_save_path : str, optional
        Path to save the results CSV. If None, uses default path (default: None)
    verbose : bool, optional
        Whether to print progress messages (default: True)
    top_n_features : int, optional
        Number of top features to show in the visualization. If None, shows all features (default: None)
    exclude_columns : list, optional
        List of column names to exclude from features. These columns will not be used as predictors.
        By default, subject_col and target_col are always excluded. Default: None
    cv_alphas : int or array-like, optional
        Number of alphas along the regularization path (if int), or array of alphas to try.
        Used for internal cross-validation to find optimal alpha. Default: 100
    n_jobs : int or None, optional
        Number of CPUs to use during the cross-validation. None means 1. Default: None
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'results_df': DataFrame with results for each fold
        - 'display_df': DataFrame with fold and all metric columns (r2, mae, rmse) plus mean row
        - 'mean_r2': Mean R² value
        - 'std_r2': Standard deviation of R²
        - 'mean_mae': Mean MAE across folds
        - 'std_mae': Standard deviation of MAE
        - 'mean_rmse': Mean RMSE across folds
        - 'std_rmse': Standard deviation of RMSE
        - 'scaler': The fitted StandardScaler object
        - 'model': The trained lasso linear regression model (from last fold)
        - 'feature_importance_df': DataFrame with feature coefficients and importance
        - 'coefficients_mean': Array of mean coefficients across folds
        - 'coefficients_std': Array of std of coefficients across folds
        - 'feature_names': List of feature names
        - 'n_features_selected': Number of features with non-zero coefficients (mean across folds)
        - 'best_alpha': Best alpha value selected by LassoCV (mean across folds)
        - 'best_alpha_std': Standard deviation of best alpha across folds
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    import warnings
    warnings.filterwarnings('ignore')
    
    # ----------------
    # Load the dataframe
    # ----------------
    if verbose:
        print(f"Loading file: {input_file_path}")
    df = pd.read_csv(input_file_path)
    
    if verbose:
        print(f"Loaded {len(df)} subjects")
        print(f"Total columns: {len(df.columns)}")
        print(f"Columns: {list(df.columns)}")
    
    # ----------------
    # Check required columns
    # ----------------
    if subject_col not in df.columns:
        raise ValueError(f"'{subject_col}' column not found. Available columns: {list(df.columns)}")
    
    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' column not found. Available columns: {list(df.columns)}")
    
    # Convert target to numeric
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    
    # Check target distribution (for regression, show summary statistics)
    if verbose:
        print(f"\nTarget distribution ({target_col}):")
        print(f"  Mean: {df[target_col].mean():.4f}")
        print(f"  Std:  {df[target_col].std():.4f}")
        print(f"  Min:  {df[target_col].min():.4f}")
        print(f"  Max:  {df[target_col].max():.4f}")
    
    # ----------------
    # Prepare features and target
    # ----------------
    if verbose:
        print(f"\n{'='*70}")
        print("Preparing features and target...")
        print(f"{'='*70}")
    
    # Get feature columns (all columns except subject_code, target, and excluded columns)
    columns_to_exclude = [subject_col, target_col]
    if exclude_columns is not None:
        # Add exclude_columns to the exclusion list
        if isinstance(exclude_columns, str):
            exclude_columns = [exclude_columns]  # Convert single string to list
        columns_to_exclude.extend(exclude_columns)
    
    feature_columns = [col for col in df.columns if col not in columns_to_exclude]
    
    if len(feature_columns) == 0:
        raise ValueError("No feature columns found! Check your column names.")
    
    if verbose:
        print(f"Excluded columns: {columns_to_exclude}")
        print(f"Feature columns ({len(feature_columns)}): {feature_columns}")
    
    # Extract features (X) and target (y)
    X = df[feature_columns].copy()
    y = df[target_col].copy()
    
    # Convert features to numeric
    for col in feature_columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    
    # Remove rows with any missing values
    valid_mask = X.notna().all(axis=1) & y.notna()
    X_clean = X[valid_mask].copy()
    y_clean = y[valid_mask].copy()
    
    if verbose:
        print(f"\nValid subjects (after removing missing): {len(X_clean)}")
        print(f"Removed {len(X) - len(X_clean)} subjects with missing values")
    
    # Check if we have sufficient data for regression
    if len(y_clean) < n_folds:
        raise ValueError(f"Not enough samples ({len(y_clean)}) for {n_folds}-fold cross-validation.")
    
    if verbose:
        print(f"Valid samples: {len(y_clean)}")
    
    # ----------------
    # Standardize features
    # ----------------
    if verbose:
        print(f"\n{'='*70}")
        print("Standardizing features...")
        print(f"{'='*70}")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    y_values = y_clean.values
    
    if verbose:
        print(f"Features standardized: {X_scaled.shape}")
    
    # ----------------
    # Lasso Linear Regression with Cross-Validation (alpha auto-selected)
    # ----------------
    if verbose:
        print(f"\n{'='*70}")
        print("LASSO LINEAR REGRESSION: Regression Analysis (Alpha Auto-Selected)")
        print(f"Target: {target_col}")
        print(f"Features: {len(feature_columns)} columns")
        print(f"Outer cross-validation: {n_folds}-fold")
        print(f"LassoCV will automatically find optimal alpha using internal CV")
        print(f"Alpha candidates: {cv_alphas}")
        print(f"{'='*70}")
    
    # Initialize lasso linear regression model with automatic alpha selection
    # LassoCV performs internal cross-validation to find the best alpha
    lasso_model = LassoCV(
        alphas=cv_alphas,
        cv=5,  # Internal CV folds for alpha selection
        max_iter=1000,
        random_state=cv_random_state,
        n_jobs=n_jobs
    )
    
    # Use KFold for cross-validation (standard for regression)
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=cv_random_state)
    
    # Store results for each fold
    fold_results = []
    
    # Store coefficients from each fold for feature analysis
    all_coefficients = []
    
    # Store number of selected features (non-zero coefficients) for each fold
    n_features_selected_list = []
    
    # Store best alpha for each fold
    best_alphas_list = []
    
    # Manual cross-validation to calculate metrics for each fold
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_scaled, y_values), 1):
        # Split data
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_values[train_idx], y_values[test_idx]
        
        if verbose:
            print(f"\n--- Fold {fold_idx}/{n_folds} ---")
            print(f"  Training data: {len(X_train)} samples")
            print(f"  Test data: {len(X_test)} samples")
            print(f"  Target range: [{y_train.min():.4f}, {y_train.max():.4f}]")
        
        # Train model (LassoCV will find best alpha internally)
        lasso_model.fit(X_train, y_train)
        
        # Store best alpha selected for this fold
        best_alpha_fold = lasso_model.alpha_
        best_alphas_list.append(best_alpha_fold)
        
        # Store coefficients for this fold (LassoCV returns coef_ as 1D array)
        all_coefficients.append(lasso_model.coef_)
        
        # Count non-zero coefficients (selected features)
        n_selected = np.sum(np.abs(lasso_model.coef_) > 1e-6)  # Features with non-zero coefficients
        n_features_selected_list.append(n_selected)
        
        if verbose:
            print(f"  Best alpha selected: {best_alpha_fold:.6f}")
            print(f"  Features selected (non-zero coefficients): {n_selected} / {len(feature_columns)}")
        
        # Predict values
        y_pred = lasso_model.predict(X_test)
        
        # Calculate regression metrics
        r2_fold = r2_score(y_test, y_pred)
        mae_fold = mean_absolute_error(y_test, y_pred)
        rmse_fold = np.sqrt(mean_squared_error(y_test, y_pred))
        
        fold_results.append({
            'fold': fold_idx,
            'r2': r2_fold,
            'mae': mae_fold,
            'rmse': rmse_fold,
            'n_train': len(y_train),
            'n_test': len(y_test),
            'n_features_selected': n_selected,
            'best_alpha': best_alpha_fold
        })
    
    # ----------------
    # Calculate average metrics
    # ----------------
    if verbose:
        print(f"\n{'='*70}")
        print("CROSS-VALIDATION RESULTS SUMMARY")
        print(f"{'='*70}")
    
    results_df = pd.DataFrame(fold_results)
    
    # Calculate mean and std for all metrics
    mean_r2 = results_df['r2'].mean()
    std_r2 = results_df['r2'].std()
    mean_mae = results_df['mae'].mean()
    std_mae = results_df['mae'].std()
    mean_rmse = results_df['rmse'].mean()
    std_rmse = results_df['rmse'].std()
    mean_n_features_selected = results_df['n_features_selected'].mean()
    std_n_features_selected = results_df['n_features_selected'].std()
    mean_best_alpha = results_df['best_alpha'].mean()
    std_best_alpha = results_df['best_alpha'].std()
    
    if verbose:
        print(f"\nALL PARAMETERS EVALUATION (Mean across {n_folds} folds):")
        print(f"  R²:                        {mean_r2:.4f} (+/- {std_r2:.4f})")
        print(f"  MAE (Mean Absolute Error): {mean_mae:.4f} (+/- {std_mae:.4f})")
        print(f"  RMSE (Root Mean Squared):  {mean_rmse:.4f} (+/- {std_rmse:.4f})")
        print(f"  Features selected:         {mean_n_features_selected:.1f} (+/- {std_n_features_selected:.1f}) out of {len(feature_columns)}")
        print(f"  Best alpha (mean):         {mean_best_alpha:.6f} (+/- {std_best_alpha:.6f})")
    
    # ----------------
    # Display detailed results table
    # ----------------
    if verbose:
        print(f"\n{'='*70}")
        print("Detailed Results by Fold:")
        print(f"{'='*70}")
    
    # Create a copy of results_df with all metric columns, and add mean row
    metric_columns = ['fold', 'r2', 'mae', 'rmse', 'n_features_selected', 'best_alpha']
    display_df = results_df[metric_columns].copy()
    
    # Create mean row with all metrics
    mean_row = {
        'fold': 'Mean',
        'r2': mean_r2,
        'mae': mean_mae,
        'rmse': mean_rmse,
        'n_features_selected': mean_n_features_selected,
        'best_alpha': mean_best_alpha
    }
    
    # Append mean row to display dataframe
    display_df = pd.concat([display_df, pd.DataFrame([mean_row])], ignore_index=True)
    
    if verbose:
        print(display_df.to_string(index=False))
    
    # ----------------
    # Feature Importance Analysis
    # ----------------
    # Calculate average coefficients across all folds
    coefficients_array = np.array(all_coefficients)
    mean_coefficients = np.mean(coefficients_array, axis=0)
    std_coefficients = np.std(coefficients_array, axis=0)
    
    # Count how many folds each feature was selected (non-zero) in
    n_folds_selected = np.sum(np.abs(coefficients_array) > 1e-6, axis=0)
    
    # Create feature importance dataframe
    feature_importance_df = pd.DataFrame({
        'feature': feature_columns,
        'coefficient_mean': mean_coefficients,
        'coefficient_std': std_coefficients,
        'abs_coefficient': np.abs(mean_coefficients),
        'n_folds_selected': n_folds_selected,
        'pct_folds_selected': (n_folds_selected / n_folds) * 100
    })
    
    # Sort by absolute coefficient value (most important first)
    feature_importance_df = feature_importance_df.sort_values('abs_coefficient', ascending=False).reset_index(drop=True)
    
    if verbose:
        print(f"\n{'='*70}")
        print("FEATURE IMPORTANCE (sorted by absolute coefficient)")
        print(f"{'='*70}")
        print(feature_importance_df.to_string(index=False))
    
    # ----------------
    # Create Feature Importance Visualization
    # ----------------
    # Select top N features if specified
    plot_df = feature_importance_df.copy()
    if top_n_features is not None and top_n_features > 0:
        plot_df = plot_df.head(top_n_features)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, max(6, len(plot_df) * 0.4)))
    
    # Color bars based on positive/negative coefficients
    colors = ['#2E86AB' if x >= 0 else '#A23B72' for x in plot_df['coefficient_mean']]
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(plot_df)), plot_df['coefficient_mean'], color=colors)
    
    # Add a vertical line at zero
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')
    
    # Set y-axis labels to feature names
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['feature'], fontsize=16)
    
    # Labels and title
    ax.set_xlabel('Coefficient Value', fontsize=18, fontweight='bold')
    ax.set_ylabel('Features', fontsize=18, fontweight='bold')
    ax.set_title(f'Feature Importance - Lasso Linear Regression (Auto-Selected Alpha)\nTarget: {target_col} (Top {len(plot_df)} features)', 
                 fontsize=20, fontweight='bold', pad=20)
    
    # Add grid for better readability
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        value = row['coefficient_mean']
        x_pos = value + (0.02 * max(abs(plot_df['coefficient_mean']))) if value >= 0 else value - (0.02 * max(abs(plot_df['coefficient_mean'])))
        ax.text(x_pos, i, f'{value:.4f}', va='center', 
                ha='left' if value >= 0 else 'right', fontsize=14)
    
    # Set x-axis limits with padding on both sides for better visibility
    x_min = plot_df['coefficient_mean'].min()
    x_max = plot_df['coefficient_mean'].max()
    x_range = x_max - x_min
    padding = max(0.15 * abs(x_range), 0.15 * max(abs(x_min), abs(x_max)))  # 15% padding or 15% of max absolute value
    ax.set_xlim(x_min - padding, x_max + padding)
    
    # Invert y-axis so highest importance is at top
    ax.invert_yaxis()
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E86AB', label='Positive (increases target value)'),
        Patch(facecolor='#A23B72', label='Negative (decreases target value)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=15)
    
    plt.tight_layout()
    
    # Display the figure in the notebook
    plt.show()
    
    # ----------------
    # Save results to CSV
    # ----------------
    if results_save_path is None:
        results_save_path = r"only_Q_outputs/combined/lasso_linear_regression_cv_results.csv"
    
    results_df.to_csv(results_save_path, index=False)
    if verbose:
        print(f"\n{'='*70}")
        print(f"Results saved to: {results_save_path}")
        print(f"{'='*70}")
    
    # Save feature importance to CSV
    if results_save_path:
        feature_importance_path = results_save_path.replace('.csv', '_feature_importance.csv')
        feature_importance_df.to_csv(feature_importance_path, index=False)
        if verbose:
            print(f"Feature importance saved to: {feature_importance_path}")
    
    return {
        'results_df': results_df,
        'display_df': display_df,
        'mean_r2': mean_r2,
        'std_r2': std_r2,
        'mean_mae': mean_mae,
        'std_mae': std_mae,
        'mean_rmse': mean_rmse,
        'std_rmse': std_rmse,
        'scaler': scaler,
        'model': lasso_model,
        'feature_importance_df': feature_importance_df,
        'coefficients_mean': mean_coefficients,
        'coefficients_std': std_coefficients,
        'feature_names': feature_columns,
        'n_features_selected': mean_n_features_selected,
        'best_alpha': mean_best_alpha,
        'best_alpha_std': std_best_alpha
    }
# Create trajectory pattern string for sorting
def create_trajectory_pattern(row, timepoint_cols):
    """Create a string representation of the trajectory for sorting"""
    return '-'.join([str(int(row[col])) if not pd.isna(row[col]) else 'N' 
                     for col in timepoint_cols])


# Count cluster changes for sorting
def count_cluster_changes(row, timepoint_cols):
    """Count the number of times cluster changes between consecutive timepoints"""
    changes = 0
    values = [row[col] for col in timepoint_cols]
    for i in range(len(values) - 1):
        if not (pd.isna(values[i]) or pd.isna(values[i+1])):
            if values[i] != values[i+1]:
                changes += 1
    return changes



# Find which transition has the most changes
def get_transition_changes(row, timepoint_cols):
    """Get list of which transitions have changes (1 if change, 0 if no change)"""
    changes = []
    values = [row[col] for col in timepoint_cols]
    for i in range(len(values) - 1):
        if not (pd.isna(values[i]) or pd.isna(values[i+1])):
            changes.append(1 if values[i] != values[i+1] else 0)
        else:
            changes.append(0)
    return changes


def assign_group(n_changes):
    """Assign a group label based on number of changes."""
    change_groups = {
        0: '0 changes (stable)',
        1: '1 change',
        2: '2 changes',
        3: '3+ changes'
    }
    if n_changes == 0:
        return change_groups[0]
    elif n_changes == 1:
        return change_groups[1]
    elif n_changes == 2:
        return change_groups[2]
    else:
        return change_groups[3]


####################################################
import pandas as pd

# Function to load Excel and print column names for ALL sheets
def print_excel_columns_all_sheets(excel_path):
    """
    Load an Excel file and print all column names for each sheet separately.
    
    Parameters:
    -----------
    excel_path : str
        Path to the Excel file
    """
    # Get all sheet names first
    excel_file = pd.ExcelFile(excel_path)
    sheet_names = excel_file.sheet_names
    
    print(f"Excel file: {excel_path}")
    print(f"Total sheets: {len(sheet_names)}")
    print("=" * 70)
    
    # Dictionary to store all dataframes
    all_dfs = {}
    
    # Iterate through each sheet
    for sheet_idx, sheet_name in enumerate(sheet_names, 1):
        # Load the sheet
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        all_dfs[sheet_name] = df
        
        # Print sheet information
        print(f"\n{'='*70}")
        print(f"SHEET {sheet_idx}/{len(sheet_names)}: {sheet_name}")
        print(f"{'='*70}")
        print(f"Total columns: {len(df.columns)}")
        print(f"\nColumn names:")
        print("-" * 70)
        
        # Print each column name
        for i, col in enumerate(df.columns, 1):
            print(f"{i:3d}. {col}")
        
        print()  # Empty line between sheets
    
    print("=" * 70)
    print(f"Finished processing {len(sheet_names)} sheet(s)")
    
    return all_dfs


def create_trajectory_heatmap(output_df, subject_col="Subject_Code", filter_by_after=None):
    """
    Create a heatmap visualization showing the trajectory of subjects between clusters.
    
    Parameters:
    -----------
    output_df : pandas.DataFrame
        DataFrame containing subject codes and timepoint cluster columns (b, t1, t2, t3, after)
    subject_col : str, optional
        Name of the subject identifier column (default: "Subject_Code")
    filter_by_after : int or None, optional
        Filter by final cluster state: None (show all), 0 (only cluster 0), 1 (only cluster 1) (default: None)
    
    Returns:
    --------
    pandas.DataFrame
        The processed dataframe with additional columns (trajectory_pattern, n_changes, change_group, etc.)
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from matplotlib.colors import ListedColormap
    
    # Make a copy to avoid modifying the original
    df = output_df.copy()
    
    # Filter by final cluster state if requested
    if filter_by_after is not None:
        if 'after' not in df.columns:
            raise ValueError("'after' column not found in dataframe. Cannot filter by final cluster state.")
        n_before = len(df)
        df = df[df['after'] == filter_by_after].copy().reset_index(drop=True)
        n_after = len(df)
        filter_label = f"subjects with after == {filter_by_after}"
        print(f"\nFiltering by final cluster state (after == {filter_by_after})...")
        print(f"Subjects before filtering: {n_before}")
        print(f"Subjects after filtering: {n_after}")
        print(f"Removed: {n_before - n_after} subjects")
    else:
        filter_label = "ALL subjects"
    
    print(f"\n{'='*70}")
    print(f"Creating trajectory heatmap for {filter_label}...")
    print(f"{'='*70}")
    print(f"Total subjects: {len(df)}")
    
    # Prepare data for plotting
    timepoint_labels = ["b", "t1", "t2", "t3", "after"]
    transition_labels = ["b→t1", "t1→t2", "t2→t3", "t3→after"]
    
    # Check which timepoint columns exist
    existing_timepoint_cols = [col for col in timepoint_labels if col in df.columns]
    if not existing_timepoint_cols:
        raise ValueError(f"None of the timepoint columns {timepoint_labels} found in dataframe.")
    
    # Convert timepoint columns to numeric
    for col in existing_timepoint_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Create trajectory pattern string for sorting
    def create_trajectory_pattern(row, timepoint_cols):
        """Create a string representation of the trajectory for sorting"""
        return '-'.join([str(int(row[col])) if not pd.isna(row[col]) else 'N' 
                         for col in timepoint_cols])
    
    df['trajectory_pattern'] = df.apply(
        lambda row: create_trajectory_pattern(row, timepoint_labels), axis=1
    )
    
    # Count cluster changes for sorting
    def count_cluster_changes(row, timepoint_cols):
        """Count the number of times cluster changes between consecutive timepoints"""
        changes = 0
        values = [row[col] for col in timepoint_cols]
        for i in range(len(values) - 1):
            if not (pd.isna(values[i]) or pd.isna(values[i+1])):
                if values[i] != values[i+1]:
                    changes += 1
        return changes
    
    df['n_changes'] = df.apply(
        lambda row: count_cluster_changes(row, timepoint_labels), axis=1
    )
    
    # Find which transition has the most changes
    def get_transition_changes(row, timepoint_cols):
        """Get list of which transitions have changes (1 if change, 0 if no change)"""
        changes = []
        values = [row[col] for col in timepoint_cols]
        for i in range(len(values) - 1):
            if not (pd.isna(values[i]) or pd.isna(values[i+1])):
                changes.append(1 if values[i] != values[i+1] else 0)
            else:
                changes.append(0)
        return changes
    
    # Add columns for each transition
    for i, trans_label in enumerate(transition_labels):
        df[f'change_{i}'] = df.apply(
            lambda row: get_transition_changes(row, timepoint_labels)[i], axis=1
        )
    
    # Calculate number of changes at each transition point
    transition_counts = []
    for i in range(len(transition_labels)):
        count = df[f'change_{i}'].sum()
        transition_counts.append(count)
    
    # Find which transition has the most changes
    max_transition_idx = np.argmax(transition_counts)
    max_transition = transition_labels[max_transition_idx]
    max_count = transition_counts[max_transition_idx]
    
    print(f"\nTransition Analysis:")
    print(f"{'='*70}")
    for i, (trans, count) in enumerate(zip(transition_labels, transition_counts)):
        marker = " <-- MOST CHANGES" if i == max_transition_idx else ""
        print(f"{trans}: {count} changes ({count/len(df)*100:.1f}% of subjects){marker}")
    
    # Sort subjects to better visualize transitions:
    # 1. First by number of changes (prioritize 1 change at the top)
    # 2. Then by starting cluster (b)
    # 3. Then by trajectory pattern (to group similar trajectories)
    # Create a custom sort key: 1 change first, then others
    df['sort_key'] = df['n_changes'].apply(lambda x: 0 if x == 1 else 1)
    df_sorted = df.sort_values(
        by=['sort_key', 'b', 'trajectory_pattern', 'n_changes', subject_col]
    ).reset_index(drop=True)
    
    # Calculate proportions for each timepoint
    proportions = []
    for col in timepoint_labels:
        prop = df[col].mean()
        proportions.append(prop)
    
    # Prepare data for heatmap
    heatmap_data = df_sorted[timepoint_labels].values
    
    # Add proportions row at the bottom
    proportions_row = np.array([proportions])
    heatmap_data_with_props = np.vstack([heatmap_data, proportions_row])
    
    # Create labels (subjects + "Proportion" row)
    subject_labels = list(df_sorted[subject_col].values) + ['Proportion']
    subject_labels_array = np.array(subject_labels)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, max(10, len(df) * 0.35)))
    
    # Create custom colormap: green for 0, red for 1
    colors = ['green', 'red']  # 0 = green (good), 1 = red (bad)
    n_bins = 2
    cmap = ListedColormap(colors)
    
    # Create annotation array - show integer values for subjects, proportions for last row
    annot_data = []
    for i in range(len(heatmap_data_with_props)):
        if i < len(df):
            # Subject rows: show integers (0 or 1)
            annot_data.append([f'{int(val)}' for val in heatmap_data_with_props[i]])
        else:
            # Proportion row: show 2 decimal places
            annot_data.append([f'{val:.2f}' for val in heatmap_data_with_props[i]])
    annot_data = np.array(annot_data)
    
    # Create heatmap
    heatmap_obj = sns.heatmap(heatmap_data_with_props, 
                annot=annot_data,  # Show formatted values
                fmt='',            # Empty fmt since we're providing formatted strings
                cmap=cmap,         # Custom colormap: green=0/good, red=1/bad
                vmin=0, vmax=1,
                cbar_kws={'label': 'Cluster (0=Good, 1=Bad)', 'ticks': [0, 1], 'labelsize': 18},
                yticklabels=subject_labels_array,
                xticklabels=timepoint_labels,
                linewidths=0.5,
                linecolor='gray',
                ax=ax,
                annot_kws={'size': 20, 'weight': 'bold'})  # Bigger, bold font for annotations
    
    # Recolor the proportions row to light blue
    proportions_row_idx = len(df)  # Index of proportions row (0-based)
    # Get the QuadMesh collection from the heatmap
    quadmesh = heatmap_obj.collections[0]
    # Get the face colors array
    face_colors = quadmesh.get_facecolors()
    # Calculate the number of columns
    n_cols = len(timepoint_labels)
    # The colors are stored in row-major order: row i starts at index i * n_cols
    # Change colors for the proportions row (last row)
    start_idx = proportions_row_idx * n_cols
    end_idx = start_idx + n_cols
    # Set light blue color (RGBA)
    lightblue = [0.68, 0.85, 1.0, 1.0]
    for idx in range(start_idx, min(end_idx, len(face_colors))):
        face_colors[idx] = lightblue
    # Update the QuadMesh with new colors
    quadmesh.set_facecolors(face_colors)
    
    ax.set_ylabel('Subject', fontsize=22)
    ax.set_xlabel('Timepoint', fontsize=22)

    # Rotate y-axis labels for better readability
    plt.yticks(rotation=0, fontsize=18)
    plt.xticks(rotation=0, fontsize=20)
    
    plt.tight_layout()
    plt.show()
    
    # Group subjects by number of changes
    change_groups = {
        0: '0 changes (stable)',
        1: '1 change',
        2: '2 changes',
        3: '3+ changes'
    }
    
    df['change_group'] = df['n_changes'].apply(assign_group)
    
    # ----------------
    # Display group counts
    # ----------------
    print(f"\n{'='*70}")
    print("Group Distribution by Number of Cluster Changes")
    print(f"{'='*70}")
    
    group_counts = df['change_group'].value_counts()
    # Reorder to match the group order
    ordered_groups = [change_groups[0], change_groups[1], change_groups[2], change_groups[3]]
    group_counts_ordered = group_counts.reindex([g for g in ordered_groups if g in group_counts.index], fill_value=0)
    group_percentages = (group_counts_ordered / len(df) * 100).round(1)
    
    summary_df = pd.DataFrame({
        'Group': group_counts_ordered.index,
        'Count': group_counts_ordered.values,
        'Percentage': group_percentages.values
    })
    
    print("\n" + summary_df.to_string(index=False))
    print(f"\nTotal subjects: {len(df)}")
    
    # Show some examples from each group
    print(f"\n{'='*70}")
    print("Example subjects from each group:")
    print(f"{'='*70}")
    
    for group_name in ordered_groups:
        group_subjects = df[df['change_group'] == group_name]
        if len(group_subjects) > 0:
            print(f"\n{group_name} (n={len(group_subjects)}):")
            # Show first 3 subjects from this group
            examples = group_subjects.head(3)
            for idx, row in examples.iterrows():
                trajectory_str = ' -> '.join([f"{row[col]:.0f}" if not pd.isna(row[col]) else "N" 
                                             for col in timepoint_labels])
                print(f"  {row[subject_col]}: {trajectory_str} (changes: {row['n_changes']})")
    
    return df


def analyze_transition_to_after(file_path=None, df=None, transition_cols=None, after_col="after", 
                                 subject_col="Subject_Code", return_summaries=False):
    """
    Analyze how each transition type relates to the final 'after' outcome (0 or 1).
    Creates crosstab tables and stacked bar plots for each transition.
    
    Parameters:
    -----------
    file_path : str, optional
        Path to CSV file containing transition data. If None, df must be provided.
    df : pandas.DataFrame, optional
        DataFrame containing transition data. If None, file_path must be provided.
    transition_cols : list of str, optional
        List of transition column names to analyze. 
        Default: ["B_TO_T1", "T1_TO_T2", "T2_TO_T3", "T3_TO_AFTER"]
    after_col : str, optional
        Name of the 'after' outcome column (default: "after")
    subject_col : str, optional
        Name of the subject identifier column (default: "Subject_Code")
    return_summaries : bool, optional
        If True, returns a dictionary of summary DataFrames for each transition (default: False)
    
    Returns:
    --------
    pandas.DataFrame or dict
        If return_summaries=False: returns the filtered dataframe
        If return_summaries=True: returns a dict with transition names as keys and summary DataFrames as values
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Load data
    if df is None:
        if file_path is None:
            raise ValueError("Either file_path or df must be provided")
        df = pd.read_csv(file_path)
    else:
        df = df.copy()
    
    # Convert after column to numeric
    if after_col not in df.columns:
        raise ValueError(f"'{after_col}' column not found in dataframe. Available columns: {list(df.columns)}")
    
    df[after_col] = pd.to_numeric(df[after_col], errors="coerce")
    df2 = df[df[after_col].isin([0, 1])].copy()
    
    # Set default transition columns if not provided
    if transition_cols is None:
        transition_cols = ["B_TO_T1", "T1_TO_T2", "T2_TO_T3", "T3_TO_AFTER"]
    
    # Check which transition columns exist
    existing_transition_cols = [col for col in transition_cols if col in df2.columns]
    missing_transition_cols = [col for col in transition_cols if col not in df2.columns]
    
    if missing_transition_cols:
        print(f"Warning: These transition columns not found: {missing_transition_cols}")
    
    if not existing_transition_cols:
        raise ValueError(f"None of the transition columns {transition_cols} found in dataframe.")
    
    # Dictionary to store summaries if requested
    summaries_dict = {}
    
    # Analyze each transition column
    for col in existing_transition_cols:
        # Check if column exists
        if col not in df2.columns:
            print(f"Warning: Column '{col}' not found, skipping...")
            continue
        
        # Counts
        counts = pd.crosstab(df2[col], df2[after_col]).reindex(columns=[0, 1], fill_value=0)
        counts.columns = ["after_0_n", "after_1_n"]
        counts["n_total"] = counts["after_0_n"] + counts["after_1_n"]
        
        # Percentages
        pct = counts[["after_0_n", "after_1_n"]].div(counts["n_total"], axis=0) * 100
        pct.columns = ["after_0_%", "after_1_%"]
        
        # Combined table
        summary = pd.concat([counts, pct], axis=1)
        summary = summary[["after_0_n", "after_0_%", "after_1_n", "after_1_%", "n_total"]]
        
        # Store summary if requested
        if return_summaries:
            summaries_dict[col] = summary
        
        print(f"\n=== {col} (COUNTS + PERCENT) ===")
        print(summary.round({"after_0_%": 1, "after_1_%": 1}))
        
        # Plot percentages
        ax = pct.plot(kind="bar", stacked=True)
        ax.set_title(f"After (0/1) distribution by {col} (percent)")
        ax.set_xlabel(col)
        ax.set_ylabel("Percent of subjects")
        plt.xticks(rotation=45, ha="right")
        
        # Annotate percentages inside bars
        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                if height <= 0:
                    continue
                if height < 5:  # optional: skip very small percentages
                    continue
                x = bar.get_x() + bar.get_width() / 2
                y = bar.get_y() + height / 2
                ax.text(x, y, f"{height:.1f}%", ha="center", va="center", fontsize=14)
        
        # Annotate n above each bar
        group_n = counts["n_total"]
        for i, n in enumerate(group_n.values):
            ax.text(i, 102, f"n={int(n)}", ha="center", va="bottom", fontsize=15)
        
        ax.set_ylim(0, 110)
        plt.tight_layout()
        plt.show()
    
    if return_summaries:
        return summaries_dict
    else:
        return df2


def lasso_logistic_regression_cv(
    input_file_path,
    subject_col="subject_code",
    target_col="after",
    n_folds=5,
    cv_random_state=42,
    results_save_path=None,
    verbose=True,
    top_n_features=None,
    use_upsampling=True,
    C_values=None,
    solver='liblinear',
    use_cv_for_C=True
):
    """
    Perform Lasso (L1) logistic regression with cross-validation.
    Lasso regularization helps with feature selection by driving coefficients to zero.
    
    Parameters:
    -----------
    input_file_path : str
        Path to the input CSV file (normalized dataframe)
    subject_col : str, optional
        Subject identifier column name (will be excluded from features, default: "subject_code")
    target_col : str, optional
        Target column for binary classification (default: "after")
    n_folds : int, optional
        Number of CV folds (default: 5)
    cv_random_state : int, optional
        Random state for cross-validation (default: 42)
    results_save_path : str, optional
        Path to save the results CSV. If None, uses default path (default: None)
    verbose : bool, optional
        Whether to print progress messages (default: True)
    top_n_features : int, optional
        Number of top features to show in the visualization. If None, shows all features (default: None)
    use_upsampling : bool, optional
        Whether to apply SMOTE upsampling to training data (default: True)
    C_values : list, optional
        List of C values (inverse of regularization strength) to test. 
        If None, uses default: [0.001, 0.01, 0.1, 1, 10, 100, 1000] (default: None)
        Smaller C = stronger regularization (more features removed)
    solver : str, optional
        Solver to use. Options: 'liblinear' (fast, good for small datasets) or 'saga' (for larger datasets).
        Only these solvers support L1 penalty (default: 'liblinear')
    use_cv_for_C : bool, optional
        If True, uses LogisticRegressionCV to automatically find best C. 
        If False, uses manual CV with specified C_values (default: True)
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'results_df': DataFrame with results for each fold
        - 'display_df': DataFrame with fold and all metric columns plus mean row
        - 'mean_accuracy': Mean accuracy across folds
        - 'std_accuracy': Standard deviation of accuracy
        - 'mean_precision': Mean precision across folds
        - 'std_precision': Standard deviation of precision
        - 'mean_recall': Mean recall across folds
        - 'std_recall': Standard deviation of recall
        - 'mean_f1': Mean F1 score across folds
        - 'std_f1': Standard deviation of F1 score
        - 'mean_roc_auc': Mean ROC-AUC across folds
        - 'std_roc_auc': Standard deviation of ROC-AUC
        - 'scaler': The fitted StandardScaler object
        - 'model': The trained logistic regression model (from last fold)
        - 'feature_importance_df': DataFrame with feature coefficients and importance
        - 'coefficients_mean': Array of mean coefficients across folds
        - 'coefficients_std': Array of std of coefficients across folds
        - 'feature_names': List of feature names
        - 'best_C': Best C value found (if use_cv_for_C=True)
        - 'selected_features': List of features with non-zero coefficients (Lasso feature selection)
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from imblearn.over_sampling import SMOTE
    import warnings
    warnings.filterwarnings('ignore')
    
    # ----------------  
    # Load the dataframe
    # ----------------
    if verbose:
        print(f"Loading file: {input_file_path}")
    df = pd.read_csv(input_file_path)
    
    if verbose:
        print(f"Loaded {len(df)} subjects")
        print(f"Total columns: {len(df.columns)}")
        print(f"Columns: {list(df.columns)}")
    
    # ----------------  
    # Check required columns
    # ----------------
    if subject_col not in df.columns:
        raise ValueError(f"'{subject_col}' column not found. Available columns: {list(df.columns)}")
    
    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' column not found. Available columns: {list(df.columns)}")
    
    # Convert target to numeric
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    
    # Check target distribution
    target_counts = df[target_col].value_counts().sort_index()
    if verbose:
        print(f"\nTarget distribution ({target_col}):")
        for val, count in target_counts.items():
            print(f"  {target_col} = {val}: {count} subjects ({100*count/len(df):.1f}%)")
    
    # ----------------  
    # Prepare features and target
    # ----------------
    if verbose:
        print(f"\n{'='*70}")
        print("Preparing features and target...")
        print(f"{'='*70}")
    
    # Get feature columns (all columns except subject_code and target)
    feature_columns = [col for col in df.columns if col not in [subject_col, target_col]]
    
    if len(feature_columns) == 0:
        raise ValueError("No feature columns found! Check your column names.")
    
    if verbose:
        print(f"Feature columns ({len(feature_columns)}): {feature_columns}")
    
    # Extract features (X) and target (y)
    X = df[feature_columns].copy()
    y = df[target_col].copy()
    
    # Convert features to numeric
    for col in feature_columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    
    # Remove rows with any missing values
    valid_mask = X.notna().all(axis=1) & y.notna()
    X_clean = X[valid_mask].copy()
    y_clean = y[valid_mask].copy()
    
    if verbose:
        print(f"\nValid subjects (after removing missing): {len(X_clean)}")
        print(f"Removed {len(X) - len(X_clean)} subjects with missing values")
    
    # Check if we have both classes
    unique_classes = y_clean.unique()
    if len(unique_classes) < 2:
        raise ValueError(f"Target has only {len(unique_classes)} unique class(es). Need at least 2 for binary classification.")
    
    if verbose:
        print(f"Target classes: {sorted(unique_classes)}")
    
    # ----------------  
    # Standardize features
    # ----------------
    if verbose:
        print(f"\n{'='*70}")
        print("Standardizing features...")
        print(f"{'='*70}")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    y_values = y_clean.values
    
    if verbose:
        print(f"Features standardized: {X_scaled.shape}")
    
    # ----------------  
    # Set default C values if not provided
    # ----------------
    if C_values is None:
        C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    
    # ----------------  
    # Lasso Logistic Regression with Cross-Validation
    # ----------------
    if verbose:
        print(f"\n{'='*70}")
        if use_upsampling:
            print("LASSO LOGISTIC REGRESSION: Binary Classification with SMOTE Upsampling")
        else:
            print("LASSO LOGISTIC REGRESSION: Binary Classification (no upsampling)")
        print(f"Target: {target_col}")
        print(f"Features: {len(feature_columns)} columns")
        print(f"Cross-validation: {n_folds}-fold stratified")
        print(f"Upsampling: {'Enabled (SMOTE)' if use_upsampling else 'Disabled'}")
        print(f"Penalty: L1 (Lasso)")
        print(f"Solver: {solver}")
        print(f"C values to test: {C_values}")
        print(f"{'='*70}")
    
    # Use StratifiedKFold to ensure balanced class distribution in each fold
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cv_random_state)
    
    # Store results for each fold
    fold_results = []
    
    # Store coefficients from each fold for feature analysis
    all_coefficients = []
    best_C_per_fold = []
    
    # Manual cross-validation
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_scaled, y_values), 1):
        # Split data
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_values[train_idx], y_values[test_idx]
        
        # Apply upsampling (SMOTE) to training data only if requested
        if use_upsampling:
            smote = SMOTE(random_state=cv_random_state)
            X_train_final, y_train_final = smote.fit_resample(X_train, y_train)
            if verbose:
                print(f"\n--- Fold {fold_idx}/{n_folds} ---")
                print(f"  Training data before upsampling: {len(X_train)} samples")
                print(f"  Training data after upsampling: {len(X_train_final)} samples")
                unique, counts = np.unique(y_train_final, return_counts=True)
                class_dist = dict(zip(unique, counts))
                print(f"  Class distribution after upsampling: {class_dist}")
        else:
            X_train_final, y_train_final = X_train, y_train
            if verbose:
                print(f"\n--- Fold {fold_idx}/{n_folds} ---")
                print(f"  Training data: {len(X_train)} samples")
                unique, counts = np.unique(y_train, return_counts=True)
                class_dist = dict(zip(unique, counts))
                print(f"  Class distribution: {class_dist}")
        
        # Find best C using cross-validation on training data if requested
        if use_cv_for_C:
            # Use LogisticRegressionCV to find best C
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=cv_random_state)
            lasso_cv = LogisticRegressionCV(
                Cs=C_values,
                penalty='l1',
                solver=solver,
                cv=inner_cv,
                max_iter=1000,
                random_state=cv_random_state,
                scoring='roc_auc',
                class_weight='balanced' if not use_upsampling else None
            )
            lasso_cv.fit(X_train_final, y_train_final)
            best_C = lasso_cv.C_[0]  # Get the best C value
            best_C_per_fold.append(best_C)
            if verbose:
                print(f"  Best C (from inner CV): {best_C}")
            
            # Train final model with best C
            logistic_model = LogisticRegression(
                C=best_C,
                penalty='l1',
                solver=solver,
                max_iter=1000,
                random_state=cv_random_state,
                class_weight='balanced' if not use_upsampling else None
            )
        else:
            # Use first C value (or you could do manual grid search)
            best_C = C_values[0]
            logistic_model = LogisticRegression(
                C=best_C,
                penalty='l1',
                solver=solver,
                max_iter=1000,
                random_state=cv_random_state,
                class_weight='balanced' if not use_upsampling else None
            )
            if verbose:
                print(f"  Using C: {best_C}")
        
        # Train the model
        logistic_model.fit(X_train_final, y_train_final)
        
        # Make predictions
        y_pred = logistic_model.predict(X_test)
        y_pred_proba = logistic_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store results
        fold_results.append({
            'fold': fold_idx,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'C': best_C
        })
        
        # Store coefficients
        all_coefficients.append(logistic_model.coef_[0])
        
        if verbose:
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1: {f1:.4f}")
            print(f"  ROC-AUC: {roc_auc:.4f}")
            # Count non-zero coefficients (selected features)
            n_selected = np.sum(np.abs(logistic_model.coef_[0]) > 1e-6)
            print(f"  Selected features (non-zero coefficients): {n_selected}/{len(feature_columns)}")
    
    # ----------------  
    # Create results DataFrame
    # ----------------
    results_df = pd.DataFrame(fold_results)
    
    # Calculate mean and std for each metric
    mean_accuracy = results_df['accuracy'].mean()
    std_accuracy = results_df['accuracy'].std()
    mean_precision = results_df['precision'].mean()
    std_precision = results_df['precision'].std()
    mean_recall = results_df['recall'].mean()
    std_recall = results_df['recall'].std()
    mean_f1 = results_df['f1'].mean()
    std_f1 = results_df['f1'].std()
    mean_roc_auc = results_df['roc_auc'].mean()
    std_roc_auc = results_df['roc_auc'].std()
    
    # Create display DataFrame with mean row
    display_df = results_df.copy()
    mean_row = pd.DataFrame({
        'fold': ['Mean'],
        'accuracy': [mean_accuracy],
        'precision': [mean_precision],
        'recall': [mean_recall],
        'f1': [mean_f1],
        'roc_auc': [mean_roc_auc],
        'C': [np.mean(best_C_per_fold) if best_C_per_fold else C_values[0]]
    })
    display_df = pd.concat([display_df, mean_row], ignore_index=True)
    
    # ----------------  
    # Feature importance analysis
    # ----------------
    coefficients_array = np.array(all_coefficients)
    coefficients_mean = coefficients_array.mean(axis=0)
    coefficients_std = coefficients_array.std(axis=0)
    
    # Create feature importance DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': feature_columns,
        'coefficient_mean': coefficients_mean,
        'abs_coefficient': np.abs(coefficients_mean)
    }).sort_values('abs_coefficient', ascending=False)
    
    # Identify selected features (non-zero coefficients)
    selected_features = feature_importance_df[
        feature_importance_df['abs_coefficient'] > 1e-6
    ]['feature'].tolist()
    
    if verbose:
        print(f"\n{'='*70}")
        print("CROSS-VALIDATION RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"Mean Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")
        print(f"Mean Precision: {mean_precision:.4f} (+/- {std_precision:.4f})")
        print(f"Mean Recall: {mean_recall:.4f} (+/- {std_recall:.4f})")
        print(f"Mean F1: {mean_f1:.4f} (+/- {std_f1:.4f})")
        print(f"Mean ROC-AUC: {mean_roc_auc:.4f} (+/- {std_roc_auc:.4f})")
        if best_C_per_fold:
            print(f"Mean Best C: {np.mean(best_C_per_fold):.4f}")
        print(f"\nSelected Features (non-zero coefficients): {len(selected_features)}/{len(feature_columns)}")
        print(f"Selected features: {selected_features}")
    
    # ----------------  
    # Visualize feature importance
    # ----------------
    if top_n_features is None:
        top_n_features = len(feature_columns)
    
    top_features = feature_importance_df.head(top_n_features)
    
    fig, ax = plt.subplots(figsize=(10, max(6, top_n_features * 0.4)))
    
    # Create horizontal bar plot
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_features['coefficient_mean'], xerr=top_features['coefficient_std'], 
            capsize=3, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'], fontsize=16)
    ax.set_xlabel('Coefficient Value', fontsize=18, fontweight='bold')
    ax.set_title(f'Lasso Logistic Regression: Feature Coefficients (Top {top_n_features})', 
                 fontsize=20, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # ----------------  
    # Save results if requested
    # ----------------
    if results_save_path is None:
        results_save_path = r"only_Q_outputs/combined/lasso_logistic_regression_cv_results.csv"
    
    display_df.to_csv(results_save_path, index=False)
    if verbose:
        print(f"\nResults saved to: {results_save_path}")
    
    # ----------------  
    # Return results dictionary
    # ----------------
    # Train final model on all data for return
    if use_upsampling:
        smote_final = SMOTE(random_state=cv_random_state)
        X_final, y_final = smote_final.fit_resample(X_scaled, y_values)
    else:
        X_final, y_final = X_scaled, y_values
    
    if use_cv_for_C:
        inner_cv_final = StratifiedKFold(n_splits=3, shuffle=True, random_state=cv_random_state)
        lasso_cv_final = LogisticRegressionCV(
            Cs=C_values,
            penalty='l1',
            solver=solver,
            cv=inner_cv_final,
            max_iter=1000,
            random_state=cv_random_state,
            scoring='roc_auc',
            class_weight='balanced' if not use_upsampling else None
        )
        lasso_cv_final.fit(X_final, y_final)
        best_C_final = lasso_cv_final.C_[0]
        final_model = LogisticRegression(
            C=best_C_final,
            penalty='l1',
            solver=solver,
            max_iter=1000,
            random_state=cv_random_state,
            class_weight='balanced' if not use_upsampling else None
        )
    else:
        best_C_final = C_values[0]
        final_model = LogisticRegression(
            C=best_C_final,
            penalty='l1',
            solver=solver,
            max_iter=1000,
            random_state=cv_random_state,
            class_weight='balanced' if not use_upsampling else None
        )
    
    final_model.fit(X_final, y_final)
    
    return {
        'results_df': results_df,
        'display_df': display_df,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'mean_precision': mean_precision,
        'std_precision': std_precision,
        'mean_recall': mean_recall,
        'std_recall': std_recall,
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'mean_roc_auc': mean_roc_auc,
        'std_roc_auc': std_roc_auc,
        'scaler': scaler,
        'model': final_model,
        'feature_importance_df': feature_importance_df,
        'coefficients_mean': coefficients_mean,
        'coefficients_std': coefficients_std,
        'feature_names': feature_columns,
        'best_C': best_C_final if use_cv_for_C else C_values[0],
        'selected_features': selected_features
    }


