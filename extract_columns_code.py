# ----------------  
# Generate tuples list and extract columns from Excel files
# ----------------
import pandas as pd
from pathlib import Path
import os

# Configuration
# ----------------
# Sheet name to use (modify this based on your actual sheet name)
SHEET_NAME = "cortical_vol_lr"  # CHANGE THIS to your actual sheet name

# Input folder containing Excel files
INPUT_FOLDER = r"YOUR_FOLDER_PATH_HERE"  # CHANGE THIS to your folder path

# Output CSV file path
OUTPUT_CSV = r"extracted_columns_output.csv"  # CHANGE THIS if needed

# Subject identifier column name (usually first column or "Subject_Code")
SUBJECT_COL = "Subject_Code"  # CHANGE THIS if different

# ----------------  
# Step 1: Create list of tuples from filtered_df
# (This assumes filtered_df exists from the previous cell)
# ----------------
# Generate tuples: (parameter_hemisphere, sheet_name)
tuples_list = []

# Make sure filtered_df exists (from previous cell)
# If not, you'll need to re-run the filtering cell first
if 'filtered_df' not in locals() and 'filtered_df' not in globals():
    raise ValueError("Please run the previous cell first to create filtered_df")

for idx, row in filtered_df.iterrows():
    param = row[param_col]
    hemi = row['Hemisphere'].lower()  # Ensure lowercase (lh/rh)
    param_hemisphere = f"{param}_{hemi}"
    tuples_list.append((param_hemisphere, SHEET_NAME))

print(f"Generated {len(tuples_list)} tuples:")
print("First 5 tuples:")
for i, tup in enumerate(tuples_list[:5], 1):
    print(f"  {i}. {tup}")

print(f"\nAll {len(tuples_list)} tuples:")
for i, tup in enumerate(tuples_list, 1):
    print(f"  {i}. {tup}")

# ----------------  
# Step 2: Extract columns from Excel files
# ----------------
def pick_column(df, candidates):
    """Return the first matching column name in df (case-insensitive), else None."""
    cols_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_map:
            return cols_map[c.lower()]
    for c in candidates:
        for k, v in cols_map.items():
            if c.lower() in k:
                return v
    return None

def extract_columns_from_excel_files(folder_path, tuples_list, sheet_name, subject_col="Subject_Code"):
    """
    Extract columns from Excel files based on the tuples list.
    
    Args:
        folder_path: Path to folder containing Excel files
        tuples_list: List of tuples (column_name, sheet_name)
        sheet_name: Name of the sheet to read from
        subject_col: Name of the subject identifier column
    
    Returns:
        DataFrame with one row per subject and columns for each tuple's column_name
    """
    folder = Path(folder_path)
    excel_files = list(folder.glob("*.xlsx")) + list(folder.glob("*.xls"))
    
    if not excel_files:
        print(f"No Excel files found in {folder_path}")
        return pd.DataFrame()
    
    print(f"Found {len(excel_files)} Excel files")
    
    # Get all unique column names from tuples
    column_names = [tup[0] for tup in tuples_list]
    unique_columns = sorted(set(column_names))
    
    print(f"\nLooking for {len(unique_columns)} unique columns:")
    for col in unique_columns[:10]:  # Show first 10
        print(f"  - {col}")
    if len(unique_columns) > 10:
        print(f"  ... and {len(unique_columns) - 10} more")
    
    rows_by_subject = {}
    
    for excel_file in excel_files:
        print(f"\nProcessing: {excel_file.name}")
        
        try:
            # Read the specified sheet
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            print(f"  Loaded sheet '{sheet_name}' with {len(df)} rows")
            print(f"  Columns: {list(df.columns)}")
            
            # Find subject column
            subject_col_found = pick_column(df, [subject_col, "Subject_Code", "subject_code", "Subject", "subject"])
            if not subject_col_found:
                # Try first column as subject identifier
                subject_col_found = df.columns[0]
                print(f"  Warning: Using first column '{subject_col_found}' as subject identifier")
            
            # Process each row in the Excel file
            for idx, row in df.iterrows():
                subject_id = str(row[subject_col_found]).strip()
                
                # Initialize subject row if not exists
                if subject_id not in rows_by_subject:
                    rows_by_subject[subject_id] = {SUBJECT_COL: subject_id}
                    # Initialize all columns with NaN
                    for col in unique_columns:
                        rows_by_subject[subject_id][col] = pd.NA
                
                # Extract values for each column in tuples_list
                for col_name, sheet in tuples_list:
                    if col_name in df.columns:
                        rows_by_subject[subject_id][col_name] = row[col_name]
                    else:
                        # Try case-insensitive match
                        col_found = pick_column(df, [col_name])
                        if col_found:
                            rows_by_subject[subject_id][col_name] = row[col_found]
                        # If still not found, leave as NaN (already initialized)
            
        except Exception as e:
            print(f"  ERROR processing {excel_file.name}: {e}")
            continue
    
    # Convert to DataFrame
    if not rows_by_subject:
        print("\nNo data extracted. Please check:")
        print("  1. Folder path is correct")
        print("  2. Sheet name exists in Excel files")
        print("  3. Column names match what's in the Excel files")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(list(rows_by_subject.values()))
    
    # Reorder columns: subject_col first, then the requested columns in order
    ordered_cols = [SUBJECT_COL] + unique_columns
    # Only include columns that exist
    ordered_cols = [col for col in ordered_cols if col in result_df.columns]
    result_df = result_df[ordered_cols]
    
    print(f"\n{'='*70}")
    print(f"Extraction complete!")
    print(f"{'='*70}")
    print(f"Total subjects: {len(result_df)}")
    print(f"Total columns: {len(result_df.columns)}")
    print(f"\nFirst few rows:")
    print(result_df.head())
    
    return result_df

# ----------------  
# Step 3: Run the extraction
# ----------------
# Uncomment and modify the path below to run the extraction
# result_df = extract_columns_from_excel_files(
#     folder_path=INPUT_FOLDER,
#     tuples_list=tuples_list,
#     sheet_name=SHEET_NAME,
#     subject_col=SUBJECT_COL
# )
# 
# # Save to CSV
# if not result_df.empty:
#     result_df.to_csv(OUTPUT_CSV, index=False)
#     print(f"\nResults saved to: {OUTPUT_CSV}")
# else:
#     print("\nNo data to save.")













