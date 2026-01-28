# ----------------  
# Generate tuples list and extract columns from Excel files
# ----------------
import pandas as pd
from pathlib import Path

# Configuration
# ----------------
# Sheet name to use (modify this based on your actual sheet name)
SHEET_NAME = "cortical_vol_lr"  # CHANGE THIS to your actual sheet name

# Input folder containing Excel files
INPUT_FOLDER = r"YOUR_FOLDER_PATH_HERE"  # CHANGE THIS to your folder path

# Output CSV file path
OUTPUT_CSV = r"extracted_columns_output.csv"  # CHANGE THIS if needed

# Subject identifier column name
SUBJECT_COL = "Subject_Code"  # CHANGE THIS if different

# ----------------  
# Step 1: Create list of tuples from filtered_df
# ----------------
# Generate tuples: (parameter_hemisphere, sheet_name)
tuples_list = []

for idx, row in filtered_df.iterrows():
    param = row[param_col]
    hemi = row['Hemisphere'].lower()  # Ensure lowercase (lh/rh)
    param_hemisphere = f"{param}_{hemi}"
    tuples_list.append((param_hemisphere, SHEET_NAME))

print(f"Generated {len(tuples_list)} tuples:")
print("\nAll tuples:")
for i, tup in enumerate(tuples_list, 1):
    print(f"  {i}. {tup}")

# Display as a list you can copy
print(f"\n{'='*70}")
print("TUPLES LIST (copy this):")
print(f"{'='*70}")
print("tuples_list = [")
for tup in tuples_list:
    print(f"    {tup},")
print("]")

# ----------------  
# Step 2: Helper function to find columns (case-insensitive)
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

# ----------------  
# Step 3: Extract columns from Excel files
# ----------------
def extract_columns_from_excel_files(folder_path, tuples_list, sheet_name, subject_col="Subject_Code"):
    """
    Extract columns from Excel files based on the tuples list.
    
    Returns:
        DataFrame with one row per subject and columns for each tuple's column_name
    """
    folder = Path(folder_path)
    excel_files = list(folder.glob("*.xlsx")) + list(folder.glob("*.xls"))
    
    if not excel_files:
        print(f"No Excel files found in {folder_path}")
        return pd.DataFrame()
    
    print(f"\n{'='*70}")
    print(f"Found {len(excel_files)} Excel files")
    print(f"{'='*70}")
    
    # Get all unique column names from tuples
    column_names = [tup[0] for tup in tuples_list]
    unique_columns = sorted(set(column_names))
    
    print(f"\nLooking for {len(unique_columns)} unique columns in sheet '{sheet_name}'")
    
    rows_by_subject = {}
    
    for excel_file in excel_files:
        print(f"\nProcessing: {excel_file.name}")
        
        try:
            # Read the specified sheet
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            
            # Find subject column
            subject_col_found = pick_column(df, [subject_col, "Subject_Code", "subject_code", "Subject", "subject"])
            if not subject_col_found:
                # Try first column as subject identifier
                subject_col_found = df.columns[0]
                print(f"  Using first column '{subject_col_found}' as subject identifier")
            
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
                    # Try exact match first
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
# Step 4: Run the extraction
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













