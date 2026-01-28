# ----------------  
# Generate tuples list from filtered_df
# ----------------

# Sheet name to use (modify this based on your actual sheet name)
SHEET_NAME = "cortical_vol_lr"  # CHANGE THIS to your actual sheet name

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

# Display as a Python list you can copy
print(f"\n{'='*70}")
print("TUPLES LIST (Python format - copy this):")
print(f"{'='*70}")
print("tuples_list = [")
for tup in tuples_list:
    print(f"    {tup},")
print("]")

# Also print just the list for easy copying
print(f"\n{'='*70}")
print("Just the list:")
print(f"{'='*70}")
print(tuples_list)













