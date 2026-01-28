# Code Snippet to Function Mapping

This document maps specific code snippets from `transition_analsyis.ipynb` to functions that should be extracted to `transition_analsyis_functions.py`.

## 1. Trajectory Analysis Functions

### Function: `create_trajectory_pattern(row, timepoint_cols)`
**Location:** Cell 6 (lines ~273-277) and Cell 10 (lines ~1112-1116) - **DUPLICATE CODE**
```python
def create_trajectory_pattern(row, timepoint_cols):
    """Create a string representation of the trajectory for sorting"""
    return '-'.join([str(int(row[col])) if not pd.isna(row[col]) else 'N' 
                     for col in timepoint_cols])
```
**Usage:** Applied to dataframe to create trajectory pattern column
```python
output_df['trajectory_pattern'] = output_df.apply(
    lambda row: create_trajectory_pattern(row, timepoint_labels), axis=1
)
```

### Function: `count_cluster_changes(row, timepoint_cols)`
**Location:** Cell 6 (lines ~283-293) and Cell 10 (lines ~1122-1132) - **DUPLICATE CODE**
```python
def count_cluster_changes(row, timepoint_cols):
    """Count the number of times cluster changes between consecutive timepoints"""
    changes = 0
    values = [row[col] for col in timepoint_cols]
    for i in range(len(values) - 1):
        if not (pd.isna(values[i]) or pd.isna(values[i+1])):
            if values[i] != values[i+1]:
                changes += 1
    return changes
```
**Usage:** Applied to dataframe to count changes
```python
output_df['n_changes'] = output_df.apply(
    lambda row: count_cluster_changes(row, timepoint_labels), axis=1
)
```

### Function: `get_transition_changes(row, timepoint_cols)`
**Location:** Cell 6 (lines ~298-308) and Cell 10 (lines ~1137-1147) - **DUPLICATE CODE**
```python
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
```
**Usage:** Used to create change columns for each transition
```python
for i, trans_label in enumerate(transition_labels):
    output_df[f'change_{i}'] = output_df.apply(
        lambda row: get_transition_changes(row, timepoint_labels)[i], axis=1
    )
```

### Function: `calculate_trajectory_metrics(df, timepoint_cols, subject_col)`
**Location:** Cell 6 and Cell 10 - **COMBINE MULTIPLE OPERATIONS**
**Extract this entire block:**
```python
# Create trajectory pattern
output_df['trajectory_pattern'] = output_df.apply(
    lambda row: create_trajectory_pattern(row, timepoint_labels), axis=1
)

# Count cluster changes
output_df['n_changes'] = output_df.apply(
    lambda row: count_cluster_changes(row, timepoint_labels), axis=1
)

# Add columns for each transition
for i, trans_label in enumerate(transition_labels):
    output_df[f'change_{i}'] = output_df.apply(
        lambda row: get_transition_changes(row, timepoint_labels)[i], axis=1
    )

# Calculate number of changes at each transition point
transition_counts = []
for i in range(len(transition_labels)):
    count = output_df[f'change_{i}'].sum()
    transition_counts.append(count)
```
**Returns:** DataFrame with added columns: `trajectory_pattern`, `n_changes`, `change_0`, `change_1`, `change_2`, `change_3`, plus dict of transition_counts

---

## 2. Transition Column Creation

### Function: `add_transition_col(df, from_col, to_col, new_col_name)`
**Location:** Cell 4 (lines ~60-75) - **ALREADY EXISTS AS LOCAL FUNCTION**
```python
def add_transition_col(df_, from_col, to_col, new_col_name):
    if from_col not in df_.columns or to_col not in df_.columns:
        raise ValueError(
            f"Missing '{from_col}' or '{to_col}' for {new_col_name}. "
            f"Available columns: {list(df_.columns)}"
        )

    a = pd.to_numeric(df_[from_col], errors="coerce")
    b = pd.to_numeric(df_[to_col], errors="coerce")

    conditions = [
        (a == 0) & (b == 0),
        (a == 0) & (b == 1),
        (a == 1) & (b == 0),
        (a == 1) & (b == 1),
    ]
    choices = ["stay_good", "worsen", "improve", "stay_bad"]

    df_[new_col_name] = np.select(conditions, choices, default="missing")
    return df_
```

### Function: `add_all_transition_columns(df, timepoint_cols, prefix="")`
**Location:** Cell 4 (lines ~77-85)
**Extract this pattern:**
```python
transitions = [
    ("b", "t1", "B_TO_T1"),
    ("t1", "t2", "T1_TO_T2"),
    ("t2", "t3", "T2_TO_T3"),
    ("t3", "after", "T3_TO_AFTER"),
]

for f, t, name in transitions:
    out_df = add_transition_col(out_df, f, t, name)
```
**Should auto-generate transition pairs from timepoint_cols list**

---

## 3. Visualization Functions

### Function: `create_trajectory_heatmap(df, timepoint_cols, subject_col, title_suffix="", show_proportions=True)`
**Location:** Cell 6 (lines ~320-420) and Cell 10 (lines ~1160-1260) - **LARGE DUPLICATE BLOCK**

**Extract this entire visualization block:**
```python
# Prepare data for plotting
timepoint_labels = ["b", "t1", "t2", "t3", "after"]
transition_labels = ["b→t1", "t1→t2", "t2→t3", "t3→after"]

# Sort subjects
output_df['sort_key'] = output_df['n_changes'].apply(lambda x: 0 if x == 1 else 1)
output_df_sorted = output_df.sort_values(
    by=['sort_key', 'b', 'trajectory_pattern', 'n_changes', subject_col]
).reset_index(drop=True)

# Calculate proportions
proportions = []
for col in timepoint_labels:
    prop = output_df[col].mean()
    proportions.append(prop)

# Prepare data for heatmap
heatmap_data = output_df_sorted[timepoint_labels].values
proportions_row = np.array([proportions])
heatmap_data_with_props = np.vstack([heatmap_data, proportions_row])

# Create labels
subject_labels = list(output_df_sorted[subject_col].values) + ['Proportion']
subject_labels_array = np.array(subject_labels)

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(10, max(10, len(output_df) * 0.35)))

# Create custom colormap
colors = ['green', 'red']
cmap = ListedColormap(colors)

# Create annotation array
annot_data = []
for i in range(len(heatmap_data_with_props)):
    if i < len(output_df):
        annot_data.append([f'{int(val)}' for val in heatmap_data_with_props[i]])
    else:
        annot_data.append([f'{val:.2f}' for val in heatmap_data_with_props[i]])
annot_data = np.array(annot_data)

# Create heatmap
heatmap_obj = sns.heatmap(heatmap_data_with_props, 
            annot=annot_data,
            fmt='',
            cmap=cmap,
            vmin=0, vmax=1,
            cbar_kws={'label': 'Cluster (0=Good, 1=Bad)', 'ticks': [0, 1]},
            yticklabels=subject_labels_array,
            xticklabels=timepoint_labels,
            linewidths=0.5,
            linecolor='gray',
            ax=ax,
            annot_kws={'size': 14, 'weight': 'bold'})

# Recolor proportions row
proportions_row_idx = len(output_df)
quadmesh = heatmap_obj.collections[0]
face_colors = quadmesh.get_facecolors()
n_cols = len(timepoint_labels)
start_idx = proportions_row_idx * n_cols
end_idx = start_idx + n_cols
lightblue = [0.68, 0.85, 1.0, 1.0]
for idx in range(start_idx, min(end_idx, len(face_colors))):
    face_colors[idx] = lightblue
quadmesh.set_facecolors(face_colors)

ax.set_ylabel('Subject', fontsize=16)
ax.set_xlabel('Timepoint', fontsize=16)
plt.yticks(rotation=0, fontsize=14)
plt.xticks(rotation=0, fontsize=16)
plt.tight_layout()
plt.show()
```

### Function: `plot_transition_distribution(df, transition_col, after_col, title_suffix="")`
**Location:** Cell 8 (entire cell) - **LOOP PATTERN TO EXTRACT**
**Extract the plotting logic inside the loop:**
```python
for col in transition_cols:
    # counts
    counts = pd.crosstab(df2[col], df2["after"]).reindex(columns=[0, 1], fill_value=0)
    counts.columns = ["after_0_n", "after_1_n"]
    counts["n_total"] = counts["after_0_n"] + counts["after_1_n"]

    # percentages
    pct = counts[["after_0_n", "after_1_n"]].div(counts["n_total"], axis=0) * 100
    pct.columns = ["after_0_%", "after_1_%"]

    # combined table
    summary = pd.concat([counts, pct], axis=1)
    summary = summary[["after_0_n", "after_0_%", "after_1_n", "after_1_%", "n_total"]]

    print(f"\n=== {col} (COUNTS + PERCENT) ===")
    print(summary.round({"after_0_%": 1, "after_1_%": 1}))

    # plot percentages
    ax = pct.plot(kind="bar", stacked=True)
    ax.set_title(f"After (0/1) distribution by {col} (percent)")
    ax.set_xlabel(col)
    ax.set_ylabel("Percent of subjects")
    plt.xticks(rotation=45, ha="right")

    # annotate percentages inside bars
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            if height <= 0:
                continue
            if height < 5:
                continue
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_y() + height / 2
            ax.text(x, y, f"{height:.1f}%", ha="center", va="center", fontsize=9)

    # annotate n above each bar
    group_n = counts["n_total"]
    for i, n in enumerate(group_n.values):
        ax.text(i, 102, f"n={int(n)}", ha="center", va="bottom", fontsize=10)

    ax.set_ylim(0, 110)
    plt.tight_layout()
    plt.show()
```

### Function: `plot_network_means_by_group(df, cluster_col, network_cols, subject_col, save_path=None)`
**Location:** Cell 7 (lines ~100-200)
**Extract this entire block:**
```python
# Calculate means by group
group_means = df.groupby(cluster_col)[network_columns].mean()

# Prepare data for plotting
group_means_plot = group_means.reset_index()
df_plot = pd.melt(
    group_means_plot, 
    id_vars=[cluster_col], 
    value_vars=network_columns,
    var_name='Network', 
    value_name='Mean'
)

# Create the plot
fig, ax = plt.subplots(figsize=(14, 8))
sns.barplot(
    data=df_plot,
    x='Network',
    y='Mean',
    hue=cluster_col,
    ax=ax,
    palette='Set2'
)

# Customize the plot
ax.set_xlabel('Network', fontsize=14, fontweight='bold')
ax.set_ylabel('Mean Value', fontsize=14, fontweight='bold')
ax.set_title(f'Network Means by {cluster_col} Groups', fontsize=16, fontweight='bold')
plt.xticks(rotation=45, ha='right')
ax.legend(title=f"'{cluster_col}' Group", title_fontsize=12, fontsize=10)
plt.tight_layout()
plt.show()

# Print summary statistics
for group in sorted(df[cluster_col].unique()):
    group_data = df[df[cluster_col] == group]
    print(f"\nGroup {cluster_col} = {group}:")
    print(f"  Number of subjects: {len(group_data)}")
    print(f"  Network means:")
    for net in network_columns:
        mean_val = group_data[net].mean()
        std_val = group_data[net].std()
        print(f"    {net}: {mean_val:.4f} ± {std_val:.4f}")

# Save the means to CSV if desired
if save_path:
    group_means.to_csv(save_path)
```

---

## 4. Data Processing Functions

### Function: `filter_subjects_by_criteria(df, criteria_dict, subject_col)`
**Location:** Cell 9 (lines ~30-60)
**Extract the filtering pattern:**
```python
# Convert columns to numeric
df[after_col] = pd.to_numeric(df[after_col], errors="coerce")
df[subject_col] = df[subject_col].astype(str)

# Filter subjects with after = 1
n_before = len(df)
df_filtered = df[df[after_col] == 1].copy()
n_after = len(df_filtered)

print(f"Subjects before filtering: {n_before}")
print(f"Subjects with {after_col} = 1: {n_after}")
print(f"Subjects removed: {n_before - n_after}")
```
**Should accept criteria_dict like:** `{"after": 1}` or `{"after": [0, 1]}`

### Function: `extract_subject_clusters(df, subject_col, cluster_cols)`
**Location:** Cell 15 (entire cell)
**Extract this pattern:**
```python
# Check which cluster columns exist
existing_cluster_cols = [col for col in cluster_cols if col in df.columns]
missing_cluster_cols = [col for col in cluster_cols if col not in df.columns]

# Select only subject code and cluster columns
result_df = df[[subject_col] + existing_cluster_cols].copy()

# Convert subject code to string
result_df[subject_col] = result_df[subject_col].astype(str)

# Convert cluster columns to numeric
for col in existing_cluster_cols:
    result_df[col] = pd.to_numeric(result_df[col], errors="coerce")
```

---

## 5. Analysis Functions

### Function: `analyze_transition_distribution(df, transition_cols, after_col)`
**Location:** Cell 8 (analysis part)
**Extract the analysis logic:**
```python
# counts
counts = pd.crosstab(df2[col], df2["after"]).reindex(columns=[0, 1], fill_value=0)
counts.columns = ["after_0_n", "after_1_n"]
counts["n_total"] = counts["after_0_n"] + counts["after_1_n"]

# percentages
pct = counts[["after_0_n", "after_1_n"]].div(counts["n_total"], axis=0) * 100
pct.columns = ["after_0_%", "after_1_%"]

# combined table
summary = pd.concat([counts, pct], axis=1)
summary = summary[["after_0_n", "after_0_%", "after_1_n", "after_1_%", "n_total"]]
```
**Returns:** Dictionary with summary DataFrames for each transition

### Function: `summarize_trajectory_groups(df, timepoint_cols, subject_col, change_groups=None)`
**Location:** Cell 6 (lines ~430-480) and Cell 10 (lines ~1270-1320)
**Extract this block:**
```python
# Group subjects by number of changes
change_groups = {
    0: '0 changes (stable)',
    1: '1 change',
    2: '2 changes',
    3: '3+ changes'
}

output_df['change_group'] = output_df['n_changes'].apply(assign_group)

# Display group counts
group_counts = output_df['change_group'].value_counts()
ordered_groups = [change_groups[0], change_groups[1], change_groups[2], change_groups[3]]
group_counts_ordered = group_counts.reindex([g for g in ordered_groups if g in group_counts.index], fill_value=0)
group_percentages = (group_counts_ordered / len(output_df) * 100).round(1)

summary_df = pd.DataFrame({
    'Group': group_counts_ordered.index,
    'Count': group_counts_ordered.values,
    'Percentage': group_percentages.values
})

# Show examples from each group
for group_name in ordered_groups:
    group_subjects = output_df[output_df['change_group'] == group_name]
    if len(group_subjects) > 0:
        print(f"\n{group_name} (n={len(group_subjects)}):")
        examples = group_subjects.head(3)
        for idx, row in examples.iterrows():
            trajectory_str = ' -> '.join([f"{row[col]:.0f}" for col in timepoint_labels])
            print(f"  {row[subject_col]}: {trajectory_str} (changes: {row['n_changes']})")
```
**Note:** Uses `assign_group` function which already exists in `transition_analsyis_functions.py` (line 1516)

---

## 6. Transition Analysis Summary

### Function: `print_transition_analysis(df, transition_labels, change_cols)`
**Location:** Cell 6 (lines ~310-320) and Cell 10 (lines ~1150-1160)
**Extract this block:**
```python
# Calculate number of changes at each transition point
transition_counts = []
for i in range(len(transition_labels)):
    count = output_df[f'change_{i}'].sum()
    transition_counts.append(count)

# Find which transition has the most changes
max_transition_idx = np.argmax(transition_counts)
max_transition = transition_labels[max_transition_idx]

print(f"\nTransition Analysis:")
print(f"{'='*70}")
for i, (trans, count) in enumerate(zip(transition_labels, transition_counts)):
    marker = " <-- MOST CHANGES" if i == max_transition_idx else ""
    print(f"{trans}: {count} changes ({count/len(output_df)*100:.1f}% of subjects){marker}")
```

---

## Summary

### Functions to Create (Priority Order):

1. **High Priority (Duplicated Code):**
   - `create_trajectory_pattern()` - Used in Cells 6 & 10
   - `count_cluster_changes()` - Used in Cells 6 & 10
   - `get_transition_changes()` - Used in Cells 6 & 10
   - `create_trajectory_heatmap()` - Large duplicate block in Cells 6 & 10

2. **Medium Priority (Reusable Patterns):**
   - `add_transition_col()` - Already exists locally in Cell 4
   - `add_all_transition_columns()` - Wrapper for transition creation
   - `plot_transition_distribution()` - Cell 8 loop pattern
   - `plot_network_means_by_group()` - Cell 7 visualization

3. **Low Priority (Single Use but Useful):**
   - `filter_subjects_by_criteria()` - Cell 9 pattern
   - `extract_subject_clusters()` - Cell 15 pattern
   - `calculate_trajectory_metrics()` - Combines multiple operations
   - `analyze_transition_distribution()` - Cell 8 analysis
   - `summarize_trajectory_groups()` - Cells 6 & 10 summary
   - `print_transition_analysis()` - Cells 6 & 10 analysis

### Existing Functions (Already in transition_analsyis_functions.py):
- `assign_group()` - Already exists (line 1516), used in Cells 6 & 10

