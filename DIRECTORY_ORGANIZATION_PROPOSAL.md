# Directory Organization Proposal for Technion Project

## Recommended Structure

```
technion/
│
├── 00_raw_data/                    # Original uploaded data (DO NOT MODIFY)
│   ├── questionnaires/
│   │   └── original_Q_data/        # Move: data/original_Q_data/
│   ├── mri/
│   │   ├── T1/                     # Move: T1/
│   │   ├── RSFC/                   # Move: RSFC/
│   │   └── cat12_outputs/          # Move: cat12 outputs/
│   └── clinical/
│       └── [Excel files from root]
│
├── 01_preprocessing/               # Data preprocessing outputs
│   ├── T1/
│   │   ├── ses1/                   # Move: data/stats_t1_fs/ses1/
│   │   ├── ses2/                   # Move: data/stats_t1_fs/ses2/
│   │   └── longitudinal/          # Move: longitude_stats_without_normalizations/
│   ├── questionnaires/
│   │   ├── timepoints/             # Split questionnaire data by timepoint
│   │   └── merged/                 # Merged questionnaire data
│   └── diffusion_maps/
│       └── SCHAEFER_mat_cor/       # Move: data/SCHAEFER_mat_cor/
│
├── 02_analysis/                     # Main analysis notebooks
│   ├── questionnaires/
│   │   ├── only_q_combined.ipynb
│   │   ├── only_q.ipynb
│   │   ├── only_q_2_time_points.ipynb
│   │   ├── only_q_3_time_points.ipynb
│   │   └── only_q_seperated.ipynb
│   ├── regression/
│   │   ├── regression_combined.ipynb
│   │   ├── regression_combined_2.ipynb
│   │   ├── regression_clinical_combined.ipynb
│   │   └── regression_without_pca.ipynb
│   ├── T1/
│   │   ├── prep_T1_first_ses.ipynb
│   │   ├── prep_T1_2_ses.ipynb
│   │   ├── T1_yeo_diffution.ipynb
│   │   └── transition_analsyis.ipynb
│   ├── clustering/
│   │   └── clustering_reiman.ipynb
│   └── other/
│       ├── trajectory_corr.ipynb
│       ├── corr_manifold.ipynb
│       └── analsyis_in the combined_datatset.ipynb
│
├── 03_outputs/                      # Generated analysis outputs
│   ├── questionnaires/
│   │   ├── combined/               # Move: only_Q_outputs/combined/
│   │   ├── 5_timepoints/          # Move: only_Q_outputs/5_timepoints/
│   │   ├── 2_timepoints/         # Move: only_Q_outputs/2_timepoints/
│   │   └── separated/             # Move: only_Q_outputs/seperated/
│   ├── regression/
│   │   ├── summaries/             # Regression summary tables
│   │   ├── feature_importance/    # Feature importance CSVs
│   │   └── ttest_results/        # T-test results
│   ├── clustering/
│   │   └── cluster_assignments/   # Cluster CSV files
│   └── T1/
│       ├── normalized/             # Normalized T1 data
│       └── diffusion/             # Diffusion map outputs
│
├── 04_results/                      # Final results and summaries
│   ├── tables/                     # Final summary tables
│   ├── figures/                    # Final publication-ready figures
│   └── reports/                    # Analysis reports
│
├── 05_scripts/                      # Reusable Python functions
│   ├── preprocessing/
│   │   ├── preprocessing_functions.py
│   │   └── subject_check.py
│   ├── analysis/
│   │   ├── clustring_functions.py
│   │   ├── pca_functions.py
│   │   ├── regression_functions.py
│   │   └── questionnaire_functions.py
│   ├── visualization/
│   │   ├── vizualizations_functions.py
│   │   └── viz_diff_map.py
│   └── diffusion/
│       ├── diffution_map_functions.py
│       └── diffution_clustring.py
│
├── 06_archive/                      # Old/unused scripts
│   └── archive_scripts/            # Move: archive_scripts/
│
├── 07_documentation/                # Documentation files
│   ├── csv_analysis_report.txt
│   └── README.md
│
└── 08_temp/                         # Temporary/intermediate files
    └── [Files that can be regenerated]
```

## Detailed Organization Plan

### 1. Raw Data (00_raw_data/)
**Purpose**: Store original, unmodified data files
- **DO NOT** modify files here
- These are your source data
- Keep backups of this directory

**Files to move**:
- `data/original_Q_data/` → `00_raw_data/questionnaires/original_Q_data/`
- `T1/` → `00_raw_data/mri/T1/`
- `RSFC/` → `00_raw_data/mri/RSFC/`
- `cat12 outputs/` → `00_raw_data/mri/cat12_outputs/`
- Root Excel files → `00_raw_data/clinical/`

### 2. Preprocessing (01_preprocessing/)
**Purpose**: Outputs from data preprocessing steps
- Cleaned/processed versions of raw data
- Can be regenerated from raw data

**Files to move**:
- `data/stats_t1_fs/` → `01_preprocessing/T1/`
- `longitude_stats_without_normalizations/` → `01_preprocessing/T1/longitudinal/`
- `data/merged_no_nan.csv` → `01_preprocessing/questionnaires/merged/`
- `data/SCHAEFER_mat_cor/` → `01_preprocessing/diffusion_maps/SCHAEFER_mat_cor/`

### 3. Analysis Notebooks (02_analysis/)
**Purpose**: All Jupyter notebooks organized by analysis type
- Group related analyses together
- Makes it easy to find the right notebook

**Organization**:
- Questionnaires: All `only_q*.ipynb` files
- Regression: All `regression*.ipynb` files
- T1: All T1-related notebooks
- Clustering: Clustering analyses
- Other: Miscellaneous analyses

### 4. Outputs (03_outputs/)
**Purpose**: Generated analysis outputs organized by type
- Can be regenerated by running notebooks
- Organized by analysis type

**Files to move**:
- `only_Q_outputs/combined/` → `03_outputs/questionnaires/combined/`
- `only_Q_outputs/5_timepoints/` → `03_outputs/questionnaires/5_timepoints/`
- `only_Q_outputs/2_timepoints/` → `03_outputs/questionnaires/2_timepoints/`
- `only_Q_outputs/seperated/` → `03_outputs/questionnaires/separated/`
- Regression outputs → `03_outputs/regression/`
- Root CSV files (if outputs) → `03_outputs/`

### 5. Results (04_results/)
**Purpose**: Final, curated results
- Summary tables
- Publication-ready figures
- Final reports
- Files you want to keep long-term

**Files to move**:
- Important summary CSVs
- Final figures (move from root)
- Analysis reports

### 6. Scripts (05_scripts/)
**Purpose**: Reusable Python functions
- Organized by functionality
- Easy to import in notebooks

**Files to move**:
- All `*_functions.py` files → appropriate subdirectories
- `subject_check.py` → `05_scripts/preprocessing/`
- `viz_diff_map.py` → `05_scripts/visualization/`

### 7. Archive (06_archive/)
**Purpose**: Old/unused code
- Keep for reference but not active use

**Files to move**:
- `archive_scripts/` → `06_archive/archive_scripts/`

### 8. Documentation (07_documentation/)
**Purpose**: Project documentation
- Reports
- README files
- Analysis documentation

### 9. Temp (08_temp/)
**Purpose**: Temporary/intermediate files
- Files that can be safely deleted
- Can be regenerated

## Migration Strategy

### Phase 1: Create New Structure
1. Create all new directories
2. Don't move files yet

### Phase 2: Move Raw Data First
1. Move `00_raw_data/` files
2. Verify nothing breaks

### Phase 3: Move Scripts
1. Move Python scripts to `05_scripts/`
2. Update import paths in notebooks

### Phase 4: Move Notebooks
1. Move notebooks to `02_analysis/`
2. Update file paths in notebooks

### Phase 5: Move Outputs
1. Move generated outputs
2. Update paths in notebooks

### Phase 6: Clean Up
1. Move final results
2. Delete temporary files
3. Update documentation

## Path Update Checklist

After moving files, you'll need to update paths in:

1. **Notebooks**:
   - `only_q_combined.ipynb` - Update paths to `03_outputs/questionnaires/combined/`
   - `transition_analsyis.ipynb` - Update T1 and output paths
   - `regression_*.ipynb` - Update regression output paths
   - All notebooks using `data/` paths

2. **Python Scripts**:
   - Update import statements
   - Update file paths in functions

3. **Configuration**:
   - Update any config files
   - Update .gitignore if needed

## Benefits of This Structure

1. **Clear Separation**: Raw data vs. processed vs. outputs
2. **Easy Navigation**: Find files by purpose
3. **Safe Deletion**: Know what can be regenerated
4. **Scalability**: Easy to add new analyses
5. **Collaboration**: Others can understand the structure
6. **Backup Strategy**: Know what to backup (raw data + code)

## Alternative: Simpler Structure

If the above is too complex, here's a simpler version:

```
technion/
├── data/
│   ├── raw/              # Original data
│   └── processed/        # Preprocessed data
├── notebooks/            # All notebooks
├── outputs/              # All outputs
├── scripts/             # Python functions
└── results/             # Final results
```

## Recommendations

1. **Start Small**: Begin with moving raw data and scripts
2. **Update Gradually**: Move one category at a time
3. **Test After Each Move**: Run notebooks to ensure paths work
4. **Keep Backup**: Backup before major reorganization
5. **Document Changes**: Update README with new structure

Would you like me to:
1. Create a script to automate the file moves?
2. Create a detailed migration plan for specific files?
3. Help update paths in notebooks after reorganization?


















