def logistic_regression_cv(
    input_file_path,
    subject_col="subject_code",
    target_col="after",
    n_folds=5,
    cv_random_state=42,
    results_save_path=None,
    verbose=True,
    top_n_features=None
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
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'results_df': DataFrame with results for each fold
        - 'display_df': DataFrame with fold and R² columns plus mean row
        - 'mean_r2': Mean R² value
        - 'std_r2': Standard deviation of R²
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
    # Logistic Regression with Cross-Validation and Upsampling
    # ----------------
    if verbose:
        print(f"\n{'='*70}")
        print("LOGISTIC REGRESSION: Binary Classification with SMOTE Upsampling")
        print(f"Target: {target_col}")
        print(f"Features: {len(feature_columns)} columns")
        print(f"Cross-validation: {n_folds}-fold stratified")
        print(f"{'='*70}")
    
    # Initialize logistic regression model (no class_weight since we're using upsampling)
    logistic_model = LogisticRegression(
        max_iter=1000,
        random_state=cv_random_state,
        solver='lbfgs'
    )
    
    # Initialize SMOTE for upsampling
    smote = SMOTE(random_state=cv_random_state)
    
    # Use StratifiedKFold to ensure balanced class distribution in each fold
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cv_random_state)
    
    # Store results for each fold
    fold_results = []
    
    # Store coefficients from each fold for feature analysis
    all_coefficients = []
    
    # Manual cross-validation to calculate R² for each fold
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_scaled, y_values), 1):
        # Split data
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_values[train_idx], y_values[test_idx]
        
        # Apply upsampling (SMOTE) to training data only
        X_train_upsampled, y_train_upsampled = smote.fit_resample(X_train, y_train)
        
        if verbose:
            print(f"\n--- Fold {fold_idx}/{n_folds} ---")
            print(f"  Training data before upsampling: {len(X_train)} samples")
            print(f"  Training data after upsampling: {len(X_train_upsampled)} samples")
            unique, counts = np.unique(y_train_upsampled, return_counts=True)
            class_dist = dict(zip(unique, counts))
            print(f"  Class distribution after upsampling: {class_dist}")
        
        # Train model on upsampled data
        logistic_model.fit(X_train_upsampled, y_train_upsampled)
        
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
            'n_train_upsampled': len(X_train_upsampled),
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
    
    # Average R² (negative values already set to 0)
    mean_r2 = results_df['r2'].mean()
    std_r2 = results_df['r2'].std()
    
    if verbose:
        print(f"\nR² (pseudo R², negative values set to 0):")
        print(f"  Mean: {mean_r2:.4f} (+/- {std_r2:.4f})")
        print(f"  Range: [{results_df['r2'].min():.4f}, {results_df['r2'].max():.4f}]")
    
    # ----------------
    # Display detailed results table
    # ----------------
    if verbose:
        print(f"\n{'='*70}")
        print("Detailed Results by Fold:")
        print(f"{'='*70}")
    
    # Create a copy of results_df with only fold and r2 columns, and add mean row
    display_df = results_df[['fold', 'r2']].copy()
    
    # Create mean row with only fold and r2
    mean_row = {
        'fold': 'Mean',
        'r2': mean_r2
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
    ax.set_yticklabels(plot_df['feature'])
    
    # Labels and title
    ax.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
    ax.set_ylabel('Features', fontsize=12, fontweight='bold')
    ax.set_title(f'Feature Importance - Logistic Regression\nTarget: {target_col} (Top {len(plot_df)} features)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add grid for better readability
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        value = row['coefficient_mean']
        x_pos = value + (0.02 * max(abs(plot_df['coefficient_mean']))) if value >= 0 else value - (0.02 * max(abs(plot_df['coefficient_mean'])))
        ax.text(x_pos, i, f'{value:.4f}', va='center', 
                ha='left' if value >= 0 else 'right', fontsize=9)
    
    # Invert y-axis so highest importance is at top
    ax.invert_yaxis()
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E86AB', label='Positive (increases class 1 probability)'),
        Patch(facecolor='#A23B72', label='Negative (decreases class 1 probability)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
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
        'scaler': scaler,
        'model': logistic_model,
        'feature_importance_df': feature_importance_df,
        'coefficients_mean': mean_coefficients,
        'coefficients_std': std_coefficients,
        'feature_names': feature_columns
    }


