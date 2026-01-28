# Updated function call with feature importance and upsampling info

results = logistic_regression_cv(
    input_file_path=r"only_Q_outputs/combined\T1_networks_all_brain_all_subjects_with_after_normalized_by_etiv_NO_NAN.csv",
    subject_col="subject_code",
    target_col="after",
    n_folds=5,
    cv_random_state=42,
    results_save_path=r"only_Q_outputs/combined/my_custom_results.csv",
    verbose=True
)

# Access the results
print(f"Mean R²: {results['mean_r2']:.4f}")
print(f"\nResults DataFrame:\n{results['results_df']}")
print(f"\nDisplay DataFrame:\n{results['display_df']}")

# Access feature importance (already printed by function, but you can also access it here)
print(f"\n{'='*70}")
print("FEATURE IMPORTANCE (from results):")
print(f"{'='*70}")
print(results['feature_importance_df'])

# Show upsampling statistics from results_df
print(f"\n{'='*70}")
print("UPSAMPLING STATISTICS:")
print(f"{'='*70}")
print(f"Average training samples before upsampling: {results['results_df']['n_train'].mean():.1f}")
print(f"Average training samples after upsampling: {results['results_df']['n_train_upsampled'].mean():.1f}")
print(f"Upsampling ratio: {results['results_df']['n_train_upsampled'].mean() / results['results_df']['n_train'].mean():.2f}x")

# Show top 5 most important features
print(f"\n{'='*70}")
print("TOP 5 MOST IMPORTANT FEATURES:")
print(f"{'='*70}")
top_features = results['feature_importance_df'].head(5)
for idx, row in top_features.iterrows():
    direction = "increases" if row['coefficient_mean'] > 0 else "decreases"
    print(f"{row['feature']}: coefficient = {row['coefficient_mean']:.4f} (abs: {row['abs_coefficient']:.4f})")
    print(f"  -> Higher {row['feature']} {direction} probability of class 1")
















