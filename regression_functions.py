




from scipy.stats import ttest_ind
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LassoCV

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV


def clean_variable_name(var):
    """
    Example:
        after_DERS_total -> DERS
        after_DES_total  -> DES
    """
    parts = var.split("_")
    if len(parts) >= 3:
        return parts[1]  # the middle part
    return var
def compare_clusters_no_fdr(
    df,
    cluster_col,
    value_cols,
    time_point,                    # <--- NEW: pass the time point directly
    n_cols=4,
    csv_path="ttest_results.csv"   # <--- NEW: CSV output file
):
    """
    Compare two clusters using t-tests, create barplots, and export CSV results.

    Parameters
    ----------
    df          : DataFrame
    cluster_col : column with cluster labels
    value_cols  : list of variables to compare
    time_point  : the time point (str or int) that data belongs to
    n_cols      : number of columns in subplot grid
    csv_path    : where to save the CSV results
    """

    df = df.dropna(subset=[cluster_col]).copy()

    # Colors
    cluster_values = sorted(df[cluster_col].unique())
    base_colors = ["#1f77b4", "#17becf"]
    color_map = {cluster_values[i]: base_colors[i % 2] for i in range(len(cluster_values))}

    stats_list = []

    # ---------------------------------------
    # 1. COLLECT ALL STATISTICS
    # ---------------------------------------
    for col in value_cols:
        g0 = df[df[cluster_col] == cluster_values[0]][col].dropna()
        g1 = df[df[cluster_col] == cluster_values[1]][col].dropna()

        t_stat, p = ttest_ind(g0, g1, equal_var=False, nan_policy="omit")

        stats_list.append({
            "parameter": col,
            "time_point": time_point,
            "t_stat": t_stat,
            "p_raw": round(p, 3),
            "ttest_result": f"T = {t_stat:.3f}, p = {p:.3f}",
            "stars": "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        })

    # ---------------------------------------
    # 2. PLOT FIGURE GRID
    # ---------------------------------------
    import math
    n_rows = math.ceil(len(value_cols) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    axes = axes.flatten()

    for ax, s in zip(axes, stats_list):
        col = s["parameter"]
        g0 = df[df[cluster_col] == cluster_values[0]][col].dropna()
        g1 = df[df[cluster_col] == cluster_values[1]][col].dropna()

        means = [g0.mean(), g1.mean()]
        stds  = [g0.std(), g1.std()]
        ns    = [g0.size, g1.size]

        x = np.arange(2)
        bar_colors = [color_map[cluster_values[0]], color_map[cluster_values[1]]]

        ax.bar(x, means, yerr=stds, capsize=6, color=bar_colors, edgecolor="black")

        ax.set_xticks(x)
        ax.set_xticklabels([
            f"{cluster_values[0]}\n(n={ns[0]})",
            f"{cluster_values[1]}\n(n={ns[1]})"
        ])

        y_top = max(m + sd for m, sd in zip(means, stds))
        ax.set_ylim(0, y_top * 1.35)

        ax.text(0.5, y_top * 1.18, s["stars"], ha="center", fontsize=28)

        for xi, m, sd in zip(x, means, stds):
            ax.text(xi, m + sd * 1.02, f"{m:.2f}\n({sd:.2f})",
                    ha="center", color="gray")

        ax.set_title(f"{col} (p={s['p_raw']:.3f})", fontsize=18)
        ax.grid(axis="y", alpha=0.3)

    # Hide unused axes
    for j in range(len(stats_list), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

    # ---------------------------------------
    # 3. SAVE TO CSV
    # ---------------------------------------
    results_df = pd.DataFrame(stats_list)
    results_df.to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path}")

    return results_df


def run_regression_and_plot(
        df,
        pc_cols,
        targets,
        cluster_col="cluster",
        n_cols=3,
        title_prefix="",
        timepoint="",
        save_csv_path=None
):
    """
    Runs LassoCV regression on multiple target variables,
    plots cluster-based barplots,
    and saves regression results to CSV.

    Added:
    -------
    - timepoint column in results table
    - results saved in long format CSV (1 row per target)
    - R2_cv column included
    """

    # -------------------------------
    # 1. PREPARE STORAGE
    # -------------------------------
    prediction_df = df[["Subject_Code", cluster_col]].copy()
    results = {}
    rows_for_csv = []   # <--- New storage for CSV export

    # -------------------------------
    # 2. RUN REGRESSION FOR EACH TARGET
    # -------------------------------
    for target in targets:
        valid_df = df[pc_cols + [target]].dropna()
        X = valid_df[pc_cols]
        y = valid_df[target]

        n_subjects = len(valid_df)    # <--- NEW

        if n_subjects < 5:
            print(f"⚠️ Skipping {target}: too few valid rows ({n_subjects})")
            continue

        model = LassoCV(cv=5, random_state=42)
        model.fit(X, y)

        preds = model.predict(df[pc_cols])
        r2 = model.score(X, y)

        prediction_df[f"pred_{target}"] = preds
        prediction_df[f"R2_{target}"] = r2

        results[target] = {
            "alpha": model.alpha_,
            "intercept": float(model.intercept_),
            "coefficients": list(map(float, model.coef_)),
            "R2_cv": r2,
            "n_subjects": n_subjects      # <--- NEW
        }

        rows_for_csv.append({
            "timepoint": timepoint,
            "target": target,
            "alpha": model.alpha_,
            "intercept": float(model.intercept_),
            "coeff_PC1_PC13": list(map(float, model.coef_)),
            "R2_cv": r2,
            "n_subjects": n_subjects      # <--- NEW
        })


    # -------------------------------
    # 3. SAVE RESULTS TO CSV
    # -------------------------------
    results_df = pd.DataFrame(rows_for_csv)

    if save_csv_path is not None:
        results_df.to_csv(save_csv_path, index=False)
        print(f"📁 Saved regression summary to: {save_csv_path}")

    # # -------------------------------
    # # 4. PLOTTING
    # # -------------------------------
    # pred_cols = [c for c in prediction_df.columns if c.startswith("pred_")]
    # r2_cols = {p: p.replace("pred_", "R2_") for p in pred_cols}
    #
    # n_plots = len(pred_cols)
    # n_rows = math.ceil(n_plots / n_cols)
    #
    # fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    # axes = axes.flatten()
    #
    # for idx, pred_col in enumerate(pred_cols):
    #     ax = axes[idx]
    #
    #     target = pred_col.replace("pred_", "")
    #     r2_col = r2_cols[pred_col]
    #     r2_value = prediction_df[r2_col].iloc[0]
    #
    #     stats = prediction_df.groupby(cluster_col)[pred_col].agg(["mean", "std"])
    #
    #     bar_colors = ["#1f77b4", "#17becf"][:len(stats.index)]
    #
    #     ax.bar(stats.index.astype(str),
    #            stats["mean"],
    #            yerr=stats["std"],
    #            color=bar_colors,
    #            capsize=6,
    #            edgecolor="black")
    #
    #     for i, (cl, row) in enumerate(stats.iterrows()):
    #         ax.text(i, row["mean"] + row["std"] * 1.1,
    #                 f"{row['mean']:.2f} ± {row['std']:.2f}",
    #                 ha="center", fontsize=9)
    #
    #     ax.set_title(f"{title_prefix}{target}\nR²={r2_value:.3f}", fontsize=20)
    #     ax.set_xlabel("Cluster")
    #     ax.set_ylabel("Predicted value")
    #     ax.grid(alpha=0.3)
    #
    # for j in range(len(pred_cols), len(axes)):
    #     axes[j].axis("off")
    #
    # plt.tight_layout()
    # plt.show()

    return prediction_df, results, results_df


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
# def run_lasso_regression_and_plot(
#         df,
#         pc_cols,
#         targets,
#         cluster_col="cluster",
#         alpha=0.1,
#         n_cols=4,
#         plot=True,
#         timepoint="",
#         save_csv_path=None
# ):
#     """
#     Runs LASSO regression for multiple target variables and
#     optionally plots cluster-based barplots. Saves regression results to CSV.

#     Added:
#     -------
#     - timepoint column in results table
#     - results saved in long format CSV (1 row per target)
#     """



#     # -------------------------------
#     # 1. Prepare prediction DF
#     # -------------------------------
#     prediction_df = df[["Subject_Code", cluster_col]].copy()
#     results = {}
#     rows_for_csv = []   # <--- store one row per target for CSV output

#     # -------------------------------
#     # 2. Run regression per target
#     # -------------------------------
#     for target in targets:

#         valid_df = df[pc_cols + [target]].dropna()
#         X = valid_df[pc_cols]
#         y = valid_df[target]

#         if len(valid_df) < 2:
#             print(f"⚠️ Skipping {target}: not enough valid rows ({len(valid_df)}).")
#             continue

#         model = Lasso(alpha=alpha)
#         model.fit(X, y)

#         preds = model.predict(df[pc_cols])
#         r2 = model.score(X, y)

#         prediction_df[f"pred_{target}"] = preds
#         prediction_df[f"R2_{target}"] = r2

#         results[target] = dict(
#             alpha=alpha,
#             intercept=float(model.intercept_),
#             coefficients=list(map(float, model.coef_)),
#             R2=r2
#         )

#         # Save one row per target for CSV
#         rows_for_csv.append({
#             "timepoint": timepoint,
#             "target": target,
#             "alpha": alpha,
#             "intercept": float(model.intercept_),
#             "coeff_PC1_PC13": list(map(float, model.coef_)),
#             "R2_without_cv": r2
#         })

#     # -------------------------------
#     # 3. Convert and save CSV results
#     # -------------------------------
#     results_df = pd.DataFrame(rows_for_csv)

#     if save_csv_path is not None:
#         results_df.to_csv(save_csv_path, index=False)
#         print(f"📁 Saved regression summary to: {save_csv_path}")

#     # -------------------------------
#     # 4. If no plotting requested
#     # -------------------------------
#     if not plot:
#         return prediction_df, results, results_df

#     # -------------------------------
#     # 5. Plotting section
#     # -------------------------------
#     pred_cols = [c for c in prediction_df.columns if c.startswith("pred_")]
#     r2_cols = {p: p.replace("pred_", "R2_") for p in pred_cols}

#     n_plots = len(pred_cols)
#     n_rows = int(np.ceil(n_plots / n_cols))

#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
#     axes = axes.flatten()

#     colors = {0: "blue", 1: "lightblue", "0": "blue", "1": "lightblue"}

#     for idx, pred_col in enumerate(pred_cols):

#         ax = axes[idx]
#         target = pred_col.replace("pred_", "")
#         r2_val = prediction_df[r2_cols[pred_col]].iloc[0]

#         stats = prediction_df.groupby(cluster_col)[pred_col].agg(["mean", "std"])
#         bar_colors = [colors.get(cl, "gray") for cl in stats.index]

#         ax.bar(
#             stats.index.astype(str),
#             stats["mean"],
#             yerr=stats["std"],
#             capsize=6,
#             color=bar_colors,
#             edgecolor="black"
#         )

#         ax.set_title(f"{target} — R²={r2_val:.3f}", fontsize=20)
#         ax.set_xlabel("Cluster")
#         ax.set_ylabel("Predicted score")

#         for i, (cluster_value, row) in enumerate(stats.iterrows()):
#             ax.text(
#                 i,
#                 row["mean"] + row["std"] + 0.02 * stats["mean"].max(),
#                 f"{row['mean']:.2f} ± {row['std']:.2f}",
#                 ha="center",
#                 fontsize=10
#             )

#         ax.grid(alpha=0.3)

#     # Hide unused axes
#     for i in range(len(pred_cols), len(axes)):
#         axes[i].axis("off")

#     plt.tight_layout()
#     plt.show()

#     return prediction_df, results, results_df

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def run_lasso_cv_scatter(
        df,
        pc_cols,
        targets,
        n_cols=4,
        save_predictions_path=None,
        save_summary_path=None,
        plot=True
):
    """
    Runs LassoCV regression on multiple target variables.
    Saves a summary CSV with:
        - target
        - R2_cv

    Optionally plots true vs predicted scatter plots.
    """


    # -----------------------------------------
    # 1. Prepare prediction DF
    # -----------------------------------------
    prediction_df = df[["Subject_Code"]].copy()
    results = {}
    summary_rows = []  # rows for summary CSV


    # -----------------------------------------
    # 2. Fit LassoCV for each target
    # -----------------------------------------
    for target in targets:

        valid_df = df[pc_cols + [target]].dropna()

        if valid_df.empty:
            print(f"⚠️ Skipping {target}: no valid rows.")
            continue

        X = valid_df[pc_cols]
        y = valid_df[target]

        # --- Updated model: StandardScaler + LassoCV ---
        model = make_pipeline(
            StandardScaler(),
            LassoCV(cv=5, random_state=42, max_iter=50000)
        )

        model.fit(X, y)

        # Pipeline handles scaling automatically
        preds = model.predict(df[pc_cols])
        r2 = model.score(X, y)

        prediction_df[f"pred_{target}"] = preds
        prediction_df[f"R2_{target}"] = r2

        results[target] = dict(R2_cv=r2)

        # Save ONLY target + R2_cv into summary
        summary_rows.append({
            "target": target,
            "R2_cv": r2
        })


    # -----------------------------------------
    # 3. Create summary DataFrame
    # -----------------------------------------
    results_df = pd.DataFrame(summary_rows)


    # -----------------------------------------
    # 4. Save files if requested
    # -----------------------------------------
    if save_predictions_path is not None:
        prediction_df.to_csv(save_predictions_path, index=False)
        print("Saved predictions to:", save_predictions_path)

    if save_summary_path is not None:
        # Always save as .csv, add .csv if missing
        if not save_summary_path.lower().endswith(".csv"):
            save_summary_path += ".csv"

        results_df.to_csv(save_summary_path, index=False)
        print("Saved summary (target, R2_cv) to:", save_summary_path)


    # -----------------------------------------
    # 5. Scatter plotting (optional)
    # -----------------------------------------
    if not plot:
        return prediction_df, results, results_df

    pred_cols = [c for c in prediction_df.columns if c.startswith("pred_")]
    r2_cols = {p: p.replace("pred_", "R2_") for p in pred_cols}

    df_pred = prediction_df.copy()
    df_true = df[targets].copy()

    n_plots = len(pred_cols)
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    axes = axes.flatten()

    for idx, pred_col in enumerate(pred_cols):

        ax = axes[idx]
        target = pred_col.replace("pred_", "")

        r2_col = r2_cols[pred_col]

        if r2_col not in df_pred.columns:
            ax.axis("off")
            continue

        r2_value = df_pred[r2_col].iloc[0]

        # Extract true + predicted
        if target not in df_true.columns:
            ax.axis("off")
            continue

        y_true = df_true[target]
        y_pred = df_pred[pred_col]

        mask = y_true.notna() & y_pred.notna()
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        ax.scatter(y_true, y_pred, alpha=0.6)

        # Regression line
        if len(y_true) > 1:
            coef = np.polyfit(y_true, y_pred, 1)
            xline = np.linspace(y_true.min(), y_true.max(), 100)
            yline = coef[0] * xline + coef[1]
            ax.plot(xline, yline, color="red", linewidth=2)

        ax.set_xlabel(f"True {target}", fontsize=18)
        ax.set_ylabel(f"Predicted {target}", fontsize=18)
        ax.set_title(f"{target}\nR² = {r2_value:.3f}", fontsize=20)
        ax.grid(alpha=0.3)

    # Hide unused axes
    for j in range(len(pred_cols), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

    return prediction_df, results, results_df

def run_lasso_scatter(
        df,
        pc_cols,
        targets,
        alpha=0.1,
        n_cols=4,
        save_predictions_path=None,
        save_summary_path=None,
        plot=True
):
    """
    Runs Lasso regression (NO cross-validation) on multiple target variables
    and optionally plots scatter plots.

    Summary CSV now contains ONLY:
        - target
        - R2_without_cv
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import Lasso

    # ---------------------------------------------------------
    # 1. Prepare prediction DF
    # ---------------------------------------------------------
    prediction_df = df[["Subject_Code"]].copy()
    results = {}
    summary_rows = []   # <--- ONLY target + R2_without_cv

    # ---------------------------------------------------------
    # 2. Fit Lasso for each target
    # ---------------------------------------------------------
    for target in targets:

        valid_df = df[pc_cols + [target]].dropna()

        if valid_df.empty:
            print(f"⚠️ Skipping {target}: no valid rows.")
            continue

        X = valid_df[pc_cols]
        y = valid_df[target]

        model = Lasso(alpha=alpha)
        model.fit(X, y)

        preds = model.predict(df[pc_cols])
        r2 = model.score(X, y)

        prediction_df[f"pred_{target}"] = preds
        prediction_df[f"R2_{target}"] = r2

        # store full results internally
        results[target] = dict(
            R2_without_cv=r2
        )

        # store minimal summary row
        summary_rows.append({
            "target": target,
            "R2_without_cv": r2
        })

    # ---------------------------------------------------------
    # 3. Create long-format summary DataFrame
    # ---------------------------------------------------------
    results_df = pd.DataFrame(summary_rows)

    # ---------------------------------------------------------
    # 4. Save CSV files if requested
    # ---------------------------------------------------------
    if save_predictions_path is not None:
        prediction_df.to_csv(save_predictions_path, index=False)
        print("Saved predictions to:", save_predictions_path)

    if save_summary_path is not None:
        results_df.to_csv(save_summary_path, index=False)
        print("Saved summary (target, R2_without_cv) to:", save_summary_path)

    # ---------------------------------------------------------
    # 5. PLOTTING
    # ---------------------------------------------------------
    if not plot:
        return prediction_df, results, results_df

    pred_cols = [c for c in prediction_df.columns if c.startswith("pred_")]
    r2_cols = {p: p.replace("pred_", "R2_") for p in pred_cols}

    df_pred = prediction_df.copy()
    df_true = df[targets].copy()

    n_plots = len(pred_cols)
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    axes = axes.flatten()

    for idx, pred_col in enumerate(pred_cols):

        ax = axes[idx]
        target = pred_col.replace("pred_", "")
        r2_col = r2_cols[pred_col]

        if r2_col not in df_pred.columns:
            ax.axis("off")
            continue

        r2_value = df_pred[r2_col].iloc[0]

        if target not in df_true.columns:
            ax.axis("off")
            continue

        y_true = df_true[target]
        y_pred = df_pred[pred_col]

        mask = y_true.notna() & y_pred.notna()
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        ax.scatter(y_true, y_pred, alpha=0.6)

        # regression line
        if len(y_true) > 1:
            coef = np.polyfit(y_true, y_pred, 1)
            xline = np.linspace(y_true.min(), y_true.max(), 100)
            yline = coef[0] * xline + coef[1]
            ax.plot(xline, yline, color="red", linewidth=2)

        ax.set_title(f"{target}\nR² = {r2_value:.3f}", fontsize=20)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.grid(alpha=0.3)

    # hide unused panels
    for j in range(len(pred_cols), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

    return prediction_df, results, results_df

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
import os 
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
import os

def run_regression_with_feature_importance(
        df,
        feature_cols,
        targets,
        title_prefix="",
        save_csv_path=None,  # This will be used for the final summary (results_df)
        save_plots=False,
        # --- NEW ARGUMENT ADDED ---
        save_target_csv_dir=None # Directory to save individual target importance CSV files
        # --------------------------
):
    """
    Runs LassoCV regression on multiple targets.
    Saves a separate feature importance CSV for each target if save_target_csv_dir is provided.
    """

    results = []
    prediction_df = df[["Subject_Code"]].copy()

    for target in targets:

        # --- Prepare data ---
        valid_df = df[feature_cols + [target]].dropna()
        X = valid_df[feature_cols]
        y = valid_df[target]

        n_subjects = len(valid_df)
        if n_subjects < 5:
            print(f"⚠️ Skipping {target}: fewer than 5 rows ({n_subjects})")
            continue

        # --- Fit model ---
        model = LassoCV(cv=5, random_state=42)
        model.fit(X, y)

        preds = model.predict(df[feature_cols])
        r2 = model.score(X, y)

        prediction_df[f"pred_{target}"] = preds
        prediction_df[f"R2_{target}"] = r2

        # --- Feature importance DF for the CURRENT target ---
        coefs = np.array(model.coef_, dtype=float)
        abs_importance = coefs

        # This DataFrame holds the results for the current target only
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "coefficient": coefs,
            "importance": abs_importance
        })
        
        # FILTER: keep only non-zero importance for display/plotting/saving
        importance_nonzero = importance_df[importance_df["importance"] != 0].copy()

        # ---------------------------
        # NEW: Save individual TARGET CSV file
        # ---------------------------
        if save_target_csv_dir:
            os.makedirs(save_target_csv_dir, exist_ok=True)
            
            # Use the full importance_df, but you might prefer to save only non-zero features:
            df_to_save = importance_df # Save all features (zero and non-zero)
            # OR: df_to_save = importance_nonzero # Save only non-zero features
            
            safe_target_name = target.replace(" ", "_").replace("/", "-")
            file_path = os.path.join(save_target_csv_dir, f"{safe_target_name}_feature_importance.csv")
            df_to_save.to_csv(file_path, index=False)
            print(f"📁 Saved feature importance for Target '{target}' to: {file_path}")
            
        # Print the value for each non-zero predictor for this target
        print(f"\n✅ Non-Zero Predictors for Target: {target}")
        for _, row in importance_nonzero.iterrows():
            print(f"   Feature: {row['feature']:<20} | Coefficient: {row['coefficient']:>10.4f} | Importance: {row['importance']:>8.4f}")


        # Save results for target-level summary table (the original results list)
        results.append({
            "target": target,
            "alpha": model.alpha_,
            "intercept": float(model.intercept_),
            "R2_cv": r2,
            "n_subjects": n_subjects,
            "coefficients": list(coefs),
            "importance": list(abs_importance),
            "feature_names": feature_cols
        })

        # ---------------------------
        # PLOT ONLY IF non-zero importance exists AND we are saving plots
        # ---------------------------
        if importance_nonzero.empty:
            print(f"📉 No non-zero features for {target} → SKIPPING plot.")
            continue
        
        if save_plots: # Plotting logic only runs if save_plots is True
            importance_nonzero = importance_nonzero.sort_values(
                "importance", ascending=False
            )
    
            plt.figure(figsize=(10, 6))
            plt.barh(
                importance_nonzero["feature"],
                importance_nonzero["importance"]
            )
            plt.title(f"{title_prefix}{target} - Non-zero Feature Importance")
            plt.xlabel("Absolute Importance")
            plt.gca().invert_yaxis()
            plt.grid(alpha=0.3)
    
            # save_plots is True, so we save the file
            plt.savefig(f"feature_importance_{target}.png", dpi=200, bbox_inches='tight')
            
            # Explicitly close the plot to manage resources
            plt.close()
            
        # If save_plots is False, the plot is not created, saved, or shown, fulfilling the requirement.


    # ---------------------------
    # Save FINAL SUMMARY CSV
    # ---------------------------
    results_df = pd.DataFrame(results)

    if save_csv_path:
        results_df.to_csv(save_csv_path, index=False)
        print(f"📁 Saved regression (target summary) to: {save_csv_path}")

    return prediction_df, results_df



import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

def run_pca_regression_with_feature_importance(
    pc_df,
    targets_df,
    pc_loadings_df,
    targets,
    pca_prefix='PC',
    save_csv_path=None,
    save_target_csv_dir=None
):
    """
    Runs LassoCV regression using PC trajectories as features and back-calculates 
    feature importance to the original features using PC loadings.
    """
    
    # 1. Identify PC Feature Columns
    pc_feature_cols = [col for col in pc_df.columns if col.startswith(pca_prefix)]
    if not pc_feature_cols:
        print(f"❌ Error: No PC columns found starting with '{pca_prefix}' in the PC Trajectories DataFrame.")
        return None, None
    
    # Identify the unique PC axes (e.g., PC1, PC2) without timepoints
    pc_axes = sorted(list(set(col.split('_')[0] for col in pc_feature_cols)))
    
    # 2. Merge PC data with Targets
    merged_df = pd.merge(
        pc_df,
        targets_df[['Subject_Code'] + targets],
        on="Subject_Code",
        how="inner"
    )
    
    results = []
    
    for target in targets:

        # --- Prepare data ---
        valid_df = merged_df[pc_feature_cols + [target]].dropna()
        X = valid_df[pc_feature_cols]
        y = valid_df[target]

        n_subjects = len(valid_df)
        if n_subjects < 5:
            print(f"⚠️ Skipping {target}: fewer than 5 rows ({n_subjects})")
            continue

        # Standardize features for LassoCV
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # --- Fit model ---
        model = LassoCV(cv=5, random_state=42, max_iter=10000)
        model.fit(X_scaled, y)

        r2 = model.score(X_scaled, y)
        coefs = np.array(model.coef_, dtype=float)

        print(f"\n✅ Regression results for Target: {target} (R2: {r2:.4f})")

        # 3. Calculate "True" Original Feature Importance

        # Map PC column name (e.g., 'PC1_b') to its Lasso coefficient
        pc_coef_map = dict(zip(pc_feature_cols, coefs))

        # Initialize the final feature importance DataFrame
        final_importance_df = pc_loadings_df[['feature']].copy()
        final_importance_df['importance'] = 0.0

        for pc_axis in pc_axes:
            # 3a. Get the column in the Loadings file corresponding to this PC (e.g., 'PC1')
            if pc_axis not in pc_loadings_df.columns:
                 print(f"❌ Missing column '{pc_axis}' in PC Loadings file. Skipping importance calculation for this PC axis.")
                 continue

            # 3b. Sum the absolute Lasso coefficients across all timepoints for this PC axis (e.g., PC1_b, PC1_t1, etc.)
            abs_lasso_coeffs_sum = 0
            for timepoint in ['b', 't1', 't2', 't3', 'after']:
                pc_col = f'{pc_axis}_{timepoint}'
                if pc_col in pc_coef_map:
                    abs_lasso_coeffs_sum += np.abs(pc_coef_map[pc_col])
            
            # If no timepoint PCs were included in the model (sum is zero), skip
            if abs_lasso_coeffs_sum == 0:
                continue

            # 3c. Calculate importance contribution for this PC: |PC Loading| * Sum(|Lasso Coeffs|)
            # Note: We use the absolute loading and multiply by the summed absolute Lasso coeffs.
            contribution = np.abs(pc_loadings_df[pc_axis]) * abs_lasso_coeffs_sum
            
            # Add to the running total importance
            final_importance_df['importance'] += contribution

        # 4. Save/Store Results
        
        # Save individual Target CSV file with final feature importance
        if save_target_csv_dir:
            os.makedirs(save_target_csv_dir, exist_ok=True)
            safe_target_name = target.replace(" ", "_").replace("/", "-")
            file_path = os.path.join(save_target_csv_dir, f"{safe_target_name}_feature_importance_pca_backcalculated.csv")
            final_importance_df.sort_values('importance', ascending=False).to_csv(file_path, index=False)
            print(f"📁 Saved back-calculated importance for Target '{target}' to: {file_path}")

        # Save results for target-level summary table (R2, coefficients of PCs)
        results.append({
            "target": target,
            "R2_cv": r2,
            "n_subjects": n_subjects,
            "pc_coefficients": list(coefs),
            "pc_features": pc_feature_cols,
            "original_feature_importance": final_importance_df.set_index('feature')['importance'].to_dict()
        })

    # Save FINAL SUMMARY CSV
    results_df = pd.DataFrame(results)

    if save_csv_path:
        results_df.to_csv(save_csv_path, index=False)
        print(f"📁 Saved PCA regression summary to: {save_csv_path}")

    return results_df

