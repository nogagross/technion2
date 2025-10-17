

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib import cm
from preprocessing_functions import (_load_mapping,_load_clustered_df,_ensure_subject_col,_ensure_pc_axes)
from clustring_functions import (_palette_from_dataset)
import matplotlib.colors as mcolors
import math
import colorsys

GRAY_MISSING = "#9e9e9e"  # gray for subjects not in that period's clusters

# Vibrant colors per period (Cluster 0, Cluster 1)
PERIOD_COLORS = {
    "before": ("#1f77b4", "#4fa8ff"),  # vivid dark/light blue
    "t1":     ("#8B4513", "#FFB470"),  # saddle brown / peach
    "t2":     ("#2ca02c", "#7CFC7C"),  # vivid green / light green
    "t3":     ("#e31a1c", "#ff7f7f"),  # vivid red / light red
    "after":  ("#6a51a3", "#b07cff"),  # vivid purple / light purple
}
GRAY_MISSING = "#9e9e9e"
def plot_grouped_bars(groups, s1, s2, s3,
                      s1_label="Column 1",
                      s2_label="Column 2",
                      s3_label="Column 3",
                      title="Grouped Bar Chart: 3 Groups × 3 Columns",
                      ylabel="Value",
                      annotate=True):
    """
    groups : list[str] of length 3 – group names (e.g., ["G1", "G2", "G3", "G4", "G5"])
    s1, s2, s3 : list[float] each of length 5 – values for the 3 bars in each group
    """

    # Validate inputs
    n = len(groups)

    x = np.arange(n)           # positions of groups on X-axis
    k = 3                      # bars per group
    width = 0.8 / k            # width of each bar

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars
    b1 = ax.bar(x - width, s1, width, label=s1_label)
    b2 = ax.bar(x,         s2, width, label=s2_label)
    b3 = ax.bar(x + width, s3, width, label=s3_label)

    # Formatting
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, groups)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    # Formatting
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0,120)
    ax.set_xticks(x, groups)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    # Legend בטור אנכי בצד ימין
    ax.legend(frameon=True,loc = 'upper left')  # ברירת מחדל בטור

    # Annotate values above bars
    if annotate:
        for bars in (b1, b2, b3):
            for rect in bars:
                h = rect.get_height()
                ax.annotate(f"{h:.2f}",
                            xy=(rect.get_x() + rect.get_width()/2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    plt.show()
    return fig, ax






def plot_clusters_with_group_overlay(
    clustered_path,              # path to clustered PCA CSV/XLSX
    *,
    mapping_path,
    mapping_subject_col="Subject_Code",
    mapping_group_col="trajectory_group",
    wanted_groups=None,          # e.g. ["increasing", "stable"] or a single string
    subject_col=None,            # name OR integer position of subject column; 0 if first col with no header
    title=None,
    dataset_name=None,           # for palette title only
    save_marked_csv_path=None,
    alpha_all=0.9,
    s_all=50,
    s_marked=120,
    edgecolor="black",
    linewidth=0.9
):
    """Show the original cluster scatter and overlay only subjects in wanted_groups."""
    # --- load clustered df from disk ---
    clustered_path = Path(clustered_path)
    if clustered_path.suffix.lower() in {".xlsx", ".xls"}:
        clustered_df = pd.read_excel(clustered_path)
    else:
        clustered_df = pd.read_csv(clustered_path)

    # Work on a copy
    df = clustered_df.copy()

    # --- required columns for plotting ---
    if "Cluster" not in df.columns:
        raise ValueError(f"clustered_df must include 'Cluster'. Found: {list(df.columns)}")

    # --- choose PC1/PC2 (use explicit names if present; else first two numeric) ---
    if {"PC1","PC2"}.issubset(df.columns):
        xcol, ycol = "PC1", "PC2"
    else:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(num_cols) < 2:
            raise ValueError("Could not find two numeric columns for plotting.")
        xcol, ycol = num_cols[0], num_cols[1]

    # --- create a guaranteed subject column: __SUBJECT__ ---
    # 1) If user gave integer -> take by POSITION
    if isinstance(subject_col, int):
        if subject_col < 0 or subject_col >= df.shape[1]:
            raise IndexError(f"subject_col index {subject_col} out of range for columns: {list(df.columns)}")
        subj_series = df.iloc[:, subject_col].astype(str).str.strip()
        df["__SUBJECT__"] = subj_series

    # 2) If user gave a name -> use it (from column or, if needed, from index)
    elif isinstance(subject_col, str):
        if subject_col in df.columns:
            df["__SUBJECT__"] = df[subject_col].astype(str).str.strip()
        else:
            # maybe it's in the index name
            df = df.reset_index(drop=False)
            if subject_col in df.columns:
                df["__SUBJECT__"] = df[subject_col].astype(str).str.strip()
            else:
                raise KeyError(f"Subject column '{subject_col}' not found (even after reset_index).")

    # 3) No subject_col provided -> auto-detect
    else:
        # try common names
        candidates = ["Subject", "subject", "Subject_ID", "Subject_Code", "id", "participant"]
        found = next((c for c in candidates if c in df.columns), None)
        if found is not None:
            df["__SUBJECT__"] = df[found].astype(str).str.strip()
        else:
            # look for 'Unnamed: ...' style first column
            unnamed = next((c for c in df.columns if isinstance(c, str) and c.lower().startswith("unnamed")), None)
            if unnamed is not None:
                df["__SUBJECT__"] = df[unnamed].astype(str).str.strip()
            else:
                # fallback: if index seems to be the IDs, bring it out
                if df.index.name is not None:
                    df = df.reset_index(drop=False)
                    df["__SUBJECT__"] = df[df.columns[0]].astype(str).str.strip()
                else:
                    # final fallback: use the first column by position
                    df["__SUBJECT__"] = df.iloc[:, 0].astype(str).str.strip()

    # --- load mapping ---
    mapping_path = Path(mapping_path)
    mapping = pd.read_excel(mapping_path) if mapping_path.suffix.lower() in {".xlsx",".xls"} else pd.read_csv(mapping_path)
    mapping = mapping.rename(columns={mapping_subject_col: "MAP_SUBJ", mapping_group_col: "MAP_GROUP"})
    if "MAP_SUBJ" not in mapping or "MAP_GROUP" not in mapping:
        raise ValueError(f"Check mapping_subject_col/mapping_group_col. Got: {list(mapping.columns)}")
    mapping["MAP_SUBJ"] = mapping["MAP_SUBJ"].astype(str).str.strip()
    mapping["MAP_GROUP"] = mapping["MAP_GROUP"].astype(str).str.strip()

    # --- filter groups & join ---
    if wanted_groups is None:
        wanted_groups = mapping["MAP_GROUP"].dropna().unique().tolist()
    if isinstance(wanted_groups, str):
        wanted_groups = [wanted_groups]

    mapping_sel = mapping[mapping["MAP_GROUP"].isin(wanted_groups)]
    marked = df.merge(mapping_sel[["MAP_SUBJ","MAP_GROUP"]],
                      left_on="__SUBJECT__", right_on="MAP_SUBJ",
                      how="inner").drop_duplicates(subset=["__SUBJECT__","MAP_GROUP"])

    # --- palette (to match prior look) ---
    def _palette_from_dataset(k, dataset_name=None, cmap=None):
        if cmap is None:
            name_to_cmap = {"VCSF":"Blues","VGMM":"Reds","VWM":"Greens"}
            cmap_name = name_to_cmap.get(str(dataset_name).upper() if dataset_name else None, "tab10")
            cmap = cm.get_cmap(cmap_name)
        elif isinstance(cmap, str):
            cmap = cm.get_cmap(cmap)
        return [cmap(x) for x in np.linspace(0.35, 0.95, k)]

    k = int(df["Cluster"].max()) + 1
    base_colors = _palette_from_dataset(k, dataset_name=dataset_name)

    # --- plot original clusters ---
    plt.figure(figsize=(8,6))
    for cid in range(k):
        m = df["Cluster"] == cid
        plt.scatter(df.loc[m, xcol], df.loc[m, ycol],
                    s=s_all, color=base_colors[cid % len(base_colors)],
                    alpha=alpha_all, label=f"Cluster {cid} (n={int(m.sum())})")

    # --- overlay wanted groups as hollow markers ---
    if not marked.empty:
        groups = sorted(marked["MAP_GROUP"].unique())
        marker_cycle = ["o","s","^","D","P","X","*","v","<",">"]
        marker_map = {g: marker_cycle[i % len(marker_cycle)] for i, g in enumerate(groups)}
        for g in groups:
            gdf = marked[marked["MAP_GROUP"] == g]
            plt.scatter(gdf[xcol], gdf[ycol],
                        s=s_marked, facecolors="none", edgecolors=edgecolor,
                        linewidths=linewidth, marker=marker_map[g],
                        label=f"{g} (marked n={len(gdf)})")

    plt.title(title or f"{dataset_name or ''} — Original Clusters with Group Overlay".strip(" —"))
    plt.xlabel(xcol); plt.ylabel(ycol)
    plt.grid(True, linestyle="--", alpha=0.25)
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.show()

    if save_marked_csv_path:
        cols_to_save = [c for c in ["__SUBJECT__", xcol, ycol, "Cluster", "MAP_GROUP"] if c in marked.columns]
        marked[cols_to_save].to_csv(save_marked_csv_path, index=False)

    return marked



def subplot_clusters_by_group(
    datasets,                      # Ordered dict/list of tuples: [(name, path), ...] length should be 3
    groups,                        # list of 5 group names (columns)
    *,
    mapping_path,
    mapping_subject_col="Subject_Code",
    mapping_group_col="trajectory_group",
    subject_col=0,                 # first column by position (no header)
    alpha_all=0.9,
    s_all=20,
    s_marked=80,
    edgecolor="black",
    linewidth=1.5,
    figsize=(18, 10)              # width, height
):
    """
    Make a 3x5 grid: rows=datasets, cols=groups. Each subplot shows original clusters + overlay of that single group.
    """
    # Load mapping once
    mapping = _load_mapping(mapping_path, mapping_subject_col, mapping_group_col)

    # Prepare figure
    nrows = len(datasets)
    ncols = len(groups)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)

    for r, (ds_name, ds_path) in enumerate(datasets):
        df = _load_clustered_df(ds_path)
        if "Cluster" not in df.columns:
            raise ValueError(f"'Cluster' column missing in {ds_path}. Found: {list(df.columns)}")
        df = _ensure_subject_col(df,subject_col= subject_col)
        xcol, ycol = _ensure_pc_axes(df)

        # Compute shared limits per row for consistent scale across the 5 groups
        xmin, xmax = df[xcol].min(), df[xcol].max()
        ymin, ymax = df[ycol].min(), df[ycol].max()
        xpad = 0.05 * (xmax - xmin if xmax > xmin else 1.0)
        ypad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
        xlim = (xmin - xpad, xmax + xpad)
        ylim = (ymin - ypad, ymax + ypad)

        k = int(df["Cluster"].max()) + 1
        base_colors = _palette_from_dataset(k, dataset_name=ds_name)

        for c, grp in enumerate(groups):
            ax = axes[r, c]

            # base clusters
            for cid in range(k):
                m = df["Cluster"] == cid
                ax.scatter(
                    df.loc[m, xcol], df.loc[m, ycol],
                    s=s_all, color=base_colors[cid % len(base_colors)],
                    alpha=alpha_all, linewidths=0, label=None
                )

            # overlay this single group
            gsel = mapping[mapping["MAP_GROUP"] == grp]
            marked = df.merge(gsel[["MAP_SUBJ","MAP_GROUP"]],
                              left_on="__SUBJECT__", right_on="MAP_SUBJ",
                              how="inner").drop_duplicates(subset=["__SUBJECT__","MAP_GROUP"])

            if not marked.empty:
                ax.scatter(
                    marked[xcol], marked[ycol],
                    s=s_marked, facecolors="none", edgecolors=edgecolor,
                    linewidths=linewidth, marker="o", label=None
                )
            else:
                # optional: annotate when empty
                ax.text(0.5, 0.5, "No members", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, alpha=0.7)

            # cosmetics
            if r == 0:
                ax.set_title(grp, fontsize=50)
            if c == 0:
                ax.set_ylabel(ds_name, fontsize=50)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            ax.grid(True, linestyle="--", alpha=0.2)
            if r < nrows - 1:
                ax.set_xticklabels([])
            if c > 0:
                ax.set_yticklabels([])

    fig.suptitle("Original Clusters with Group Overlays (rows=datasets, cols=groups)", fontsize=50)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def save_top_loadings(pca, feature_names, pc_index=0, top_n=20, out_csv=None):
    """
    Plot and (optionally) save the top loadings of a given PC.

    Args:
        pca          : fitted PCA object
        feature_names: list of feature names (aligned to PCA input)
        pc_index     : which principal component (0=PC1, 1=PC2, etc.)
        top_n        : how many top features by |loading| to include
        out_csv      : if given (str or Path), save results to CSV
    """
    weights = pca.components_[pc_index, :]
    order = np.argsort(np.abs(weights))[::-1][:top_n]
    top_features = [(feature_names[i], weights[i]) for i in order]

    # --- plot ---
    plt.figure(figsize=(10, 6))
    plt.bar(range(top_n), [weights[i] for i in order])
    plt.xticks(range(top_n), [feature_names[i] for i in order],
               rotation=75, ha='right')
    plt.ylabel("Loading weight")
    plt.title(f"Top {top_n} loadings for PC{pc_index+1}")
    plt.tight_layout()
    plt.show()

    # --- print ---
    print(f"\nTop {top_n} loadings for PC{pc_index+1}:")
    for feat, val in top_features:
        print(f"{feat:30s} {val:+.4f}")

    # --- save ---
    if out_csv is not None:
        df = pd.DataFrame({
            "feature": [f for f, _ in top_features],
            f"PC{pc_index+1}_loading": [v for _, v in top_features]
        })
        df.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"✅ Saved loadings to {out_csv}")

    return top_features


def plot_top_loadings(pca, feature_names, pc_index=0, top_n=20):
    # get weights for this PC
    weights = pca.components_[pc_index, :]

    # sort by absolute weight, but keep original sign
    order = np.argsort(np.abs(weights))[::-1][:top_n]
    top_features = [(feature_names[i], weights[i]) for i in order]

    # --- plot ---
    plt.figure(figsize=(10, 6))
    plt.bar(range(top_n), [weights[i] for i in order])
    plt.xticks(range(top_n), [feature_names[i] for i in order],
               rotation=75, ha='right')
    plt.ylabel("Loading weight")
    plt.title(f"Top {top_n} loadings for PC{pc_index+1}")
    plt.tight_layout()
    plt.show()

    # --- print top features with signed weights ---
    print(f"\nTop {top_n} loadings for PC{pc_index+1}:")
    for feat, val in top_features:
        print(f"{feat:30s} {val:+.4f}")

def make_colors_for_period(clusters_series, period_label):
    c0, c1 = PERIOD_COLORS.get(period_label.lower(), ("#444444", "#bbbbbb"))
    # sort unique non-null labels so strings ('A','B') map consistently
    uniq = sorted(clusters_series.dropna().astype(str).unique())
    def color_for(v):
        if pd.isna(v): return GRAY_MISSING
        s = str(v).strip()
        if s in {"0","Cluster 0"}: return c0
        if s in {"1","Cluster 1"}: return c1
        return c0 if s == (uniq[0] if uniq else s) else c1
    return [color_for(v) for v in clusters_series]
def plot_pc12_colored(scores, explained, clusters_series, period_label, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,5))
    colors = make_colors_for_period(clusters_series, period_label)
    ax.scatter(scores[:,0], scores[:,1],
               c=colors, s=50, alpha=0.9, edgecolor="white", linewidth=0.6)
    titles = {"before":"Pre pregnancy","t1":"1st trimester","t2":"2nd trimester",
              "t3":"3rd trimester","after":"Post pregnancy"}
    ax.set_title(titles.get(period_label.lower(), period_label))
    ax.set_xlabel(f"PC1 ({explained[0]*100:.2f}% var)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.2f}% var)")

    # legend
    c0, c1 = PERIOD_COLORS.get(period_label.lower(), ("#444","##bbb"))
    handles = [
        plt.Line2D([], [], marker='o', linestyle='', color=c0, label="Cluster 0", markeredgecolor="white"),
        plt.Line2D([], [], marker='o', linestyle='', color=c1, label="Cluster 1", markeredgecolor="white"),
    ]
    if clusters_series.isna().any():
        handles.append(plt.Line2D([], [], marker='o', linestyle='', color=GRAY_MISSING,
                                  label="Missing in this period", markeredgecolor="white"))
    ax.legend(handles=handles, frameon=True, loc="best")
    ax.grid(True, alpha=0.3)
    return ax

def plot_multi_dataset_scatters_colored(
    datasets,
    titles=None,
    labels_list=None,
    x_idx=0,
    y_idx=1,
    figsize=(18, 10),
    point_size=30,
    alpha=0.85,
    base_colors=None,
    wspace=0.35,
    hspace=0.55,
    axes_bg="white",
    max_cols=3,
    legend_loc="best",
    sort_labels=True,
    show_counts=True,      # NEW: add n per cluster in legend
    show_percent=False     # NEW: also show % of subjects per cluster
):
    """
    Draw multiple PCA scatter panels, coloring by clusters. Adds cluster counts to legend.
    """
    n = len(datasets)
    if base_colors is None:
        base_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red",
                       "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan"]

    if len(base_colors) < n:
        cycles_needed = math.ceil(n / len(base_colors))
        base_colors = (base_colors * cycles_needed)[:n]

    nrows, ncols = _auto_layout(n, max_cols=max_cols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize,
        gridspec_kw={"wspace": wspace, "hspace": hspace}
    )
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = np.array([axes])

    for i, data in enumerate(datasets):
        ax = axes[i]
        ax.set_facecolor(axes_bg)

        labels = None if labels_list is None else labels_list[i]
        x = data[:, x_idx]
        y = data[:, y_idx]

        if labels is None:
            ax.scatter(x, y, s=point_size, alpha=alpha, color=base_colors[i])
            total_n = len(x)
        else:
            labels = np.asarray(labels)
            # exclude NaNs from unique list (if any)
            unique_labels = [lab for lab in np.unique(labels) if lab == lab]
            if sort_labels:
                try:
                    unique_labels = np.sort(unique_labels)
                except Exception:
                    pass

            total_n = int(np.sum(~np.isnan(labels))) if np.issubdtype(labels.dtype, np.floating) else len(labels)
            palette = _pick_palette_for_labels(base_colors[i], unique_labels)

            for lab in unique_labels:
                mask = labels == lab
                n_lab = int(np.sum(mask))
                if n_lab == 0:
                    continue
                legend_text = f"Cluster {lab}"
                if show_counts:
                    if show_percent and total_n > 0:
                        legend_text += f" (n={n_lab}, {n_lab/total_n:.0%})"
                    else:
                        legend_text += f" (n={n_lab})"

                ax.scatter(
                    x[mask], y[mask],
                    s=point_size, alpha=alpha,
                    color=palette[lab], label=legend_text
                )

            ax.legend(fontsize=8, frameon=True, framealpha=0.85, loc=legend_loc)

        ax.set_xlabel(f"PC{x_idx+1}")
        ax.set_ylabel(f"PC{y_idx+1}")
        if titles is not None and i < len(titles):
            ttl = titles[i]
            # annotate total N in the title for quick context
            ttl = f"{ttl}  •  N={total_n}" if show_counts else ttl
            ax.set_title(ttl, pad=8)
        ax.grid(alpha=0.3)

    for j in range(len(datasets), nrows * ncols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()




def _generate_shades_from_base(base_color, k, min_l=0.35, max_l=0.8, min_s=0.6, max_s=1.0):
    """
    יוצר k גוונים שונים מתוך צבע בסיס יחיד באמצעות HLS:
    משנים בעיקר את ה-Lightness וקצת את ה-Saturation כדי לקבל הבחנה נעימה.
    """
    base_rgb = np.array(mcolors.to_rgb(base_color))
    h, l, s = colorsys.rgb_to_hls(*base_rgb)
    if k == 1:
        return [base_rgb]
    # נבחר k נקודות אחידות בטווחים
    Ls = np.linspace(min_l, max_l, k)
    Ss = np.linspace(min_s, max_s, k)
    cols = []
    for i in range(k):
        # נשמור על Hue הבסיסי, נשחק עם L ו-S
        rgb = colorsys.hls_to_rgb(h, Ls[i], Ss[i])
        cols.append(rgb)
    return cols

def _pick_palette_for_labels(base_color, unique_labels):
    """
    ממפה לייבלים ייחודיים -> צבעים (גוונים מה-base).
    """
    k = len(unique_labels)
    shades = _generate_shades_from_base(base_color, k)
    return {lab: shades[i] for i, lab in enumerate(unique_labels)}

def _auto_layout(n, max_cols=3):
    """
    סידור דינמי של תתי-התרשימים עד 3 בעמודה (ברירת מחדל).
    """
    ncols = min(max_cols, max(1, n))
    nrows = math.ceil(n / ncols)
    return nrows, ncols

