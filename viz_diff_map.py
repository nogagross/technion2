import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_distance_heatmap_from_csv(df, title="Distance Heatmap", cmap="magma"):
    """
    Loads a distance matrix from a CSV file (assuming subject names are the index/first column),
    and displays a heatmap of the numerical data.

    Args:
        file_path (str): The full path to the CSV file.
        title (str, optional): The title of the heatmap plot.
        cmap (str, optional): The colormap to use.
    """



    # 2. חילוץ מטריצת המספרים (כמערך NumPy)
    # הערכים (values) של DataFrame הם מטריצת המספרים הטהורה.
    distance_matrix = df.values

    # 3. יצירת מפת החום
    plt.figure(figsize=(12, 10))

    # שימוש ב-df ישירות:
    # seaborn משתמש אוטומטית בכותרות השורות והעמודות מה-DataFrame
    sns.heatmap(df, annot=False, cmap=cmap, fmt=".2f",
                linewidths=.5, linecolor='lightgray',
                # מאפשר הצגה טובה יותר של כותרות האינדקס (שמות הנבדקים)
                cbar_kws={'label': 'Riemannian Distance'})

    plt.title(title, fontsize=16)
    plt.xlabel("Subject Index (Name)", fontsize=12)
    plt.ylabel("Subject Index (Name)", fontsize=12)

    # התאמה של התוויות אם המטריצה גדולה מדי
    if df.shape[0] > 50:
        plt.yticks(rotation=0)  # סיבוב 0 עבור שמות שורות
        plt.xticks(rotation=90)  # סיבוב 90 עבור שמות עמודות

    plt.tight_layout()  # התאמת הפריסה כדי למנוע חיתוך
    plt.show()





def reorder_distance_matrix_by_clusters(distance_csv_path, cluster_csv_path):
    """
    Loads a distance matrix and cluster assignments, and reorders the distance
    matrix rows and columns based on cluster and group/prefix assignment.

    Args:
        distance_csv_path (str): Path to the distance matrix CSV.
        cluster_csv_path (str): Path to the cluster assignment CSV.

    Returns:
        pd.DataFrame: The reordered distance matrix DataFrame.
    """
    # 1. טעינת מטריצת המרחקים והאשכולות
    df_dist = pd.read_csv(distance_csv_path, index_col=0)
    df_clusters = pd.read_csv(cluster_csv_path, index_col='Subject_Code')[['Cluster']]

    # 2. מיזוג ויצירת מפתחות מיון
    # טבלת מידע שתכיל את קוד הנבדק, האשכול והקידומת (CT/NT)
    df_sorted_info = df_dist.index.to_frame(name='Subject_Code')
    df_sorted_info = df_sorted_info.merge(df_clusters, left_on='Subject_Code', right_index=True, how='left')

    # חילוץ הקידומת (לדוגמה, 'CT' או 'NT') - משמשת למיון המשני
    df_sorted_info['Group_Prefix'] = df_sorted_info['Subject_Code'].str[:2]
    df_sorted_info['Cluster'] = df_sorted_info['Cluster'].fillna(-1).astype(int)

    # 3. קביעת סדר המיון הסופי: קודם לפי אשכול, שני לפי קידומת (CT/NT)
    df_sorted_info = df_sorted_info.sort_values(by=['Cluster', 'Group_Prefix'])

    # רשימת הנבדקים בסדר החדש
    sorted_subjects = df_sorted_info['Subject_Code'].tolist()

    # 4. סידור מחדש של שורות ועמודות מטריצת המרחקים
    df_reordered = df_dist.reindex(index=sorted_subjects, columns=sorted_subjects)

    print(df_reordered)
    return df_reordered
# --- דוגמה לשימוש ---
if __name__ == '__main__':
    # 🚨 הערה: צריך להחליף את 'your_distance_matrix.csv' בנתיב לקובץ שלך

    session1_distances= pd.read_csv('data/SCHAEFER_mat_cor/csv_out/diffusion_distances_ses1_labeled.csv',index_col =0 )
    session2_distances= pd.read_csv('data/SCHAEFER_mat_cor/csv_out/diffusion_distances_ses2_labeled.csv',index_col =0 )

    plot_distance_heatmap_from_csv(session1_distances,
                                    title="Riemannian Distance Map - Session 1")

    print("Example run setup: You need to replace the file path with your actual CSV file path.")

    plot_distance_heatmap_from_csv(session2_distances,
                                    title="Riemannian Distance Map - Session 2")

    session1_reorder = reorder_distance_matrix_by_clusters('data/SCHAEFER_mat_cor/csv_out/diffusion_distances_ses1_labeled.csv','data/SCHAEFER_mat_cor/csv_out/diff_map_clusters_ses1.csv')
    session2_reorder = reorder_distance_matrix_by_clusters('data/SCHAEFER_mat_cor/csv_out/diffusion_distances_ses2_labeled.csv','data/SCHAEFER_mat_cor/csv_out/diff_map_clusters_ses2.csv')

    plot_distance_heatmap_from_csv(session1_reorder,
                                   title="Riemannian Distance Map by clusters - Session 1")

    plot_distance_heatmap_from_csv(session2_reorder,
                                   title="Riemannian Distance Map by clusters - Session 2")