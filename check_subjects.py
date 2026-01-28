import pandas as pd

# Load the clusters_after file
clusters = pd.read_csv('only_Q_outputs/combined/clusters_after.csv')
print(f"Total subjects in clusters_after: {len(clusters)}")

# Get subjects in cluster 1
cluster1_subjects = clusters[clusters['Cluster'] == 1]['Subject_Code'].tolist()
print(f"\nSubjects in cluster 1 (after): {len(cluster1_subjects)}")
print(f"Cluster 1 subjects: {sorted(cluster1_subjects)}")

# Load the 2-session T1 summary file
t1_2ses = pd.read_csv('T1/longitude_stats_without_normalizations/2_sess_T1_summary.csv')
print(f"\nTotal rows in 2_sess_T1_summary: {len(t1_2ses)}")

# Get unique subjects that have session 2
t1_2ses_subjects = t1_2ses[t1_2ses['session'] == 2]['subject_code'].unique().tolist()
print(f"Subjects with 2-session T1 data (session 2): {len(t1_2ses_subjects)}")

# Find intersection
intersection = [s for s in cluster1_subjects if s in t1_2ses_subjects]

print(f"\n{'='*70}")
print(f"INTERSECTION: Subjects in cluster 1 (after) AND have 2-session T1 data")
print(f"{'='*70}")
print(f"Count: {len(intersection)}")
print(f"\nSubjects:")
for subj in sorted(intersection):
    print(f"  {subj}")

















