import pandas as pd

# Load the clusters_after file
clusters = pd.read_csv('only_Q_outputs/combined/clusters_after.csv')
print("="*70)
print("ANALYSIS: Subjects in Cluster 1 (after) with 2-session T1 data")
print("="*70)

# Get subjects in cluster 1
cluster1_df = clusters[clusters['Cluster'] == 1]
cluster1_subjects = set(cluster1_df['Subject_Code'].tolist())
print(f"\n1. Subjects in cluster 1 (after): {len(cluster1_subjects)}")
print(f"   Subjects: {sorted(cluster1_subjects)}")

# Load the 2-session T1 summary file
t1_2ses = pd.read_csv('T1/longitude_stats_without_normalizations/2_sess_T1_summary.csv')

# Get unique subjects that have session 2
t1_2ses_subjects = set(t1_2ses[t1_2ses['session'] == 2]['subject_code'].unique().tolist())
print(f"\n2. Subjects with 2-session T1 data (session 2): {len(t1_2ses_subjects)}")
print(f"   Subjects: {sorted(t1_2ses_subjects)}")

# Find intersection
intersection = sorted(list(cluster1_subjects & t1_2ses_subjects))

print(f"\n" + "="*70)
print(f"3. INTERSECTION: Subjects in cluster 1 (after) AND have 2-session T1 data")
print("="*70)
print(f"   COUNT: {len(intersection)}")
print(f"\n   Subjects:")
for subj in intersection:
    print(f"      - {subj}")

# Also show which cluster 1 subjects DON'T have 2-session data
missing_2ses = sorted(list(cluster1_subjects - t1_2ses_subjects))
if missing_2ses:
    print(f"\n4. Cluster 1 subjects WITHOUT 2-session T1 data: {len(missing_2ses)}")
    print(f"   Subjects: {missing_2ses}")

















