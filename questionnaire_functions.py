import pandas as pd

def analyze_questionnaire(prefix,file_path):
    df = pd.read_csv(file_path)

    # סינון עמודות שמתחילות ב-prefix (ולא העמודה Subject_Code)
    data_cols = [col for col in df.columns if col.startswith(prefix) and col != 'Subject_Code']
    df_sub = df[data_cols].dropna(how='all')

    if df_sub.empty:
        print(f"⚠️ אין נבדקות פעילות בקובץ: {file_path}")
        return

    total_subjects = len(df_sub)
    print(f"\n📘 קובץ: {file_path}")
    print(f"👥 נבדקות פעילות: {total_subjects}")

    # חישוב אחוז מענה לכל עמודה
    percentages = df_sub.notna().sum() / total_subjects * 100

    # קיבוץ לפי שם שאלון (החלק האמצעי בין שני קווים תחתונים)
    group_stats = {}

    for col in df_sub.columns:
        parts = col.split('_')
        if len(parts) < 3:
            continue
        group_key = parts[1]  # החלק האמצעי
        group_stats.setdefault(group_key, []).append(percentages[col])

    # הדפסת סיכום לכל שאלון
    for group, values in group_stats.items():
        avg_percent = sum(values) / len(values)
        print(f"   🔸 {group}: ממוצע השתתפות {avg_percent:.2f}% ({len(values)} שאלות)")


def export_questionnaire(df, prefix, filename):
    # סינון עמודות שמתחילות ב-prefix
    selected_columns = df.columns[df.columns.str.startswith(prefix)].tolist()
    df_sub = df[selected_columns]


    # הסרת נבדקות שאין להן אף ערך בטווח העמודות האלה
    df_sub = df_sub.dropna(how='all')

    # אם אין נתונים, מדלגים
    if df_sub.empty:
        print(f"⚠️ אין נבדקות עם נתונים עבור {prefix}")
        return

    # הוספת עמודת Subject_Code כעמודה ראשונה
    if 'Subject_Code' in df.columns:
        subject_ids = df.loc[df_sub.index, 'Subject_Code']
        df_sub.insert(0, 'Subject_Code', subject_ids)
    else:
        print(f"⚠️ העמודה 'Subject_Code' לא קיימת ב-DataFrame")
        return

    # חישוב כמות ואחוזים
    counts = df_sub.notna().sum()
    total = len(df_sub)
    percentages = (counts / total * 100).round(2)

    # טבלת סיכום
    summary_df = pd.DataFrame([counts, percentages], index=['number of subjects', '% of subjects'])

    # כתיבה לשני גיליונות באקסל
    df_sub.to_csv(f"{filename}.csv")

    # ✅ הדפסה
    print(f"\n✅ נוצר הקובץ: {filename}.csv")
    print(f"📌 מספר נבדקות בקובץ: {total}")
    print("📋 עמודות שנכללו:")
    print(', '.join(selected_columns))



def export_questionnaire_combined(df, prefix, filename):
    # סינון עמודות שמתחילות ב-prefix
    df_sub = df


    # הסרת נבדקות שאין להן אף ערך בטווח העמודות האלה
    df_sub = df_sub.dropna(how='any')

    # אם אין נתונים, מדלגים
    if df_sub.empty:
        print(f"⚠️ אין נבדקות עם נתונים עבור {prefix}")
        return


    # חישוב כמות ואחוזים
    counts = df_sub.notna().sum()
    total = len(df_sub)
    percentages = (counts / total * 100).round(2)

    # טבלת סיכום
    summary_df = pd.DataFrame([counts, percentages], index=['number of subjects', '% of subjects'])

    # כתיבה לשני גיליונות באקסל
    df_sub.to_csv(f"{filename}.csv")

    # ✅ הדפסה
    print(f"\n✅ נוצר הקובץ: {filename}.csv")
    print(f"📌 מספר נבדקות בקובץ: {total}")



def build_timepoint_presence_matrix(timepoint_files, output_csv):
    """
    timepoint_files: dict
        keys   = timepoint names (e.g. 'b', 't1', ...)
        values = path to questionnaire CSV for that timepoint
    """

    presence = {}

    # collect subjects per timepoint
    for tp, path in timepoint_files.items():
        df = pd.read_csv(path)

        if 'Subject_Code' not in df.columns:
            raise ValueError(f"Subject_Code missing in {path}")

        presence[tp] = set(df['Subject_Code'].dropna())

    # all unique subjects across all timepoints
    all_subjects = sorted(set().union(*presence.values()))

    # build matrix
    matrix = pd.DataFrame(index=all_subjects)

    for tp, subjects in presence.items():
        matrix[tp] = matrix.index.map(lambda x: 1 if x in subjects else 0)

    # optional: nicer column order
    matrix.index.name = 'Subject_Code'

    # export
    matrix.to_csv(output_csv)

    print(f"✅ Presence matrix created: {output_csv}")
    print(f"👥 Total subjects: {len(matrix)}")
