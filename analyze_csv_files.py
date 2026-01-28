"""
Script to analyze all CSV files in the technion directory:
- Which notebook created each CSV file
- If the CSV is used anywhere
- If it was created by code or uploaded
"""

import os
import re
from pathlib import Path
from collections import defaultdict
import json

# Get the project root
PROJECT_ROOT = Path(__file__).parent

# Find all CSV files
def find_all_csv_files():
    csv_files = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Skip certain directories
        if any(skip in root for skip in ['.git', '__pycache__', '.ipynb_checkpoints']):
            continue
        for file in files:
            if file.endswith('.csv'):
                full_path = Path(root) / file
                rel_path = full_path.relative_to(PROJECT_ROOT)
                csv_files.append(str(rel_path))
    return sorted(csv_files)

# Find all notebook files
def find_all_notebooks():
    notebooks = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        if any(skip in root for skip in ['.git', '__pycache__', '.ipynb_checkpoints']):
            continue
        for file in files:
            if file.endswith('.ipynb'):
                full_path = Path(root) / file
                rel_path = full_path.relative_to(PROJECT_ROOT)
                notebooks.append(str(rel_path))
    return sorted(notebooks)

# Search for CSV creation patterns in notebooks
def search_notebook_for_csv_creation(notebook_path):
    """Search a notebook for CSV file creation patterns"""
    created_csvs = []
    try:
        with open(notebook_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Try to parse as JSON (Jupyter notebook format)
        try:
            nb = json.loads(content)
            # Extract all code cells
            code_cells = []
            for cell in nb.get('cells', []):
                if cell.get('cell_type') == 'code':
                    source = ''.join(cell.get('source', []))
                    code_cells.append(source)
            content = '\n'.join(code_cells)
        except:
            pass
        
        # Patterns to find CSV file creation
        patterns = [
            r'\.to_csv\(["\']([^"\']+\.csv)["\']',
            r'\.to_csv\(([^)]+\.csv)',
            r'save.*["\']([^"\']+\.csv)["\']',
            r'output.*["\']([^"\']+\.csv)["\']',
            r'output_path.*=.*["\']([^"\']+\.csv)["\']',
            r'csv_path.*=.*["\']([^"\']+\.csv)["\']',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                csv_path = match.group(1)
                # Clean up the path
                csv_path = csv_path.strip('"\'')
                # Handle relative paths
                if not os.path.isabs(csv_path):
                    # Try to resolve relative to notebook location
                    nb_dir = os.path.dirname(notebook_path)
                    resolved = os.path.normpath(os.path.join(nb_dir, csv_path))
                    if os.path.exists(resolved):
                        csv_path = os.path.relpath(resolved, PROJECT_ROOT)
                    else:
                        # Keep original if can't resolve
                        csv_path = csv_path.replace('\\', '/')
                else:
                    csv_path = os.path.relpath(csv_path, PROJECT_ROOT) if os.path.exists(csv_path) else csv_path
                
                created_csvs.append(csv_path)
    except Exception as e:
        print(f"Error reading {notebook_path}: {e}")
    
    return list(set(created_csvs))  # Remove duplicates

# Search for CSV usage patterns
def search_for_csv_usage(csv_file):
    """Search for usage of a CSV file in notebooks and Python files"""
    usage_locations = []
    csv_name = os.path.basename(csv_file)
    csv_name_no_ext = os.path.splitext(csv_name)[0]
    
    # Search in notebooks and Python files
    for root, dirs, files in os.walk(PROJECT_ROOT):
        if any(skip in root for skip in ['.git', '__pycache__', '.ipynb_checkpoints']):
            continue
        for file in files:
            if file.endswith(('.ipynb', '.py')):
                file_path = Path(root) / file
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Check if CSV is referenced
                    if csv_name in content or csv_name_no_ext in content or csv_file.replace('\\', '/') in content:
                        rel_path = file_path.relative_to(PROJECT_ROOT)
                        usage_locations.append(str(rel_path))
                except:
                    pass
    
    return list(set(usage_locations))

# Determine if CSV is likely uploaded (not created by code)
def is_likely_uploaded(csv_file, created_by_notebooks):
    """Determine if CSV is likely uploaded by user"""
    # If created by a notebook, it's code-generated
    if created_by_notebooks:
        return False
    
    # Check if it's in data directories (often uploaded)
    data_dirs = ['data', 'original', 'cat12', 'RSFC', 'T1']
    if any(csv_file.startswith(d) for d in data_dirs):
        return True
    
    # Check if it's in root directory (often uploaded)
    if '/' not in csv_file.replace('\\', '/'):
        return True
    
    # Default: assume code-generated if in output directories
    output_dirs = ['output', 'only_Q_outputs', 'stats', 'longitude']
    if any(csv_file.startswith(d) for d in output_dirs):
        return False
    
    # Otherwise, assume uploaded
    return True

# Main analysis
def analyze_all_csvs():
    print("Finding all CSV files...")
    all_csvs = find_all_csv_files()
    print(f"Found {len(all_csvs)} CSV files")
    
    print("\nFinding all notebooks...")
    all_notebooks = find_all_notebooks()
    print(f"Found {len(all_notebooks)} notebooks")
    
    print("\nAnalyzing notebooks for CSV creation...")
    notebook_to_csvs = defaultdict(list)
    csv_to_notebooks = defaultdict(list)
    
    for notebook in all_notebooks:
        print(f"  Analyzing {notebook}...")
        created = search_notebook_for_csv_creation(notebook)
        for csv_file in created:
            notebook_to_csvs[notebook].append(csv_file)
            csv_to_notebooks[csv_file].append(notebook)
    
    print("\nAnalyzing CSV usage...")
    results = []
    
    for csv_file in all_csvs:
        print(f"  Analyzing {csv_file}...")
        created_by = csv_to_notebooks.get(csv_file, [])
        used_in = search_for_csv_usage(csv_file)
        # Remove the CSV file itself from usage if it appears
        used_in = [u for u in used_in if u != csv_file]
        is_uploaded = is_likely_uploaded(csv_file, created_by)
        
        results.append({
            'csv_file': csv_file,
            'created_by_notebooks': created_by,
            'used_in': used_in,
            'is_code_generated': not is_uploaded,
            'is_uploaded': is_uploaded
        })
    
    return results

# Generate report
def generate_report(results):
    report = []
    report.append("=" * 100)
    report.append("CSV FILE ANALYSIS REPORT")
    report.append("=" * 100)
    report.append("")
    
    # Summary statistics
    code_generated = sum(1 for r in results if r['is_code_generated'])
    uploaded = sum(1 for r in results if r['is_uploaded'])
    has_creator = sum(1 for r in results if r['created_by_notebooks'])
    has_usage = sum(1 for r in results if r['used_in'])
    
    report.append("SUMMARY STATISTICS:")
    report.append(f"  Total CSV files: {len(results)}")
    report.append(f"  Code-generated: {code_generated}")
    report.append(f"  Likely uploaded: {uploaded}")
    report.append(f"  Created by notebooks: {has_creator}")
    report.append(f"  Used in code: {has_usage}")
    report.append("")
    report.append("=" * 100)
    report.append("")
    
    # Detailed list
    for result in sorted(results, key=lambda x: x['csv_file']):
        report.append(f"CSV File: {result['csv_file']}")
        
        if result['created_by_notebooks']:
            report.append(f"  Created by: {', '.join(result['created_by_notebooks'])}")
        else:
            report.append(f"  Created by: UNKNOWN (likely uploaded)")
        
        if result['used_in']:
            report.append(f"  Used in: {', '.join(result['used_in'][:5])}")
            if len(result['used_in']) > 5:
                report.append(f"    ... and {len(result['used_in']) - 5} more files")
        else:
            report.append(f"  Used in: NOT USED")
        
        report.append(f"  Type: {'CODE-GENERATED' if result['is_code_generated'] else 'LIKELY UPLOADED'}")
        report.append("")
    
    return '\n'.join(report)

if __name__ == '__main__':
    results = analyze_all_csvs()
    report = generate_report(results)
    
    # Save report
    output_file = PROJECT_ROOT / 'csv_analysis_report.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE!")
    print(f"Report saved to: {output_file}")
    print("=" * 100)
    
    # Also print summary
    print("\n" + report.split("=" * 100)[1])



















