#!/usr/bin/env python3
"""
Script to analyze column names in CSV/Excel files within PPMI folders.
Reads each file and extracts column information with folder mappings.
"""

import os
import json
import csv
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Any
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore')


def get_column_names_from_file(file_path: str) -> List[str]:
    """
    Extract column names from a CSV or Excel file.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        List of column names, empty list if error occurs
    """
    try:
        file_lower = file_path.lower()
        
        if file_lower.endswith('.csv'):
            # Read CSV file
            df = pd.read_csv(file_path, nrows=0)  # Only read header
            return df.columns.tolist()
        
        elif file_lower.endswith(('.xlsx', '.xls')):
            # Read Excel file
            df = pd.read_excel(file_path, nrows=0)  # Only read header
            return df.columns.tolist()
        
        else:
            return []
            
    except Exception as e:
        print(f"Warning: Could not read file {file_path}: {str(e)}")
        return []


def analyze_ppmi_columns(ppmi_path: str) -> Tuple[Dict[str, List[Tuple[str, str]]], List[str], Dict[str, List[str]]]:
    """
    Analyze column names in PPMI CSV/Excel files.
    
    Args:
        ppmi_path (str): Path to the PPMI folder
        
    Returns:
        Tuple containing:
        - Dictionary mapping column names to list of (folder_name, filename) tuples
        - List of modified_folder_name_column names
        - Dictionary mapping filename to list of modified column names
    """
    
    # Folder name mapping
    folder_mapping = {
        "Medical_History": "medical_history",
        "Motor___MDS-UPDRS": "motor_assessments",
        "Non-motor_Assessments": "non_motor_assessments",
        "_Subject_Characteristics": "subject_characteristics"
    }
    
    # Initialize return variables
    column_mapping = {}  # column_name -> [(folder, filename), ...]
    modified_column_names = []  # modified_folder_name_column format
    filename_to_columns = {}  # filename -> [modified_column_names]
    
    # Check if PPMI path exists
    if not os.path.exists(ppmi_path):
        print(f"Error: PPMI path '{ppmi_path}' does not exist!")
        return column_mapping, modified_column_names, filename_to_columns
    
    # Get all items in PPMI folder
    try:
        items = os.listdir(ppmi_path)
    except PermissionError:
        print(f"Error: Permission denied accessing '{ppmi_path}'")
        return column_mapping, modified_column_names, filename_to_columns
    
    print("Analyzing files and extracting column names...")
    print("-" * 50)
    
    # Process each folder
    total_files_processed = 0
    total_columns_found = 0
    
    for item in items:
        item_path = os.path.join(ppmi_path, item)
        
        # Skip hidden files and only process directories
        if item.startswith('.') or not os.path.isdir(item_path):
            continue
        
        # Skip folders not in our mapping
        if item not in folder_mapping:
            print(f"Skipping unmapped folder: {item}")
            continue
        
        modified_folder_name = folder_mapping[item]
        print(f"Processing folder: {item} -> {modified_folder_name}")
        
        try:
            folder_items = os.listdir(item_path)
            folder_file_count = 0
            
            for file_item in folder_items:
                file_path = os.path.join(item_path, file_item)
                
                # Only process actual files (not subdirectories)
                if not os.path.isfile(file_path):
                    continue
                
                # Check if file is CSV or Excel
                file_lower = file_item.lower()
                if not file_lower.endswith(('.csv', '.xlsx', '.xls')):
                    continue
                
                # Get column names from the file
                column_names = get_column_names_from_file(file_path)
                
                if column_names:
                    folder_file_count += 1
                    total_files_processed += 1
                    
                    # Initialize list for this file's modified column names
                    file_modified_columns = []
                    
                    # Process each column
                    for column_name in column_names:
                        # Clean column name (strip whitespace)
                        clean_column_name = str(column_name).strip()
                        
                        # Add to column mapping
                        if clean_column_name not in column_mapping:
                            column_mapping[clean_column_name] = []
                        
                        column_mapping[clean_column_name].append((item, file_item))
                        
                        # Create modified column name
                        modified_name = f"{modified_folder_name}_{clean_column_name}"
                        modified_column_names.append(modified_name)
                        file_modified_columns.append(modified_name)
                        total_columns_found += 1
                    
                    # Store modified column names for this file
                    filename_to_columns[file_item] = file_modified_columns
                
                print(f"  ✓ Processed: {file_item} ({len(column_names)} columns)")
            
            print(f"  → Processed {folder_file_count} files in {item}")
            
        except PermissionError:
            print(f"  ✗ Permission denied accessing folder '{item_path}'")
        except Exception as e:
            print(f"  ✗ Error processing folder '{item}': {str(e)}")
    
    print(f"\nProcessing complete!")
    print(f"Total files processed: {total_files_processed}")
    print(f"Total columns found: {total_columns_found}")
    print(f"Unique column names: {len(column_mapping)}")
    
    return column_mapping, modified_column_names, filename_to_columns


def save_column_analysis_to_files(column_mapping: Dict[str, List[Tuple[str, str]]], 
                                 modified_column_names: List[str],
                                 filename_to_columns: Dict[str, List[str]],
                                 output_dir: str = "output"):
    """
    Save the column analysis results to multiple file formats.
    
    Args:
        column_mapping: Dictionary mapping column names to (folder, filename) tuples
        modified_column_names: List of modified column names
        filename_to_columns: Dictionary mapping filename to modified column names
        output_dir: Directory to save output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save complete analysis as JSON file
    json_data = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_unique_columns": len(column_mapping),
            "total_column_instances": len(modified_column_names),
            "total_modified_names": len(set(modified_column_names))  # unique modified names
        },
        "column_mapping": {
            col_name: [{"folder": folder, "filename": filename} for folder, filename in locations]
            for col_name, locations in column_mapping.items()
        },
        "modified_column_names": modified_column_names,
        "filename_to_columns": filename_to_columns
    }
    
    json_file = os.path.join(output_dir, f"ppmi_column_analysis_{timestamp}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Complete analysis saved to JSON: {json_file}")
    
    # 2. Save column mapping to CSV (detailed view)
    column_mapping_file = os.path.join(output_dir, f"ppmi_column_mapping_{timestamp}.csv")
    with open(column_mapping_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Column_Name', 'Folder', 'Filename', 'File_Count'])
        
        for col_name, locations in sorted(column_mapping.items()):
            for folder, filename in locations:
                writer.writerow([col_name, folder, filename, len(locations)])
    print(f"✓ Column mapping saved to CSV: {column_mapping_file}")
    
    # 3. Save modified column names to CSV
    modified_names_file = os.path.join(output_dir, f"ppmi_modified_column_names_{timestamp}.csv")
    with open(modified_names_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Index', 'Modified_Column_Name', 'Original_Column', 'Folder_Prefix'])
        
        for i, modified_name in enumerate(modified_column_names, 1):
            # Split to get folder prefix and original column
            # Format is: folder_prefix_original_column
            parts = modified_name.split('_', 1)
            if len(parts) == 2:
                folder_prefix, original_column = parts
                writer.writerow([i, modified_name, original_column, folder_prefix])
            else:
                writer.writerow([i, modified_name, modified_name, ''])
    print(f"✓ Modified column names saved to CSV: {modified_names_file}")
    
    # 4. Save unique columns summary
    unique_columns_file = os.path.join(output_dir, f"ppmi_unique_columns_summary_{timestamp}.csv")
    with open(unique_columns_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Column_Name', 'Appears_In_Files', 'Folder_Count', 'Folders'])
        
        for col_name, locations in sorted(column_mapping.items()):
            folders = list(set([folder for folder, _ in locations]))
            writer.writerow([
                col_name, 
                len(locations), 
                len(folders),
                '; '.join(sorted(folders))
            ])
    print(f"✓ Unique columns summary saved to CSV: {unique_columns_file}")
    
    # 5. Save filename to columns mapping to CSV
    filename_columns_file = os.path.join(output_dir, f"ppmi_filename_to_columns_{timestamp}.csv")
    with open(filename_columns_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Filename', 'Column_Count', 'Modified_Column_Names'])
        
        for filename, columns in sorted(filename_to_columns.items()):
            # Join all column names with semicolon separator
            columns_str = '; '.join(columns)
            writer.writerow([filename, len(columns), columns_str])
    print(f"✓ Filename to columns mapping saved to CSV: {filename_columns_file}")
    
    # 6. Save summary report as text file
    summary_file = os.path.join(output_dir, f"ppmi_column_analysis_summary_{timestamp}.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("PPMI COLUMN ANALYSIS SUMMARY REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"PPMI Path: /Users/ahmedabdullah/Desktop/ppmi/PIE/PPMI\n\n")
        
        f.write("STATISTICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total unique column names: {len(column_mapping)}\n")
        f.write(f"Total column instances: {len(modified_column_names)}\n")
        f.write(f"Unique modified names: {len(set(modified_column_names))}\n\n")
        
        f.write("TOP 20 MOST COMMON COLUMNS:\n")
        f.write("-" * 35 + "\n")
        # Sort columns by frequency
        column_frequency = [(col, len(locations)) for col, locations in column_mapping.items()]
        column_frequency.sort(key=lambda x: x[1], reverse=True)
        
        for i, (col_name, count) in enumerate(column_frequency[:20], 1):
            f.write(f"{i:2d}. {col_name} (appears in {count} files)\n")
        
        f.write(f"\nFOLDER MAPPING USED:\n")
        f.write("-" * 25 + "\n")
        folder_mapping = {
            "Medical_History": "medical_history",
            "Motor___MDS-UPDRS": "motor_assessments",
            "Non-motor_Assessments": "non_motor_assessments",
            "_Subject_Characteristics": "subject_characteristics"
        }
        for original, modified in folder_mapping.items():
            f.write(f"{original} -> {modified}\n")
    
    print(f"✓ Summary report saved to TXT: {summary_file}")
    
    return {
        'json_file': json_file,
        'column_mapping_file': column_mapping_file,
        'modified_names_file': modified_names_file,
        'unique_columns_file': unique_columns_file,
        'filename_columns_file': filename_columns_file,
        'summary_file': summary_file
    }


def print_column_analysis_results(column_mapping: Dict[str, List[Tuple[str, str]]], 
                                 modified_column_names: List[str],
                                 filename_to_columns: Dict[str, List[str]]):
    """Print the column analysis results in a formatted way."""
    
    print("=" * 80)
    print("PPMI COLUMN ANALYSIS RESULTS")
    print("=" * 80)
    
    print(f"\n1. COLUMN MAPPING SUMMARY:")
    print("-" * 40)
    print(f"Total unique column names found: {len(column_mapping)}")
    print(f"Total column instances: {len(modified_column_names)}")
    print(f"Unique modified names: {len(set(modified_column_names))}")
    
    print(f"\n2. TOP 15 MOST COMMON COLUMNS:")
    print("-" * 40)
    # Sort columns by frequency
    column_frequency = [(col, len(locations)) for col, locations in column_mapping.items()]
    column_frequency.sort(key=lambda x: x[1], reverse=True)
    
    for i, (col_name, count) in enumerate(column_frequency[:15], 1):
        print(f"{i:2d}. '{col_name}' (appears in {count} files)")
    
    print(f"\n3. SAMPLE MODIFIED COLUMN NAMES (first 20):")
    print("-" * 50)
    unique_modified = list(set(modified_column_names))[:20]
    for i, modified_name in enumerate(unique_modified, 1):
        print(f"{i:2d}. {modified_name}")
    
    if len(set(modified_column_names)) > 20:
        print(f"... and {len(set(modified_column_names)) - 20} more unique modified names")
    
    print(f"\n4. FILENAME TO COLUMNS MAPPING (sample - first 10 files):")
    print("-" * 60)
    sample_files = list(filename_to_columns.items())[:10]
    for i, (filename, columns) in enumerate(sample_files, 1):
        print(f"{i:2d}. {filename} ({len(columns)} columns)")
        # Show first few column names
        sample_columns = columns[:3]
        if len(columns) > 3:
            sample_columns.append(f"... and {len(columns) - 3} more")
        print(f"     Columns: {', '.join(sample_columns)}")


def main():
    """Main function to run the PPMI column analysis."""
    # Define the PPMI path
    ppmi_path = "../PPMI"
    
    print(f"Starting PPMI Column Analysis")
    print(f"PPMI folder: {ppmi_path}")
    print("=" * 80)
    
    # Analyze columns
    column_mapping, modified_column_names, filename_to_columns = analyze_ppmi_columns(ppmi_path)
    
    if not column_mapping:
        print("No columns found or error occurred during analysis.")
        return column_mapping, modified_column_names, filename_to_columns
    
    # Print results to console
    print_column_analysis_results(column_mapping, modified_column_names, filename_to_columns)
    
    # Save results to files
    print(f"\n5. SAVING RESULTS TO FILES:")
    print("-" * 40)
    output_files = save_column_analysis_to_files(column_mapping, modified_column_names, filename_to_columns)
    
    print(f"\n6. OUTPUT FILES CREATED:")
    print("-" * 30)
    for file_type, file_path in output_files.items():
        print(f"• {file_type.replace('_', ' ').title()}: {os.path.basename(file_path)}")
    
    # Return the results for programmatic use
    return column_mapping, modified_column_names, filename_to_columns


if __name__ == "__main__":
    # Run the script
    column_mapping, modified_column_names, filename_to_columns = main()