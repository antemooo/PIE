#!/usr/bin/env python3
"""
Script to extract modified column names for specific MDS-UPDRS files from PPMI column analysis output.
Reads the JSON output from analyze_ppmi_columns and filters for specified files.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Set


def read_column_analysis_results(json_file_path: str) -> Dict:
    """
    Read the JSON output from analyze_ppmi_columns script.
    
    Args:
        json_file_path (str): Path to the JSON file
        
    Returns:
        Dictionary containing the analysis results
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_file_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {json_file_path}: {str(e)}")
        return {}
    except Exception as e:
        print(f"Error reading JSON file: {str(e)}")
        return {}


def extract_mds_updrs_columns(analysis_data: Dict, target_files: List[str]) -> List[str]:
    """
    Extract modified column names for specific MDS-UPDRS files.
    
    Args:
        analysis_data (Dict): The loaded JSON data from analyze_ppmi_columns
        target_files (List[str]): List of target filenames to extract columns for
        
    Returns:
        List of modified column names for the target files
    """
    if not analysis_data or 'filename_to_columns' not in analysis_data:
        print("Error: Invalid analysis data or missing filename_to_columns")
        return []
    
    filename_to_columns = analysis_data['filename_to_columns']
    extracted_columns = []
    
    print(f"Looking for {len(target_files)} target files...")
    print("-" * 50)
    
    for target_file in target_files:
        if target_file in filename_to_columns:
            file_columns = filename_to_columns[target_file]
            extracted_columns.extend(file_columns)
            print(f"✓ Found {target_file}: {len(file_columns)} columns")
        else:
            print(f"✗ File not found: {target_file}")
    
    print(f"\nTotal columns extracted: {len(extracted_columns)}")
    print(f"Unique columns: {len(set(extracted_columns))}")
    
    return extracted_columns


def save_extracted_columns(columns: List[str], output_file: str) -> None:
    """
    Save the extracted column names to a text file.
    
    Args:
        columns (List[str]): List of column names to save
        output_file (str): Path to the output text file
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write header
            f.write("MDS-UPDRS Modified Column Names\n")
            f.write("=" * 40 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total columns: {len(columns)}\n")
            f.write(f"Unique columns: {len(set(columns))}\n\n")
            
            # Write column names (one per line)
            for column in columns:
                f.write(f"{column}\n")
        
        print(f"✓ Column names saved to: {output_file}")
        
    except Exception as e:
        print(f"Error saving to file {output_file}: {str(e)}")


def find_latest_analysis_file(output_dir: str = "output") -> str:
    """
    Find the most recent column analysis JSON file.
    
    Args:
        output_dir (str): Directory to search for analysis files
        
    Returns:
        Path to the most recent analysis JSON file, empty string if none found
    """
    if not os.path.exists(output_dir):
        print(f"Error: Output directory {output_dir} does not exist")
        return ""
    
    try:
        # Look for ppmi_column_analysis_*.json files
        analysis_files = [
            f for f in os.listdir(output_dir) 
            if f.startswith('ppmi_column_analysis_') and f.endswith('.json')
        ]
        
        if not analysis_files:
            print(f"No column analysis JSON files found in {output_dir}")
            return ""
        
        # Sort by filename (which includes timestamp) to get the latest
        analysis_files.sort(reverse=True)
        latest_file = os.path.join(output_dir, analysis_files[0])
        
        print(f"Found latest analysis file: {analysis_files[0]}")
        return latest_file
        
    except Exception as e:
        print(f"Error searching for analysis files: {str(e)}")
        return ""


def main():
    """Main function to extract MDS-UPDRS column names."""
    
    # Target MDS-UPDRS files
    target_files = [
        "MDS-UPDRS_Part_III_17May2025.csv",
        "MDS-UPDRS_Part_IV__Motor_Complications_17May2025.csv",
        "MDS-UPDRS_Part_I_17May2025.csv",
        "MDS-UPDRS_Part_I_Patient_Questionnaire_17May2025.csv",
        "MDS_UPDRS_Part_II__Patient_Questionnaire_17May2025.csv"
    ]
    
    print("MDS-UPDRS Column Extractor")
    print("=" * 50)
    print(f"Target files ({len(target_files)}):")
    for i, file in enumerate(target_files, 1):
        print(f"  {i}. {file}")
    print()
    
    # Find the latest analysis file
    json_file_path = find_latest_analysis_file()
    if not json_file_path:
        print("Please run analyze_ppmi_columns.py first to generate analysis data.")
        return [], ""
    
    # Read the analysis results
    print(f"Reading analysis data from: {os.path.basename(json_file_path)}")
    analysis_data = read_column_analysis_results(json_file_path)
    
    if not analysis_data:
        print("Failed to read analysis data.")
        return [], ""
    
    # Extract columns for target files
    extracted_columns = extract_mds_updrs_columns(analysis_data, target_files)
    
    if not extracted_columns:
        print("No columns extracted. Please check if the target files exist in the analysis.")
        return [], ""
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"output/mds_updrs_columns_{timestamp}.txt"
    
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    
    # Save the extracted columns
    save_extracted_columns(extracted_columns, output_file)
    
    # Print summary
    print(f"\nSUMMARY:")
    print("-" * 20)
    print(f"Files processed: {len([f for f in target_files if f in analysis_data.get('filename_to_columns', {})])}/{len(target_files)}")
    print(f"Total columns extracted: {len(extracted_columns)}")
    print(f"Unique columns: {len(set(extracted_columns))}")
    print(f"Output file: {output_file}")
    
    # Show first few column names as preview
    if extracted_columns:
        print(f"\nPREVIEW (first 10 columns):")
        print("-" * 30)
        for i, col in enumerate(extracted_columns[:10], 1):
            print(f"{i:2d}. {col}")
        if len(extracted_columns) > 10:
            print(f"... and {len(extracted_columns) - 10} more columns")
    
    return extracted_columns, output_file


if __name__ == "__main__":
    extracted_columns, output_file = main()