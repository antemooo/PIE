#!/usr/bin/env python3
"""
Script to traverse the PPMI folder and extract CSV/Excel file information.
Only reads .csv, .xlsx, and .xls files from each folder.
"""

import os
import json
import csv
from datetime import datetime
from typing import List, Dict, Tuple


def traverse_ppmi_folder(ppmi_path: str) -> Tuple[List[str], Dict[str, List[str]], List[str]]:
    """
    Traverse the PPMI folder and extract folder names, CSV/Excel files, and combined names.
    Only processes .csv, .xlsx, and .xls files.
    
    Args:
        ppmi_path (str): Path to the PPMI folder
        
    Returns:
        Tuple containing:
        - List of folder names
        - Dictionary mapping folder names to their CSV/Excel files
        - List of combined folder_file names (CSV/Excel files only)
    """
    # Initialize return variables
    folder_names = []
    folder_files_dict = {}
    combined_names = []
    
    # Check if PPMI path exists
    if not os.path.exists(ppmi_path):
        print(f"Error: PPMI path '{ppmi_path}' does not exist!")
        return folder_names, folder_files_dict, combined_names
    
    # Get all items in PPMI folder
    try:
        items = os.listdir(ppmi_path)
    except PermissionError:
        print(f"Error: Permission denied accessing '{ppmi_path}'")
        return folder_names, folder_files_dict, combined_names
    
    # Filter out hidden files and keep only directories
    for item in items:
        item_path = os.path.join(ppmi_path, item)
        
        # Skip hidden files (starting with .)
        if item.startswith('.'):
            continue
            
        # Only process directories
        if os.path.isdir(item_path):
            folder_names.append(item)
            
            # Get CSV and Excel files in this folder
            try:
                files_in_folder = []
                folder_items = os.listdir(item_path)
                
                for file_item in folder_items:
                    file_path = os.path.join(item_path, file_item)
                    # Only include actual files, not subdirectories
                    if os.path.isfile(file_path):
                        # Check if file is CSV or Excel
                        file_lower = file_item.lower()
                        if file_lower.endswith(('.csv', '.xlsx', '.xls')):
                            files_in_folder.append(file_item)
                            # Create combined folder_file name
                            combined_names.append(f"{item}_{file_item}")
                
                # Store CSV/Excel files for this folder
                folder_files_dict[item] = files_in_folder
                
            except PermissionError:
                print(f"Warning: Permission denied accessing folder '{item_path}'")
                folder_files_dict[item] = []
    
    # Sort all lists for consistent output
    folder_names.sort()
    combined_names.sort()
    for folder in folder_files_dict:
        folder_files_dict[folder].sort()
    
    return folder_names, folder_files_dict, combined_names


def save_results_to_files(folder_names: List[str], folder_files_dict: Dict[str, List[str]], 
                         combined_names: List[str], output_dir: str = "output"):
    """
    Save the results to multiple file formats.
    
    Args:
        folder_names: List of folder names
        folder_files_dict: Dictionary of folder to files mapping
        combined_names: List of combined folder_file names
        output_dir: Directory to save output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save as JSON file
    json_data = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_folders": len(folder_names),
            "total_csv_excel_files": sum(len(files) for files in folder_files_dict.values()),
            "total_combined_names": len(combined_names)
        },
        "folder_names": folder_names,
        "folder_files_dict": folder_files_dict,
        "combined_names": combined_names
    }
    
    json_file = os.path.join(output_dir, f"ppmi_traverse_results_{timestamp}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Results saved to JSON: {json_file}")
    
    # 2. Save folder names to CSV
    folder_names_file = os.path.join(output_dir, f"ppmi_folder_names_{timestamp}.csv")
    with open(folder_names_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Index', 'Folder_Name'])
        for i, folder in enumerate(folder_names, 1):
            writer.writerow([i, folder])
    print(f"✓ Folder names saved to CSV: {folder_names_file}")
    
    # 3. Save combined names to CSV
    combined_names_file = os.path.join(output_dir, f"ppmi_combined_names_{timestamp}.csv")
    with open(combined_names_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Index', 'Combined_Name', 'Folder', 'File'])
        for i, combined in enumerate(combined_names, 1):
            # Find the folder name by matching against known folder names
            folder_name = ""
            file_name = ""
            for folder in folder_names:
                if combined.startswith(folder + "_"):
                    folder_name = folder
                    file_name = combined[len(folder) + 1:]  # +1 for the underscore
                    break
            writer.writerow([i, combined, folder_name, file_name])
    print(f"✓ Combined names saved to CSV: {combined_names_file}")
    
    # 4. Save detailed folder-files mapping to CSV
    detailed_file = os.path.join(output_dir, f"ppmi_folder_files_detailed_{timestamp}.csv")
    with open(detailed_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Folder', 'File_Index', 'File_Name', 'Combined_Name'])
        for folder, files in folder_files_dict.items():
            for i, file in enumerate(files, 1):
                combined = f"{folder}_{file}"
                writer.writerow([folder, i, file, combined])
    print(f"✓ Detailed mapping saved to CSV: {detailed_file}")
    
    # 5. Save summary report as text file
    summary_file = os.path.join(output_dir, f"ppmi_traverse_summary_{timestamp}.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("PPMI FOLDER TRAVERSAL SUMMARY REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"PPMI Path: /Users/ahmedabdullah/Desktop/ppmi/PIE/PPMI\n\n")
        
        f.write("STATISTICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total folders found: {len(folder_names)}\n")
        f.write(f"Total CSV/Excel files found: {sum(len(files) for files in folder_files_dict.values())}\n")
        f.write(f"Total combined names: {len(combined_names)}\n\n")
        
        f.write("FOLDERS AND FILE COUNTS:\n")
        f.write("-" * 30 + "\n")
        for folder, files in folder_files_dict.items():
            f.write(f"{folder}: {len(files)} files\n")
        
        f.write(f"\nFOLDER NAMES:\n")
        f.write("-" * 15 + "\n")
        for i, folder in enumerate(folder_names, 1):
            f.write(f"{i:2d}. {folder}\n")
    
    print(f"✓ Summary report saved to TXT: {summary_file}")
    
    return {
        'json_file': json_file,
        'folder_names_file': folder_names_file,
        'combined_names_file': combined_names_file,
        'detailed_file': detailed_file,
        'summary_file': summary_file
    }


def print_results(folder_names: List[str], folder_files_dict: Dict[str, List[str]], combined_names: List[str]):
    """Print the results in a formatted way."""
    
    print("=" * 80)
    print("PPMI FOLDER TRAVERSAL RESULTS")
    print("=" * 80)
    
    print(f"\n1. FOLDER NAMES ({len(folder_names)} folders found):")
    print("-" * 50)
    for i, folder in enumerate(folder_names, 1):
        print(f"{i:2d}. {folder}")
    
    print(f"\n2. FOLDER-FILES DICTIONARY (CSV/Excel files only):")
    print("-" * 50)
    for folder, files in folder_files_dict.items():
        print(f"\n'{folder}' ({len(files)} CSV/Excel files):")
        for j, file in enumerate(files, 1):
            print(f"    {j:2d}. {file}")
    
    print(f"\n3. COMBINED FOLDER_FILE NAMES ({len(combined_names)} CSV/Excel combinations):")
    print("-" * 50)
    for i, combined in enumerate(combined_names, 1):
        print(f"{i:3d}. {combined}")


def main():
    """Main function to run the PPMI folder traversal."""
    # Define the PPMI path
    ppmi_path = "/Users/ahmedabdullah/Desktop/ppmi/PIE/PPMI"
    
    print(f"Traversing PPMI folder: {ppmi_path}")
    print("=" * 80)
    
    # Traverse the folder
    folder_names, folder_files_dict, combined_names = traverse_ppmi_folder(ppmi_path)
    
    # Print results to console
    print_results(folder_names, folder_files_dict, combined_names)
    
    # Print summary statistics
    print(f"\n4. SUMMARY:")
    print("-" * 50)
    print(f"Total folders found: {len(folder_names)}")
    print(f"Total CSV/Excel files found: {sum(len(files) for files in folder_files_dict.values())}")
    print(f"Total combined names: {len(combined_names)}")
    
    # Save results to files
    print(f"\n5. SAVING RESULTS TO FILES:")
    print("-" * 50)
    output_files = save_results_to_files(folder_names, folder_files_dict, combined_names)
    
    print(f"\n6. OUTPUT FILES CREATED:")
    print("-" * 30)
    for file_type, file_path in output_files.items():
        print(f"• {file_type.replace('_', ' ').title()}: {os.path.basename(file_path)}")
    
    # Return the results for programmatic use
    return folder_names, folder_files_dict, combined_names


if __name__ == "__main__":
    # Run the script
    folder_names, folder_files_dict, combined_names = main()