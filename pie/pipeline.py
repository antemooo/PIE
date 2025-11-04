#!/usr/bin/env python3
"""
Master pipeline script for the PIE project.

This script orchestrates the entire workflow from data loading to classification,
generating reports at each stage and a final summary report.
"""

import os
import sys
import logging
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import webbrowser
from datetime import datetime
from typing import List, Dict, Optional
import time
import re

# Add the parent directory to the Python path to make the pie module importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import PIE-clean modules
from pie_clean import DataLoader
from pie_clean import ALL_MODALITIES

# Import PIE modules
from pie.data_reducer import DataReducer
from pie.feature_engineer import FeatureEngineer
from pie.classification_report import generate_report as run_classification_step
from pie.feature_selector import FeatureSelector
from pie.reporting import (
    generate_data_reduction_html_report,
    generate_feature_engineering_report_html,
    generate_feature_selection_report_html
)

# Imports for feature selection step
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectFdr, f_classif, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("PIE.pipeline")


# --- TIMING DECORATOR ---
def timing_decorator(func):
    """A simple decorator to log the execution time of a function."""
    def wrapper(*args, **kwargs):
        logger.info(f"--- Timing: Starting '{func.__name__}' ---")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"--- Timing: Finished '{func.__name__}' in {elapsed_time:.2f} seconds ---")
        return result
    return wrapper


# --- REPORTING HELPER FUNCTIONS (ADAPTED FROM TEST SCRIPTS) ---

def _calculate_dict_size(data_dict: dict) -> float:
    """Calculates the total memory usage of DataFrames in a dictionary (in MB)."""
    total_size_bytes = 0
    for key, value in data_dict.items():
        if isinstance(value, pd.DataFrame):
            try:
                total_size_bytes += value.memory_usage(deep=True, index=True).sum()
            except Exception as e:
                 logging.warning(f"Could not calculate size for DataFrame '{key}': {e}")
        elif isinstance(value, dict):
            total_size_bytes += _calculate_dict_size(value)
    return total_size_bytes / (1024 * 1024)

def _get_dict_summary(data_dict: dict) -> dict:
    """Calculates shape and basic stats for each DataFrame, returns as dict."""
    summary = {}
    total_rows_sum, total_cols_sum, total_numeric_cols_sum, total_object_cols_sum, total_nulls_sum, total_cells_sum, dataframe_count = 0, 0, 0, 0, 0, 0, 0
    for key, value in data_dict.items():
        if isinstance(value, pd.DataFrame):
            if not value.empty:
                dataframe_count += 1
                rows, cols = value.shape
                total_rows_sum += rows; total_cols_sum += cols
                null_sum = value.isnull().sum().sum(); df_cells = rows * cols
                total_nulls_sum += null_sum; total_cells_sum += df_cells
                summary[key] = {"shape": (rows, cols), "null_pct": (null_sum / df_cells) * 100 if df_cells > 0 else 0,
                                "numeric_cols": value.select_dtypes(include=np.number).shape[1],
                                "object_cols": value.select_dtypes(include='object').shape[1], "is_empty": False}
                total_numeric_cols_sum += summary[key]['numeric_cols']; total_object_cols_sum += summary[key]['object_cols']
            else:
                 summary[key] = {"shape": (0,0), "null_pct": 0, "numeric_cols": 0, "object_cols": 0, "is_empty": True}
        elif isinstance(value, dict):
            if not value: summary[key] = {"shape": "Empty Dict", "is_empty": True}; continue
            for sub_key, sub_value in value.items():
                 if isinstance(sub_value, pd.DataFrame) and not sub_value.empty:
                     dataframe_count += 1; rows, cols = sub_value.shape
                     total_rows_sum += rows; total_cols_sum += cols
                     null_sum = sub_value.isnull().sum().sum(); df_cells = rows * cols
                     total_nulls_sum += null_sum; total_cells_sum += df_cells
                     summary[f"{key}.{sub_key}"] = {"shape": (rows, cols), "null_pct": (null_sum / df_cells) * 100 if df_cells > 0 else 0,
                                                   "numeric_cols": sub_value.select_dtypes(include=np.number).shape[1],
                                                   "object_cols": sub_value.select_dtypes(include='object').shape[1], "is_empty": False}
                     total_numeric_cols_sum += summary[f"{key}.{sub_key}"]['numeric_cols']; total_object_cols_sum += summary[f"{key}.{sub_key}"]['object_cols']
                 else: summary[f"{key}.{sub_key}"] = {"shape": (0,0), "null_pct": 0, "numeric_cols": 0, "object_cols": 0, "is_empty": True}
    summary['totals'] = {"dataframe_count": dataframe_count, "total_rows_sum": total_rows_sum, "total_columns_sum": total_cols_sum,
                         "total_numeric_cols_sum": total_numeric_cols_sum, "total_object_cols_sum": total_object_cols_sum,
                         "overall_null_pct": (total_nulls_sum / total_cells_sum) * 100 if total_cells_sum > 0 else 0}
    return summary

def _generate_reduction_report_html(initial_dict_summary, reduced_dict_summary, analysis_report, initial_size_mb, reduced_size_mb, output_html_path, final_consolidated_df_shape):
    generate_data_reduction_html_report(initial_dict_summary, reduced_dict_summary, analysis_report, initial_size_mb, reduced_size_mb, output_html_path, final_consolidated_df_shape)

def _generate_feature_engineering_report_html(report_data, output_html_path):
    generate_feature_engineering_report_html(report_data, output_html_path)

def _generate_feature_selection_report_html(report_data, output_html_path):
    generate_feature_selection_report_html(report_data, output_html_path)


# --- PIPELINE STEPS ---

@timing_decorator
def run_data_reduction_step(data_dir: str, output_csv_path: Path, output_html_path: Path, modalities: Optional[List[str]] = None) -> dict:
    # Caching: store loaded data_dict as a pickle file
    cache_dir = Path("output/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "data_dict.pkl"

    import pickle
    # Load or create data_dict only in this function
    if cache_file.exists():
        logger.info(f"Loading cached data_dict from {cache_file}")
        with open(cache_file, "rb") as f:
            loaded = pickle.load(f)
        if isinstance(loaded, dict):
            # Ensure all keys are strings
            data_dict = {str(k): v for k, v in loaded.items()}
        elif hasattr(loaded, 'to_dict'):
            logger.warning("Cached data_dict is not a dict, converting using to_dict().")
            temp_dict = loaded.to_dict()
            data_dict = {str(k): v for k, v in temp_dict.items()}
        else:
            logger.error("Cached data_dict is not a dict or DataFrame. Re-running DataLoader.load...")
            loaded = DataLoader.load(data_path=data_dir, merge_output=False, modalities=modalities)
            data_dict = {str(k): v for k, v in loaded.items()}
            with open(cache_file, "wb") as f:
                pickle.dump(data_dict, f)
            logger.info(f"Cached data_dict to {cache_file}")
    else:
        logger.info("No cached data_dict found. Running DataLoader.load...")
        loaded = DataLoader.load(data_path=data_dir, merge_output=False, modalities=modalities)
        data_dict = {str(k): v for k, v in loaded.items()}
        with open(cache_file, "wb") as f:
            pickle.dump(data_dict, f)
        logger.info(f"Cached data_dict to {cache_file}")
    """Loads, reduces, merges, and consolidates data."""
    logger.info("Starting data loading and reduction step...")
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}. Step cannot proceed.")
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Caching: store loaded data_dict as a pickle file
    cache_dir = Path("output/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "data_dict.pkl"

    import pickle
    if cache_file.exists():
        logger.info(f"Loading cached data_dict from {cache_file}")
        with open(cache_file, "rb") as f:
            loaded = pickle.load(f)
        data_dict = loaded
    else:
        logger.info("No cached data_dict found. Running DataLoader.load...")
        data_dict = DataLoader.load(data_path=data_dir, merge_output=False, modalities=modalities)
        with open(cache_file, "wb") as f:
            pickle.dump(data_dict, f)
        logger.info(f"Cached data_dict to {cache_file}")

    # Ensure data_dict is a dict before downstream usage
    if not isinstance(data_dict, dict):
        if hasattr(data_dict, 'to_dict'):
            logger.warning("data_dict is not a dict, converting using to_dict().")
            data_dict = data_dict.to_dict()
        else:
            raise TypeError("data_dict is not a dict and cannot be converted.")

    initial_size_mb = _calculate_dict_size(data_dict)
    initial_summary = _get_dict_summary(data_dict)

    reducer = DataReducer(data_dict)
    analysis_results = reducer.analyze()
    drop_suggestions = reducer.get_drop_suggestions(analysis_results)
    reduced_dict = reducer.apply_drops(drop_suggestions)
    
    reduced_size_mb = _calculate_dict_size(reduced_dict)
    reduced_summary = _get_dict_summary(reduced_dict)

    merged_df = reducer.merge_reduced_data(reduced_dict, output_filename="merged_temp.csv")
    final_df = reducer.consolidate_cohort_columns(merged_df) if not merged_df.empty else pd.DataFrame()

    if not final_df.empty:
        final_df.to_csv(output_csv_path, index=False)
        logger.info(f"Final reduced and consolidated data saved to: {output_csv_path}")
    else:
        logger.warning("Final DataFrame is empty. Not saving CSV.")

    _generate_reduction_report_html(
        initial_dict_summary=initial_summary,
        reduced_dict_summary=reduced_summary,
        analysis_report=analysis_results,
        initial_size_mb=initial_size_mb,
        reduced_size_mb=reduced_size_mb,
        output_html_path=str(output_html_path),
        final_consolidated_df_shape=final_df.shape if not final_df.empty else (0, 0)
    )
    
    # Get the relative path for the main report
    report_path = Path(os.path.relpath(output_html_path, output_html_path.parent))

    return {
        "initial_tables": initial_summary.get('totals', {}).get('dataframe_count', 0),
        "reduced_tables": reduced_summary.get('totals', {}).get('dataframe_count', 0),
        "initial_size_mb": initial_size_mb,
        "reduced_size_mb": reduced_size_mb,
        "output_shape": final_df.shape if not final_df.empty else (0,0),
        "report_path": report_path
    }

@timing_decorator
def run_feature_engineering_step(input_csv_path: str, output_csv_path: Path, output_html_path: Path) -> dict:
    """Applies feature engineering to the reduced data."""
    logger.info("Starting feature engineering step...")
    if not os.path.exists(input_csv_path):
        logger.error(f"Input file not found: {input_csv_path}. Step cannot proceed.")
        raise FileNotFoundError(f"Input file not found: {input_csv_path}")

    df = pd.read_csv(input_csv_path)
    report_data = {'input_csv_path': input_csv_path, 'output_csv_path': str(output_csv_path), 'input_data_shape': df.shape}

    if df.empty:
        logger.warning("Input DataFrame is empty. Skipping feature engineering.")
        final_engineered_df = pd.DataFrame()
    else:
        engineer = FeatureEngineer(df.copy())
        engineer.one_hot_encode(auto_identify_threshold=20, max_categories_to_encode=25, min_frequency_for_category=0.01)
        engineer.scale_numeric_features(scaler_type='standard')
        final_engineered_df = engineer.get_dataframe()
        report_data.update(engineer.get_engineered_feature_summary())
    
    if not final_engineered_df.empty:
        final_engineered_df.to_csv(output_csv_path, index=False)
        logger.info(f"Engineered data saved to: {output_csv_path}")
    
    report_data['output_data_shape'] = final_engineered_df.shape if not final_engineered_df.empty else (0,0)
    _generate_feature_engineering_report_html(report_data, str(output_html_path))

    # Get the relative path for the main report
    report_path = Path(os.path.relpath(output_html_path, output_html_path.parent))

    return {
        "input_shape": report_data['input_data_shape'],
        "output_shape": report_data['output_data_shape'],
        "new_features": report_data.get('feature_engineering_summary', {}).get('newly_engineered_features_count', 0),
        "report_path": report_path
    }

@timing_decorator
def run_feature_selection_step(
    input_csv_path: str,
    train_csv_path: Path,
    test_csv_path: Path,
    output_html_path: Path,
    target_column: str,
    fs_method: str,
    fs_param_value: float,
    leakage_features_path: Optional[str] = None
) -> dict:
    """Performs feature selection on the engineered data."""
    logger.info("Starting feature selection step...")
    if not os.path.exists(input_csv_path):
        logger.error(f"Input file not found: {input_csv_path}. Step cannot proceed.")
        raise FileNotFoundError(f"Input file not found: {input_csv_path}")
        
    report_data = {'input_csv_path': input_csv_path, 'target_column': target_column}
    df = pd.read_csv(input_csv_path)
    
    # Remove leakage features before any other processing
    if leakage_features_path and Path(leakage_features_path).exists():
        with open(leakage_features_path, 'r') as f:
            leakage_features = {line.strip() for line in f if line.strip()}
        
        cols_to_drop = [col for col in df.columns if col in leakage_features]
        
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
            logger.info(f"Removed {len(cols_to_drop)} leakage features specified in {leakage_features_path}.")
            report_data['leakage_features_removed'] = ', '.join(cols_to_drop)
    
    initial_rows = len(df)
    df.dropna(subset=[target_column], inplace=True)
    report_data['rows_dropped_missing_target'] = str(initial_rows - len(df))
    report_data['clean_data_shape'] = str(df.shape)

    if 'PATNO' in df.columns:
        df['PATNO'] = df['PATNO'].astype(int)

    id_cols = ['PATNO', 'EVENT_ID']
    feature_cols = [col for col in df.columns if col not in [target_column] + id_cols]
    
    X = df[feature_cols]
    y = df[target_column]

    # Determine task type based on target dtype
    is_regression = pd.api.types.is_numeric_dtype(y)
    task_type = 'regression' if is_regression else 'classification'

    # --- Start of new code ---
    # Handle pipe-separated values in object columns that are likely numeric
    X = X.copy()  # Avoid SettingWithCopyWarning

    # Define patterns for columns to skip. PATNO and EVENT_ID are already excluded
    # from X, but this handles other potential ID/date-like columns.
    ID_DATE_PATTERNS = ['ID', 'DATE', 'TIME', 'PATNO', 'EVENT']

    for col in X.select_dtypes(include=['object']).columns:
        # Skip if it looks like an ID or date column based on name patterns
        if any(pattern in col.upper() for pattern in ID_DATE_PATTERNS):
            logger.info(f"Skipping pipe-averaging for potential ID/date column: '{col}'")
            continue

        if not X[col].astype(str).str.contains(r'\|', na=False).any():
            continue

        # This column has pipes. Let's see if it's mostly numeric.
        logger.info(f"Column '{col}' contains pipe-separated values. Analyzing...")

        def average_pipe_values(val):
            if isinstance(val, str) and '|' in val:
                try:
                    # Split, convert to float, and average
                    return np.mean([float(x) for x in val.split('|')])
                except (ValueError, TypeError):
                    # If any part isn't a number, this value is not numeric
                    return np.nan
            return val

        # Apply the averaging function to a temporary series
        converted_series = X[col].apply(average_pipe_values)
        
        # Now, try to convert the entire series to a numeric type
        numeric_series = pd.to_numeric(converted_series, errors='coerce')

        # Heuristic: If over 90% of the original non-null values can be
        # converted to a number, we'll treat the column as numeric.
        original_non_null_count = X[col].notna().sum()
        numeric_count = numeric_series.notna().sum()

        if original_non_null_count > 0 and (numeric_count / original_non_null_count) > 0.9:
            logger.info(f"Converting column '{col}' to numeric by averaging pipe-separated values.")
            X[col] = numeric_series
        else:
            logger.warning(
                f"Column '{col}' has pipe-separated values but is not consistently numeric. "
                f"({numeric_count}/{original_non_null_count} values converted). "
                "Leaving as is."
            )
    # --- End of new code ---

    # Drop non-numeric columns BEFORE imputation to avoid shape mismatch
    non_numeric_cols = X.select_dtypes(exclude=np.number).columns
    if len(non_numeric_cols) > 0:
        logger.warning(
            f"Dropping {len(non_numeric_cols)} non-numeric columns before imputation: {list(non_numeric_cols)}"
        )
        X = X.drop(columns=non_numeric_cols)

    # Refined imputation: use mean for numeric columns
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    
    # Fit and transform, handling potential column drops
    X_transformed = imputer.fit_transform(X)
    
    # CRITICAL: SimpleImputer may drop all-NaN columns silently
    # We need to identify which columns actually survived imputation
    # The number of columns in X_transformed may be less than X.columns
    if hasattr(imputer, 'feature_names_in_'):
        # Get the columns that the imputer saw during fit
        input_features = imputer.feature_names_in_
        # Check if any columns were dropped (all-NaN columns)
        if X_transformed.shape[1] < len(input_features):
            # Find which columns have all NaN values and were dropped
            all_nan_mask = X[input_features].isna().all()
            kept_columns = input_features[~all_nan_mask]
        else:
            kept_columns = input_features
    else:
        # Fallback: use X.columns but verify shape matches
        if X_transformed.shape[1] == len(X.columns):
            kept_columns = X.columns
        else:
            # Find non-all-NaN columns manually
            all_nan_mask = X.isna().all()
            kept_columns = X.columns[~all_nan_mask]
    
    # Create DataFrame with only the kept columns
    X_imputed = pd.DataFrame(X_transformed, columns=kept_columns, index=X.index)

    # Prepare target and split depending on task type
    if is_regression:
        X_train, X_test, y_train, y_test = train_test_split(
            X_imputed, y, test_size=0.2, random_state=42
        )
        y_train_encoded = y_train  # for API consistency
    else:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        X_train, X_test, y_train, y_test, y_train_encoded, _ = train_test_split(
            X_imputed, y, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

    selector = FeatureSelector(
        method=fs_method,
        task_type=task_type,
        k_or_frac=fs_param_value if fs_method == 'k_best' else None,
        alpha_fdr=fs_param_value if fs_method == 'fdr' else 0.05
    )
    
    selector.fit(X_train, y_train_encoded)
    X_train_final = selector.transform(X_train)
    X_test_final = selector.transform(X_test)

    # Combine selected features with the original target y
    train_df = pd.concat([X_train_final.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test_final.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    logger.info(f"Selected train/test data saved to {train_csv_path} and {test_csv_path}")

    report_data.update({
        'X_train_shape': str(X_train.shape),
        'X_test_shape': str(X_test.shape),
        'y_train_shape': str(y_train.shape),
        'y_test_shape': str(y_test.shape),
        'num_final_selected_features': str(X_train_final.shape[1]),
        'final_train_data_shape': str(train_df.shape),
        'final_test_data_shape': str(test_df.shape),
        'output_train_csv_path': str(train_csv_path),
        'output_test_csv_path': str(test_csv_path),
        'final_selected_feature_names': ', '.join(selector.selected_feature_names_)
    })
    _generate_feature_selection_report_html(report_data, str(output_html_path))

    report_path = Path(os.path.relpath(output_html_path, output_html_path.parent))

    return {
        "initial_features": X.shape[1],
        "final_features": X_train_final.shape[1],
        "train_shape": train_df.shape,
        "test_shape": test_df.shape,
        "report_path": report_path
    }

@timing_decorator
def run_regression_step(
    train_csv_path: str,
    test_csv_path: str,
    target_column: str,
    output_dir: Path,
    leakage_features_path: str = None,
    generate_plots: bool = True
) -> dict:
    """
    Runs comprehensive regression modeling using the regression_report module.
    
    This generates a full regression report with model comparison, visualizations,
    and detailed performance metrics similar to the classification report.
    """
    logger.info("Starting comprehensive regression modeling step...")

    # Import the regression report module
    from pie import regression_report

    # Prepare output directory
    reg_dir = output_dir / "regression"
    reg_dir.mkdir(parents=True, exist_ok=True)

    # Load leakage features to exclude
    exclude_features = []
    if leakage_features_path and Path(leakage_features_path).exists():
        try:
            with open(leakage_features_path, 'r') as f:
                exclude_features = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(exclude_features)} features to exclude from {leakage_features_path}")
        except Exception as e:
            logger.warning(f"Could not read leakage features file: {e}")

    # Generate comprehensive regression report
    report_data = regression_report.generate_report(
        train_csv_path=train_csv_path,
        test_csv_path=test_csv_path,
        target_column=target_column,
        output_dir=str(reg_dir),
        exclude_features=exclude_features,
        generate_plots=generate_plots
    )

    # Extract key metrics from report data
    r2 = report_data.get('best_r2', 'N/A')
    mae = report_data.get('best_mae', 'N/A')
    rmse = report_data.get('best_rmse', 'N/A')
    f_score = report_data.get('best_f_score', 'N/A')

    logger.info(f"Regression complete. Best model: {report_data.get('best_model_name', 'N/A')}")
    logger.info(f"R²: {r2}, MAE: {mae}, RMSE: {rmse}, F-Score: {f_score}")

    # Prepare return data for main pipeline report
    report_html_path = reg_dir / "regression_report.html"
    return {
        'report_path': Path(os.path.relpath(report_html_path, output_dir)),
        'r2': r2 if r2 != 'N/A' else 0.0,
        'mae': mae if mae != 'N/A' else 0.0,
        'rmse': rmse if rmse != 'N/A' else 0.0,
        'f_score': f_score if f_score != 'N/A' else 0.0,
        'leaderboard': report_data.get('leaderboard')
    }

def generate_main_report(report_data: dict, output_path: Path):
    """Generates the main pipeline report that links to sub-reports."""
    logger.info(f"Generating main pipeline report at: {output_path}")

    html_style = """
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; line-height: 1.6; margin: 20px; background-color: #f8f9fa; color: #212529; }
        .container { background-color: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 0 20px rgba(0,0,0,0.05); max-width: 900px; margin: 40px auto; }
        h1, h2 { color: #343a40; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
        h1 { text-align: center; color: #007bff; font-size: 2.2em; }
        table { width: 100%; border-collapse: collapse; margin: 25px 0; }
        th, td { border: 1px solid #dee2e6; padding: 12px; text-align: left; }
        th { background-color: #f2f3f5; font-weight: 600; }
        tr:nth-child(even) { background-color: #f8f9fa; }
        .stage-box { border: 1px solid #e9ecef; padding: 20px; margin-bottom: 20px; background-color: #fff; border-radius: 5px; }
        .stage-title { font-size: 1.5em; color: #495057; margin-bottom: 15px; }
        .report-link { display: inline-block; background-color: #007bff; color: white; padding: 10px 15px; text-decoration: none; border-radius: 5px; font-weight: 500; transition: background-color 0.2s; }
        .report-link:hover { background-color: #0056b3; }
        .metric { font-weight: bold; color: #28a745; }
        .timestamp { color: #6c757d; font-style: italic; text-align: right; margin-top: 20px; font-size: 0.9em; }
    </style>
    """

    html_content = f"""
    <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>PIE Pipeline Overall Report</title>{html_style}</head>
    <body><div class="container">
        <h1>PIE Pipeline Run Summary</h1>
        <p class="timestamp">Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    """

    # --- Data Reduction ---
    if 'reduction' in report_data:
        r = report_data['reduction']
        html_content += f"""
        <div class="stage-box">
            <h2 class="stage-title">1. Data Reduction</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Initial Size / Tables</td><td><span class="metric">{r.get('initial_size_mb', 0.0):.2f} MB</span> / {r.get('initial_tables', 'N/A')} tables</td></tr>
                <tr><td>Reduced Size / Tables</td><td><span class="metric">{r.get('reduced_size_mb', 0.0):.2f} MB</span> / {r.get('reduced_tables', 'N/A')} tables</td></tr>
                <tr><td>Final Consolidated Shape</td><td>{r.get('output_shape', 'N/A')}</td></tr>
            </table>
            <a href="{r['report_path']}" target="_blank" class="report-link">View Full Reduction Report</a>
        </div>
        """

    # --- Feature Engineering ---
    if 'feature_engineering' in report_data:
        fe = report_data['feature_engineering']
        html_content += f"""
        <div class="stage-box">
            <h2 class="stage-title">2. Feature Engineering</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Shape Before Engineering</td><td>{fe.get('input_shape', 'N/A')}</td></tr>
                <tr><td>Shape After Engineering</td><td><span class="metric">{fe.get('output_shape', 'N/A')}</span></td></tr>
                <tr><td>New Features Created</td><td><span class="metric">{fe.get('new_features', 'N/A')}</span></td></tr>
            </table>
            <a href="{fe['report_path']}" target="_blank" class="report-link">View Full Engineering Report</a>
        </div>
        """

    # --- Feature Selection ---
    if 'feature_selection' in report_data:
        fs = report_data['feature_selection']
        html_content += f"""
        <div class="stage-box">
            <h2 class="stage-title">3. Feature Selection</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Features Before Selection</td><td>{fs.get('initial_features', 'N/A')}</td></tr>
                <tr><td>Features After Selection</td><td><span class="metric">{fs.get('final_features', 'N/A')}</span></td></tr>
                <tr><td>Final Train / Test Shape</td><td>{fs.get('train_shape', 'N/A')} / {fs.get('test_shape', 'N/A')}</td></tr>
            </table>
            <a href="{fs['report_path']}" target="_blank" class="report-link">View Full Selection Report</a>
        </div>
        """
    
    # --- Regression ---
    if 'regression' in report_data:
        r = report_data['regression']
        html_content += f"""
        <div class="stage-box">
            <h2 class="stage-title">4. Regression</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>R2</td><td><span class=\"metric\">{r.get('r2', 'N/A'):.4f}</span></td></tr>
                <tr><td>MAE</td><td>{r.get('mae', 'N/A'):.4f}</td></tr>
                <tr><td>RMSE</td><td>{r.get('rmse', 'N/A'):.4f}</td></tr>
            </table>
            <a href="{r['report_path']}" target="_blank" class="report-link">View Full Regression Report</a>
        </div>
        """
    
    # --- Classification ---
    if 'classification' in report_data:
        c = report_data['classification']
        html_content += f"""
        <div class="stage-box">
            <h2 class="stage-title">4. Classification</h2>
            <p>The classification step compares multiple models, tunes the best one (optional), and evaluates its performance on a held-out test set.</p>
             <a href="{c['report_path']}" target="_blank" class="report-link">View Full Classification Report</a>
        </div>
        """

    html_content += "</div></body></html>"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    webbrowser.open(f"file://{os.path.realpath(output_path)}")

def run_pipeline(
    data_dir: str,
    output_dir: str,
    target_column: str,
    leakage_features_path: str,
    modalities: Optional[List[str]] = None,
    fs_method: str = 'fdr',
    fs_param_value: float = 0.05,
    n_models_to_compare: int = 5,
    tune_best_model: bool = False,
    generate_plots: bool = True,
    budget_time_minutes: float = 30.0,
    skip_to_step: Optional[str] = None
):
    """Executes the full PIE pipeline."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pipeline_report_data = {}
    
    # Define file paths
    reduced_csv = output_path / "final_reduced_consolidated_data.csv"
    engineered_csv = output_path / "final_engineered_dataset.csv"
    train_csv = output_path / "selected_train_data.csv"
    test_csv = output_path / "selected_test_data.csv"

    # --- 1. Data Reduction ---
    logger.info("\n" + "="*80)
    logger.info("--- STEP 1: DATA REDUCTION ---")
    logger.info("="*80)
    if not skip_to_step or skip_to_step == 'reduction':
        pipeline_report_data['reduction'] = run_data_reduction_step(
            data_dir,
            output_csv_path=reduced_csv,
            output_html_path=output_path / "data_reduction_report.html",
            modalities=modalities if modalities else ALL_MODALITIES
        )
    
    # --- 2. Feature Engineering ---
    logger.info("\n" + "="*80)
    logger.info("--- STEP 2: FEATURE ENGINEERING ---")
    logger.info("="*80)
    if not skip_to_step or skip_to_step in ['reduction', 'engineering']:
        if not reduced_csv.exists():
            logger.error(f"{reduced_csv} not found. Cannot run feature engineering. Please run the reduction step first.")
            return
        pipeline_report_data['feature_engineering'] = run_feature_engineering_step(
            str(reduced_csv),
            output_csv_path=engineered_csv,
            output_html_path=output_path / "feature_engineering_report.html"
        )

    # --- 3. Feature Selection ---
    logger.info("\n" + "="*80)
    logger.info("--- STEP 3: FEATURE SELECTION ---")
    logger.info("="*80)
    if not skip_to_step or skip_to_step in ['reduction', 'engineering', 'selection']:
        if not engineered_csv.exists():
            logger.error(f"{engineered_csv} not found. Cannot run feature selection. Please run the engineering step first.")
            return
        pipeline_report_data['feature_selection'] = run_feature_selection_step(
            str(engineered_csv),
            train_csv_path=train_csv,
            test_csv_path=test_csv,
            output_html_path=output_path / "feature_selection_report.html",
            target_column=target_column,
            fs_method=fs_method,
            fs_param_value=fs_param_value,
            leakage_features_path=leakage_features_path
        )

    # --- 4. Modeling (Classification or Regression) ---
    if not train_csv.exists() or not test_csv.exists():
        logger.error(f"Train/Test CSVs not found. Cannot run modeling. Please run the full pipeline.")
        return

    # Detect task type from selected training data (numeric => regression)
    selected_train_head = pd.read_csv(train_csv, nrows=5)
    if target_column not in selected_train_head.columns:
        logger.error(f"Target column '{target_column}' not found in selected train data.")
        return
    is_regression = pd.api.types.is_numeric_dtype(selected_train_head[target_column])

    if is_regression:
        logger.info("\n" + "="*80)
        logger.info("--- STEP 4: REGRESSION ---")
        logger.info("="*80)
        reg_result = run_regression_step(
            train_csv_path=str(train_csv),
            test_csv_path=str(test_csv),
            target_column=target_column,
            output_dir=output_path,
            leakage_features_path=leakage_features_path,
            generate_plots=generate_plots
        )
        pipeline_report_data['regression'] = {
            'report_path': reg_result['report_path'],
            'r2': reg_result['r2'],
            'mae': reg_result['mae'],
            'rmse': reg_result['rmse']
        }
    else:
        logger.info("\n" + "="*80)
        logger.info("--- STEP 4: CLASSIFICATION ---")
        logger.info("="*80)
        classification_output_dir = output_path / "classification"
        exclude_features = []
        if leakage_features_path and Path(leakage_features_path).exists():
            with open(leakage_features_path, 'r') as f:
                exclude_features = [line.strip() for line in f if line.strip()]

        logger.info("--- Timing: Starting 'run_classification_step' ---")
        start_time_class = time.time()
        run_classification_step(
            train_csv_path=str(train_csv),
            test_csv_path=str(test_csv),
            use_feature_selection=False,
            target_column=target_column,
            exclude_features=exclude_features,
            output_dir=str(classification_output_dir),
            n_models_to_compare=n_models_to_compare,
            tune_best_model=tune_best_model,
            generate_plots=generate_plots,
            budget_time_minutes=budget_time_minutes
        )
        end_time_class = time.time()
        logger.info(f"--- Timing: Finished 'run_classification_step' in {end_time_class - start_time_class:.2f} seconds ---")
        
        classification_report_path = classification_output_dir / "classification_report.html"
        relative_classification_report_path = Path(os.path.relpath(classification_report_path, output_path))
        pipeline_report_data['classification'] = {
            "report_path": relative_classification_report_path
        }

    # --- 5. Final Report ---
    logger.info("\n" + "="*80)
    logger.info("--- STEP 5: GENERATING FINAL REPORT ---")
    logger.info("="*80)
    generate_main_report(pipeline_report_data, output_path / "pipeline_report.html")
    logger.info("--- PIE Pipeline Finished Successfully! ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the PIE Automated ML Pipeline.")
    parser.add_argument('--data-dir', type=str, default='./PPMI', help='Path to raw PPMI data directory.')
    parser.add_argument('--output-dir', type=str, default='output/pipeline_run', help='Directory to save all pipeline outputs and reports.')
    parser.add_argument('--target-column', type=str, default='COHORT', help='Name of the target variable.')
    parser.add_argument('--leakage-features-path', type=str, default='config/leakage_features.txt', help='Path to a file containing features to exclude to prevent data leakage.')
    parser.add_argument('--modalities', type=str, default='', help='Comma/space-separated list of modalities to include. Default: all. Options: subject_characteristics, medical_history, motor_assessments, non_motor_assessments, biospecimen')
    
    # Feature Selection Params
    parser.add_argument('--fs-method', type=str, default='fdr', help="Feature selection method ('fdr' or 'k_best').")
    parser.add_argument('--fs-param', type=float, default=0.05, help="Parameter for the FS method (alpha for 'fdr', k-fraction for 'k_best').")

    # Classification Params
    parser.add_argument('--n-models', type=int, default=5, help='Number of models to compare in classification.')
    parser.add_argument('--tune', action='store_true', help='Tune the best model.')
    parser.add_argument('--no-plots', action='store_false', dest='plots', help='Disable plot generation in classification.')
    parser.add_argument('--budget', type=float, default=30.0, help='Time budget in minutes for model comparison.')

    # Pipeline Control
    parser.add_argument('--skip-to', type=str, choices=['reduction', 'engineering', 'selection', 'classification'], help='Skip to a specific step of the pipeline.')

    args = parser.parse_args()

    # Normalize modalities (case-insensitive, comma/space separated)
    modalities_list = None
    if args.modalities:
        tokens = re.split(r'[;,\s]+', args.modalities)
        normalized = [t.strip().lower() for t in tokens if t and t.strip()]
        # Filter to known modalities, warn on unknowns
        if normalized:
            unknown = [m for m in normalized if m not in ALL_MODALITIES]
            if unknown:
                logger.warning(f"Unknown modalities provided and will be ignored: {unknown}. Valid options: {ALL_MODALITIES}")
            modalities_list = [m for m in normalized if m in ALL_MODALITIES]
            if not modalities_list:
                logger.warning("No valid modalities specified after filtering; defaulting to all modalities.")
                modalities_list = None

    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        target_column=args.target_column,
        leakage_features_path=args.leakage_features_path,
        modalities=modalities_list,
        fs_method=args.fs_method,
        fs_param_value=args.fs_param,
        n_models_to_compare=args.n_models,
        tune_best_model=args.tune,
        generate_plots=args.plots,
        budget_time_minutes=args.budget,
        skip_to_step=args.skip_to
    )
