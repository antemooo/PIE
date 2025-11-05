"""
regression_report.py

Comprehensive regression pipeline reporting with model comparison,
visualization, and performance metrics.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import webbrowser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
from typing import List, Union, Optional, Any, Dict, Tuple

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PIE.regression_report")


def generate_regression_report_html(
    report_data: dict,
    output_html_path: str,
    plots_dir: str
):
    """Generates a comprehensive HTML report for the regression pipeline."""
    logger.info(f"Generating Regression HTML report at: {output_html_path}")

    html_style = """
    <style>
        body { font-family: 'Arial', sans-serif; line-height: 1.6; margin: 20px; background-color: #f4f4f4; color: #333; }
        .container { background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 15px rgba(0,0,0,0.1); max-width: 1400px; margin: 0 auto; }
        h1, h2, h3 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        h1 { text-align: center; color: #3498db; font-size: 2.5em; }
        h2 { color: #34495e; margin-top: 30px; }
        h3 { color: #7f8c8d; }
        table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #3498db; color: white; font-weight: bold; }
        tr:nth-child(even) { background-color: #ecf0f1; }
        tr:hover { background-color: #d5dbdb; }
        .summary-box { border: 2px solid #bdc3c7; padding: 20px; margin-bottom: 25px; background-color: #f8f9f9; border-radius: 8px; }
        .code { background-color: #2c3e50; color: #ecf0f1; padding: 3px 8px; border-radius: 4px; font-family: 'Courier New', Courier, monospace; }
        .highlight { color: #3498db; font-weight: bold; font-size: 1.1em; }
        .metric-value { font-weight: bold; color: #27ae60; font-size: 1.1em; }
        .warning { color: #f39c12; font-weight: bold; }
        .plot-container { margin: 25px 0; text-align: center; background-color: #ecf0f1; padding: 20px; border-radius: 8px; }
        .plot-container img { max-width: 90%; height: auto; border: 2px solid #34495e; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        .plot-title { font-weight: bold; color: #2c3e50; margin-bottom: 10px; font-size: 1.2em; }
        .leaderboard-table { font-size: 0.95em; }
        .leaderboard-table td, .leaderboard-table th { padding: 8px; }
        .top-model { background-color: #d5f4e6; font-weight: bold; }
        .best-model { background-color: #d5f4e6; padding: 20px; border-radius: 8px; margin: 20px 0; border: 2px solid #27ae60; }
        ul { list-style-type: square; padding-left: 30px; }
        li { margin-bottom: 8px; }
        .timestamp { color: #7f8c8d; font-style: italic; text-align: right; margin-top: 20px; }
    </style>
    """

    # Start building HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>PIE Regression Pipeline Report</title>
        {html_style}
    </head>
    <body>
        <div class="container">
            <h1>📊 PIE Regression Pipeline Report</h1>
            
            <div class="summary-box">
                <h2>1. Pipeline Overview</h2>
                <table>
                    <tr><th>Component</th><th>Details</th></tr>
                    <tr><td>Input Data</td><td><span class="code">{report_data.get('train_data_path', 'N/A')}</span></td></tr>
                    <tr><td>Training Data Shape</td><td>{report_data.get('train_shape', 'N/A')}</td></tr>
                    <tr><td>Test Data Shape</td><td>{report_data.get('test_shape', 'N/A')}</td></tr>
                    <tr><td>Target Variable</td><td><span class="highlight">{report_data.get('target_column', 'N/A')}</span></td></tr>
                    <tr><td>Task Type</td><td>Regression (Continuous Target)</td></tr>
                    <tr><td>Number of Features</td><td><span class="metric-value">{report_data.get('n_features', 'N/A')}</span></td></tr>
                </table>
            </div>
    """

    # Add excluded features details if any were excluded
    if report_data.get('excluded_features'):
        html_content += f"""
            <div class="summary-box">
                <h3>🚫 Excluded Features (Data Leakage Prevention)</h3>
                <p>The following features were excluded to prevent data leakage:</p>
                <ul>
        """
        for feature in report_data.get('excluded_features', []):
            html_content += f"<li><span class='code'>{feature}</span></li>"
        html_content += """
                </ul>
                <p><em>These features were excluded because they may be too closely related to the target variable 
                or contain information that would not be available at prediction time.</em></p>
            </div>
        """

    # Model comparison leaderboard
    if report_data.get('leaderboard') is not None:
        html_content += """
            <div class="summary-box">
                <h2>🏆 2. Model Comparison Leaderboard</h2>
                <p>Performance comparison of all evaluated regression models:</p>
                <div style="overflow-x: auto;">
        """
        # Convert leaderboard to HTML with custom styling for top model
        leaderboard_html = report_data['leaderboard'].to_html(
            classes='leaderboard-table',
            index=False,
            float_format=lambda x: f'{x:.4f}'
        )
        # Highlight the top row
        leaderboard_html = leaderboard_html.replace('<tr>', '<tr class="top-model">', 1)
        html_content += leaderboard_html
        html_content += """
                </div>
                <p><em>Note: Models are ranked by R² score (coefficient of determination) on test data. Higher R² indicates better model performance.</em></p>
            </div>
        """

    # Best model details
    if report_data.get('best_model_name'):
        html_content += f"""
            <div class="best-model">
                <h2>🌟 3. Best Model Performance</h2>
                <h3>Selected Model: <span class="highlight">{report_data.get('best_model_name', 'N/A')}</span></h3>
        """

        # Add metrics table
        html_content += """
                <table>
                    <tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>
        """
        
        r2 = report_data.get('best_r2', 'N/A')
        mae = report_data.get('best_mae', 'N/A')
        rmse = report_data.get('best_rmse', 'N/A')
        f_score = report_data.get('best_f_score', 'N/A')
        
        if r2 != 'N/A':
            r2_interp = "Excellent" if r2 > 0.9 else "Good" if r2 > 0.7 else "Moderate" if r2 > 0.5 else "Poor"
            html_content += f"<tr><td>R² Score</td><td class='metric-value'>{r2:.4f}</td><td>{r2_interp} (Variance explained: {r2*100:.1f}%)</td></tr>"
        
        if mae != 'N/A':
            html_content += f"<tr><td>Mean Absolute Error (MAE)</td><td class='metric-value'>{mae:.4f}</td><td>Average prediction error magnitude</td></tr>"
        
        if rmse != 'N/A':
            html_content += f"<tr><td>Root Mean Squared Error (RMSE)</td><td class='metric-value'>{rmse:.4f}</td><td>Penalizes larger errors more heavily</td></tr>"
        
        if f_score != 'N/A':
            html_content += f"<tr><td>Explained Variance</td><td class='metric-value'>{f_score:.4f}</td><td>Proportion of variance explained</td></tr>"
        
        html_content += """
                </table>
            </div>
        """

    # Model plots
    html_content += """
            <div class="summary-box">
                <h2>📈 4. Model Performance Visualizations</h2>
    """

    # Check for all possible plots
    all_possible_plots = [
        ('pred_vs_actual', 'Predicted vs Actual Values'),
        ('residuals', 'Residuals Distribution'),
        ('residuals_vs_fitted', 'Residuals vs Fitted Values'),
        ('qq_plot', 'Q-Q Plot (Normality Check)'),
        ('feature_importance', 'Feature Importance'),
        ('learning_curve', 'Learning Curve')
    ]

    plots_found = False
    for plot_file, plot_title in all_possible_plots:
        plot_path = Path(plots_dir) / f"{plot_file}.png"
        if plot_path.exists():
            plots_found = True
            relative_plot_path = f"plots/{plot_file}.png"
            html_content += f"""
                <div class="plot-container">
                    <div class="plot-title">{plot_title.upper()}</div>
                    <img src="{relative_plot_path}" alt="{plot_title}">
                </div>
            """

    if not plots_found:
        html_content += "<p><em>No visualization plots were generated.</em></p>"

    html_content += """
            </div>
    """

    # Feature importance details
    if report_data.get('feature_importance'):
        html_content += """
            <div class="summary-box">
                <h2>🎯 5. Top Predictive Features</h2>
                <p>The following features have the strongest influence on predictions:</p>
                <table>
                    <tr><th>Rank</th><th>Feature</th><th>Importance Score</th></tr>
        """
        for i, (feature, importance) in enumerate(report_data['feature_importance'][:20], 1):
            html_content += f"<tr><td>{i}</td><td><span class='code'>{feature}</span></td><td class='metric-value'>{importance:.4f}</td></tr>"
        html_content += """
                </table>
            </div>
        """

    # Recommendations
    html_content += """
            <div class="summary-box">
                <h2>💡 6. Recommendations</h2>
                <ul>
    """

    # Add dynamic recommendations based on results
    recommendations = []

    r2_val = report_data.get('best_r2', 0)
    if r2_val != 'N/A' and r2_val < 0.5:
        recommendations.append("R² score is below 0.5, indicating poor model fit. Consider: (1) Engineering additional features, (2) Collecting more data, (3) Trying non-linear models, (4) Checking for data quality issues.")
    elif r2_val != 'N/A' and r2_val > 0.8:
        recommendations.append("Excellent R² score! Model explains most of the variance. Verify there's no data leakage from the target variable.")

    mae_val = report_data.get('best_mae', 0)
    if mae_val != 'N/A' and mae_val > 0:
        target_range = report_data.get('target_range', None)
        if target_range:
            mae_pct = (mae_val / target_range) * 100
            if mae_pct > 20:
                recommendations.append(f"MAE is {mae_pct:.1f}% of target range. Consider feature engineering or ensemble methods to improve accuracy.")

    if not recommendations:
        recommendations.append("Model performance is satisfactory. Consider validating on external datasets before deployment.")
        recommendations.append("Monitor model performance over time and retrain as new data becomes available.")
        recommendations.append("Document all assumptions and limitations for stakeholders.")

    for rec in recommendations:
        html_content += f"<li>{rec}</li>"

    html_content += f"""
                </ul>
            </div>
            
            <div class="timestamp">
                Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
    </body>
    </html>
    """

    try:
        # Ensure output directory exists
        Path(output_html_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"HTML report generated successfully: {output_html_path}")
    except Exception as e:
        logger.error(f"Failed to write HTML report to {output_html_path}: {e}")


def generate_report(
    train_csv_path: str,
    test_csv_path: str,
    target_column: str,
    output_dir: str = "output/regression",
    exclude_features: List[str] = None,
    generate_plots: bool = True
) -> dict:
    """
    Runs the complete regression pipeline including model comparison,
    visualization, and comprehensive reporting.

    Parameters:
    -----------
    train_csv_path : str
        Path to training data CSV
    test_csv_path : str
        Path to test data CSV  
    target_column : str
        Name of the target column
    output_dir : str
        Directory to save outputs
    exclude_features : List[str], optional
        List of feature names to exclude from training
    generate_plots : bool
        Whether to generate visualization plots

    Returns:
    --------
    dict : Report data and metrics
    """
    logger.info("Starting regression report generation...")

    # Initialize exclude_features if None
    if exclude_features is None:
        exclude_features = []

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Initialize report data collector
    report_data = {
        'target_column': target_column,
        'train_data_path': train_csv_path,
        'test_data_path': test_csv_path,
        'plots_dir': str(plots_dir),
        'excluded_features': exclude_features
    }

    # Load data
    try:
        train_df = pd.read_csv(train_csv_path)
        test_df = pd.read_csv(test_csv_path)
        logger.info(f"Loaded training data from {train_csv_path}. Shape: {train_df.shape}")
        logger.info(f"Loaded test data from {test_csv_path}. Shape: {test_df.shape}")

        report_data['train_shape'] = train_df.shape
        report_data['test_shape'] = test_df.shape

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return report_data

    # Check target column
    if target_column not in train_df.columns:
        logger.error(f"Target column '{target_column}' not found in training data")
        return report_data

    # EXCLUDE SPECIFIED FEATURES EARLY (similar to classification_report.py)
    # This ensures leakage features are removed even if feature selection didn't catch them
    if exclude_features:
        logger.info(f"Excluding {len(exclude_features)} specified features from regression analysis...")
        
        # Check which excluded features actually exist in the data
        # CRITICAL: Never exclude the target column itself
        existing_excluded_features = [
            feat for feat in exclude_features 
            if feat in train_df.columns and feat != target_column
        ]
        missing_excluded_features = [
            feat for feat in exclude_features 
            if feat not in train_df.columns and feat != target_column
        ]
        
        if existing_excluded_features:
            logger.info(f"Excluding features from train data: {existing_excluded_features}")
            train_df = train_df.drop(columns=existing_excluded_features)
            # Also drop from test data if they exist
            existing_in_test = [feat for feat in existing_excluded_features if feat in test_df.columns]
            if existing_in_test:
                test_df = test_df.drop(columns=existing_in_test)
            logger.info(f"Train data shape after feature exclusion: {train_df.shape}")
            logger.info(f"Test data shape after feature exclusion: {test_df.shape}")
        
        if missing_excluded_features:
            logger.warning(f"Specified features not found in data (already removed or never existed): {missing_excluded_features}")
        
        # Warn if target column was in exclude list (should not happen after pipeline fix)
        if target_column in exclude_features:
            logger.warning(f"Target column '{target_column}' found in exclude_features list but was preserved.")
        
        # Update report data
        report_data['excluded_features'] = existing_excluded_features
        report_data['excluded_features_count'] = len(existing_excluded_features)

    # Prepare features and target
    X_train = train_df.drop(columns=[target_column]).select_dtypes(include=np.number)
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column]).select_dtypes(include=np.number)
    y_test = test_df[target_column]

    # Align columns
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    report_data['n_features'] = X_train.shape[1]
    report_data['target_range'] = y_train.max() - y_train.min()

    # Define regression models to compare
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    try:
        from xgboost import XGBRegressor
        has_xgboost = True
    except ImportError:
        has_xgboost = False
        logger.warning("XGBoost not available")

    try:
        from lightgbm import LGBMRegressor
        has_lightgbm = True
    except ImportError:
        has_lightgbm = False
        logger.warning("LightGBM not available")

    try:
        from catboost import CatBoostRegressor
        has_catboost = True
    except ImportError:
        has_catboost = False
        logger.warning("CatBoost not available")

    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'ElasticNet': ElasticNet(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    if has_xgboost:
        models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0)
    if has_lightgbm:
        models['LightGBM'] = LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    if has_catboost:
        models['CatBoost'] = CatBoostRegressor(n_estimators=100, random_state=42, verbose=0)

    # Compare models
    logger.info(f"Comparing {len(models)} regression models...")
    leaderboard = []

    for name, model in models.items():
        try:
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate metrics
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            f_score = explained_variance_score(y_test, y_pred)

            leaderboard.append({
                'Model': name,
                'R²': r2,
                'MAE': mae,
                'RMSE': rmse,
                'Explained Variance': f_score
            })

            logger.info(f"{name} - R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        except Exception as e:
            logger.warning(f"Model {name} failed: {e}")

    # Sort leaderboard by R² descending
    leaderboard_df = pd.DataFrame(leaderboard).sort_values('R²', ascending=False).reset_index(drop=True)
    report_data['leaderboard'] = leaderboard_df

    # Get best model
    if len(leaderboard_df) > 0:
        best_row = leaderboard_df.iloc[0]
        best_model_name = best_row['Model']
        best_model = models[best_model_name]

        # Retrain best model and get predictions
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        report_data['best_model_name'] = best_model_name
        report_data['best_r2'] = best_row['R²']
        report_data['best_mae'] = best_row['MAE']
        report_data['best_rmse'] = best_row['RMSE']
        report_data['best_f_score'] = best_row['Explained Variance']

        logger.info(f"Best model: {best_model_name} with Test R²={best_row['R²']:.4f}")

        # Save best model
        model_path = output_path / "final_regression_model.pkl"
        joblib.dump(best_model, model_path)
        logger.info(f"Saved best model to {model_path}")

        # Extract feature importance if available
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = sorted(
                zip(X_train.columns, best_model.feature_importances_),
                key=lambda x: x[1],
                reverse=True
            )
            report_data['feature_importance'] = feature_importance
        elif hasattr(best_model, 'coef_'):
            feature_importance = sorted(
                zip(X_train.columns, np.abs(best_model.coef_)),
                key=lambda x: x[1],
                reverse=True
            )
            report_data['feature_importance'] = feature_importance

        # Generate plots
        if generate_plots:
            logger.info("Generating visualization plots...")

            # 1. Predicted vs Actual
            plt.figure(figsize=(10, 8))
            plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', s=60)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
            plt.xlabel('Actual Values', fontsize=12)
            plt.ylabel('Predicted Values', fontsize=12)
            plt.title(f'Predicted vs Actual - {best_model_name}', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / 'pred_vs_actual.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 2. Residuals Distribution
            residuals = y_test - y_pred
            plt.figure(figsize=(10, 6))
            plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
            plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
            plt.xlabel('Residuals', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title('Residuals Distribution', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / 'residuals.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 3. Residuals vs Fitted
            plt.figure(figsize=(10, 6))
            plt.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', s=60)
            plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
            plt.xlabel('Fitted Values', fontsize=12)
            plt.ylabel('Residuals', fontsize=12)
            plt.title('Residuals vs Fitted Values', fontsize=14, fontweight='bold')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / 'residuals_vs_fitted.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 4. Q-Q Plot
            from scipy import stats
            plt.figure(figsize=(10, 6))
            stats.probplot(residuals, dist="norm", plot=plt)
            plt.title('Q-Q Plot (Normality Check)', fontsize=14, fontweight='bold')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / 'qq_plot.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 5. Feature Importance (if available)
            if report_data.get('feature_importance'):
                top_features = report_data['feature_importance'][:15]
                features, importances = zip(*top_features)

                plt.figure(figsize=(10, 8))
                plt.barh(range(len(features)), importances, color='steelblue', edgecolor='black')
                plt.yticks(range(len(features)), features)
                plt.xlabel('Importance', fontsize=12)
                plt.title('Top 15 Feature Importances', fontsize=14, fontweight='bold')
                plt.gca().invert_yaxis()
                plt.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                plt.savefig(plots_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
                plt.close()

            logger.info("Plots generated successfully")

    # Generate HTML report
    report_path = output_path / "regression_report.html"
    generate_regression_report_html(report_data, str(report_path), str(plots_dir))

    # Try to open the report
    try:
        report_abs_path = report_path.resolve()
        webbrowser.open(f"file://{report_abs_path}")
        logger.info(f"Opened report in browser: file://{report_abs_path}")
    except Exception as e:
        logger.info(f"Could not automatically open the report in browser: {e}")

    logger.info("Regression report generation completed successfully!")

    return report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a regression report for a given dataset.")
    parser.add_argument('--train-csv-path', type=str, required=True, help='Path to training data CSV.')
    parser.add_argument('--test-csv-path', type=str, required=True, help='Path to test data CSV.')
    parser.add_argument('--target-column', type=str, required=True, help='Name of the target column.')
    parser.add_argument('--output-dir', type=str, default='output/regression', help='Directory to save outputs.')
    parser.add_argument('--exclude-features-file', type=str, help='Path to a text file with features to exclude (one per line).')
    parser.add_argument('--generate-plots', action='store_true', help='Enable plot generation.')

    args = parser.parse_args()

    exclude_features_list = []
    if args.exclude_features_file:
        try:
            with open(args.exclude_features_file, 'r') as f:
                exclude_features_list = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(exclude_features_list)} features to exclude from {args.exclude_features_file}")
        except Exception as e:
            logger.error(f"Could not read exclude features file: {e}")
            sys.exit(1)

    # Run the report generation
    generate_report(
        train_csv_path=args.train_csv_path,
        test_csv_path=args.test_csv_path,
        target_column=args.target_column,
        output_dir=args.output_dir,
        exclude_features=exclude_features_list,
        generate_plots=args.generate_plots
    )
