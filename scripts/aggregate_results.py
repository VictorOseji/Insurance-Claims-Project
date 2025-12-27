# ============================================================================
# FILE: scripts/aggregate_results.py
# ============================================================================

"""
Aggregate results from all parallel model training runs
Creates comparison report in CSV and HTML formats
"""

import sys
from pathlib import Path
import json
import pandas as pd
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger

def main():
    logger = setup_logger("aggregate_results", "logs/aggregate_results.log")
    
    logger.info("="*60)
    logger.info("AGGREGATING MODEL RESULTS")
    logger.info("="*60)
    
    # Find all metrics files
    models_dir = Path("results/models")
    metrics_files = list(models_dir.glob("*/metrics.json"))
    
    logger.info(f"Found {len(metrics_files)} model results")
    
    if len(metrics_files) == 0:
        logger.error("No model metrics found!")
        sys.exit(1)
    
    # Load all metrics
    all_metrics = []
    for metrics_file in metrics_files:
        logger.info(f"Loading metrics from: {metrics_file}")
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
            all_metrics.append(metrics)
    
    # Create DataFrame
    df = pd.DataFrame(all_metrics)
    df = df.sort_values('r2', ascending=False)
    
    logger.info("Model rankings by R² score:")
    for idx, row in df.iterrows():
        logger.info(f"  {idx+1}. {row['model_type']}: R²={row['r2']:.4f}")
    
    # Prepare display DataFrame
    display_df = df[[
        'model_type', 'r2', 'rmse', 'mae', 'pin_name'
    ]].copy()
    display_df.columns = ['Model', 'R²', 'RMSE', 'MAE', 'Pin Name']
    
    # Save CSV
    csv_path = "results/model_comparison.csv"
    display_df.to_csv(csv_path, index=False)
    logger.info(f"✓ Saved comparison to: {csv_path}")
    
    # Generate HTML report
    logger.info("Generating HTML report...")
    
    best_model = display_df.iloc[0]
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Comparison Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section h2 {{
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
            font-size: 1.8em;
        }}
        
        .best-model {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        }}
        
        .best-model h3 {{
            font-size: 2em;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
        }}
        
        .best-model h3::before {{
            content: "🏆";
            font-size: 1.2em;
            margin-right: 15px;
        }}
        
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        
        .metric-card {{
            background: rgba(255, 255, 255, 0.2);
            padding: 15px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }}
        
        .metric-card .label {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 5px;
        }}
        
        .metric-card .value {{
            font-size: 1.8em;
            font-weight: bold;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        
        thead {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        
        th, td {{
            padding: 15px;
            text-align: left;
        }}
        
        th {{
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.9em;
            letter-spacing: 0.5px;
        }}
        
        tbody tr {{
            border-bottom: 1px solid #e0e0e0;
            transition: background 0.3s ease;
        }}
        
        tbody tr:hover {{
            background: #f5f5f5;
        }}
        
        tbody tr:first-child {{
            background: #e8f5e9;
            font-weight: 600;
        }}
        
        tbody tr:first-child td:first-child::before {{
            content: "🥇 ";
        }}
        
        tbody tr:nth-child(2) td:first-child::before {{
            content: "🥈 ";
        }}
        
        tbody tr:nth-child(3) td:first-child::before {{
            content: "🥉 ";
        }}
        
        .metric-value {{
            font-family: 'Courier New', monospace;
            font-weight: 600;
            color: #667eea;
        }}
        
        .footer {{
            background: #f5f5f5;
            padding: 20px 40px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
        
        .badge {{
            display: inline-block;
            background: #4CAF50;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            margin-left: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Insurance Claims Prediction</h1>
            <p>Model Comparison Report</p>
            <p style="font-size: 0.9em; margin-top: 10px;">
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </p>
        </div>
        
        <div class="content">
            <div class="section">
                <div class="best-model">
                    <h3>{best_model['Model']} <span class="badge">BEST MODEL</span></h3>
                    <div class="metric-grid">
                        <div class="metric-card">
                            <div class="label">R² Score</div>
                            <div class="value">{best_model['R²']:.4f}</div>
                        </div>
                        <div class="metric-card">
                            <div class="label">RMSE</div>
                            <div class="value">{best_model['RMSE']:.2f}</div>
                        </div>
                        <div class="metric-card">
                            <div class="label">MAE</div>
                            <div class="value">{best_model['MAE']:.2f}</div>
                        </div>
                        <div class="metric-card">
                            <div class="label">Pin Name</div>
                            <div class="value" style="font-size: 1em;">{best_model['Pin Name']}</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>All Models Performance</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>R² Score</th>
                            <th>RMSE</th>
                            <th>MAE</th>
                            <th>Pin Name</th>
                        </tr>
                    </thead>
                    <tbody>
"""
    
    for _, row in display_df.iterrows():
        html += f"""
                        <tr>
                            <td>{row['Model']}</td>
                            <td class="metric-value">{row['R²']:.4f}</td>
                            <td class="metric-value">{row['RMSE']:.2f}</td>
                            <td class="metric-value">{row['MAE']:.2f}</td>
                            <td style="font-size: 0.85em;">{row['Pin Name']}</td>
                        </tr>
"""
    
    html += """
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2>📊 Model Insights</h2>
                <ul style="list-style: none; padding: 0;">
                    <li style="padding: 10px; margin: 10px 0; background: #f5f5f5; border-radius: 5px;">
                        <strong>Highest R² Score:</strong> {:.4f} ({})
                    </li>
                    <li style="padding: 10px; margin: 10px 0; background: #f5f5f5; border-radius: 5px;">
                        <strong>Lowest RMSE:</strong> {:.2f} ({})
                    </li>
                    <li style="padding: 10px; margin: 10px 0; background: #f5f5f5; border-radius: 5px;">
                        <strong>Models Trained:</strong> {} models in parallel
                    </li>
                </ul>
            </div>
            
            <div class="section">
                <h2>🚀 Next Steps</h2>
                <ol style="line-height: 2; color: #555;">
                    <li>Review model performance metrics above</li>
                    <li>Load best model from Pins: <code style="background: #f5f5f5; padding: 2px 8px; border-radius: 3px;">{}</code></li>
                    <li>Validate model on additional test data</li>
                    <li>Deploy model to production environment</li>
                    <li>Set up monitoring and retraining pipeline</li>
                </ol>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated by Insurance Claims ML Pipeline | Snakemake Workflow</p>
            <p style="margin-top: 5px;">
                View detailed logs: <code>mlflow ui</code> | 
                Access models: <code>model_pins_board/</code>
            </p>
        </div>
    </div>
</body>
</html>
""".format(
        display_df.iloc[0]['R²'], display_df.iloc[0]['Model'],
        display_df.iloc[0]['RMSE'], display_df.iloc[0]['Model'],
        len(display_df),
        best_model['Pin Name']
    )
    
    html_path = "results/model_comparison.html"
    with open(html_path, 'w') as f:
        f.write(html)
    
    logger.info(f"✓ Saved HTML report to: {html_path}")
    
    logger.info("="*60)
    logger.info("✓ RESULTS AGGREGATION COMPLETE!")
    logger.info("="*60)
    logger.info(f"Best model: {best_model['Model']} (R²={best_model['R²']:.4f})")

if __name__ == "__main__":
    main()
