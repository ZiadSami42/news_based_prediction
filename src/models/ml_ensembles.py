import duckdb
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import shap
import matplotlib.pyplot as plt
import os

DB_PATH = 'data/prediction.db'
PLOTS_DIR = 'plots'
ASSET_ORDER = ['NVDA', 'USO', 'EGX']

def run_ml_ensembles():
    con = duckdb.connect(DB_PATH)
    df = con.execute("SELECT * FROM model_features ORDER BY trading_session").df()
    
    all_predictions = []
    results = []
    
    features_baseline = ['log_return_1d', 'log_return_3d', 'log_return_5d', 'vol_roc_1d']
    features_finbert = features_baseline + ['finbert_mean']
    features_gemma = features_baseline + [
        'gemma_dispersion', 'sentiment_persistence_3d', 
        'article_impact_index', 'gemma_conf_weighted',
        'mean_score_direct', 'mean_score_indirect'
    ]
    
    # Ensure columns exist, if indirect/direct had no matches in duckdb pivot, they might be missing
    for col in features_gemma:
        if col not in df.columns:
            print(f"Warning: Column {col} missing, filling with 0.")
            df[col] = 0
            
    fig_sum = plt.figure(figsize=(24, 6))
    fig_dep = plt.figure(figsize=(24, 6))
    
    target = 'realized_vol_22d'
    
    for asset_idx, asset in enumerate(ASSET_ORDER):
        asset_df = df[df['asset_tag'] == asset].copy()
        asset_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        asset_df = asset_df.dropna(subset=[target] + features_gemma)
        
        if asset_df.empty:
            continue
            
        # Chronological split (80% train, 20% test)
        train_size = int(len(asset_df) * 0.8)
        train_df = asset_df.iloc[:train_size]
        test_df = asset_df.iloc[train_size:].copy()
        
        variations = {
            'Baseline': features_baseline,
            'FinBERT': features_finbert,
            'Gemma': features_gemma
        }
        
        for var_name, features in variations.items():
            X_train = train_df[features]
            y_train = train_df[target]
            X_test = test_df[features]
            y_test = test_df[target]
            
            # XGBoost
            xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
            xgb_model.fit(X_train, y_train)
            xgb_pred = xgb_model.predict(X_test)
            
            # Random Forest
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict(X_test)
            
            test_df[f'xgb_pred_{var_name.lower()}'] = xgb_pred
            test_df[f'rf_pred_{var_name.lower()}'] = rf_pred
            
            for model_name, preds in [('XGBoost', xgb_pred), ('RandomForest', rf_pred)]:
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                mae = mean_absolute_error(y_test, preds)
                results.append({'Asset': asset, 'Variation': var_name, 'Model': model_name, 'RMSE': rmse, 'MAE': mae})
                
            # Generate SHAP for XGBoost Reasoning Model
            if var_name == 'Gemma':
                explainer = shap.TreeExplainer(xgb_model)
                shap_values = explainer.shap_values(X_test)
                
                plt.figure(fig_sum.number)
                plt.subplot(1, 3, asset_idx + 1)
                shap.summary_plot(shap_values, X_test, show=False)
                plt.title(f'SHAP Summary: ({asset})', fontweight='bold')
                
                if 'gemma_dispersion' in features:
                    plt.figure(fig_dep.number)
                    ax = plt.subplot(1, 3, asset_idx + 1)
                    shap.dependence_plot('gemma_dispersion', shap_values, X_test, ax=ax, show=False)
                    plt.title(f'SHAP Dep: Dispersion ({asset})', fontweight='bold')

        all_predictions.append(test_df)
        
    plt.figure(fig_sum.number)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/shap_summary_combined.png')
    plt.close(fig_sum)

    plt.figure(fig_dep.number)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/shap_dep_dispersion_combined.png')
    plt.close(fig_dep)
        
    res_df = pd.DataFrame(results)
    print("\n--- ML Ensembles Evaluation Summary ---")
    print(res_df.to_string(index=False))
    
    os.makedirs('artifacts', exist_ok=True)
    res_df.to_markdown('artifacts/ml_results.md', index=False)
    
    final_preds = pd.concat(all_predictions, ignore_index=True)
    pred_cols = ['trading_session', 'asset_tag', 'close', 'log_return_1d', target] + [c for c in final_preds.columns if 'pred' in c]
    eval_df = final_preds[pred_cols]
    
    con.execute("DROP TABLE IF EXISTS ml_predictions")
    con.execute("CREATE TABLE ml_predictions AS SELECT * FROM eval_df")
    print(f"\nSaved {len(eval_df)} predictions to ml_predictions table.")
    con.close()

if __name__ == '__main__':
    run_ml_ensembles()
