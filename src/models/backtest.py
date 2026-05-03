import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

DB_PATH = 'data/prediction.db'
PLOTS_DIR = 'plots'
ASSET_ORDER = ['NVDA', 'USO', 'EGX']
ASSET_COLORS = {'NVDA': '#76b900', 'USO': '#eb6e2b', 'EGX': '#FFCE00'}

def max_drawdown(equity_curve):
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve / peak) - 1
    return drawdown.min()

def calculate_sharpe(returns, risk_free_rate=0.0):
    if returns.std() == 0:
        return 0
    return np.sqrt(252) * (returns.mean() - risk_free_rate) / returns.std()

def run_backtest():
    con = duckdb.connect(DB_PATH)
    
    ml_df = con.execute("SELECT * FROM ml_predictions").df()
    try:
        dl_df = con.execute("SELECT trading_session, asset_tag, lstm_base_pred, lstm_hybrid_pred FROM dl_predictions").df()
        df = pd.merge(ml_df, dl_df, on=['trading_session', 'asset_tag'], how='left')
    except:
        df = ml_df
        
    results = []
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 15), sharex=False)
    
    for i, asset in enumerate(ASSET_ORDER):
        asset_df = df[df['asset_tag'] == asset].copy()
        if asset_df.empty: continue
        
        asset_df.sort_values('trading_session', inplace=True)
        
        # Target daily volatility = 1%
        target_daily_vol = 0.01
        
        bnh_returns = asset_df['log_return_1d']
        bnh_equity = (1 + bnh_returns).cumprod()
        
        asset_results = {
            'Asset': asset,
            'Model': 'Buy & Hold',
            'Sharpe': calculate_sharpe(bnh_returns),
            'Max Drawdown': max_drawdown(bnh_equity)
        }
        results.append(asset_results)
        
        axes[i].plot(pd.to_datetime(asset_df['trading_session']), bnh_equity, label='Buy & Hold', color='black', alpha=0.6)
        
        models_to_test = [
            ('XGBoost Baseline', 'xgb_pred_baseline', 'gray'),
            ('XGBoost FinBERT', 'xgb_pred_finbert', 'blue'),
            ('XGBoost Gemma', 'xgb_pred_gemma', ASSET_COLORS[asset])
        ]
        if 'lstm_hybrid_pred' in asset_df.columns:
            models_to_test.append(('LSTM-Hybrid', 'lstm_hybrid_pred', 'purple'))
            
        for name, pred_col, color in models_to_test:
            if pred_col not in asset_df.columns: continue
            
            # The predictions are ALREADY daily volatility (since we didn't annualize in build_feature_store)
            pred_daily_vol = np.clip(asset_df[pred_col], 0.005, 0.1) # Clip between 0.5% and 10% daily vol
            
            position_size = target_daily_vol / pred_daily_vol
            position_size = np.clip(position_size, 0, 2.0)
            
            strat_returns = position_size.shift(1) * asset_df['log_return_1d']
            strat_returns.fillna(0, inplace=True)
            
            strat_equity = (1 + strat_returns).cumprod()
            
            axes[i].plot(pd.to_datetime(asset_df['trading_session']), strat_equity, label=name, color=color, alpha=0.8)
            
            results.append({
                'Asset': asset,
                'Model': name,
                'Sharpe': calculate_sharpe(strat_returns),
                'Max Drawdown': max_drawdown(strat_equity)
            })
            
        axes[i].set_title(f'{asset}: Volatility-Targeting Equity Curves', fontweight='bold', fontsize=12)
        axes[i].set_ylabel('Cumulative Return')
        axes[i].legend(loc='upper left')
        
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/equity_curves_combined.png')
    plt.close()
    
    res_df = pd.DataFrame(results)
    print("\n--- Backtest Evaluation Summary ---")
    print(res_df.to_string(index=False))
    
    os.makedirs('artifacts', exist_ok=True)
    res_df.to_csv('artifacts/backtest_results.csv', index=False)
    
if __name__ == '__main__':
    run_backtest()

