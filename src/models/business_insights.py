import duckdb
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates

DB_PATH = 'data/prediction.db'
PLOTS_DIR = 'plots'
ASSET_ORDER = ['NVDA', 'USO', 'EGX']
ASSET_COLORS = {'NVDA': '#76b900', 'USO': '#eb6e2b', 'EGX': '#FFCE00'}

def generate_insights():
    con = duckdb.connect(DB_PATH)
    df = con.execute("SELECT * FROM model_features ORDER BY trading_session").df()
    try:
        dl_df = con.execute("SELECT trading_session, asset_tag, lstm_hybrid_pred FROM dl_predictions").df()
        df = pd.merge(df, dl_df, on=['trading_session', 'asset_tag'], how='left')
    except Exception as e:
        print(f"Warning: Could not load dl_predictions ({e})")
        df['lstm_hybrid_pred'] = df['realized_vol_22d']
        
    features = [
        'log_return_1d', 'log_return_3d', 'log_return_5d', 'vol_roc_1d',
        'gemma_mean', 'gemma_dispersion', 'gemma_conf_weighted', 
        'sentiment_persistence_3d', 'article_impact_index'
    ]
    
    results = []
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for i, asset in enumerate(ASSET_ORDER):
        asset_df = df[df['asset_tag'] == asset].copy()
        if asset_df.empty: continue
        
        asset_df.sort_values('trading_session', inplace=True)
        asset_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Calculate forward 5-day return for training
        asset_df['target_return_5d'] = np.log(asset_df['close'].shift(-5) / asset_df['close'])
        
        # We can't train on the last 5 days
        train_df = asset_df.dropna(subset=['target_return_5d'] + features)
        
        if train_df.empty: continue
            
        X_train = train_df[features]
        y_train = train_df['target_return_5d']
        
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict on the VERY LAST available day
        latest_row = asset_df.iloc[-1:]
        X_latest = latest_row[features].fillna(0)
        
        pred_return = model.predict(X_latest)[0]
        
        # Get volatility prediction
        pred_vol = latest_row['lstm_hybrid_pred'].values[0]
        if pd.isna(pred_vol):
            pred_vol = latest_row['realized_vol_22d'].values[0]
            
        current_price = latest_row['close'].values[0]
        latest_date = pd.to_datetime(latest_row['trading_session'].values[0])
        
        # Confidence interval: 1 standard deviation for 5 days
        # pred_vol is daily volatility
        sigma_5d = pred_vol * np.sqrt(5)
        lower_bound = current_price * np.exp(pred_return - sigma_5d)
        upper_bound = current_price * np.exp(pred_return + sigma_5d)
        expected_price = current_price * np.exp(pred_return)
        
        # Categorize Volatility
        vol_env = "High Swing/Risk" if pred_vol > 0.03 else "Moderate" if pred_vol > 0.015 else "Low Risk/Stable"
        
        results.append({
            'Asset': asset,
            'Latest Date': latest_date.strftime('%Y-%m-%d'),
            'Current Price': f"{current_price:.2f}",
            'Expected 5d Return': f"{pred_return*100:.2f}%",
            'Predicted Volatility': vol_env,
            'Confidence Interval': f"[{lower_bound:.2f}, {upper_bound:.2f}]"
        })
        
        # Plotting the dashboard
        ax = axes[i]
        
        # Plot last 30 days
        recent_df = asset_df.iloc[-30:]
        dates = pd.to_datetime(recent_df['trading_session'])
        prices = recent_df['close']
        
        ax.plot(dates, prices, color='black', linewidth=2, label='Historical Price')
        ax.scatter(latest_date, current_price, color=ASSET_COLORS[asset], s=100, zorder=5)
        
        # Plot projection cone
        future_dates = [latest_date, latest_date + pd.Timedelta(days=5)]
        
        ax.plot(future_dates, [current_price, expected_price], color=ASSET_COLORS[asset], linestyle='--', linewidth=2, label='Expected Trajectory')
        ax.fill_between(future_dates, 
                        [current_price, lower_bound], 
                        [current_price, upper_bound], 
                        color=ASSET_COLORS[asset], alpha=0.2, label='68% Confidence Interval')
        
        ax.set_title(f"{asset} Business Insights\nExpected: {pred_return*100:.1f}% | Vol: {vol_env}", fontweight='bold', fontsize=14)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/business_insights_dashboard.png')
    plt.close()
    
    res_df = pd.DataFrame(results)
    print("\n" + "="*80)
    print(" "*25 + "FINAL BUSINESS INSIGHTS")
    print("="*80)
    print(res_df.to_string(index=False))
    print("="*80 + "\n")
    
    os.makedirs('artifacts', exist_ok=True)
    res_df.to_csv('artifacts/business_insights.csv', index=False)
    
if __name__ == '__main__':
    generate_insights()
