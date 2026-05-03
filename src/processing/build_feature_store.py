import duckdb
import pandas as pd
import numpy as np
import os

DB_PATH = 'data/prediction.db'

def build_features():
    con = duckdb.connect(DB_PATH)
    
    print("1. Loading OHLCV and Mapping Symbols...")
    ohlcv = con.execute("SELECT time, symbol, open, high, low, close, volume FROM ohlcv ORDER BY time").df()
    symbol_map = {
        'NASDAQ:NVDA': 'NVDA',
        'ICEEUR:BRN1!': 'USO', 
        'EGX:EGX70EWI': 'EGX'
    }
    ohlcv['asset_tag'] = ohlcv['symbol'].map(symbol_map)
    ohlcv['trading_session'] = pd.to_datetime(ohlcv['time']).dt.date
    
    # Sort for time-series operations
    ohlcv = ohlcv.sort_values(['asset_tag', 'trading_session']).reset_index(drop=True)
    
    print("2. Calculating Price Features...")
    # Log Returns
    ohlcv['log_return_1d'] = ohlcv.groupby('asset_tag')['close'].transform(lambda x: np.log(x / x.shift(1)))
    ohlcv['log_return_3d'] = ohlcv.groupby('asset_tag')['close'].transform(lambda x: np.log(x / x.shift(3)))
    ohlcv['log_return_5d'] = ohlcv.groupby('asset_tag')['close'].transform(lambda x: np.log(x / x.shift(5)))
    
    # Target: 22-day Realized Volatility
    ohlcv['realized_vol_22d'] = ohlcv.groupby('asset_tag')['log_return_1d'].transform(lambda x: x.rolling(22).std())
    
    # Volume ROC
    ohlcv['vol_roc_1d'] = ohlcv.groupby('asset_tag')['volume'].transform(lambda x: x.pct_change(1))
    
    print("3. Loading & Aggregating Sentiment Features...")
    news = con.execute("""
        SELECT trading_session, asset_tag, gemma_sentiment_score, gemma_confidence_score, gemma_impact_type, finbert_sentiment
        FROM news_articles
        WHERE trading_session IS NOT NULL
    """).df()
    news['trading_session'] = pd.to_datetime(news['trading_session']).dt.date
    
    # Base daily aggregations
    daily_sent = news.groupby(['asset_tag', 'trading_session']).apply(lambda x: pd.Series({
        'news_volume': len(x),
        'finbert_mean': x['finbert_sentiment'].mean(),
        'gemma_mean': x['gemma_sentiment_score'].mean(),
        'gemma_dispersion': x['gemma_sentiment_score'].std(ddof=0),
        'gemma_conf_weighted': (x['gemma_sentiment_score'] * x['gemma_confidence_score']).sum() / x['gemma_confidence_score'].sum() if x['gemma_confidence_score'].sum() > 0 else 0
    })).reset_index()
    
    # Article Impact Index
    daily_sent['article_impact_index'] = daily_sent['gemma_mean'] * np.log(1 + daily_sent['news_volume'])
    
    # Separated by Impact Type (Direct/Indirect)
    impact_group = news.groupby(['asset_tag', 'trading_session', 'gemma_impact_type']).agg(
        mean_score=('gemma_sentiment_score', 'mean'),
        count=('gemma_sentiment_score', 'count')
    ).reset_index()
    
    impact_group['impact_index'] = impact_group['mean_score'] * np.log(1 + impact_group['count'])
    
    pivot_impact = impact_group.pivot_table(
        index=['asset_tag', 'trading_session'], 
        columns='gemma_impact_type', 
        values=['mean_score', 'impact_index'],
        fill_value=0
    )
    # Flatten columns
    pivot_impact.columns = [f"{col[1].lower()}_{col[0]}" for col in pivot_impact.columns]
    pivot_impact = pivot_impact.reset_index()
    
    # Merge sentiment aggregations
    all_sent = pd.merge(daily_sent, pivot_impact, on=['asset_tag', 'trading_session'], how='left')
    all_sent.fillna(0, inplace=True) # Fill missing indirect/direct components with 0
    
    print("4. Joining OHLCV with Sentiment...")
    # Find max news date per asset
    max_news_dates = news.groupby('asset_tag')['trading_session'].max().to_dict()
    
    model_df = pd.merge(ohlcv, all_sent, on=['asset_tag', 'trading_session'], how='left')
    
    # Filter rows beyond the max news date for each asset
    filtered_dfs = []
    for asset, max_date in max_news_dates.items():
        asset_df = model_df[(model_df['asset_tag'] == asset) & (pd.to_datetime(model_df['trading_session']) <= pd.to_datetime(max_date))]
        filtered_dfs.append(asset_df)
    
    model_df = pd.concat(filtered_dfs, ignore_index=True)
    
    # Extract the sentiment columns to fillna with 0
    sentiment_cols = ['news_volume', 'finbert_mean', 'gemma_mean', 'gemma_dispersion', 'gemma_conf_weighted', 'article_impact_index'] + list(pivot_impact.columns[2:])
    for col in sentiment_cols:
        model_df[col] = model_df[col].fillna(0)
        
    # Calculate Sentiment Persistence (3-day SMA of daily sentiment mean) AFTER filling 0s
    model_df['sentiment_persistence_3d'] = model_df.groupby('asset_tag')['gemma_mean'].transform(lambda x: x.rolling(3).mean())
    model_df['sentiment_persistence_3d'] = model_df['sentiment_persistence_3d'].fillna(0)

    # Shift Next-Day Absolute Return for the Scatter Plot target
    model_df['next_day_abs_return'] = model_df.groupby('asset_tag')['log_return_1d'].shift(-1).abs()
    
    # Drop records where Target Volatility is NA (first 22 days per asset)
    model_df = model_df.dropna(subset=['realized_vol_22d'])
    
    print("5. Saving to DuckDB Master Feature Store...")
    con.execute("DROP TABLE IF EXISTS model_features")
    con.execute("CREATE TABLE model_features AS SELECT * FROM model_df")
    
    print(f"Successfully built model_features table with {len(model_df)} rows and {len(model_df.columns)} features.")
    
    # Preview schema
    schema = con.execute("DESCRIBE model_features").df()
    print("\nTable Schema Preview:")
    print(schema[['column_name', 'column_type']].head(20).to_string())
    
    con.close()

if __name__ == '__main__':
    build_features()
