import duckdb
import json
import pandas as pd
from datetime import timedelta
from pandas.tseries.offsets import CustomBusinessDay
import exchange_calendars as xcals

DB_PATH = 'data/prediction.db'

def main():
    print("Loading calendars...")
    nyse = xcals.get_calendar('NYSE')
    bday_eg = CustomBusinessDay(weekmask='Sun Mon Tue Wed Thu')
    
    print("Connecting to database...")
    con = duckdb.connect(DB_PATH)
    
    # 1. Fetch current news articles to map
    df_db = con.execute("SELECT rowid, asset_tag, headline, news_id FROM news_articles").df()
    
    print("Loading raw JSON files to restore original timestamps...")
    raw_data = []
    for tag, path in [('NVDA', 'raw_data/nvda_news_data.json'), 
                      ('USO', 'raw_data/uso_news_data.json'), 
                      ('EGX', 'raw_data/egypt_news_data.json')]:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                for item in data:
                    time_str = item.get('created_at') or item.get('time') or item.get('date')
                    headline = item.get('headline') or item.get('title')
                    news_id = item.get('id')
                    if headline and time_str:
                        raw_data.append({
                            'asset_tag': tag,
                            'headline': headline,
                            'news_id': news_id,
                            'raw_time_str': time_str
                        })
        except FileNotFoundError:
            print(f"Warning: {path} not found.")

    raw_df = pd.DataFrame(raw_data)
    
    # Deduplicate raw_df
    # For NVDA/USO use news_id if present
    raw_df_id = raw_df[raw_df['news_id'].notna()].drop_duplicates(subset=['asset_tag', 'news_id'])
    raw_df_no_id = raw_df[raw_df['news_id'].isna()].drop_duplicates(subset=['asset_tag', 'headline'])
    
    print("Matching original timestamps...")
    # Create mapping dictionaries
    map_id = raw_df_id.set_index(['asset_tag', 'news_id'])['raw_time_str'].to_dict()
    map_headline = raw_df_no_id.set_index(['asset_tag', 'headline'])['raw_time_str'].to_dict()
    
    updates = []
    
    print(f"Processing {len(df_db)} rows to compute correct trading sessions...")
    
    for _, row in df_db.iterrows():
        tag = row['asset_tag']
        raw_time_str = None
        
        if pd.notna(row['news_id']):
            raw_time_str = map_id.get((tag, row['news_id']))
            
        if not raw_time_str: # Fallback to headline
            raw_time_str = map_headline.get((tag, row['headline']))
            
        if not raw_time_str:
            continue
            
        dt_utc = pd.to_datetime(raw_time_str, utc=True)
        
        session_date = None
        if tag in ['NVDA', 'USO']:
            # Use NYSE calendar (Standard for USO/NVDA)
            start_str = str((dt_utc - timedelta(days=7)).date())
            end_str = str((dt_utc + timedelta(days=7)).date())
            window = nyse.schedule.loc[start_str:end_str]
            
            for idx, cal_row in window.iterrows():
                s_date = idx.date()
                if dt_utc.date() < s_date:
                    session_date = s_date
                    break
                if dt_utc.date() == s_date:
                    close_time = cal_row['close'] # tz-aware UTC
                    if dt_utc <= close_time:
                        session_date = s_date
                        break
                    else:
                        # After hours, falls through to next session iteration
                        pass
        elif tag == 'EGX':
            # Use CustomBusinessDay (Sun-Thu)
            dt_naive = dt_utc.tz_localize(None)
            shifted_dt = dt_naive + 0 * bday_eg
            session_date = shifted_dt.date()
            
        if session_date:
            updates.append({
                'rowid': row['rowid'],
                'original_time': dt_utc.tz_localize(None),
                'trading_session': session_date
            })
            
    print(f"Updating {len(updates)} rows in database...")
    updates_df = pd.DataFrame(updates)
    
    con.execute("CREATE TEMP TABLE time_updates AS SELECT * FROM updates_df")
    
    # Ensure trading_session column exists
    cols = con.execute("PRAGMA table_info('news_articles')").df()
    if 'trading_session' not in cols['name'].values:
        con.execute("ALTER TABLE news_articles ADD COLUMN trading_session DATE")
        
    con.execute("""
        UPDATE news_articles 
        SET time = tu.original_time,
            trading_session = tu.trading_session
        FROM time_updates tu
        WHERE news_articles.rowid = tu.rowid
    """)
    
    print("\nVerification (100 rows sample):")
    sample = con.execute("SELECT time, trading_session, asset_tag, SUBSTRING(headline, 1, 40) as headline FROM news_articles LIMIT 100").df()
    print(sample)
    
    print("\nSentiment Not-Null Counts:")
    counts = con.execute("""
        SELECT asset_tag, 
               COUNT(finbert_sentiment) as finbert_count, 
               COUNT(gemma_sentiment_score) as gemma_count 
        FROM news_articles 
        GROUP BY asset_tag
    """).df()
    print(counts)
    
    con.close()
    print("\nUpdate completed successfully.")

if __name__ == '__main__':
    main()
