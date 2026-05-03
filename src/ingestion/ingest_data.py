
import duckdb
import json
import os
import pandas as pd
from datetime import datetime, time
import pandas_market_calendars as mcal
from bs4 import BeautifulSoup

DB_PATH = 'data/prediction.db'

# Initialize Calendars
nyse = mcal.get_calendar('NYSE')
# EGX might not be in mcal, let's check or use a custom schedule
try:
    egx_cal = mcal.get_calendar('EGX')
except:
    egx_cal = None

def get_con():
    return duckdb.connect(DB_PATH)

def clean_html(text):
    if text is None:
        return None
    # Remove HTML tags using BeautifulSoup
    soup = BeautifulSoup(text, "html.parser")
    # Get text and replace multiple newlines/spaces with single ones
    cleaned = soup.get_text(separator=' ')
    return ' '.join(cleaned.split())

def shift_timestamp(dt, asset_tag):
    """
    Shifts timestamp to next trading day if it arrives after market close.
    """
    # Ensure dt is timezone-aware or handled correctly
    # For simplicity, we convert to the local exchange time
    
    if asset_tag in ['NVDA', 'USO']:
        # US Markets close at 4:00 PM EST
        # We check if it's after 16:00
        # For a robust approach, we use mcal
        if dt.time() >= time(16, 0):
            # Find next business day
            next_days = nyse.valid_days(start_date=dt, end_date=dt + pd.Timedelta(days=7))
            # If today is in next_days and it's after close, take the second element
            # If today is NOT in next_days (weekend/holiday), the first element is the next open day
            if dt.date() in next_days.date:
                # It's a trading day, but after close
                target_date = next_days[1]
            else:
                # It's a weekend or holiday
                target_date = next_days[0]
            # Set to 09:30 AM of the next trading day
            return target_date.replace(hour=9, minute=30, second=0)
    
    elif asset_tag == 'EGX':
        # Egypt close is approx 14:30 Cairo
        if dt.time() >= time(14, 30):
            # Fallback if EGX cal not available: add 1 day and skip Fri/Sat
            new_dt = dt + pd.Timedelta(days=1)
            while new_dt.weekday() in [4, 5]: # Friday (4), Saturday (5)
                new_dt += pd.Timedelta(days=1)
            return new_dt.replace(hour=10, minute=0, second=0) # Open at 10 AM
            
    return dt

def ingest_ohlcv(file_path):
    print(f"Ingesting OHLCV from {file_path}...")
    con = get_con()
    con.execute("TRUNCATE ohlcv;")
    try:
        con.execute(f"""
        INSERT INTO ohlcv 
        SELECT 
            CAST(datetime AS TIMESTAMP) as time,
            symbol,
            open,
            high,
            low,
            close,
            volume
        FROM read_csv_auto('{file_path}')
        ON CONFLICT (time, symbol) DO NOTHING;
        """)
        count = con.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
        print(f"Total rows in ohlcv: {count}")
    except Exception as e:
        print(f"Error ingesting OHLCV: {e}")
    finally:
        con.close()

def ingest_news(file_path, asset_tag):
    print(f"Ingesting news for {asset_tag} from {file_path}...")
    con = get_con()
    # No truncate here yet, we'll do it in main
    
    with open(file_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in {file_path}: {e}")
            return

    batch = []
    cutoff_date = datetime(2020, 1, 1)
    
    for item in data:
        time_str = item.get('created_at') or item.get('time') or item.get('date')
        if not time_str: continue
            
        try:
            dt = pd.to_datetime(time_str, utc=True).replace(tzinfo=None)
        except: continue
            
        if dt < cutoff_date: continue
            
        # 1. Shift Timestamp to avoid Look-Ahead Bias
        dt = shift_timestamp(dt, asset_tag)
            
        news_id = item.get('id')
        headline = item.get('headline') or item.get('title')
        content = item.get('content') or item.get('body')
        summary = item.get('summary') or item.get('lead')
        
        if not headline: continue
            
        # 2. Clean HTML
        content = clean_html(content)
        summary = clean_html(summary)
            
        batch.append({
            "time": dt,
            "news_id": news_id,
            "asset_tag": asset_tag,
            "headline": headline,
            "content": content,
            "summary": summary
        })
        
        if len(batch) >= 1000:
            df = pd.DataFrame(batch)
            # Ensure time is correctly formatted for DuckDB
            df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_localize(None)
            con.execute("INSERT INTO news_articles SELECT * FROM df ON CONFLICT DO NOTHING")
            batch = []
            
    if batch:
        df = pd.DataFrame(batch)
        df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_localize(None)
        con.execute("INSERT INTO news_articles SELECT * FROM df ON CONFLICT DO NOTHING")

    count = con.execute(f"SELECT COUNT(*) FROM news_articles WHERE asset_tag = ?", (asset_tag,)).fetchone()[0]
    print(f"Total articles for {asset_tag} (2020+, Cleaned, Shifted): {count}")
    con.close()

if __name__ == "__main__":
    from init_duckdb import init_db
    init_db()
    
    con = get_con()
    con.execute("TRUNCATE news_articles;")
    con.close()
    
    if os.path.exists('raw_data/ohlcv_data_2020_2026.csv'):
        ingest_ohlcv('raw_data/ohlcv_data_2020_2026.csv')
        
    news_files = {
        'NVDA': 'raw_data/nvda_news_data.json',
        'USO': 'raw_data/uso_news_data.json',
        'EGX': 'raw_data/egypt_news_data.json'
    }
    
    for tag, path in news_files.items():
        if os.path.exists(path):
            ingest_news(path, tag)
