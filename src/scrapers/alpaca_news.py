import requests
import pandas as pd
import time
import logging
import json
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Alpaca API Credentials
API_KEY = os.environ.get("ALPACA_API_KEY")
API_SECRET = os.environ.get("ALPACA_API_SECRET")
BASE_URL = "https://data.alpaca.markets/v1beta1/news"

def fetch_alpaca_news(ticker: str, start_date: str) -> list:
    """
    Fetches all news articles for a specific ticker and returns a list of dictionaries.
    Implements pagination and rate-limit backoff handling.
    """
    headers = {
        "Apca-Api-Key-Id": API_KEY,
        "Apca-Api-Secret-Key": API_SECRET
    }
    
    params = {
        "symbols": ticker,
        "start": start_date,
        "limit": 50,          # Maximum allowed per request
        "sort": "ASC",        # Chronological order
        "include_content": True 
    }
    
    all_news_data = []
    page_token = None
    request_count = 0

    logging.info(f"Starting JSON data extraction for {ticker} from {start_date}...")

    while True:
        if page_token:
            params["page_token"] = page_token

        try:
            response = requests.get(BASE_URL, headers=headers, params=params)
            
            if response.status_code == 429:
                logging.warning("Rate limit reached. Sleeping for 60 seconds...")
                time.sleep(60)
                continue
                
            response.raise_for_status()
            
        except requests.exceptions.RequestException as e:
            logging.error(f"API Connection Error: {e}")
            break

        data = response.json()
        news_items = data.get("news", [])
        
        if not news_items:
            break
            
        all_news_data.extend(news_items)
        request_count += 1
        
        if request_count % 10 == 0:
            logging.info(f"Retrieved {len(all_news_data)} records for {ticker}...")

        page_token = data.get("next_page_token")
        if not page_token:
            logging.info(f"Extraction complete for {ticker}. Total: {len(all_news_data)}")
            break

        # Gentle throttle for API stability
        time.sleep(0.3)

    return all_news_data

def main():
    start_date = "2020-01-01T00:00:00Z"
    tickers = ["NVDA", "USO"]
    
    for ticker in tickers:
        news_list = fetch_alpaca_news(ticker, start_date)
        
        if news_list:
            current_date = datetime.now().strftime("%Y%m%d")
            filename = f"{ticker}_news_{start_date[:10]}_to_{current_date}.json"
            
            # Save as a formatted JSON file
            # 'indent=4' makes it human-readable; 'ensure_ascii=False' preserves non-English characters
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(news_list, f, indent=4, ensure_ascii=False)
                
            logging.info(f"Successfully saved {ticker} data to {filename}\n")
        else:
            logging.warning(f"No news data found for {ticker}.\n")

if __name__ == "__main__":
    main()