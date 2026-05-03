import duckdb
import pandas as pd
from transformers import pipeline
import logging
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DB_PATH = 'data/prediction.db'

def setup_db():
    con = duckdb.connect(DB_PATH)
    # Add finbert_sentiment column if it doesn't exist
    cols = con.execute("PRAGMA table_info('news_articles')").df()
    if 'finbert_sentiment' not in cols['name'].values:
        logging.info("Adding 'finbert_sentiment' column to news_articles table...")
        con.execute("ALTER TABLE news_articles ADD COLUMN finbert_sentiment REAL")
    con.close()

def get_sentiment_pipeline():
    logging.info("Loading FinBERT model...")
    # ProsusAI/finbert is specifically trained for financial sentiment
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1) # -1 for CPU

def process_sentiment(batch_size=50, limit=None, test_mode=False):
    setup_db()
    
    con = duckdb.connect(DB_PATH)
    
    # Select rows where sentiment is NULL
    query = "SELECT rowid, headline, summary FROM news_articles WHERE finbert_sentiment IS NULL"
    if limit:
        query += f" LIMIT {limit}"
        
    df = con.execute(query).df()
    
    if df.empty:
        logging.info("No news articles found for sentiment processing.")
        con.close()
        return

    nlp = get_sentiment_pipeline()
    logging.info(f"Processing sentiment for {len(df)} articles...")

    results = []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        
        # Prepare input text: Summary (fallback to Headline)
        texts = []
        for _, row in batch.iterrows():
            text = row['summary'] if (pd.notna(row['summary']) and row['summary'] != '') else row['headline']
            # Truncate text for model limits (usually 512 tokens)
            texts.append(text[:1000]) 
            
        # Run inference
        outputs = nlp(texts)
        
        # Calculate polar score: positive - negative
        for j, out in enumerate(outputs):
            # ProsusAI/finbert labels: positive, negative, neutral
            label = out['label']
            score = out['score']
            
            polar_score = 0
            if label == 'positive':
                polar_score = score
            elif label == 'negative':
                polar_score = -score
            else:
                # neutral contributes 0 to the polar score
                polar_score = 0
                
            results.append((float(polar_score), int(batch.iloc[j]['rowid'])))

        if not test_mode and (i + batch_size) % 100 == 0:
            logging.info(f"Processed {i + batch_size} articles...")

    # Update DB
    logging.info("Updating database with sentiment scores...")
    for score, rowid in results:
        con.execute("UPDATE news_articles SET finbert_sentiment = ? WHERE rowid = ?", [score, rowid])
    
    con.close()
    logging.info("Sentiment processing complete.")

def test_finbert():
    logging.info("--- Starting FinBERT Test Run ---")
    # Process 3 articles to verify integration
    process_sentiment(batch_size=3, limit=3, test_mode=True)
    
    con = duckdb.connect(DB_PATH)
    results = con.execute("SELECT headline, finbert_sentiment FROM news_articles WHERE finbert_sentiment IS NOT NULL LIMIT 3").df()
    print("\nTest Results:")
    print(results)
    con.close()

if __name__ == "__main__":
    process_sentiment(batch_size=50)
