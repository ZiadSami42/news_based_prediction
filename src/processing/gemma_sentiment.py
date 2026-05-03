import duckdb
import pandas as pd
import requests
import json
import time
import re
import logging
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

DB_PATH = 'data/prediction.db'

# Google API Keys
API_KEYS = [
    os.environ.get("GEMINI_API_KEY_1"),
    os.environ.get("GEMINI_API_KEY_2"),
    os.environ.get("GEMINI_API_KEY_3"),
    os.environ.get("GEMINI_API_KEY_4"),
    os.environ.get("GEMINI_API_KEY_5"),
    os.environ.get("GEMINI_API_KEY_6"),
    os.environ.get("GEMINI_API_KEY_7"),
    os.environ.get("GEMINI_API_KEY_8"),
    os.environ.get("GEMINI_API_KEY_9")
]
API_KEYS = [k for k in API_KEYS if k] # Filter out missing keys

MODEL_NAME = "models/gemma-3-27b-it"

def setup_db():
    con = duckdb.connect(DB_PATH)
    cols = con.execute("PRAGMA table_info('news_articles')").df()
    new_cols = {
        'gemma_sentiment_score': 'REAL',
        'gemma_confidence_score': 'REAL',
        'gemma_impact_type': 'VARCHAR',
        'gemma_reasoning': 'TEXT'
    }
    for col, dtype in new_cols.items():
        if col not in cols['name'].values:
            logging.info(f"Adding '{col}' column to news_articles table...")
            con.execute(f"ALTER TABLE news_articles ADD COLUMN {col} {dtype}")
    con.close()

def get_prompt(asset_tag, text):
    # NVDA Prompt
    if "NVDA" in asset_tag:
        system_prompt = """You are a quantitative financial analyst specializing in the semiconductor and technology sectors. Your task is to perform Target-Based Financial Sentiment Analysis.
TARGET ASSET: NVIDIA Corporation (NVDA)
ASSET CONTEXT: AI hardware, data centers, semiconductor supply chain, and global tech sector movements.
INSTRUCTIONS:
1. Entity Isolation: Determine if the news directly impacts NVDA, its supply chain, its competitors (e.g., AMD, Intel), or the broader tech macro-environment.
2. Reasoning: Provide a concise, step-by-step logical link between the news event and NVDA's potential stock price volatility.
3. Scoring: Assign a sentiment score from -1.0 (highly negative for NVDA) to 1.0 (highly positive for NVDA). Assign a confidence score from 0.0 (uncertain) to 1.0 (highly certain).
4. Strict Output: You must output ONLY a valid JSON object. Do not include markdown formatting, backticks, or any conversational text before or after the JSON.
OUTPUT SCHEMA:
{
"asset": "NVDA",
"reasoning": "string",
"impact_type": "Direct" | "Indirect" | "None",
"sentiment_score": float,
"confidence_score": float
}"""
    
    # BRN1! (USO/Brent) Prompt
    elif "USO" in asset_tag or "BRN" in asset_tag:
        system_prompt = """You are a quantitative financial analyst specializing in global commodities and energy markets. Your task is to perform Target-Based Financial Sentiment Analysis.
TARGET ASSET: Brent Crude Oil (BRN1!)
ASSET CONTEXT: Global energy markets, OPEC+ production decisions, geopolitical tensions, macroeconomic supply and demand dynamics.
INSTRUCTIONS:
1. Entity Isolation: Determine if the news directly impacts global oil supply/demand, geopolitical stability in oil-producing regions, or currency impacts (like the USD) that affect commodity pricing.
2. Reasoning: Provide a concise, step-by-step logical link between the news event and Brent Crude's potential price volatility.
3. Scoring: Assign a sentiment score from -1.0 (highly bearish for Brent prices) to 1.0 (highly bullish for Brent prices). Assign a confidence score from 0.0 (uncertain) to 1.0 (highly certain).
4. Strict Output: You must output ONLY a valid JSON object. Do not include markdown formatting, backticks, or any conversational text before or after the JSON.
OUTPUT SCHEMA:
{
"asset": "BRN1!",
"reasoning": "string",
"impact_type": "Direct" | "Indirect" | "None",
"sentiment_score": float,
"confidence_score": float
}"""
    
    # EGX70 Prompt
    elif "EGX" in asset_tag:
        system_prompt = """You are a quantitative financial analyst specializing in emerging markets and Middle Eastern equities. Your task is to perform Target-Based Financial Sentiment Analysis. If the input text is in Arabic, internally translate and analyze it, but generate the final output strictly in English.
TARGET ASSET: EGX70 EWI (Egyptian Exchange 70 Equal Weight Index)
ASSET CONTEXT: Egyptian small and medium-sized enterprises, local inflation rates, Egyptian Pound (EGP) currency fluctuations, Central Bank of Egypt policies, and regional geopolitical stability.
INSTRUCTIONS:
1. Entity Isolation: Determine if the news impacts the Egyptian macro-economy, specific local sectors (like real estate, agriculture, or local manufacturing), or currency liquidity. Remember this index excludes the top 30 largest companies, so focus on broader SME economic conditions.
2. Reasoning: Provide a concise, step-by-step logical link between the news event and the EGX70's potential price volatility.
3. Scoring: Assign a sentiment score from -1.0 (highly negative for the Egyptian local market) to 1.0 (highly positive for the Egyptian local market). Assign a confidence score from 0.0 (uncertain) to 1.0 (highly certain).
4. Strict Output: You must output ONLY a valid JSON object. Do not include markdown formatting, backticks, or any conversational text before or after the JSON.
OUTPUT SCHEMA:
{
"asset": "EGX70",
"reasoning": "string",
"impact_type": "Direct" | "Indirect" | "None",
"sentiment_score": float,
"confidence_score": float
}"""
    
    else:
        # Fallback
        system_prompt = "Perform target-based financial sentiment analysis. Output ONLY a JSON object with keys: asset, reasoning, impact_type, sentiment_score, confidence_score."

    return f"{system_prompt}\n\nINPUT TEXT:\n{text}"

def extract_json(text):
    try:
        # Try finding JSON block
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(text)
    except:
        return None

def call_gemma(prompt, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemma-3-27b-it:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "topP": 0.95,
            "maxOutputTokens": 512
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 429:
            return None, "RATE_LIMIT"
        response.raise_for_status()
        data = response.json()
        return data['candidates'][0]['content']['parts'][0]['text'], None
    except Exception as e:
        return None, str(e)

def _process_single_article(row, key_group, key_group_idx, db_lock):
    """Process a single article using a specific key group. Returns (rowid, success)."""
    summary_text = row['summary'] if (pd.notna(row['summary']) and row['summary'] != '') else row['headline']
    content_text = row['content'] if ('content' in row and pd.notna(row['content']) and row['content'] != '') else ''
    
    combined_text = f"{summary_text}\n\n{content_text}".strip()
    
    # Truncation: 10,000 chars ≈ 2,500 tokens. Each key gets ~5 req/min = 12,500 TPM (under 15k limit)
    final_text = combined_text[:10000]
    prompt = get_prompt(row['asset_tag'], final_text)
    
    success = False
    retries = 0
    local_key_idx = 0  # Round-robin within this group
    
    while not success and retries < len(key_group):
        api_key = key_group[local_key_idx]
        response_text, error = call_gemma(prompt, api_key)
        
        if error:
            if error == "RATE_LIMIT" or "429" in error or "503" in error:
                logging.warning(f"[Group {key_group_idx}] Rate limit/503 on key {local_key_idx}. Switching...")
            else:
                logging.error(f"[Group {key_group_idx}] Error for row {row['rowid']}: {error}")
                if "400" in error and len(final_text) > 2000:
                    logging.info(f"[Group {key_group_idx}] Falling back to shorter text...")
                    final_text = final_text[:2000]
                    prompt = get_prompt(row['asset_tag'], final_text)
                    retries += 1
                    time.sleep(1)
                    continue
                else:
                    break
            
            # Switch to next key within this group
            local_key_idx = (local_key_idx + 1) % len(key_group)
            retries += 1
            time.sleep(1.5)
            continue
        
        data = extract_json(response_text)
        if data:
            with db_lock:
                try:
                    con = duckdb.connect(DB_PATH)
                    con.execute("""
                        UPDATE news_articles 
                        SET gemma_sentiment_score = ?,
                            gemma_confidence_score = ?,
                            gemma_impact_type = ?,
                            gemma_reasoning = ?
                        WHERE rowid = ?
                    """, [
                        data.get('sentiment_score'),
                        data.get('confidence_score'),
                        data.get('impact_type'),
                        data.get('reasoning'),
                        row['rowid']
                    ])
                    con.close()
                except Exception as e:
                    logging.error(f"[Group {key_group_idx}] DB write error for row {row['rowid']}: {e}")
            success = True
        else:
            logging.error(f"[Group {key_group_idx}] Failed to parse JSON for row {row['rowid']}")
            break
    
    return row['rowid'], success


def _worker(worker_id, articles, key_group, db_lock, progress_counter, total_articles):
    """Worker thread: processes its assigned slice of articles using its own key group."""
    local_key_idx = 0
    batch_start = time.time()
    local_count = 0
    
    for _, row in articles.iterrows():
        iter_start = time.time()
        
        # Round-robin key within this group for each request
        local_key_idx = (local_key_idx + 1) % len(key_group)
        
        rowid, success = _process_single_article(row, key_group, worker_id, db_lock)
        
        local_count += 1
        
        # Update shared progress counter
        with db_lock:
            progress_counter[0] += 1
            current_progress = progress_counter[0]
        
        # Log progress every 50 articles (globally)
        if current_progress % 50 == 0:
            elapsed = time.time() - batch_start
            avg = elapsed / local_count if local_count > 0 else 0
            logging.info(
                f"[Global] Processed {current_progress}/{total_articles} articles. "
                f"Worker {worker_id}: {local_count} done (Avg: {avg:.2f}s/req)"
            )
        
        # Dynamic delay: ensure each worker stays at ~5 req/min per key (12s per 3 keys = 4s per req)
        # Since API latency is ~4s, usually no extra sleep is needed
        elapsed_iter = time.time() - iter_start
        sleep_time = max(0, 4.0 - elapsed_iter)
        time.sleep(sleep_time)
    
    logging.info(f"[Worker {worker_id}] Finished processing {local_count} articles.")


def process_gemma(limit=None, test_mode=False):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    
    setup_db()
    
    con = duckdb.connect(DB_PATH)
    query = "SELECT rowid, asset_tag, headline, summary, content FROM news_articles WHERE gemma_sentiment_score IS NULL"
    if limit:
        query += f" LIMIT {limit}"
    
    df = con.execute(query).df()
    con.close()
    
    if df.empty:
        logging.info("No articles found for Gemma processing.")
        return

    # Split 9 keys into 3 groups of 3
    num_workers = min(3, len(API_KEYS) // 3) if len(API_KEYS) >= 3 else 1
    keys_per_worker = len(API_KEYS) // num_workers
    key_groups = [
        API_KEYS[i * keys_per_worker : (i + 1) * keys_per_worker]
        for i in range(num_workers)
    ]
    # Assign any remaining keys to the last group
    remainder = len(API_KEYS) % num_workers
    if remainder:
        key_groups[-1].extend(API_KEYS[-remainder:])
    
    # Split articles evenly across workers
    article_chunks = [df.iloc[i::num_workers].copy() for i in range(num_workers)]
    
    total = len(df)
    logging.info(f"Processing {total} articles with {num_workers} concurrent workers ({keys_per_worker} keys each)...")
    for i, kg in enumerate(key_groups):
        logging.info(f"  Worker {i}: {len(article_chunks[i])} articles, {len(kg)} keys")
    
    db_lock = threading.Lock()
    progress_counter = [0]  # Mutable list for shared counter
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(num_workers):
            f = executor.submit(
                _worker, i, article_chunks[i], key_groups[i],
                db_lock, progress_counter, total
            )
            futures.append(f)
        
        # Wait for all workers and propagate exceptions
        for f in futures:
            f.result()
    
    elapsed = time.time() - start_time
    logging.info(f"Gemma processing complete. {total} articles in {elapsed:.1f}s ({elapsed/total:.2f}s/req effective)")

def test_gemma():
    logging.info("--- Starting Gemma 3 27B Test Run ---")
    # Test one article from each category if possible
    con = duckdb.connect(DB_PATH)
    assets = con.execute("SELECT DISTINCT asset_tag FROM news_articles").df()['asset_tag'].tolist()
    con.close()
    
    for asset in assets:
        logging.info(f"Testing for asset: {asset}")
        con = duckdb.connect(DB_PATH)
        row = con.execute(f"SELECT rowid, asset_tag, headline, summary, content FROM news_articles WHERE asset_tag = '{asset}' LIMIT 1").df()
        con.close()
        
        if not row.empty:
            process_gemma(limit=1, test_mode=True)
    
    con = duckdb.connect(DB_PATH)
    results = con.execute("SELECT asset_tag, gemma_sentiment_score, gemma_impact_type, gemma_reasoning FROM news_articles WHERE gemma_sentiment_score IS NOT NULL").df()
    print("\nTest Results:")
    print(results)
    con.close()

if __name__ == "__main__":
    process_gemma()
