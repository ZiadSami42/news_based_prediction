
import duckdb
import os

def init_db(db_path='data/prediction.db'):
    # Ensure data directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Connect to DuckDB (creates the file if it doesn't exist)
    con = duckdb.connect(db_path)
    
    # Create ohlcv table
    con.execute("""
    CREATE TABLE IF NOT EXISTS ohlcv (
        time        TIMESTAMP         NOT NULL,
        symbol      TEXT              NOT NULL,
        open        DOUBLE,
        high        DOUBLE,
        low         DOUBLE,
        close       DOUBLE,
        volume      DOUBLE,
        PRIMARY KEY (time, symbol)
    );
    """)
    
    # Create news_articles table
    con.execute("""
    CREATE TABLE IF NOT EXISTS news_articles (
        time        TIMESTAMP         NOT NULL,
        news_id     BIGINT,
        asset_tag   TEXT,
        headline    TEXT,
        content     TEXT,
        summary     TEXT,
        PRIMARY KEY (time, headline, asset_tag)
    );
    """)
    
    print(f"DuckDB database initialized at {db_path}")
    con.close()

if __name__ == "__main__":
    init_db()
