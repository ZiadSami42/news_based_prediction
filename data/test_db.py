import duckdb
import pandas as pd
con = duckdb.connect('prediction.db')
df = con.execute("SELECT asset_tag, COUNT(*) as total, SUM(CASE WHEN content IS NULL OR content = '' THEN 1 ELSE 0 END) as missing_content, SUM(CASE WHEN summary IS NULL OR summary = '' THEN 1 ELSE 0 END) as missing_summary FROM news_articles GROUP BY asset_tag").df()
print(df)
con.close()
