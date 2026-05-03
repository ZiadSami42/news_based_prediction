import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plotting style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12

DB_PATH = 'data/prediction.db'
PLOTS_DIR = 'plots'

# Consistent Color Mapping & Order
ASSET_COLORS = {
    'NVDA': '#76b900',   # NVIDIA Green
    'USO': '#eb6e2b',    # Energy Orange
    'EGX': '#FFCE00'     # Egypt Yellow
}
ASSET_ORDER = ['NVDA', 'USO', 'EGX']

def get_con():
    return duckdb.connect(DB_PATH)

def plot_price_trends():
    con = get_con()
    df = con.execute("SELECT time, symbol, close FROM ohlcv ORDER BY time").df()
    con.close()
    
    if df.empty:
        print("No OHLCV data found for price trends.")
        return

    # Map symbols to asset tags for consistent coloring
    symbol_map = {
        'NASDAQ:NVDA': 'NVDA',
        'ICEEUR:BRN1!': 'USO',
        'EGX:EGX70EWI': 'EGX'
    }
    df['asset_tag'] = df['symbol'].map(symbol_map)
    
    plt.figure(figsize=(15, 8))
    # Filter to only known symbols and sort by ASSET_ORDER
    df = df[df['asset_tag'].isin(ASSET_ORDER)]
    
    sns.lineplot(data=df, x='time', y='close', hue='asset_tag', 
                 hue_order=ASSET_ORDER, palette=ASSET_COLORS)
    plt.title('Asset Close Prices Over Time (2020-2026)', fontsize=16, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Price (USD/Local)')
    plt.yscale('log')
    plt.legend(title='Asset', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/eda_price_trends.png')
    plt.close()
    print("Saved eda_price_trends.png")

def plot_news_frequency():
    con = get_con()
    df = con.execute("""
        SELECT 
            date_trunc('month', time) as month, 
            asset_tag, 
            COUNT(*) as article_count 
        FROM news_articles 
        GROUP BY 1, 2 
        ORDER BY 1, 2
    """).df()
    con.close()
    
    if df.empty:
        return
        
    fig, axes = plt.subplots(len(ASSET_ORDER), 1, figsize=(15, 4 * len(ASSET_ORDER)), sharex=True)
    
    for i, asset in enumerate(ASSET_ORDER):
        asset_df = df[df['asset_tag'] == asset]
        if asset_df.empty: continue
        
        sns.barplot(data=asset_df, x='month', y='article_count', ax=axes[i], color=ASSET_COLORS[asset])
        axes[i].set_title(f'Monthly News Frequency: {asset}', fontsize=14, fontweight='bold')
        axes[i].set_ylabel('Article Count')
        
        labels = [pd.to_datetime(d).strftime('%Y-%m') for d in asset_df['month']]
        axes[i].set_xticks(range(len(labels)))
        axes[i].set_xticklabels(labels, rotation=45)
        
    axes[-1].set_xlabel('Month')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/eda_news_frequency.png')
    plt.close()
    print("Saved eda_news_frequency.png")

def plot_content_distribution():
    con = get_con()
    df = con.execute("""
        SELECT asset_tag, LENGTH(content) as content_length, LENGTH(summary) as summary_length 
        FROM news_articles 
        WHERE content IS NOT NULL AND LENGTH(content) > 0 AND LENGTH(summary) > 0
    """).df()
    con.close()
    
    if df.empty:
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Content Length
    sns.histplot(data=df, x='content_length', hue='asset_tag', 
                 hue_order=ASSET_ORDER, palette=ASSET_COLORS,
                 kde=True, log_scale=True, ax=axes[0])
    axes[0].set_title('News Content Length Distribution (Log Scale)')
    axes[0].set_xlabel('Length (characters)')
    
    # Summary Length
    sns.histplot(data=df, x='summary_length', hue='asset_tag', 
                 hue_order=ASSET_ORDER, palette=ASSET_COLORS,
                 kde=True, log_scale=True, ax=axes[1])
    axes[1].set_title('News Summary Length Distribution (Log Scale)')
    axes[1].set_xlabel('Length (characters)')
    
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/eda_length_distribution.png')
    plt.close()
    print("Saved eda_length_distribution.png")

def plot_missing_values_news():
    con = get_con()
    df = con.execute("""
        SELECT asset_tag, 
               COUNT(*) as total, 
               SUM(CASE WHEN summary IS NULL OR summary = '' THEN 1 ELSE 0 END) as missing_summary,
               SUM(CASE WHEN content IS NULL OR content = '' THEN 1 ELSE 0 END) as missing_content
        FROM news_articles 
        GROUP BY asset_tag
    """).df()
    con.close()
    
    if df.empty:
        return
        
    # Calculate percentages
    df['summary'] = (df['missing_summary'] / df['total']) * 100
    df['content'] = (df['missing_content'] / df['total']) * 100
    
    # Melt for plotting
    plot_df = pd.melt(df, id_vars=['asset_tag'], 
                      value_vars=['summary', 'content'],
                      var_name='Field', value_name='Missing Percentage')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x='asset_tag', y='Missing Percentage', hue='Field', order=ASSET_ORDER)
    plt.title('Missing Values Percentage by Asset Tag (News Table)', fontsize=16, fontweight='bold')
    plt.ylabel('Missing Percentage (%)')
    plt.xlabel('Asset Tag')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/missing_values_news.png')
    plt.close()
    print("Saved missing_values_news.png")

def plot_sentiment_distribution():
    con = get_con()
    df = con.execute("""
        SELECT asset_tag, finbert_sentiment, gemma_sentiment_score 
        FROM news_articles 
        WHERE gemma_sentiment_score IS NOT NULL
    """).df()
    con.close()
    
    if df.empty:
        return
        
    fig, axes = plt.subplots(len(ASSET_ORDER), 2, figsize=(16, 4 * len(ASSET_ORDER)))
    
    for i, asset in enumerate(ASSET_ORDER):
        asset_df = df[df['asset_tag'] == asset]
        if asset_df.empty: continue
        
        # FinBERT
        sns.histplot(data=asset_df, x='finbert_sentiment', ax=axes[i][0], bins=20, kde=True, color=ASSET_COLORS[asset])
        axes[i][0].set_title(f'{asset}: FinBERT Sentiment Distribution', fontweight='bold')
        axes[i][0].set_xlabel('Sentiment Score')
        
        # Gemma
        sns.histplot(data=asset_df, x='gemma_sentiment_score', ax=axes[i][1], bins=20, kde=True, color=ASSET_COLORS[asset], alpha=0.6)
        axes[i][1].set_title(f'{asset}: Gemma Sentiment Distribution', fontweight='bold')
        axes[i][1].set_xlabel('Sentiment Score')
        
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/sentiment_distribution.png')
    plt.close()
    print("Saved sentiment_distribution.png")

def plot_sentiment_over_time():
    con = get_con()
    df = con.execute("""
        SELECT trading_session, asset_tag, 
               AVG(finbert_sentiment) as avg_finbert,
               AVG(gemma_sentiment_score) as avg_gemma
        FROM news_articles
        WHERE gemma_sentiment_score IS NOT NULL
        GROUP BY 1, 2
        ORDER BY 1
    """).df()
    con.close()
    
    if df.empty:
        return
        
    fig, axes = plt.subplots(len(ASSET_ORDER), 1, figsize=(15, 5 * len(ASSET_ORDER)), sharex=True)
    
    for i, asset in enumerate(ASSET_ORDER):
        asset_df = df[df['asset_tag'] == asset].set_index('trading_session')
        if asset_df.empty: continue
        
        asset_df.index = pd.to_datetime(asset_df.index)
        asset_df = asset_df.sort_index()
        
        rolling_df = asset_df[['avg_finbert', 'avg_gemma']].rolling(window='30D', min_periods=1).mean()
        
        sns.lineplot(data=rolling_df, x=rolling_df.index, y='avg_finbert', ax=axes[i], 
                     label='FinBERT (30d avg)', color='black', alpha=0.4, linestyle='--')
        sns.lineplot(data=rolling_df, x=rolling_df.index, y='avg_gemma', ax=axes[i], 
                     label='Gemma (30d avg)', color=ASSET_COLORS[asset], linewidth=2.5)
        
        axes[i].set_title(f'{asset}: 30-Day Rolling Sentiment', fontsize=14, fontweight='bold')
        axes[i].set_ylabel('Average Sentiment Score')
        axes[i].legend(loc='upper right')
        
    axes[-1].set_xlabel('Trading Session Date')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/sentiment_over_time.png')
    plt.close()
    print("Saved sentiment_over_time.png")

def plot_impact_types():
    con = get_con()
    df = con.execute("""
        SELECT asset_tag, gemma_impact_type, COUNT(*) as count, AVG(gemma_sentiment_score) as avg_sentiment
        FROM news_articles
        WHERE gemma_impact_type IS NOT NULL
        GROUP BY 1, 2
        ORDER BY 1, 2
    """).df()
    con.close()
    
    if df.empty:
        return

    fig, axes = plt.subplots(1, len(ASSET_ORDER), figsize=(18, 6), sharey=True)
    
    for i, asset in enumerate(ASSET_ORDER):
        asset_df = df[df['asset_tag'] == asset]
        if asset_df.empty:
            axes[i].set_title(f'Impact Type: {asset} (No Data)', fontsize=14, fontweight='bold')
            continue
        
        # Ensure 'Direct' and 'Indirect' order
        asset_df = asset_df.set_index('gemma_impact_type').reindex(['Direct', 'Indirect']).fillna(0).reset_index()
        
        sns.barplot(data=asset_df, x='gemma_impact_type', y='count', ax=axes[i], 
                    palette=[ASSET_COLORS[asset], ASSET_COLORS[asset]], alpha=0.9)
        
        # Differentiate Indirect with transparency and overlay average sentiment
        for j, bar in enumerate(axes[i].patches):
            impact_type = asset_df.iloc[j]['gemma_impact_type']
            avg_sent = asset_df.iloc[j]['avg_sentiment']
            
            if impact_type == 'Indirect':
                bar.set_alpha(0.4)
                bar.set_edgecolor(ASSET_COLORS[asset])
                bar.set_linewidth(2)
                
            # Overlay average sentiment score
            axes[i].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f"Avg Score:\n{avg_sent:.3f}",
                ha='center', va='center', color='black', fontweight='bold', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5')
            )
        
        axes[i].set_title(f'Impact Type: {asset}', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Article Count' if i == 0 else '')

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/impact_type_distribution.png')
    plt.close()
    print("Saved impact_type_distribution.png")

if __name__ == "__main__":
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    print("Generating updated visual explorations...")
    plot_price_trends()
    plot_news_frequency()
    plot_content_distribution()
    plot_missing_values_news()
    plot_sentiment_distribution()
    plot_sentiment_over_time()
    plot_impact_types()
    
    print(f"\nAll visualizations successfully saved to '{PLOTS_DIR}/' folder.")
