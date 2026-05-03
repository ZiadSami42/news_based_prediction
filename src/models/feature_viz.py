import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DB_PATH = 'data/prediction.db'
PLOTS_DIR = 'plots'
ASSET_COLORS = {'NVDA': '#76b900', 'USO': '#eb6e2b', 'EGX': '#FFCE00'}
ASSET_ORDER = ['NVDA', 'USO', 'EGX']

def plot_volatility_vs_sentiment():
    con = duckdb.connect(DB_PATH)
    df = con.execute("SELECT asset_tag, trading_session, realized_vol_22d, sentiment_persistence_3d FROM model_features ORDER BY trading_session").df()
    con.close()
    
    if df.empty: return
    
    fig, axes = plt.subplots(len(ASSET_ORDER), 1, figsize=(15, 6 * len(ASSET_ORDER)), sharex=True)
    
    for i, asset in enumerate(ASSET_ORDER):
        asset_df = df[df['asset_tag'] == asset].set_index('trading_session')
        if asset_df.empty: continue
        
        asset_df.index = pd.to_datetime(asset_df.index)
        
        ax1 = axes[i]
        ax2 = ax1.twinx()
        
        # Plot Volatility on primary Y-axis (left)
        ax1.plot(asset_df.index, asset_df['realized_vol_22d'], color='black', alpha=0.5, label='22d Realized Volatility')
        ax1.set_ylabel('Realized Volatility', color='black', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='black')
        
        # Plot Sentiment Persistence on secondary Y-axis (right)
        ax2.plot(asset_df.index, asset_df['sentiment_persistence_3d'], color=ASSET_COLORS[asset], alpha=0.9, linewidth=1.5, label='Sentiment Persistence (SMA3)')
        ax2.set_ylabel('Sentiment Persistence', color=ASSET_COLORS[asset], fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=ASSET_COLORS[asset])
        
        # Add a horizontal line at 0 for sentiment
        ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
        
        axes[i].set_title(f'{asset}: Volatility vs. Sentiment Persistence', fontsize=14, fontweight='bold')
        
        # Combine legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
        
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/volatility_vs_sentiment.png')
    plt.close()
    print("Saved volatility_vs_sentiment.png")

def plot_dispersion_scatter():
    con = duckdb.connect(DB_PATH)
    # We drop NA because some days might not have next_day_abs_return
    df = con.execute("SELECT asset_tag, gemma_dispersion, next_day_abs_return FROM model_features WHERE gemma_dispersion > 0 AND next_day_abs_return IS NOT NULL").df()
    con.close()
    
    if df.empty: return
    
    fig, axes = plt.subplots(1, len(ASSET_ORDER), figsize=(18, 6))
    
    for i, asset in enumerate(ASSET_ORDER):
        asset_df = df[df['asset_tag'] == asset]
        if asset_df.empty: continue
        
        sns.regplot(data=asset_df, x='gemma_dispersion', y='next_day_abs_return', 
                    ax=axes[i], color=ASSET_COLORS[asset], 
                    scatter_kws={'alpha': 0.5}, line_kws={'color': 'black', 'linestyle': '--'})
        
        axes[i].set_title(f'{asset}: Dispersion vs Next-Day Return', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Sentiment Dispersion')
        axes[i].set_ylabel('Next-Day Absolute Return' if i == 0 else '')
        
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/dispersion_vs_return.png')
    plt.close()
    print("Saved dispersion_vs_return.png")

if __name__ == '__main__':
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_volatility_vs_sentiment()
    plot_dispersion_scatter()
