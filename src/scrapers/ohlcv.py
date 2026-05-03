import pandas as pd
from tvDatafeed import TvDatafeed, Interval
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Initialize tvDatafeed
tv = TvDatafeed()

# Define assets to pull
assets = [
    {'symbol': 'NVDA', 'exchange': 'NASDAQ'},
    {'symbol': 'BRN1!', 'exchange': 'ICEEUR'},
    {'symbol': 'EGX70EWI', 'exchange': 'EGX'}
]

# Define consistent colors
color_map = {
    'NVDA': '#76B900',
    'BRN1!': '#964B00',
    'EGX70EWI': '#E9D66B'
}

all_dfs = []

print("Fetching data...")
for asset in assets:
    try:
        # Pull data (n_bars is used as a limit, we will filter by date after)
        df = tv.get_hist(symbol=asset['symbol'], exchange=asset['exchange'], interval=Interval.in_daily, n_bars=2500)
        if df is not None and not df.empty:
            df['Symbol'] = asset['symbol']
            all_dfs.append(df)
            print(f"Successfully fetched {asset['symbol']}")
        else:
            print(f"Warning: No data returned for {asset['symbol']}")
        time.sleep(1) # Small delay to prevent timeouts
    except Exception as e:
        print(f"Error fetching {asset['symbol']}: {e}")

if all_dfs:
    # Combine and filter by date
    full_df = pd.concat(all_dfs)
    full_df.index = pd.to_datetime(full_df.index).tz_localize(None)
    mask = (full_df.index >= '2020-01-01') & (full_df.index <= '2026-04-26')
    filtered_df = full_df.loc[mask].copy()

    # Save to CSV
    csv_filename = 'ohlcv_data_2020_2026.csv'
    filtered_df.to_csv(csv_filename)
    print(f"Data saved to {csv_filename}")

    # Visualization
    sns.set_theme(style='whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Normalize for comparison
    pivot_df = filtered_df.pivot_table(index=filtered_df.index, columns='Symbol', values='close').ffill().dropna()
    if not pivot_df.empty:
        normalized = (pivot_df / pivot_df.iloc[0]) * 100

        for col in normalized.columns:
            ax1.plot(normalized.index, normalized[col], label=col, color=color_map.get(col))
        ax1.set_title('Normalized Price Comparison (Jan 2020 - April 2026)')
        ax1.legend()

        # Volume
        for sym in filtered_df['Symbol'].unique():
            subset = filtered_df[filtered_df['Symbol'] == sym]
            ax2.plot(subset.index, subset['volume'], label=sym, alpha=0.6, color=color_map.get(sym))
        ax2.set_yscale('log')
        ax2.set_title('Trading Volume (Log Scale)')
        ax2.legend()

        plt.tight_layout()
        plt.show()
    else:
        print("No overlapping data found for normalization.")
else:
    print("No data was collected for any symbols.")