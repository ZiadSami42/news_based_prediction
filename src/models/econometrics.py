import duckdb
import pandas as pd
import numpy as np
import yfinance as yf
from tvDatafeed import TvDatafeed, Interval
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from arch import arch_model
import os

DB_PATH = 'data/prediction.db'

def get_benchmarks(start_date, end_date):
    print("Fetching benchmark indices...")
    # Fetch SPY and VUG for NVDA
    spy = yf.download('SPY', start=start_date, end=end_date)['Close'].reset_index()
    spy.columns = ['Date', 'Close']
    spy['symbol'] = 'SPY'
    
    vug = yf.download('VUG', start=start_date, end=end_date)['Close'].reset_index()
    vug.columns = ['Date', 'Close']
    vug['symbol'] = 'VUG'
    
    # Fetch XLE for USO
    xle = yf.download('XLE', start=start_date, end=end_date)['Close'].reset_index()
    xle.columns = ['Date', 'Close']
    xle['symbol'] = 'XLE'
    
    # Fetch EGX30 for EGX
    tv = TvDatafeed()
    egx30_data = tv.get_hist(symbol='EGX30', exchange='EGX', interval=Interval.in_daily, n_bars=3000)
    egx30 = egx30_data[['close']].reset_index()
    egx30.rename(columns={'datetime': 'Date', 'close': 'Close'}, inplace=True)
    egx30['symbol'] = 'EGX30'
    
    # Calculate log returns for benchmarks
    benchmarks = pd.concat([spy, vug, xle, egx30], ignore_index=True)
    
    # Flatten multi-index columns from yfinance if they exist
    if isinstance(benchmarks.columns, pd.MultiIndex):
        benchmarks.columns = ['Date', 'Close', 'symbol']
        
    benchmarks['Date'] = pd.to_datetime(benchmarks['Date']).dt.tz_localize(None)
    benchmarks['log_return'] = benchmarks.groupby('symbol')['Close'].transform(lambda x: np.log(x / x.shift(1)))
    benchmarks.dropna(subset=['log_return'], inplace=True)
    return benchmarks

def run_econometrics():
    con = duckdb.connect(DB_PATH)
    df = con.execute("SELECT * FROM model_features ORDER BY trading_session").df()
    con.close()
    
    start_date = df['trading_session'].min()
    end_date = df['trading_session'].max()
    
    benchmarks = get_benchmarks(start_date, end_date)
    
    results = []
    
    for asset in ['NVDA', 'USO', 'EGX']:
        print(f"\n--- Processing Econometric Models for {asset} ---")
        asset_df = df[df['asset_tag'] == asset].copy()
        asset_df.dropna(subset=['log_return_1d'], inplace=True)
        
        # Merge Benchmarks
        asset_df['merge_date'] = pd.to_datetime(asset_df['trading_session']).dt.strftime('%Y-%m-%d')
        benchmarks['merge_date'] = pd.to_datetime(benchmarks['Date']).dt.strftime('%Y-%m-%d')
        
        if asset == 'NVDA':
            mkt_proxy = benchmarks[benchmarks['symbol'] == 'SPY'][['merge_date', 'log_return']].rename(columns={'log_return': 'Rm'})
            factor_proxy = benchmarks[benchmarks['symbol'] == 'VUG'][['merge_date', 'log_return']].rename(columns={'log_return': 'Growth_Factor'})
        elif asset == 'EGX':
            mkt_proxy = benchmarks[benchmarks['symbol'] == 'EGX30'][['merge_date', 'log_return']].rename(columns={'log_return': 'Rm'})
            factor_proxy = mkt_proxy.copy() # We just use Rm as the size baseline for Egypt
            factor_proxy.rename(columns={'Rm': 'Size_Factor'}, inplace=True)
        else: # USO
            mkt_proxy = benchmarks[benchmarks['symbol'] == 'XLE'][['merge_date', 'log_return']].rename(columns={'log_return': 'Rm'})
            factor_proxy = mkt_proxy.copy()
            factor_proxy.rename(columns={'Rm': 'Commodity_Factor'}, inplace=True)
            
        asset_df = pd.merge(asset_df, mkt_proxy, on='merge_date', how='left')
        asset_df = pd.merge(asset_df, factor_proxy, on='merge_date', how='left')
        asset_df.dropna(subset=['Rm'], inplace=True) # Ensure benchmark data exists

        
        y_ret = asset_df['log_return_1d'].values
        
        # 1. CAPM / Benchmark OLS (Phase 3)
        X = sm.add_constant(asset_df['Rm'].values)
        model_capm = sm.OLS(y_ret, X).fit()
        results.append({'Asset': asset, 'Model': 'CAPM', 'AIC': model_capm.aic, 'BIC': model_capm.bic, 'Note': f"Beta: {model_capm.params[1]:.4f}"})
        
        # 2. Factor OLS
        factor_col = 'Growth_Factor' if asset == 'NVDA' else 'Size_Factor' if asset == 'EGX' else 'Commodity_Factor'
        X_factor = sm.add_constant(asset_df[['Rm', factor_col]].values)
        model_factor = sm.OLS(y_ret, X_factor).fit()
        results.append({'Asset': asset, 'Model': f'Factor ({factor_col})', 'AIC': model_factor.aic, 'BIC': model_factor.bic})

        # 3. ARIMA (on log returns)
        try:
            model_arima = ARIMA(y_ret, order=(1,0,1)).fit()
            results.append({'Asset': asset, 'Model': 'ARIMA(1,0,1)', 'AIC': model_arima.aic, 'BIC': model_arima.bic})
        except:
            pass
            
        # 4. Prophet (Decomposition)
        # Prophet wants 'ds' and 'y'
        prophet_df = pd.DataFrame({'ds': pd.to_datetime(asset_df['trading_session']), 'y': asset_df['close']})
        try:
            m = Prophet(daily_seasonality=False, yearly_seasonality=True)
            m.fit(prophet_df)
            results.append({'Asset': asset, 'Model': 'Prophet', 'AIC': np.nan, 'BIC': np.nan, 'Note': 'Decomposed trends saved.'})
        except Exception as e:
            print(f"Warning: Prophet failed to fit for {asset} - {e}")
        
        # 5. GARCH(1,1)
        # Rescale returns for GARCH stability
        y_garch = y_ret * 100 
        am_garch = arch_model(y_garch, vol='Garch', p=1, q=1, dist='Normal')
        res_garch = am_garch.fit(disp='off')
        results.append({'Asset': asset, 'Model': 'GARCH(1,1)', 'AIC': res_garch.aic, 'BIC': res_garch.bic})
        
        # 6. EGARCH
        am_egarch = arch_model(y_garch, vol='EGARCH', p=1, o=1, q=1, dist='Normal')
        res_egarch = am_egarch.fit(disp='off')
        results.append({'Asset': asset, 'Model': 'EGARCH(1,1,1)', 'AIC': res_egarch.aic, 'BIC': res_egarch.bic})
        
        # 7. TGARCH
        am_tgarch = arch_model(y_garch, vol='GARCH', p=1, o=1, q=1, power=1.0, dist='Normal')
        res_tgarch = am_tgarch.fit(disp='off')
        results.append({'Asset': asset, 'Model': 'TGARCH(1,1,1)', 'AIC': res_tgarch.aic, 'BIC': res_tgarch.bic})
        
        # 8. GARCH-X (using sentiment dispersion and impact index)
        X_exo = asset_df[['gemma_dispersion', 'article_impact_index']].values
        am_garchx = arch_model(y_garch, x=X_exo, vol='Garch', p=1, q=1, dist='Normal')
        try:
            res_garchx = am_garchx.fit(disp='off')
            results.append({'Asset': asset, 'Model': 'GARCH-X(Sentiment)', 'AIC': res_garchx.aic, 'BIC': res_garchx.bic})
        except:
            print(f"Warning: GARCH-X failed to converge for {asset}")
            
        # Save GARCH residuals for the Hybrid model later
        # We use the standard GARCH(1,1) residuals
        asset_df['garch_residual'] = res_garch.resid
        
        # We need to save these residuals back to the duckdb later
        # Let's save them as a CSV for now, or just update the DuckDB table.
        # It's easier to collect them in a list of DataFrames and write at the end.
        if 'all_residuals' not in locals():
            all_residuals = []
        all_residuals.append(asset_df[['trading_session', 'asset_tag', 'garch_residual']])
            
    res_df = pd.DataFrame(results)
    print("\n--- Econometric Models Evaluation Summary ---")
    print(res_df.to_string(index=False))
    
    # Save to a markdown file
    os.makedirs('artifacts', exist_ok=True)
    res_df.to_markdown('artifacts/econometric_results.md', index=False)
    print("\nSaved evaluation to artifacts/econometric_results.md")

    # Save residuals to DB
    resid_df = pd.concat(all_residuals, ignore_index=True)
    con = duckdb.connect(DB_PATH)
    # create a temporary table and merge it
    con.execute("DROP TABLE IF EXISTS temp_residuals")
    con.execute("CREATE TABLE temp_residuals AS SELECT * FROM resid_df")
    
    # Add column if not exists
    try:
        con.execute("ALTER TABLE model_features ADD COLUMN garch_residual DOUBLE")
    except:
        pass # Column already exists
        
    con.execute("""
        UPDATE model_features 
        SET garch_residual = temp_residuals.garch_residual
        FROM temp_residuals 
        WHERE model_features.trading_session = temp_residuals.trading_session 
        AND model_features.asset_tag = temp_residuals.asset_tag
    """)
    con.execute("DROP TABLE temp_residuals")
    con.close()
    print("Saved GARCH residuals to model_features table.")

if __name__ == '__main__':
    run_econometrics()
