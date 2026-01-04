import sqlite3
import pandas as pd
import numpy as np
import os
from .config import TRADE_HISTORY_DB, ETF_DATA_DB, INITIAL_CAPITAL
from .logger import logger

def get_etf_prices(etf_codes, start_date, end_date):
    price_db_path = ETF_DATA_DB
    if not os.path.isabs(price_db_path):
        price_db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), price_db_path)
    
    prices = {}
    if not os.path.exists(price_db_path):
        logger.error(f"Price database not found at {price_db_path}")
        return prices

    conn = sqlite3.connect(price_db_path)
    try:
        for code in etf_codes:
            try:
                query = f"SELECT 日期, 收盘 FROM etf_{code} WHERE 日期 >= '{start_date}' AND 日期 <= '{end_date}'"
                df = pd.read_sql_query(query, conn, index_col='日期', parse_dates=['日期'])
                prices[code] = df['收盘']
            except Exception as e:
                logger.warning(f"Could not load prices for {code}: {e}")
    finally:
        conn.close()
    return prices

def get_daily_portfolio_valuation(trades, etf_prices):
    trades['date_only'] = trades['date'].dt.date
    portfolio_values = []
    
    # Create a date range from first trade to last trade
    if trades.empty:
        return pd.DataFrame(columns=['date', 'value'])
        
    all_dates = pd.to_datetime(sorted(list(trades['date_only'].unique())))
    date_range = pd.date_range(start=all_dates[0], end=all_dates[-1], freq='D')
    
    cash = INITIAL_CAPITAL
    positions = {} # code -> quantity

    for dt in date_range:
        day_str = dt.strftime('%Y-%m-%d')
        
        # Process trades for the day
        day_trades = trades[trades['date_only'] == dt.date()]
        if not day_trades.empty:
            cash = day_trades.iloc[-1]['capital_after']
            
            # Reconstruct positions at end of day based on ALL trades up to this day
            temp_positions = {}
            for _, trade in trades[trades['date'] <= dt].iterrows():
                code = trade['etf_code']
                action = trade['action']
                qty = trade['quantity']
                if action == 'buy':
                    temp_positions[code] = temp_positions.get(code, 0) + qty
                elif action == 'sell':
                    temp_positions[code] = temp_positions.get(code, 0) - qty
            positions = {k: v for k, v in temp_positions.items() if v > 0.01}


        # Calculate portfolio value at end of day
        holdings_value = 0
        for code, quantity in positions.items():
            if code in etf_prices and not etf_prices[code].empty:
                # Find the last available price on or before the current date
                price_series = etf_prices[code]
                latest_price = price_series[price_series.index <= dt]
                if not latest_price.empty:
                    holdings_value += quantity * latest_price.iloc[-1]
        
        total_value = cash + holdings_value
        portfolio_values.append({'date': dt, 'value': total_value})

    return pd.DataFrame(portfolio_values)

def calculate_performance(db_path=TRADE_HISTORY_DB):
    if not os.path.isabs(db_path):
        db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), db_path)
    
    default_result = {
        "initial_capital": INITIAL_CAPITAL,
        "total_trades": 0, "total_sells": 0, "win_rate": 0, "total_return": 0,
        "annualized_return": 0, "max_drawdown": 0, "current_capital": INITIAL_CAPITAL,
        "sharpe_ratio": 0,
        "volatility": 0,
        "portfolio_history": {'dates': [], 'values': []},
        "benchmark_history": {'dates': [], 'values': []},
        "underwater_history": {'dates': [], 'values': []},
        "weekly_win_rate": [],
        "trade_pnl": [],
        "max_drawdown_window": {"peak": None, "trough": None, "recovery": None, "max_drawdown": 0},
        "strategy_profile": {"labels": [], "scores": [], "raw": {}}
    }

    if not os.path.exists(db_path):
        return default_result

    conn = sqlite3.connect(db_path)
    try:
        trades = pd.read_sql_query("SELECT * FROM trades ORDER BY date, id", conn)
    except Exception:
        trades = pd.DataFrame()
    finally:
        conn.close()

    if trades.empty:
        return default_result

    trades['date'] = pd.to_datetime(trades['date'])
    
    sell_events = []
    buy_fifo = {}
    for _, trade in trades.iterrows():
        action = trade.get('action')
        code = trade.get('etf_code')
        qty = float(trade.get('quantity', 0) or 0)
        price = float(trade.get('price', 0) or 0)
        dt = trade.get('date')
        if not code or qty <= 0 or price <= 0 or pd.isna(dt):
            continue

        if action == 'buy':
            buy_fifo.setdefault(code, []).append({'price': price, 'quantity': qty})
            continue

        if action != 'sell':
            continue

        remaining = qty
        cost = 0.0
        proceeds = 0.0
        lots = buy_fifo.get(code, [])
        while remaining > 1e-9 and lots:
            lot = lots[0]
            match_qty = min(remaining, float(lot['quantity']))
            cost += match_qty * float(lot['price'])
            proceeds += match_qty * price
            lot['quantity'] = float(lot['quantity']) - match_qty
            remaining -= match_qty
            if lot['quantity'] <= 1e-6:
                lots.pop(0)
        buy_fifo[code] = lots

        if cost <= 0:
            continue

        pnl = proceeds - cost
        pnl_pct = (pnl / cost) * 100 if cost > 0 else 0.0
        sell_events.append(
            {
                "date": dt.strftime('%Y-%m-%d'),
                "datetime": dt.strftime('%Y-%m-%d %H:%M:%S'),
                "etf_code": code,
                "quantity": qty,
                "sell_price": price,
                "cost": cost,
                "proceeds": proceeds,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
            }
        )

    total_sells = len(sell_events)
    wins = sum(1 for e in sell_events if float(e.get('pnl', 0) or 0) > 0)
    win_rate = (wins / total_sells) if total_sells > 0 else 0.0

    # --- Portfolio Valuation and Metrics ---
    etf_codes = trades['etf_code'].unique()
    start_date = trades['date'].min().strftime('%Y-%m-%d')
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    etf_prices = get_etf_prices(etf_codes, start_date, end_date)
    
    portfolio_history = get_daily_portfolio_valuation(trades, etf_prices)

    if portfolio_history.empty:
        return default_result

    # Prepend initial capital to correctly calculate max drawdown from the start
    first_trade_date = portfolio_history['date'].iloc[0]
    day_before = first_trade_date - pd.Timedelta(days=1)
    initial_state = pd.DataFrame([{'date': day_before, 'value': INITIAL_CAPITAL}])
    portfolio_history = pd.concat([initial_state, portfolio_history], ignore_index=True)

    portfolio_history['value'] = portfolio_history['value'].replace(0, np.nan).ffill()
    
    # Max Drawdown
    peak = portfolio_history['value'].cummax()
    drawdown = (portfolio_history['value'] - peak) / peak
    max_drawdown = drawdown.min() if not drawdown.empty else 0

    # Total Return
    current_capital = portfolio_history['value'].iloc[-1]
    total_return = (current_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL

    # Annualized Return
    days = (portfolio_history['date'].iloc[-1] - portfolio_history['date'].iloc[0]).days
    annualized_return = total_return * (365 / days) if days > 0 else 0

    returns = portfolio_history['value'].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        sharpe_ratio = 0
        volatility = 0
    else:
        std = float(returns.std(ddof=0)) if not np.isnan(returns.std(ddof=0)) else 0.0
        mean = float(returns.mean()) if not np.isnan(returns.mean()) else 0.0
        if std > 0:
            sharpe_ratio = (mean / std) * float(np.sqrt(252))
            volatility = std * float(np.sqrt(252)) * 100
        else:
            sharpe_ratio = 0
            volatility = 0

    underwater = ((portfolio_history['value'] / peak) - 1) * 100
    underwater = underwater.replace([np.inf, -np.inf], np.nan).fillna(0)

    benchmark_code = "510300"
    benchmark_values = []
    try:
        benchmark_prices = get_etf_prices(
            [benchmark_code],
            day_before.strftime('%Y-%m-%d'),
            end_date
        ).get(benchmark_code)
        if benchmark_prices is not None and not benchmark_prices.empty:
            aligned = benchmark_prices.reindex(portfolio_history['date']).sort_index()
            aligned = aligned.bfill().ffill()
            base = float(aligned.iloc[0]) if len(aligned) > 0 else 0.0
            if base > 0:
                benchmark_values = (aligned / base * INITIAL_CAPITAL).astype(float).tolist()
            else:
                benchmark_values = []
    except Exception as e:
        logger.warning(f"Could not calculate benchmark: {e}")
        benchmark_values = []

    weekly_win_rate = []
    if sell_events:
        sells_df = pd.DataFrame(sell_events)
        sells_df['date_dt'] = pd.to_datetime(sells_df['date'])
        sells_df['week_start'] = sells_df['date_dt'] - pd.to_timedelta(sells_df['date_dt'].dt.weekday, unit='D')
        sells_df['is_win'] = sells_df['pnl'].astype(float) > 0
        grouped = sells_df.groupby(sells_df['week_start'].dt.strftime('%Y-%m-%d'), sort=True).agg(
            total=('is_win', 'size'),
            wins=('is_win', 'sum')
        )
        for week_start, row in grouped.iterrows():
            total = int(row['total'])
            wins_n = int(row['wins'])
            weekly_win_rate.append(
                {
                    "week_start": week_start,
                    "win_rate": (wins_n / total) * 100 if total > 0 else 0.0,
                    "total_sells": total,
                }
            )

    trade_pnl = []
    for idx, e in enumerate(sell_events, start=1):
        trade_pnl.append(
            {
                "idx": idx,
                "date": e.get("date"),
                "etf_code": e.get("etf_code"),
                "pnl": float(e.get("pnl", 0) or 0),
                "pnl_pct": float(e.get("pnl_pct", 0) or 0),
            }
        )

    peak_date = None
    trough_date = None
    recovery_date = None
    try:
        if not drawdown.empty:
            trough_idx = int(drawdown.idxmin())
            if trough_idx >= 0:
                peak_before = portfolio_history['value'].iloc[:trough_idx + 1].idxmax()
                peak_date = portfolio_history['date'].iloc[int(peak_before)].strftime('%Y-%m-%d')
                trough_date = portfolio_history['date'].iloc[trough_idx].strftime('%Y-%m-%d')
                peak_value = float(portfolio_history['value'].iloc[int(peak_before)])
                after = portfolio_history.iloc[trough_idx + 1:]
                recovered = after[after['value'] >= peak_value]
                if not recovered.empty:
                    recovery_date = recovered['date'].iloc[0].strftime('%Y-%m-%d')
    except Exception:
        peak_date = None
        trough_date = None
        recovery_date = None

    profit_factor = None
    avg_win_pct = None
    avg_loss_pct = None
    if sell_events:
        pnl_amounts = np.array([float(e.get('pnl', 0) or 0) for e in sell_events], dtype=float)
        pnl_pcts = np.array([float(e.get('pnl_pct', 0) or 0) for e in sell_events], dtype=float)
        wins_mask = pnl_amounts > 0
        losses_mask = pnl_amounts < 0
        gross_profit = float(pnl_amounts[wins_mask].sum()) if wins_mask.any() else 0.0
        gross_loss = float(-pnl_amounts[losses_mask].sum()) if losses_mask.any() else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
        avg_win_pct = float(pnl_pcts[pnl_pcts > 0].mean()) if (pnl_pcts > 0).any() else 0.0
        avg_loss_pct = float(pnl_pcts[pnl_pcts < 0].mean()) if (pnl_pcts < 0).any() else 0.0

    def clamp(x, lo, hi):
        return max(lo, min(hi, x))

    def score_linear(x, x0, x1):
        if x1 == x0:
            return 0.0
        t = (x - x0) / (x1 - x0)
        return clamp(t * 100.0, 0.0, 100.0)

    total_return_pct = float(total_return * 100)
    annualized_return_pct = float(annualized_return * 100)
    max_dd_pct = float(abs(max_drawdown) * 100)
    volatility_pct = float(volatility)
    win_rate_pct = float(win_rate * 100)
    pf = float(profit_factor or 0.0)
    unique_assets = int(trades['etf_code'].nunique()) if 'etf_code' in trades else 0

    score_return = score_linear(annualized_return_pct, -30.0, 30.0)
    score_risk = 100.0 - score_linear(max_dd_pct, 0.0, 35.0)
    score_stability = 100.0 - score_linear(volatility_pct, 0.0, 120.0)
    score_winrate = clamp(win_rate_pct, 0.0, 100.0)
    score_pf = score_linear(pf, 0.8, 2.0)
    score_capacity = clamp(unique_assets / 3.0 * 60.0 + min(40.0, len(trades) * 2.0), 0.0, 100.0)
    profile_labels = ["收益能力", "抗风险", "稳定性", "胜率", "盈亏比", "策略容量"]
    profile_scores = [score_return, score_risk, score_stability, score_winrate, score_pf, score_capacity]
    
    return {
        "initial_capital": INITIAL_CAPITAL,
        "total_trades": len(trades),
        "total_sells": total_sells,
        "win_rate": win_rate * 100,
        "total_return": total_return * 100,
        "annualized_return": annualized_return * 100,
        "max_drawdown": abs(max_drawdown) * 100,
        "current_capital": current_capital,
        "sharpe_ratio": sharpe_ratio,
        "volatility": volatility,
        "portfolio_history": {
            'dates': portfolio_history['date'].dt.strftime('%Y-%m-%d').tolist(),
            'values': portfolio_history['value'].tolist()
        },
        "benchmark_history": {
            'dates': portfolio_history['date'].dt.strftime('%Y-%m-%d').tolist(),
            'values': benchmark_values
        },
        "underwater_history": {
            'dates': portfolio_history['date'].dt.strftime('%Y-%m-%d').tolist(),
            'values': underwater.astype(float).tolist()
        },
        "weekly_win_rate": weekly_win_rate,
        "trade_pnl": trade_pnl,
        "max_drawdown_window": {
            "peak": peak_date,
            "trough": trough_date,
            "recovery": recovery_date,
            "max_drawdown": max_dd_pct
        },
        "strategy_profile": {
            "labels": profile_labels,
            "scores": profile_scores,
            "raw": {
                "total_return": total_return_pct,
                "annualized_return": annualized_return_pct,
                "max_drawdown": max_dd_pct,
                "volatility": volatility_pct,
                "win_rate": win_rate_pct,
                "profit_factor": pf,
                "avg_win_pct": avg_win_pct,
                "avg_loss_pct": avg_loss_pct,
                "unique_assets": unique_assets
            }
        }
    }
