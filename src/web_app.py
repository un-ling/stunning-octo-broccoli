import glob
import json
import os
import sqlite3

import pandas as pd
from flask import Flask, jsonify, send_from_directory

from .config import (
    DATA_DIR,
    DECISIONS_DIR,
    ETF_DATA_DB,
    PROJECT_ROOT,
    STATIC_DIR,
    TEMPLATE_DIR,
    TRADE_HISTORY_DB,
)
from .logger import logger
from .performance import calculate_performance

app = Flask(__name__, static_folder=STATIC_DIR, template_folder=TEMPLATE_DIR)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))

def _score_linear(x, x0, x1):
    if x1 == x0:
        return 0.0
    t = (x - x0) / (x1 - x0)
    return _clamp(t * 100.0, 0.0, 100.0)

def _calc_sell_events(trades_df: pd.DataFrame):
    if trades_df is None or trades_df.empty:
        return []

    df = trades_df.copy()
    if "date" not in df.columns:
        return []
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if df.empty:
        return []

    if "id" in df.columns:
        df = df.sort_values(["date", "id"], ascending=[True, True])
    else:
        df = df.sort_values(["date"], ascending=[True])

    sell_events = []
    buy_fifo = {}
    for _, trade in df.iterrows():
        action = trade.get("action")
        code = trade.get("etf_code")
        qty = float(trade.get("quantity", 0) or 0)
        price = float(trade.get("price", 0) or 0)
        dt = trade.get("date")
        if not code or qty <= 0 or price <= 0 or pd.isna(dt):
            continue

        if action == "buy":
            buy_fifo.setdefault(code, []).append({"price": price, "quantity": qty})
            continue

        if action != "sell":
            continue

        remaining = qty
        cost = 0.0
        proceeds = 0.0
        lots = buy_fifo.get(code, [])
        while remaining > 1e-9 and lots:
            lot = lots[0]
            match_qty = min(remaining, float(lot["quantity"]))
            cost += match_qty * float(lot["price"])
            proceeds += match_qty * price
            lot["quantity"] = float(lot["quantity"]) - match_qty
            remaining -= match_qty
            if lot["quantity"] <= 1e-6:
                lots.pop(0)
        buy_fifo[code] = lots

        if cost <= 0:
            continue

        pnl = proceeds - cost
        pnl_pct = (pnl / cost) * 100 if cost > 0 else 0.0
        sell_events.append(
            {
                "date": dt.strftime("%Y-%m-%d"),
                "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "etf_code": code,
                "quantity": qty,
                "sell_price": price,
                "cost": cost,
                "proceeds": proceeds,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
            }
        )
    return sell_events

def _calc_weekly_win_rate(sell_events):
    if not sell_events:
        return []
    sells_df = pd.DataFrame(sell_events)
    if sells_df.empty:
        return []
    sells_df["date_dt"] = pd.to_datetime(sells_df["date"], errors="coerce")
    sells_df = sells_df.dropna(subset=["date_dt"])
    if sells_df.empty:
        return []
    sells_df["week_start"] = sells_df["date_dt"] - pd.to_timedelta(sells_df["date_dt"].dt.weekday, unit="D")
    sells_df["is_win"] = sells_df["pnl"].astype(float) > 0
    grouped = sells_df.groupby(sells_df["week_start"].dt.strftime("%Y-%m-%d"), sort=True).agg(
        total=("is_win", "size"),
        wins=("is_win", "sum"),
    )
    out = []
    for week_start, row in grouped.iterrows():
        total = int(row["total"])
        wins_n = int(row["wins"])
        out.append(
            {
                "week_start": week_start,
                "win_rate": (wins_n / total) * 100 if total > 0 else 0.0,
                "total_sells": total,
            }
        )
    return out

def _calc_trade_pnl(sell_events):
    out = []
    for idx, e in enumerate(sell_events, start=1):
        out.append(
            {
                "idx": idx,
                "date": e.get("date"),
                "etf_code": e.get("etf_code"),
                "pnl": float(e.get("pnl", 0) or 0),
                "pnl_pct": float(e.get("pnl_pct", 0) or 0),
            }
        )
    return out

def _calc_max_drawdown_window(history):
    dates = (history or {}).get("dates") or []
    values = (history or {}).get("values") or []
    n = min(len(dates), len(values))
    if n < 2:
        return {"peak": None, "trough": None, "recovery": None, "max_drawdown": 0}
    vals = []
    for i in range(n):
        try:
            v = float(values[i])
        except Exception:
            v = None
        vals.append(v)

    peak = None
    peak_idx = 0
    max_dd = 0.0
    trough_idx = 0
    peak_before_idx = 0
    for i, v in enumerate(vals):
        if v is None:
            continue
        if peak is None or v > peak:
            peak = v
            peak_idx = i
        dd = ((v - peak) / peak) if peak and peak > 0 else 0.0
        if dd < max_dd:
            max_dd = dd
            trough_idx = i
            peak_before_idx = peak_idx

    peak_date = dates[peak_before_idx] if 0 <= peak_before_idx < n else None
    trough_date = dates[trough_idx] if 0 <= trough_idx < n else None
    recovery_date = None
    peak_value = vals[peak_before_idx] if 0 <= peak_before_idx < n else None
    if peak_value is not None and trough_idx + 1 < n:
        for i in range(trough_idx + 1, n):
            v = vals[i]
            if v is None:
                continue
            if v >= peak_value:
                recovery_date = dates[i]
                break

    return {
        "peak": peak_date,
        "trough": trough_date,
        "recovery": recovery_date,
        "max_drawdown": abs(float(max_dd)) * 100,
    }

def _calc_strategy_profile(perf, trades_df, sell_events):
    annualized_return_pct = float(perf.get("annualized_return", 0) or 0)
    max_dd_pct = float(perf.get("max_drawdown", 0) or 0)
    volatility_pct = float(perf.get("volatility", 0) or 0)
    win_rate_pct = float(perf.get("win_rate", 0) or 0)

    pnl_amounts = [float(e.get("pnl", 0) or 0) for e in (sell_events or [])]
    gross_profit = sum(x for x in pnl_amounts if x > 0)
    gross_loss = sum(-x for x in pnl_amounts if x < 0)
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

    unique_assets = int(trades_df["etf_code"].nunique()) if isinstance(trades_df, pd.DataFrame) and "etf_code" in trades_df.columns else 0
    trades_n = int(len(trades_df)) if isinstance(trades_df, pd.DataFrame) else 0

    score_return = _score_linear(annualized_return_pct, -30.0, 30.0)
    score_risk = 100.0 - _score_linear(max_dd_pct, 0.0, 35.0)
    score_stability = 100.0 - _score_linear(volatility_pct, 0.0, 120.0)
    score_winrate = _clamp(win_rate_pct, 0.0, 100.0)
    score_pf = _score_linear(float(profit_factor), 0.8, 2.0)
    score_capacity = _clamp(unique_assets / 3.0 * 60.0 + min(40.0, trades_n * 2.0), 0.0, 100.0)

    labels = ["收益能力", "抗风险", "稳定性", "胜率", "盈亏比", "策略容量"]
    scores = [score_return, score_risk, score_stability, score_winrate, score_pf, score_capacity]
    return {
        "labels": labels,
        "scores": scores,
        "raw": {
            "annualized_return": annualized_return_pct,
            "max_drawdown": max_dd_pct,
            "volatility": volatility_pct,
            "win_rate": win_rate_pct,
            "profit_factor": float(profit_factor),
            "unique_assets": unique_assets,
            "total_trades": trades_n,
        },
    }

def _ensure_perf_fields(perf: dict, db_path: str):
    if not isinstance(perf, dict):
        perf = {}

    missing = []
    for k in ("weekly_win_rate", "trade_pnl", "max_drawdown_window", "strategy_profile"):
        if k not in perf:
            missing.append(k)
    if not missing:
        return perf

    trades_df = pd.DataFrame()
    try:
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            try:
                trades_df = pd.read_sql_query("SELECT * FROM trades ORDER BY date, id", conn)
            finally:
                conn.close()
    except Exception:
        trades_df = pd.DataFrame()

    sell_events = _calc_sell_events(trades_df)
    if "weekly_win_rate" in missing:
        perf["weekly_win_rate"] = _calc_weekly_win_rate(sell_events)
    if "trade_pnl" in missing:
        perf["trade_pnl"] = _calc_trade_pnl(sell_events)
    if "max_drawdown_window" in missing:
        perf["max_drawdown_window"] = _calc_max_drawdown_window(perf.get("portfolio_history"))
    if "strategy_profile" in missing:
        perf["strategy_profile"] = _calc_strategy_profile(perf, trades_df, sell_events)

    return perf

@app.route('/api/summary')
def get_summary():
    db_path = os.path.join(DATA_DIR, 'trade_history.db')
    perf = calculate_performance(db_path=db_path)
    perf = _ensure_perf_fields(perf, db_path)
    return jsonify(perf)

def get_db_connection(db_name='trade_history.db'):
    db_path = os.path.join(DATA_DIR, db_name)
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    return send_from_directory(os.path.join(PROJECT_ROOT, 'trades'), 'index.html')

@app.route('/api/trades')
def get_trades():
    conn = get_db_connection()
    if conn is None:
        return jsonify([])
    try:
        trades = pd.read_sql_query("SELECT * FROM trades ORDER BY date DESC LIMIT 100", conn)
        return jsonify(trades.to_dict('records'))
    except Exception as e:
        print(f"Error reading trades: {e}")
        return jsonify([])
    finally:
        conn.close()

@app.route('/api/performance')
def get_performance():
    perf = calculate_performance(db_path=os.path.join(DATA_DIR, 'trade_history.db'))
    return jsonify(perf.get('portfolio_history', {'dates': [], 'values': []}))

@app.route('/api/positions')
def get_positions():
    trade_conn = get_db_connection()
    if trade_conn is None:
        return jsonify([])

    try:
        positions_df = pd.read_sql_query("SELECT * FROM positions", trade_conn)
    except Exception as e:
        logger.error(f"Error reading positions: {e}")
        return jsonify([])
    finally:
        trade_conn.close()

    if positions_df.empty:
        return jsonify([])

    price_db_path = ETF_DATA_DB
    if not os.path.isabs(price_db_path):
        price_db_path = os.path.join(PROJECT_ROOT, price_db_path)

    results = []
    price_conn = sqlite3.connect(price_db_path) if os.path.exists(price_db_path) else None
    try:
        for _, row in positions_df.iterrows():
            etf_code = row.get('etf_code')
            quantity = float(row.get('quantity', 0) or 0)
            entry_price = float(row.get('entry_price', 0) or 0)
            stop_loss = row.get('stop_loss', 0)
            take_profit = row.get('take_profit', 0)

            current_price = None
            current_price_date = None
            if price_conn is not None and etf_code:
                try:
                    df_price = pd.read_sql_query(
                        f"SELECT 日期, 收盘 FROM etf_{etf_code} ORDER BY 日期 DESC LIMIT 1",
                        price_conn,
                    )
                    if not df_price.empty:
                        current_price = float(df_price.iloc[0]['收盘'])
                        current_price_date = str(df_price.iloc[0]['日期'])
                except Exception:
                    current_price = None
                    current_price_date = None

            market_value = quantity * current_price if current_price is not None else None
            pnl = (current_price - entry_price) * quantity if current_price is not None else None
            pnl_pct = ((current_price / entry_price) - 1) * 100 if (current_price is not None and entry_price) else None

            results.append(
                {
                    "etf_code": etf_code,
                    "quantity": quantity,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "current_price_date": current_price_date,
                    "market_value": market_value,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                }
            )
    finally:
        if price_conn is not None:
            price_conn.close()

    results.sort(key=lambda x: (x.get("market_value") is None, -(x.get("market_value") or 0)))
    return jsonify(results)

@app.route('/api/decisions')
def get_decisions():
    decisions = []
    search_pattern = os.path.join(DECISIONS_DIR, '*.json')
    files = glob.glob(search_pattern)
    
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as file:
                d = json.load(file)
                filename = os.path.basename(f)
                # Expected filename format: {date}_{etf}.json (e.g. 20231201_510050.json)
                parts = filename.replace('.json', '').split('_')
                date_str = parts[0] if len(parts) > 0 else 'Unknown'
                etf_code = parts[1] if len(parts) > 1 else 'Unknown'
                
                decisions.append({
                    'file': filename,
                    'etf': etf_code,
                    'date': date_str,
                    'decision': d.get('decision', 'unknown'),
                    'confidence': d.get('confidence', 0),
                    'reasoning': d.get('reasoning', '')
                })
        except Exception as e:
            logger.error(f"Error reading decision file {f}: {e}")
            
    # Sort by date descending
    decisions.sort(key=lambda x: x['date'], reverse=True)
    return jsonify(decisions[:50])

@app.route('/api/status')
def get_status():
    status_path = os.path.join(DATA_DIR, 'status.json')
    paused_flag = os.path.join(DATA_DIR, 'paused.flag')
    status = {
        'last_run': None,
        'paused': os.path.exists(paused_flag),
        'total_value': None
    }
    try:
        if os.path.exists(status_path):
            with open(status_path, 'r', encoding='utf-8') as f:
                s = json.load(f)
                status.update(s)
    except Exception as e:
        logger.error(f"Error reading status: {e}")
    return jsonify(status)

@app.route('/api/pause', methods=['POST'])
def pause_system():
    try:
        with open(os.path.join(DATA_DIR, 'paused.flag'), 'w') as f:
            f.write('1')
        # update status file
        status_path = os.path.join(DATA_DIR, 'status.json')
        status = {
            'paused': True
        }
        try:
            if os.path.exists(status_path):
                with open(status_path, 'r', encoding='utf-8') as f:
                    s = json.load(f)
                    s.update(status)
                with open(status_path, 'w', encoding='utf-8') as f:
                    json.dump(s, f, ensure_ascii=False)
            else:
                with open(status_path, 'w', encoding='utf-8') as f:
                    json.dump(status, f, ensure_ascii=False)
        except Exception:
            pass
        return jsonify({'ok': True})
    except Exception as e:
        logger.error(f"Pause error: {e}")
        return jsonify({'ok': False})

@app.route('/api/resume', methods=['POST'])
def resume_system():
    try:
        flag = os.path.join(DATA_DIR, 'paused.flag')
        if os.path.exists(flag):
            os.remove(flag)
        # update status file
        status_path = os.path.join(DATA_DIR, 'status.json')
        try:
            if os.path.exists(status_path):
                with open(status_path, 'r', encoding='utf-8') as f:
                    s = json.load(f)
                s['paused'] = False
                with open(status_path, 'w', encoding='utf-8') as f:
                    json.dump(s, f, ensure_ascii=False)
        except Exception:
            pass
        return jsonify({'ok': True})
    except Exception as e:
        logger.error(f"Resume error: {e}")
        return jsonify({'ok': False})

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
