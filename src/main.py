import sqlite3
import pandas as pd
from datetime import datetime
import os
import sys
import json
from apscheduler.schedulers.blocking import BlockingScheduler
from .config import ETF_LIST, ETF_DATA_DB, DATA_DIR, MIN_DECISION_CONFIDENCE
from .logger import logger

# Add current directory to path to ensure imports work
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .data_fetcher import fetch_etf_data, save_to_db
from .ai_decision import get_ai_decision
from .trade_executor import TradeExecutor

# Initialize Executor
executor = TradeExecutor()

def daily_task():
    paused_flag = os.path.join(DATA_DIR, 'paused.flag')
    if os.path.exists(paused_flag):
        logger.info("系统处于暂停状态，跳过本次任务")
        return
    logger.info(f"每日任务开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Fetch Data
    for etf in ETF_LIST:
        logger.info(f"获取 {etf} 数据")
        df = fetch_etf_data(etf, days=700)
        if not df.empty:
            save_to_db(df, etf)
        else:
            logger.warning(f"获取 {etf} 数据失败，跳过")
            continue
        # 礼貌性延时，避免被数据源限流
        try:
            import time
            time.sleep(1.5)
        except Exception:
            pass

        # Step 2: AI Decision
        # Read from DB to ensure we have the latest state
        db_path = ETF_DATA_DB
        if not os.path.isabs(db_path):
            db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), db_path)
        conn = sqlite3.connect(db_path)
        try:
            df_db = pd.read_sql_query(f"SELECT * FROM etf_{etf}", conn)
        except Exception as e:
            logger.error(f"读取数据库失败: {e}")
            conn.close()
            continue
        conn.close()
        
        if df_db.empty:
            continue
            
        current_price = df_db.iloc[-1]['收盘']
        
        logger.info(f"生成 {etf} 的AI决策")
        decision = get_ai_decision(etf, df_db)
        logger.info(f"{etf} AI决策: {decision.get('decision')} 置信度: {decision.get('confidence')}")

        try:
            conf = float(decision.get('confidence', 0) or 0)
        except Exception:
            conf = 0.0
        if conf < float(MIN_DECISION_CONFIDENCE):
            decision = dict(decision or {})
            decision['decision'] = 'hold'
            decision['reasoning'] = f"{decision.get('reasoning', '')}（置信度{conf:.2f}低于阈值{MIN_DECISION_CONFIDENCE}，本次不交易）".strip()
        
        # Step 3: Execute Trade
        executor.execute_trade(etf, decision, current_price)
        
    # Step 4: Calculate Portfolio Value & Risk Control
    # We need current prices for all held ETFs
    current_prices = {}
    price_db = ETF_DATA_DB
    if not os.path.isabs(price_db):
        price_db = os.path.join(os.path.dirname(os.path.dirname(__file__)), price_db)
    conn = sqlite3.connect(price_db)
    for etf in ETF_LIST:
        try:
            # 这里的查询假设最新数据在最后
            # 由于我们现在使用增量更新，可能需要按日期排序取最新
            df_temp = pd.read_sql_query(f"SELECT 收盘 FROM etf_{etf} ORDER BY 日期 DESC LIMIT 1", conn)
            if not df_temp.empty:
                current_prices[etf] = df_temp.iloc[0]['收盘']
        except:
            pass
    conn.close()
    
    # Check Stop Loss / Take Profit
    logger.info("检查止损止盈")
    executor.check_stop_loss_take_profit(current_prices)
    
    total_value = executor.get_portfolio_value(current_prices)
    logger.info(f"当前总资产(含持仓): {total_value:.2f} 元")

    status = {
        "last_run": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "paused": False,
        "total_value": total_value
    }
    try:
        with open(os.path.join(DATA_DIR, 'status.json'), 'w', encoding='utf-8') as f:
            json.dump(status, f, ensure_ascii=False)
    except Exception:
        pass
    logger.info("每日任务结束")

if __name__ == "__main__":
    logger.info("AI ETF Trader 启动中...")
    daily_task()
    scheduler = BlockingScheduler()
    scheduler.add_job(daily_task, 'cron', hour=15, minute=30)
    logger.info("定时任务已启动 (每天 15:30 执行)")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("定时任务已停止")
