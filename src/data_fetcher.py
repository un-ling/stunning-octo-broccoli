import akshare as ak
import pandas as pd
import sqlite3
import os
import time
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv
from .config import ETF_LIST, ETF_DATA_DB
from .logger import logger

def fetch_etf_data(etf_code, days=700):
    """获取ETF日线数据（带重试与退避）"""
    # Load proxies from .env if provided (HTTP_PROXY/HTTPS_PROXY)
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path)

    logger.info(f"正在获取 {etf_code} 的数据...")
    end_date = datetime.now().strftime('%Y%m%d')

    # Progressive smaller windows to reduce server pressure on retries
    window_days = [days, max(300, days // 2), max(120, days // 4)]
    for attempt, win in enumerate(window_days, start=1):
        start_date = (datetime.now() - timedelta(days=win)).strftime('%Y%m%d')
        try:
            df = ak.fund_etf_hist_em(symbol=etf_code, period="daily",
                                     start_date=start_date, end_date=end_date, adjust="qfq")
            if isinstance(df, pd.DataFrame) and not df.empty:
                logger.info(f"成功获取 {etf_code} 数据（尝试{attempt}）：{len(df)} 条")
                # 礼貌性延时，降低被限流风险
                time.sleep(random.uniform(1.0, 2.0))
                return df
            else:
                logger.warning(f"{etf_code} 返回空数据（尝试{attempt}）")
        except Exception as e:
            logger.error(f"获取 {etf_code} 数据失败（尝试{attempt}）: {e}")
            # 退避延时，逐步增加
            delay = 1.5 * attempt
            time.sleep(delay)

    logger.error(f"多次尝试后仍无法获取 {etf_code} 数据")
    return pd.DataFrame()

def save_to_db(df, etf_code, db_path=ETF_DATA_DB):
    """保存数据到SQLite（增量更新）"""
    if df.empty:
        return
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    table_name = f'etf_{etf_code}'
    
    try:
        # Check if table exists
        cursor = conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        table_exists = cursor.fetchone()
        
        if not table_exists:
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            logger.info(f"已创建并保存 {etf_code} 数据到数据库")
        else:
            # Get existing dates
            try:
                existing_dates = pd.read_sql_query(f"SELECT 日期 FROM {table_name}", conn)
                if not existing_dates.empty:
                    existing_dates_set = set(existing_dates['日期'].astype(str))
                    # Filter new rows
                    df['日期'] = df['日期'].astype(str)
                    new_rows = df[~df['日期'].isin(existing_dates_set)]
                    
                    if not new_rows.empty:
                        new_rows.to_sql(table_name, conn, if_exists='append', index=False)
                        logger.info(f"已增量更新 {etf_code} 数据: {len(new_rows)} 条新记录")
                    else:
                        logger.info(f"{etf_code} 数据已是最新")
                else:
                    df.to_sql(table_name, conn, if_exists='replace', index=False)
                    logger.info(f"表为空，已重新保存 {etf_code} 数据")
            except Exception as e:
                logger.warning(f"读取现有数据失败，尝试全量覆盖: {e}")
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                logger.info(f"已覆盖保存 {etf_code} 数据")
                
    except Exception as e:
        logger.error(f"数据库操作失败: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    for etf in ETF_LIST:
        data = fetch_etf_data(etf)
        save_to_db(data, etf)
