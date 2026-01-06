import pandas as pd
import os
from .config import PROMPTS_DIR

SYSTEM_PROMPT = """
你是一位积极进取的短线交易员。你的目标是捕捉市场中的短期波动机会来获利。你对风险有更高的容忍度，并愿意为了潜在收益进行更频繁的交易。
分析时请同时参考 MA20/MA60、成交量、20日波动率以及 MACD(12,26,9)、RSI(14) 的最新数值与变化。
决策必须输出以下JSON格式：
{
"decision": "buy" | "sell" | "hold",
"confidence": 0.0-1.0,
"reasoning": "分析理由（中文，50-100字）",
"target_price": 数值（目标价，基于当前价±波动率/阻力位计算），
"stop_loss": 数值（止损价，基于当前价-波动率/支撑位计算），
"take_profit": 数值（止盈价，基于当前价+波动率/阻力位计算）
}
"""

def build_user_message(etf_code, df):
    """构建用户消息"""
    if df.empty:
        return "无数据"
        
    latest = df.iloc[-1]
    
    # 修复日期获取逻辑：优先使用 '日期' 列，否则尝试 'date'，最后使用 index
    date_val = "未知"
    if '日期' in latest:
        date_val = latest['日期']
    elif 'date' in latest:
        date_val = latest['date']
    else:
        # 如果是 RangeIndex (数字)，尝试转换或寻找其他列
        if isinstance(latest.name, (int, float)):
             # 检查是否有 datetime 类型的列但未被设为 index
             # 这里简单处理：如果没有明确日期列，就标记为“未知”或尝试找 datetime 列
             date_cols = df.select_dtypes(include=['datetime64']).columns
             if len(date_cols) > 0:
                 date_val = latest[date_cols[0]]
             else:
                 # 尝试把 name 转字符串
                 date_val = str(latest.name)
        else:
            date_val = latest.name

    # 格式化日期
    if hasattr(date_val, 'strftime'):
        date_str = date_val.strftime('%Y-%m-%d')
    else:
        date_str = str(date_val)

    close_price = latest['收盘']
    volume = latest['成交量']
    
    # Calculate moving averages if not present (though they should ideally be pre-calculated)
    if 'ma20' not in df.columns:
        df['ma20'] = df['收盘'].rolling(window=20).mean()
    if 'ma60' not in df.columns:
        df['ma60'] = df['收盘'].rolling(window=60).mean()
        
    ma20 = df['ma20'].iloc[-1]
    ma60 = df['ma60'].iloc[-1]
    
    trend = '上涨趋势' if ma20 > ma60 else '下跌趋势' if ma20 < ma60 else '震荡趋势'
    
    vol_ma5 = df['成交量'].rolling(window=5).mean().shift(1).iloc[-1] if len(df) > 5 else volume
    vol_change = volume / vol_ma5 - 1 if vol_ma5 > 0 else 0
    
    vol_status = '成交量显著放大' if vol_change > 0.3 else '成交量显著萎缩' if vol_change < -0.3 else '成交量正常'
    
    # Calculate price position relative to MA20
    price_pos = '高于20日均线，偏强' if close_price > ma20 else '低于20日均线，偏弱'
    
    # Historical stats
    last_5_change = ((df['收盘'].iloc[-1] / df['收盘'].iloc[-5]) - 1) * 100 if len(df) >= 5 else 0
    volatility_20 = df['收盘'].pct_change().rolling(window=20).std().iloc[-1] * 100 if len(df) >= 20 else 0
    if 'macd' not in df.columns or 'macd_signal' not in df.columns or 'macd_hist' not in df.columns:
        ema12 = df['收盘'].ewm(span=12, adjust=False).mean()
        ema26 = df['收盘'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
    macd = df['macd'].iloc[-1]
    macd_signal = df['macd_signal'].iloc[-1]
    macd_hist = df['macd_hist'].iloc[-1]
    if 'rsi14' not in df.columns:
        delta = df['收盘'].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.rolling(14).mean()
        roll_down = down.rolling(14).mean()
        rs = roll_up / roll_down
        df['rsi14'] = 100 - (100 / (1 + rs))
    rsi14 = df['rsi14'].iloc[-1]
    
    # Create the message
    message = f"""
## ETF信息
- 代码: {etf_code}
- 名称: {latest['名称'] if '名称' in latest else '未知'}
## 最新价格数据
- 日期: {date_str}
- 收盘价: {close_price:.3f}
- 成交量: {volume:.0f} 手 ({vol_status})
## 技术指标
- 20日均线: {ma20:.3f}
- 60日均线: {ma60:.3f}
- MACD(12,26,9): DIF {macd:.3f}, DEA {macd_signal:.3f}, 柱体 {macd_hist:.3f}
- RSI(14): {rsi14:.2f}
- 趋势: {trend}
- 价格位置: {price_pos}
## 历史数据统计
- 最近5日涨跌幅: {last_5_change:.2f}%
- 最近20日波动率: {volatility_20:.2f}%
## 任务
基于以上数据，分析{etf_code}的走势，决定是买入、卖出还是持有。
特别注意：
1. 目标价应参考当前价格及阻力位，或设置为当前价 + 2~3倍波动率。
2. 止损价应参考支撑位，或设置为当前价 - 1.5~2倍波动率。
3. 止盈价应高于目标价，确保盈亏比合理。
请输出JSON格式决策。
"""
    return message
