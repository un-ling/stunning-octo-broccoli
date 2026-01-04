import sqlite3
import pandas as pd
from datetime import datetime
import os
from .config import (
    INITIAL_CAPITAL,
    MAX_SINGLE_TRADE_PERCENT,
    MAX_POSITION_PERCENT,
    MIN_TRADE_UNIT,
    MAX_BUY_QTY_PER_TRADE,
    PYRAMID_ADD_QTY,
    PARTIAL_TAKE_PROFIT_QTY,
)
from .logger import logger

class TradeExecutor:
    def __init__(self, initial_capital=INITIAL_CAPITAL, db_path='data/trade_history.db'):
        self.capital = initial_capital
        # Resolve absolute path for db_path
        if not os.path.isabs(db_path):
            self.db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), db_path)
        else:
            self.db_path = db_path
            
        self.init_db()
        self.load_state()

    def init_db(self):
        """初始化数据库，创建交易表和持仓表"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        
        # 交易记录表
        conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            etf_code TEXT,
            action TEXT,
            price REAL,
            quantity REAL,
            value REAL,
            capital_after REAL,
            reasoning TEXT
        )
        """)
        
        # 持仓表 (持久化持仓状态)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            etf_code TEXT PRIMARY KEY,
            quantity REAL,
            entry_price REAL,
            date TEXT,
            stop_loss REAL,
            take_profit REAL
        )
        """)
        
        conn.commit()
        conn.close()

    def load_state(self):
        """从数据库加载持仓和资金状态"""
        conn = sqlite3.connect(self.db_path)
        try:
            # 加载资金状态 (从最后一笔交易)
            trades = pd.read_sql_query("SELECT capital_after FROM trades ORDER BY id DESC LIMIT 1", conn)
            if not trades.empty:
                self.capital = trades.iloc[0]['capital_after']
            
            # 加载持仓
            self.positions = {}
            pos_df = pd.read_sql_query("SELECT * FROM positions", conn)
            for _, row in pos_df.iterrows():
                self.positions[row['etf_code']] = {
                    'quantity': row['quantity'],
                    'entry_price': row['entry_price'],
                    'date': row['date'],
                    'stop_loss': row.get('stop_loss', 0),
                    'take_profit': row.get('take_profit', 0)
                }
                
        except Exception as e:
            logger.error(f"Error loading state: {e}")
        finally:
            conn.close()

    def execute_trade(self, etf_code, decision, current_price):
        conn = sqlite3.connect(self.db_path)
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        action = decision['decision']
        reasoning = decision['reasoning']
        target_price = decision.get('target_price', 0)
        stop_loss = decision.get('stop_loss', 0)
        take_profit = decision.get('take_profit', 0)
        
        executed = False
        
        if action == 'buy':
            max_single_trade = self.capital * MAX_SINGLE_TRADE_PERCENT
            
            current_portfolio_value = self.get_portfolio_value_fast()
            max_position_value = current_portfolio_value * MAX_POSITION_PERCENT
            current_position_value = 0.0
            existing_pos = self.positions.get(etf_code)
            if existing_pos and current_price > 0:
                current_position_value = float(existing_pos.get('quantity', 0) or 0) * current_price

            allowed_position_value = max(0.0, max_position_value - current_position_value)
            trade_cap = min(max_single_trade, allowed_position_value, self.capital)

            if current_price > 0 and trade_cap >= current_price * MIN_TRADE_UNIT:
                if existing_pos is None:
                    quantity = trade_cap / current_price
                    quantity = min(float(MAX_BUY_QTY_PER_TRADE), quantity)
                    quantity = int(quantity // MIN_TRADE_UNIT) * MIN_TRADE_UNIT
                    if quantity >= MIN_TRADE_UNIT:
                        value = quantity * current_price
                        if value <= self.capital:
                            self.capital -= value
                            self.positions[etf_code] = {
                                'quantity': quantity,
                                'entry_price': current_price,
                                'date': date,
                                'stop_loss': stop_loss,
                                'take_profit': take_profit
                            }

                            conn.execute("""
                            INSERT OR REPLACE INTO positions (etf_code, quantity, entry_price, date, stop_loss, take_profit)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """, (etf_code, quantity, current_price, date, stop_loss, take_profit))

                            conn.execute("""
                            INSERT INTO trades (date, etf_code, action, price, quantity, value, capital_after, reasoning) 
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """, (date, etf_code, 'buy', current_price, quantity, value, self.capital, reasoning))

                            logger.info(f"买入 {etf_code}: {quantity:.2f} 单位 @ {current_price:.3f}, 总值: {value:.2f}")
                            executed = True
                else:
                    entry_price = float(existing_pos.get('entry_price', 0) or 0)
                    if current_price > entry_price:
                        desired_add = int(PYRAMID_ADD_QTY // MIN_TRADE_UNIT) * MIN_TRADE_UNIT
                        max_add_by_cap = int((trade_cap / current_price) // MIN_TRADE_UNIT) * MIN_TRADE_UNIT
                        add_qty = min(desired_add, max_add_by_cap)
                        if add_qty >= MIN_TRADE_UNIT:
                            value = add_qty * current_price
                            if value <= self.capital:
                                old_qty = float(existing_pos.get('quantity', 0) or 0)
                                new_qty = old_qty + add_qty
                                new_entry = ((old_qty * entry_price) + (add_qty * current_price)) / new_qty if new_qty > 0 else current_price
                                existing_pos['quantity'] = new_qty
                                existing_pos['entry_price'] = new_entry
                                existing_pos['date'] = date
                                if stop_loss:
                                    existing_pos['stop_loss'] = stop_loss
                                if take_profit:
                                    existing_pos['take_profit'] = take_profit

                                self.capital -= value

                                conn.execute("""
                                INSERT OR REPLACE INTO positions (etf_code, quantity, entry_price, date, stop_loss, take_profit)
                                VALUES (?, ?, ?, ?, ?, ?)
                                """, (
                                    etf_code,
                                    float(existing_pos.get('quantity', 0) or 0),
                                    float(existing_pos.get('entry_price', 0) or 0),
                                    date,
                                    float(existing_pos.get('stop_loss', 0) or 0),
                                    float(existing_pos.get('take_profit', 0) or 0),
                                ))

                                conn.execute("""
                                INSERT INTO trades (date, etf_code, action, price, quantity, value, capital_after, reasoning) 
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """, (date, etf_code, 'buy', current_price, add_qty, value, self.capital, reasoning))

                                logger.info(f"加仓买入 {etf_code}: {add_qty:.2f} 单位 @ {current_price:.3f}, 总值: {value:.2f}")
                                executed = True

        elif action == 'sell' and etf_code in self.positions:
            pos = self.positions.get(etf_code)
            entry_price = float(pos.get('entry_price', 0) or 0) if pos else 0.0
            if entry_price > 0 and current_price >= entry_price:
                desired = float(PARTIAL_TAKE_PROFIT_QTY)
                ok = self._sell_partial(conn, etf_code, desired, current_price, date, reasoning)
                if ok:
                    executed = True
                else:
                    self._sell_position(conn, etf_code, current_price, date, reasoning)
                    executed = True
            else:
                stop_loss = float(pos.get('stop_loss', 0) or 0) if pos else 0.0
                if stop_loss > 0 and current_price <= stop_loss:
                    self._sell_position(conn, etf_code, current_price, date, reasoning)
                    executed = True
            
        conn.commit()
        conn.close()
        return self.capital

    def _sell_position(self, conn, etf_code, current_price, date, reasoning):
        pos = self.positions[etf_code]
        quantity = float(pos['quantity'])
        value = quantity * current_price
        profit = value - (quantity * pos['entry_price'])
        self.capital += value
        
        # 记录交易
        conn.execute("""
        INSERT INTO trades (date, etf_code, action, price, quantity, value, capital_after, reasoning) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (date, etf_code, 'sell', current_price, quantity, value, self.capital, reasoning))
        
        # 删除数据库持仓
        conn.execute("DELETE FROM positions WHERE etf_code = ?", (etf_code,))
        
        # 删除内存持仓
        del self.positions[etf_code]
        
        logger.info(f"卖出 {etf_code}: {quantity:.2f} 单位 @ {current_price:.3f}, 总值: {value:.2f}, 盈亏: {profit:.2f}")

    def _sell_partial(self, conn, etf_code, sell_quantity, current_price, date, reasoning):
        pos = self.positions.get(etf_code)
        if pos is None:
            return False

        total_qty = float(pos.get('quantity', 0) or 0)
        sell_qty = float(sell_quantity or 0)
        if total_qty <= 0 or sell_qty <= 0:
            return False

        sell_qty = min(total_qty, sell_qty)
        sell_qty = int(sell_qty // MIN_TRADE_UNIT) * MIN_TRADE_UNIT
        if sell_qty < MIN_TRADE_UNIT:
            return False

        entry_price = float(pos.get('entry_price', 0) or 0)
        value = sell_qty * current_price
        profit = value - (sell_qty * entry_price)
        self.capital += value

        conn.execute("""
        INSERT INTO trades (date, etf_code, action, price, quantity, value, capital_after, reasoning) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (date, etf_code, 'sell', current_price, sell_qty, value, self.capital, reasoning))

        remaining = total_qty - sell_qty
        if remaining < MIN_TRADE_UNIT:
            conn.execute("DELETE FROM positions WHERE etf_code = ?", (etf_code,))
            del self.positions[etf_code]
        else:
            pos['quantity'] = remaining
            pos['date'] = date
            conn.execute("""
            INSERT OR REPLACE INTO positions (etf_code, quantity, entry_price, date, stop_loss, take_profit)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                etf_code,
                float(pos.get('quantity', 0) or 0),
                float(pos.get('entry_price', 0) or 0),
                date,
                float(pos.get('stop_loss', 0) or 0),
                float(pos.get('take_profit', 0) or 0),
            ))

        logger.info(f"分批卖出 {etf_code}: {sell_qty:.2f} 单位 @ {current_price:.3f}, 总值: {value:.2f}, 盈亏: {profit:.2f}")
        return True

    def check_stop_loss_take_profit(self, current_prices):
        """检查是否触发止损止盈"""
        conn = sqlite3.connect(self.db_path)
        triggered = False
        
        for etf_code, pos in list(self.positions.items()): # Use list to allow modification during iteration
            if etf_code in current_prices:
                current_price = current_prices[etf_code]
                stop_loss = pos.get('stop_loss', 0)
                take_profit = pos.get('take_profit', 0)
                
                reason = ""
                if stop_loss > 0 and current_price <= stop_loss:
                    reason = f"触发止损: 当前价 {current_price} <= 止损价 {stop_loss}"
                elif take_profit > 0 and current_price >= take_profit:
                    reason = f"触发止盈: 当前价 {current_price} >= 止盈价 {take_profit}"
                    
                if reason:
                    logger.warning(f"{etf_code} {reason}，执行卖出")
                    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    if stop_loss > 0 and current_price <= stop_loss:
                        self._sell_position(conn, etf_code, current_price, date, reason)
                    else:
                        desired = float(PARTIAL_TAKE_PROFIT_QTY)
                        ok = self._sell_partial(conn, etf_code, desired, current_price, date, reason)
                        if not ok:
                            self._sell_position(conn, etf_code, current_price, date, reason)
                    triggered = True
        
        if triggered:
            conn.commit()
        conn.close()

    def get_portfolio_value(self, current_prices):
        total_value = self.capital
        for etf_code, pos in self.positions.items():
            if etf_code in current_prices:
                total_value += pos['quantity'] * current_prices[etf_code]
            else:
                # Fallback to entry price if current price not available
                total_value += pos['quantity'] * pos['entry_price']

        return total_value

    def get_portfolio_value_fast(self):
        total_value = self.capital
        for etf_code, pos in self.positions.items():
            total_value += pos['quantity'] * pos['entry_price']
        return total_value
