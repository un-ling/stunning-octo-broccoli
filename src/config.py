import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
PROMPTS_DIR = os.path.join(PROJECT_ROOT, 'prompts')
DECISIONS_DIR = os.path.join(PROJECT_ROOT, 'decisions')
TEMPLATE_DIR = os.path.join(PROJECT_ROOT, 'templates')
STATIC_DIR = os.path.join(PROJECT_ROOT, 'static')

ETF_DATA_DB = os.path.join(DATA_DIR, 'etf_data.db')
TRADE_HISTORY_DB = os.path.join(DATA_DIR, 'trade_history.db')

ETF_LIST = ["510050", "159915", "510300"]
INITIAL_CAPITAL = 100000.0
MAX_SINGLE_TRADE_PERCENT = 0.1
MAX_POSITION_PERCENT = 0.4
MIN_TRADE_UNIT = 100
MAX_BUY_QTY_PER_TRADE = 1500
PYRAMID_ADD_QTY = 200
MIN_DECISION_CONFIDENCE = 0.6
PARTIAL_TAKE_PROFIT_QTY = 500

DEFAULT_MODEL = "qwen-turbo"

for directory in [DATA_DIR, LOGS_DIR, PROMPTS_DIR, DECISIONS_DIR]:
    os.makedirs(directory, exist_ok=True)
