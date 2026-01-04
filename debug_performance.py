
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

from performance import calculate_performance

try:
    # Simulate the path used in web_app.py
    db_path = os.path.join(os.getcwd(), 'data', 'trade_history.db')
    print(f"Testing database path: {db_path}")
    
    result = calculate_performance(db_path)
    print("\n--- Result from calculate_performance ---")
    print(result)
    print("-----------------------------------------")
except Exception as e:
    print(f"\nError occurred: {e}")
    import traceback
    traceback.print_exc()
