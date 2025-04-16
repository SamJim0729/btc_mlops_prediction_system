# ✅ services/monthly_update_service.py
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
from btc_data_service import update_historical_prices
from market_metrics_service import update_market_metrics
from macro_data_service import update_macro_metrics

def update_all_data(start:str=None, end:str=None):
    """
    執行每日 BTC 數據更新流程，涵蓋以下模組：
    1. BTC 歷史價格資料
    2. 鏈上市場指標（如交易量、哈希率等）
    3. 總體經濟指標（如 CPI、M2、利率等）

    Args:
        start (str, optional): 起始日期，格式為 'YYYY-MM-DD'
        end (str, optional): 結束日期，格式為 'YYYY-MM-DD'
    """
    
    print("🚀 開始每日 BTC 數據更新流程...")
    update_market_metrics()
    if start and end:
        update_historical_prices(start, end)
        update_macro_metrics(start, end)
    else:
        print('scrape api data沒提供區間，區間自動調整為前月1日~前天')
        update_historical_prices()
        update_macro_metrics()
    
    print("✅ 每日數據更新完成！")

if __name__=='__main__':
    pass