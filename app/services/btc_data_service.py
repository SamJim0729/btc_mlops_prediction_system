# ✅ services/btc_data_service.py
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from services.local_fetcher import BTCDataLocalFetcher
from models.database.mysql_handler import Mysql

load_dotenv()

def update_historical_prices(start=None, end=None):
    """
    抓取 BTC 現貨與期貨數據，並存入 btc_price_history 表。
    """
    fetcher = BTCDataLocalFetcher(start=start, end=end)
    db = Mysql(
        host=os.getenv("SQL_HOST"),
        user=os.getenv("SQL_USER"),
        password=os.getenv("SQL_PASSWORD"),
        database=os.getenv("SQL_DB")
    )

    print("📡 抓取現貨 & 期貨數據...")
    df = fetcher.fetch_historical_prices()

    if df is None or df.empty:
        print("❌ 沒有獲取到歷史價格數據")
        return

    print(f"✅ 獲取 {len(df)} 筆歷史價格數據，開始寫入 MySQL...")
    for _, row in df.iterrows():
        db.insert_data("btc_price_history", row.to_dict())

    print("✅ 歷史價格數據已更新到 btc_price_history")
