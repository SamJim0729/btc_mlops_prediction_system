# ✅ services/macro_data_service.py
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from services.local_fetcher import MacroDataFetcher
from models.database.mysql_handler import Mysql


load_dotenv()

def update_macro_metrics(start=None, end=None):
    """
    抓取總經數據並存入 macro_market_metrics 表。
    """
    fetcher = MacroDataFetcher(start=start, end=end)
    db = Mysql(
        host=os.getenv("SQL_HOST"),
        user=os.getenv("SQL_USER"),
        password=os.getenv("SQL_PASSWORD"),
        database=os.getenv("SQL_DB")
    )

    print("📡 抓取總經數據...")
    df = fetcher.fetch_all_macro_data()

    if df is None or df.empty:
        print("❌ 沒有獲取到總經數據")
        return

    print(f"✅ 獲取 {len(df)} 筆總經數據，開始寫入 MySQL...")
    for _, row in df.iterrows():
        db.insert_data("macro_market_metrics", row.to_dict())

    print("✅ 總經數據已更新到 macro_market_metrics")
