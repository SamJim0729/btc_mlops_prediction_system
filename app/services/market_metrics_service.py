import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from services.s3_fetcher import BTCDataAWSFetcher
from models.database.mysql_handler import Mysql

load_dotenv()

def update_market_metrics():
    """
    抓取 AWS S3 上鏈數據並存入 btc_market_metrics 表。
    """
    fetcher = BTCDataAWSFetcher()
    db = Mysql(
        host=os.getenv("SQL_HOST"),
        user=os.getenv("SQL_USER"),
        password=os.getenv("SQL_PASSWORD"),
        database=os.getenv("SQL_DB")
    )

    print("📡 抓取 AWS S3 數據...")
    df = fetcher.process_s3_data()

    if df is None or df.empty:
        print("❌ 沒有獲取到 S3 數據")
        return

    print(f"✅ 獲取 {len(df)} 筆 S3 數據，開始寫入 MySQL...")
    for _, row in df.iterrows():
        db.insert_data("btc_market_metrics", row.to_dict())

    print("✅ 鏈上市場數據已更新到 btc_market_metrics")
