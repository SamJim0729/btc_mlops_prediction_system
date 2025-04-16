import boto3
import json
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv


load_dotenv()

class BTCDataAWSFetcher:
    """負責抓取 S3 JSON 檔案、計算每日指標平均值，並適配 MySQL 存儲格式"""

    def __init__(self):
        """初始化 S3 連線"""
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "ap-northeast-1")
        )
        self.bucket_name = os.getenv("AWS_S3_BUCKET_NAME", "btc-scraper-storage")

    def _list_s3_files(self):
        """列出 S3 `btc_data_*.json` 檔案"""
        response = self.s3.list_objects_v2(Bucket=self.bucket_name)
        if "Contents" not in response:
            return []

         # 過濾文件名稱，只保留日期 <= 昨天的檔案
        all_file_names = [
            obj["Key"] for obj in response["Contents"] # 確保是 YYYYMMDD 格式 # 過濾條件
        ]
        return all_file_names

    def _fetch_s3_data(self, file_names):
        """下載 S3 檔案並轉換為 DataFrame"""
        data_list = []

        for file_key in file_names:
            obj = self.s3.get_object(Bucket=self.bucket_name, Key=file_key)
            file_data = json.loads(obj["Body"].read().decode("utf-8"))
            # 解析 JSON 格式
            record = {
                "trade_date": datetime.strptime(file_data["timestamp"], "%Y/%m/%d %H:%M:%S").strftime("%Y-%m-%d"),
                "funding_rate": float(file_data["funding_rate"]["fundingRate"]) if file_data.get("funding_rate") else None,
                "open_interest": float(file_data["open_interest"]) if file_data.get("open_interest") else None,
                "leverage_ratio": float(file_data["leverage_ratio"]) if file_data.get("leverage_ratio") else None,
                "fear_greed_value": int(file_data["fear_greed_index"]["value"]) if file_data.get("fear_greed_index") else None,
                "fear_greed_cls": file_data["fear_greed_index"]["value_classification"] if file_data.get("fear_greed_index") else None,
                "usdt_supply": float(file_data["usdt_supply"]["circulating_supply"]) if file_data.get("usdt_supply") else None,
                "usdt_market_cap": float(file_data["usdt_supply"]["market_cap"]) if file_data.get("usdt_supply") else None,
                "spot_price": float(file_data["spot_price"]) if file_data.get("spot_price") else None
            }

            data_list.append(record)

        return pd.DataFrame(data_list)
    
    def _aggregate_daily_metrics(self, df):
        """對同一天的數據取平均，生成 `btc_market_metrics` 表格格式"""
        # 數值欄位取平均
        numeric_cols = ["funding_rate", "open_interest", "leverage_ratio", "fear_greed_value", "usdt_supply", "usdt_market_cap"]
        #Pandas 的 groupby().mean() 不需要 額外的 skipna=True 參數，因為 mean() 本身 預設 就會忽略 NaN
        df_numeric_avg = df.groupby("trade_date")[numeric_cols].mean().reset_index()

        # 根據 `fear_greed_value` 平均值來決定 `fear_greed_cls`
        df_numeric_avg["fear_greed_cls"] = df_numeric_avg["fear_greed_value"].apply(self._map_fear_greed_classification)
        return df_numeric_avg
    
    def _map_fear_greed_classification(self, value):
        """根據 `fear_greed_index["value"]` 來決定 `value_classification`"""
        if value >= 75:
            return "Extreme Greed"
        elif value >= 55:
            return "Greed"
        elif value >= 45:
            return "Neutral"
        elif value >= 25:
            return "Fear"
        else:
            return "Extreme Fear"
        
    def _delete_s3_files(self, file_names):
        """刪除已處理的 S3 檔案"""
        for file_key in file_names:
            self.s3.delete_object(Bucket=self.bucket_name, Key=file_key)
            print(f"🗑️ {file_key} 已刪除")

    def process_s3_data(self):
        """主函數：抓取 S3 數據、計算每日平均值"""
        all_file_names = self._list_s3_files()
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        filtered_file_names = [
            obj for obj in all_file_names if obj.startswith("2") and obj.split("_")[0] <= yesterday
        ]

        if not filtered_file_names:
            print("✅ S3 沒有新數據")
            return None

        df = self._fetch_s3_data(filtered_file_names)
        if df.empty:
            print("❌ 無有效數據")
            return None

        # 計算每日平均值
        daily_metrics = self._aggregate_daily_metrics(df)
        print(f"✅ 處理完成，共 {len(daily_metrics)} 筆每日數據")
        
        # 刪除 S3 上已處理的檔案
        self._delete_s3_files(filtered_file_names)

        return daily_metrics
    
    def get_latest_s3_data(self):
        all_file_names = self._list_s3_files()  # prefix 可根據日期處理
        if not all_file_names:
            return None  # 沒有檔案
            # 依照 (日期, 編號) 排序
        def sort_key(key_name):
            parts = key_name.replace(".json", "").split("_")
            date = parts[0]
            index = int(parts[-1])
            return (date, index)
        latest_file_name = sorted(all_file_names, key=sort_key)[-1]  # 最後一個為最新
        return self._fetch_s3_data([latest_file_name])

if __name__ == "__main__":
    pass


