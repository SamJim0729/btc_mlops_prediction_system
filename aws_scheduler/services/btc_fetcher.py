import json
import os
import datetime
import requests
import boto3
import pytz

class BTCDataFetcher:
    """封裝 BTC 相關 API 抓取與 S3 存儲的類別"""

    def __init__(self, bucket_name=None):
        """初始化 S3 客戶端與 Bucket 設定"""
        self.s3_client = boto3.client("s3")
        self.bucket_name = bucket_name or os.environ.get("S3_BUCKET_NAME", "btc-scraper-storage")

    def fetch_api_data(self, url, params=None):
        """通用 API 請求函式，增加錯誤處理與超時"""
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ API 請求失敗: {e}")
            return None

    def get_funding_rate(self, symbol="BTCUSDT"):
        """獲取 BTC 期貨資金費率（Funding Rate）"""
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1"
        data = self.fetch_api_data(url)
        return data[0] if data else None

    def get_open_interest(self, symbol="BTCUSDT"):
        """獲取 BTC 期貨未平倉量（Open Interest）"""
        url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}"
        data = self.fetch_api_data(url)
        return float(data["openInterest"]) if data else None

    def get_leverage_ratio(self, symbol="BTCUSDT"):
        """獲取 BTC 槓桿比率（Leverage Ratio）"""
        url = f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={symbol}&period=1h&limit=1"
        data = self.fetch_api_data(url)
        return float(data[0]["longShortRatio"]) if data else None

    def get_fear_greed_index(self):
        """獲取市場恐懼與貪婪指數（Fear & Greed Index）"""
        url = "https://api.alternative.me/fng/?limit=1"
        data = self.fetch_api_data(url)
        return data["data"][0] if data else None
    
    def get_spot_price(self, symbol="BTCUSDT"):
        """獲取 BTC 現貨價格"""
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        data = self.fetch_api_data(url)
        return float(data["price"]) if data else None

    def get_stablecoin_supply(self, stablecoin_id="tether"):
        """從 CoinGecko API 獲取穩定幣的流通供應量"""
        url = f"https://api.coingecko.com/api/v3/coins/{stablecoin_id}"
        data = self.fetch_api_data(url)
        if data:
            return {
                "symbol": data["symbol"].upper(),
                "circulating_supply": data["market_data"]["circulating_supply"],
                "market_cap": data["market_data"]["market_cap"]["usd"]
            }
        return None

    def get_file_index(self):
        """獲取當天已上傳的檔案數量，確保檔名唯一"""
        tz = pytz.timezone("Asia/Taipei")
        taiwan_time = datetime.datetime.now(tz)
        today_str = taiwan_time.strftime("%Y%m%d")
        prefix = f"{today_str}_btc_data_"
        
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            count = response.get("KeyCount", 0)
            return count + 1  # 下一個檔案的編號
        except Exception as e:
            print(f"❌ 無法獲取 S3 檔案列表: {e}")
            return 1  # 如果有錯誤，從 1 開始

    def save_to_s3(self, data):
        """將 JSON 數據存入 S3"""
        # 設定台灣時區 (Asia/Taipei)
        tz = pytz.timezone("Asia/Taipei")
        taiwan_time = datetime.datetime.now(tz)

        # 取得 YYYYMMDD 格式的日期
        today_str = taiwan_time.strftime("%Y%m%d")
        file_index = self.get_file_index()
        file_name = f"{today_str}_btc_data_{file_index}.json"

        # 轉換 JSON
        json_data = json.dumps(data, indent=4)

        # 上傳到 S3
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=file_name,
                Body=json_data,
                ContentType="application/json"
            )
            print(f"✅ 成功上傳檔案到 S3: {file_name}")
            return file_name
        except Exception as e:
            print(f"❌ 上傳 S3 失敗: {e}")
            return None
