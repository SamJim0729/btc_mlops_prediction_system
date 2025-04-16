from services.btc_fetcher import BTCDataFetcher
import datetime
import pytz

def lambda_handler(event, context):
    """AWS Lambda 主函式"""
    btc_fetcher = BTCDataFetcher()  # 初始化類別
    tz = pytz.timezone("Asia/Taipei")
    taiwan_time = datetime.datetime.now(tz)
    # 抓取 BTC 數據
    data = {
        "timestamp": taiwan_time.strftime("%Y/%m/%d %H:%M:%S"),
        "spot_price": btc_fetcher.get_spot_price(),  
        "funding_rate": btc_fetcher.get_funding_rate(),
        "open_interest": btc_fetcher.get_open_interest(),
        "leverage_ratio": btc_fetcher.get_leverage_ratio(),
        "fear_greed_index": btc_fetcher.get_fear_greed_index(),
        "usdt_supply": btc_fetcher.get_stablecoin_supply()
    }

    # 儲存到 S3
    file_name = btc_fetcher.save_to_s3(data)

    # 返回結果
    return {
        "statusCode": 200,
        "message": "Data fetched and saved successfully",
        "file_name": file_name
    }
