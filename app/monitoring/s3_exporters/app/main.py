from services.s3_fetcher import BTCDataAWSFetcher
from services.prometheus_handler import update_realtime_metrics
from prometheus_client import start_http_server


def update_latest_s3_metrics():
    fetcher = BTCDataAWSFetcher()
    df = fetcher.get_latest_s3_data()

    if df is None or df.empty:
        print("❌ 無最新即時資料，跳過推送")
        return

    latest_row = df.iloc[-1]
    spot_price = float(latest_row.get("spot_price", 0))
    fear_greed_value = float(latest_row.get("fear_greed_value", 0))
    funding_rate = float(latest_row.get("funding_rate", 0))
    leverage_ratio = float(latest_row.get("leverage_ratio", 0))

    # 更新 Prometheus 指標
    update_realtime_metrics(spot_price, fear_greed_value, funding_rate, leverage_ratio)
    print(f"✅ 已更新最新指標資料（現貨價格: {spot_price}, 恐慌指數: {fear_greed_value}, 資金費率: {funding_rate}, 槓桿比: {leverage_ratio}）")


if __name__=='__main__':
    # 啟動 HTTP Server 提供 Prometheus 抓取（預設埠 8000）
    start_http_server(8000)
    print("🚀 S3 Exporter 啟動成功，監聽 http://localhost:8000")
    import time
    while True:
        update_latest_s3_metrics()
        time.sleep(1800)  # 每小時更新一次