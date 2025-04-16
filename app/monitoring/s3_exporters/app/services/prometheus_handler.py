from prometheus_client import Gauge


#如果用start_http_server，就不用打包registry，pushgateway才需要
s3_hourly_btc_price = Gauge('btc_price_usdt', 'Current BTC Price in USDT')
s3_hourly_fear_greed_index = Gauge('fear_greed_value', 'Current BTC Fear & Greed value')
s3_hourly_funding_rate = Gauge('funding_rate', 'Current BTC Funding Rate')
s3_hourly_leverage_ratio = Gauge('leverage_ratio', 'Current BTC Leverage Ratio')


# 更新 ML 指標
def update_realtime_metrics(price: float, fear_greed: float, funding_rate: float, leverage_ratio: float):
    s3_hourly_btc_price.set(price)
    s3_hourly_fear_greed_index.set(fear_greed)
    s3_hourly_funding_rate.set(funding_rate)
    s3_hourly_leverage_ratio.set(leverage_ratio)
    

if __name__ == '__main__':
    pass
    
