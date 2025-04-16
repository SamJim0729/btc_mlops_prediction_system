from prometheus_client import Gauge, CollectorRegistry, push_to_gateway, delete_from_gateway

# pushgateway才需要打包客製registry，start_prometheus_server會用預設不能自己設，會無法推送
registry = CollectorRegistry()

# ML 模型結果指標
ml_model_r2 = Gauge(name='ml_model_r2', documentation='R² Score of Monthly BTC Prediction', registry=registry)
ml_model_smape = Gauge(name='ml_model_smape', documentation='Mean Absolute Percentage Error (MAPE) of Monthly BTC Prediction', registry=registry)
ml_model_rmse = Gauge(name='ml_model_rmse', documentation='Root Mean Squared Error (RMSE) of Monthly BTC Prediction', registry=registry)

# ml抓出每次模型訓練影響最大的top 5變數，各自drift_score
drift_score = Gauge(name='ml_feature_drift_score', documentation='Data Drift Score per Feature',
                    labelnames=['feature_name'], registry=registry)

# #start_prometheus_server使用
# s3_hourly_btc_price = Gauge('btc_price_usdt', 'Current BTC Price in USDT')
# s3_hourly_fear_greed_index = Gauge('fear_greed_value', 'Current BTC Fear & Greed value')


# 更新 ML 指標
def update_ml_model_results(r2:float, smape:float, rmse:float):
    ml_model_r2.set(r2)
    ml_model_smape.set(smape)
    ml_model_rmse.set(rmse)

def update_data_drift_metrics(feature_name:str, score:float):
    drift_score.labels(feature_name=feature_name).set(score)

# def update_realtime_metrics(price: float, fear_greed: float):
#     s3_hourly_btc_price.set(price)
#     s3_hourly_fear_greed_index.set(fear_greed)


def push_metrics(job='crypto_proj'):
    push_to_gateway('crypto_proj_pushgateway_container:9091', job=job, registry=registry)
    print("✅ Metrics pushed to Prometheus")

# 清除 Pushgateway 中的舊資料
def clear_old_metrics(job='crypto_proj'):
    delete_from_gateway('crypto_proj_pushgateway_container:9091', job=job)
    print("🧹 已清除 Pushgateway 的舊資料")

# def start_prometheus_server(port=8000):
#     start_http_server(port)
#     print(f"🚀 Prometheus metrics server started on port {port}")

if __name__ == '__main__':
    pass
    
