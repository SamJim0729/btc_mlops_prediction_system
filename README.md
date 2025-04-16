
# BTC MLOps Prediction System

本專案為一套以 MLOps 架構實現的比特幣價格預測系統，預測每月 BTC 價格的月均變化率。系統整合資料更新、自動模型訓練、預測結果儲存、異常監控與指標可視化，並透過 Grafana 儀表板呈現模型預測與 BTC 相關指標。  
採用 Docker Compose 管理多容器服務，結合 MLflow、Prometheus、Grafana 等工具，並以 Airflow 定期排程驅動全流程任務，打造高可維護性、易擴展的自動化金融分析平台。


## 專案架構

```bash
.
├── app/                                 # 地端主服務程式目錄
│   ├── configs/                            # 執行特徵篩選 controller，產出模型特徵設定檔
│   │
│   ├── controllers/                        # 控制層，調度整體流程模組
│   │   ├── update_data_controller.py           # 控制資料抓取更新流程
│   │   ├── feature_selection_controller.py     # 控制特徵工程流程
│   │   └── model_training_controller.py        # 控制模型訓練流程
│   │
│   ├── models/                             # 資料前處理與特徵工程流程模組
│   │   ├── data/                               # 資料處理、資料載入、特徵篩選邏輯
│   │   │   ├── data_loader.py                      # 以特徵設定檔驅動的訓練資料建構模組
│   │   │   ├── data_preprocessor.py                # 特徵工程模組，支援資料合併、滯後變數、變化率、溢價率計算
│   │   │   ├── feature_dataset_builder.py          # 根據指定區間構建模型訓練用特徵資料集
│   │   │   ├── feature_engineering.py              # 特徵選擇模組，整合相關係數與 XGBoost 特徵重要性進行變數篩選模組
│   │   │   └── feature_selection_pipeline.py       # 自動化特徵選擇主流程，儲存最終變數設定為 JSON
│   │   │ 
│   │   ├── training/                       # 模型訓練與追蹤模組
│   │   │   ├── model_training_pipeline.py      # 自動化模型訓練主流程，串接訓練與評估
│   │   │   ├── Regressor.py                    # 封裝多模型迴歸訓練邏輯的模組
│   │   │   └── mlflow_handler.py               # 負責將訓練過程與結果送入 MLflow 追蹤
│   │   │
│   │   └── database/                       # 模型訓練與追蹤模組
│   │       └── mysql_handler.py                # MySQL 資料寫入/查詢/更新等封裝邏輯
│   │
│   ├── monitoring/                         # 模型與系統監控模組
│   │   ├── prometheus/                         # Prometheus、alert_rules yml設定檔
│   │   ├── alertmanager/                       # alertmanager 設定檔，串連 slack webhook
│   │   ├── s3_exporters/                       # 指標轉換為 HTTP 端點，供 Prometheus 擷取
│   │   └── prometheus_handler.py               # 整合指標推送與格式化邏輯
│   │
│   ├── services/                           # 資料服務層，負責資料抓取、整併與存取邏輯
│   │   ├── btc_data_service.py                 # BTC數據-資料抓取、存儲邏輯
│   │   ├── macro_data_service.py               # 總經數據-資料抓取、存儲邏輯
│   │   ├── market_metrics_service.py           # 鏈上數據-資料抓取(S3)、存儲邏輯
│   │   ├── monthly_update_service.py           # 自動化資料更新主流程，全資料統一入口
│   │   ├── local_fetcher.py                    # 本地資料擷取工具（支援 BTC / 總經等類別）
│   │   └── s3_fetcher.py                       # 本地資料擷取工具（支援 S3 資料抓取類別）
│   │
│   ├── utils/                              # 工具模組（無業務邏輯，僅通用輔助功能）
│   │   ├── date_utils.py                       # 計算訓練與測試的時間區間模組
│   │   ├── docker_utils.py                     # 執行 docker inspect 查容器 IP 模組
│   │   └── file_utils.py                       # 尋找最新模型特徵設定檔模組
│   │
│   └── crypto_api_app.py                   # Flask API 主應用進入點，接收 Airflow 觸發執行資料更新與模型訓練
│
├── aws_scheduler/                          # AWS Lambda 任務模組（雲端資料排程）
│   ├── lambda_package/                         # Lambda 打包上傳的壓縮檔與原始碼
│   │   ├── python/                                 # 部署到 Lambda 的 Python 套件資料夾（透過 ZIP 打包）
│   │   └── lambda_package.zip                      # Lambda 部署用的完整壓縮檔（含函數與依賴）
│   │   
│   ├── services/                               # Lambda 的核心商業邏輯模組
│   │   └── btc_fetcher.py                          # 封裝 BTC API 抓取與 S3 上傳邏輯的模組
│   │ 
│   ├── lambda_function.py                      # Lambda 入口模組 (handler)
│   └── requirements.txt                        # Lambda 執行所需的依賴清單
│
├── docker-compose/                         # 儲存 MLOps 系統的容器架構與服務部署設定檔
│   ├── mlops.yml                               # 定義 MLOps 容器架構
│   ├── dashboard.yml                           # 定義 Grafana 儀表板用容器架構
│   └── Dockerfile                              # Flask 主服務的建構說明
│
├── .env                                    # 預設環境變數
├── .gitignore                              # Git 忽略項目設定
└── requirements.txt                        # 專案依賴套件清單（本地執行使用）
```


## 功能模組

| 功能模組         | 對應容器 / 技術               | 說明 |
|------------------|-------------------------------|------|
| **Flask API**         | `crypto_proj_flask_main_container`     | 提供 `/run_data_update`、`/run_model_training` 等 RESTful 介面給 Airflow 調用 |
| **MLflow Tracking**   | `crypto_proj_mlflow_container`         | 追蹤模型實驗與參數，支援 artifact 儲存至 MinIO、metadata 儲存至 MySQL |
| **MinIO**             | `crypto_proj_minio_container`          | S3-compatible 儲存服務，存放模型 artifact、log 等 |
| **MySQL**             | `crypto_proj_mysql_container`          | 儲存資料原始記錄與處理後數據 |
| **Prometheus**        | `crypto_proj_prometheus_container`     | 擷取系統與模型訓練指標，進行監控與告警邏輯判斷 |
| **Alertmanager**      | `crypto_proj_alertmanager_container`   | 接收 Prometheus 告警，並可串接 LINE / Slack 通知 |
| **Pushgateway**       | `crypto_proj_pushgateway_container`    | 支援主動推送自定義指標供 Prometheus 擷取 |
| **S3 Exporter**       | `crypto_proj_s3_exporter_container`    | 抓取 BTC api 即時數據供 Prometheus 擷取 |
| **Grafana**           | `crypto_proj_grafana_container`        | 視覺化儀表板，從 Prometheus、MySql 擷取顯示模型訓練結果與 BTC 相關數據 |


## 啟動專案

### 1.複製專案並建立環境變數

```bash
git clone https://github.com/SamJim0729/btc_mlops_prediction_system.git
cd btc_mlops_prediction_system
cp .env.example .env
# 請填入 AWS、MinIO、MySQL 等連線資訊
```

### 2️.啟動mlops容器、dashboard容器

```bash
docker compose -f docker-compose/mlops.yml -p crypto_proj --env-file .env up -d
docker compose -f docker-compose/dashboard.yml -p crypto_proj --env-file .env up -d
```


## API 一覽
本系統透過 Flask 提供三個 API 端點，僅 run_feature_selection 開放手動觸發，其餘皆由 Airflow 定期排程執行（此 repo 未納入 DAG 配置檔）。

### 手動觸發（開發者可用於重訓模型）

| Method | Endpoint               | 說明 |
|--------|------------------------|------|
| POST   | /run_feature_selection | 執行特徵選擇流程，僅當模型效能下降或發生資料漂移時建議重新執行。會重新選變數並產出 JSON 特徵設定檔。 |

```bash
curl -X POST http://localhost:8081/run_feature_selection
```

### 自動執行（由 Airflow 定期排程）
下列 API 為系統內部使用，每月自動由 Airflow 呼叫，無需手動操作。可用於測試目的。

| Method | Endpoint             | 說明 |
|--------|----------------------|------|
| POST   | /run_data_update     | 擷取最新 BTC / 總經資料，寫入 MySQL。資料涵蓋現貨、期貨、技術指標與全球經濟數據等。 |
| POST   | /run_model_training  | 根據最新特徵設定檔訓練回歸模型，訓練結果送入 MLflow 並推送指標至 Prometheus。 |

```bash
curl -X POST http://localhost:8081/run_data_update
curl -X POST http://localhost:8081/run_model_training
```


## 監控與可視化

- Grafana 顯示 BTC 相關指標數據、模型預測結果、模型評估指標
- Prometheus 每月抓取最新訓練指標
- Alertmanager 超過門檻自動通知 (此專案整合 Slack 通知)


## 測試與部署建議

- **Airflow**：已規劃為 DAG 排程自動化資料更新與模型訓練


