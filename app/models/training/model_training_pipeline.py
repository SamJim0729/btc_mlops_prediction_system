import os
import sys
import time
import random
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from typing import List, Dict, Any
from dateutil.relativedelta import relativedelta

from sklearn.preprocessing import StandardScaler
import mlflow
from mlflow.models import infer_signature
from sklearn.ensemble import BaggingRegressor
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from models.data import data_loader
from models.data.feature_dataset_builder import fetch_big_table_from_db
from utils.docker_utils import get_container_ip
from models.training.Regressor import Regressor
from models.training import mlflow_handler
from monitoring import prometheus_handler

random.seed(42)
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = str(42)

def setup_training_env():
    load_dotenv()
    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("MINIO_ACCESS_KEY_ID")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("MINIO_SECRET_ACCESS_KEY")
    # os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL")
    minio_ip = get_container_ip("crypto_proj_minio_container")
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{minio_ip}:9000" # 避免 MLflow/Boto3 無法解析 MinIO container 名稱，動態設定 IP 確保連線成功
    prometheus_handler.clear_old_metrics(job='crypto_proj') #清除上月pushgateway紀錄

def train_and_evaluate(
                        target: str,
                        time_column: str,
                        model_list: List[str],
                        train_years: int = 3,
                        test_months: int = 3,
                        validation_method: str = "TimeSeriesSplit",
                        n_splits: int = 3,
                        use_bagging: bool = False,
                        bagging_n: int = 1,
                        bagging_fraction: float = 1.0,
                        optuna_trials: int = 50
                    ) -> Dict[str, Any]:
    
    """
    主流程：訓練預測模型並進行評估與 MLflow 記錄

    Args:
        target (str): 預測目標欄位（會 shift 成下一期）
        time_column (str): 時間欄位名稱（用於切分 train/test）
        model_list (List[str]): 使用的模型清單（如：['XGBoost']）
        train_years (int): 訓練資料年數（預設3年）
        test_months (int): 測試資料月數（預設3個月）
        validation_method (str): 驗證方法，支援 TimeSeriesSplit 或 cross-validation
        n_splits (int): 驗證資料切分數
        use_bagging (bool): 是否使用 Bagging 包裝最佳模型
        bagging_n (int): Bagging 的模型數量
        bagging_fraction (float): 每次抽樣比例（0~1）
        optuna_trials (int): Optuna 超參數搜尋次數

    Returns:
        Dict[str, Any]: 訓練與測試評估結果（RMSE、R2、MAPE等）
    """

    setup_training_env()

    # ✅ 初始化 mlflow
    mlflow_handler.init_mlflow("crypto_proj", tracking_uri="http://crypto_proj_mlflow_container:5000")

    # ✅ 載入特徵設定與組資料、切分訓練與測試資料、訓練資料處理流程看data_pipeline
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'configs')
    selected_features_dict = data_loader.load_feature_config(config_dir=config_dir, prefix="selected_features_")
    df_all = fetch_big_table_from_db(time_col=selected_features_dict['time_col'], train_years=train_years, test_months=test_months, delay_one_month=True, lag_features=True)
    df_selected_features = data_loader.build_feature_dataframe_shift_and_rename_y(df_all=df_all, feature_config=selected_features_dict, target=target)
    df_train, df_test, train_interval, test_interval = data_loader.split_train_test(df=df_selected_features, time_col=selected_features_dict['time_col'], train_years=train_years, test_months=test_months)

    # ✅ datadrift 偵測（近6個月 vs 當月）、rolling_window
    drift_report, drift_results = run_drift_analysis(df_train.iloc[-6:, :], df_test)

    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run():
        start_time = time.time()
        
        # ✅ MLflow 基礎記錄
        mlflow_handler.log_time_range(train_interval, test_interval, train_years, test_months)
        mlflow_handler.log_dataset_info(df_train, df_test)
        mlflow_handler.log_selected_features(df_selected_features.columns.tolist())
        mlflow_handler.log_data_drift_report(drift_report, drift_results)
        mlflow_handler.log_model_params({
            "target_column": f"{target}_next",
            "model_list": model_list,
            "validation_method": validation_method,
            "n_splits": n_splits,
            "bagging_n": bagging_n,
            "bagging_fraction": bagging_fraction,
            "use_bagging": use_bagging,
            "optuna_trials": optuna_trials
        })

    # 4️⃣ 訓練模型、5️⃣ 評估模型
        model = Regressor(
            df_train=df_train, 
            df_test=df_test,
            target_col=f"{target}_next", 
            drop_cols=[time_column], 
            scaler=StandardScaler(),
            model_list=model_list,
            validation_method=validation_method,
            n_splits=n_splits,
            bagging_n=bagging_n,
            bagging_fraction=bagging_fraction
        )
        df_train_X, df_train_y, df_test_X, df_test_y = model.prepare_data(scale_data=True)
        model.train_best_model(df_train_X, df_train_y, trials=optuna_trials, use_bagging=use_bagging)
        results = model.evaluate_model(df_train_X, df_train_y, df_test_X, df_test_y)

        # ✅ Data Drift - Top 5 特徵監控推送至 Prometheus
        top_model_features = extract_top_drift_features(model.best_model, df_train_X, top_n=5)
        drift_details = drift_results["metrics"][1]["result"]["drift_by_columns"]
        for col_name, info in drift_details.items():
            if col_name in top_model_features:
                # 根據實際 drift 測試類型自動命名，例如 "ks"、"psi"（或從 info["stattest_name"] 抓）
                score = round(info.get("drift_score", 0.0), 3)
                prometheus_handler.update_data_drift_metrics(feature_name=col_name, score=score)
        
        # ✅ 模型效能指標推送至 Prometheus
        prometheus_handler.update_ml_model_results(r2=results["test"]["r2"], smape=results["test"]["smape"], rmse=results["test"]["rmse"])
        prometheus_handler.push_metrics(job='crypto_proj')

        # ✅ 儲存模型與 Signature
        input_example = df_train_X.iloc[:1].to_dict(orient="records")[0]
        signature = infer_signature(df_train_X, model.best_model.predict(df_train_X[:5]))
        mlflow_handler.log_model_to_mlflow(
            model=model.best_model,
            model_name="model",
            signature=signature,
            input_example=input_example
        )

        # ✅ 紀錄預測結果與參數
        mlflow_handler.log_evaluation_metrics(results)
        mlflow_handler.log_best_params(model.best_params)
        mlflow_handler.log_predictions([float(value) for value in results["test"]["y_true"]], [float(value) for value in results["test"]["y_pred"]])
        mlflow_handler.log_training_time(start_time)

        #設定預測區間區間
        all_features = df_selected_features.columns.tolist()
        all_features.remove(f"{target}_next")
        prediction_data = df_all[all_features].iloc[[-1]]
        prediction_month = (prediction_data[time_column].iloc[-1] + relativedelta(months=1)).strftime("%Y-%m") #3月資料預測4月
        predicted_value = model.predict_from_raw(df_input=prediction_data, scale_data=True)
        mlflow_handler.log_model_final_prediction(predicted_value, prediction_month)
        



        #先去workkbench，透過 run uuid > key，最新找出對應所有值

        #抓取最新X時間(上月X預測當月Y)
        #資料庫抓取資料
        #import predict log
        #這邊要加mlhandler 紀錄預測結果
        
    # return results

def extract_top_drift_features(model, df_X, top_n=5) -> List[str]:
    """
    從模型中提取 top_n 的重要特徵名稱
    支援單一模型 (需具備 .feature_importances_) 或 BaggingRegressor (會取子模型平均)
    """
    # 若是 BaggingRegressor，對每個子模型取 feature_importance 再平均
    if isinstance(model, BaggingRegressor):
        if hasattr(model.estimators_[0], 'feature_importances_'):
            all_importances = [
                est.feature_importances_ for est in model.estimators_
            ]
            mean_importance = pd.Series(
                sum(all_importances) / len(all_importances),
                index=df_X.columns
            )
            mean_importance.sort_values(ascending=False).head(top_n).index.tolist()
            return mean_importance.sort_values(ascending=False).head(top_n).index.tolist()
        else:
            raise ValueError("Bagging 的 base_estimator 不支援 feature_importances_")
    
    # 若是一般模型，直接取用 .feature_importances_
    elif hasattr(model, 'feature_importances_'):
        importance = pd.Series(model.feature_importances_, index=df_X.columns)
        return importance.sort_values(ascending=False).head(top_n).index.tolist()

    else:
        raise ValueError("該模型不支援 feature_importances_，無法提取特徵排序")

def run_drift_analysis(reference_df, current_df, save_html_path="evidently_drift_report.html"):
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference_df, current_data=current_df)
    drift_report.save_html(save_html_path)
    return drift_report, drift_report.as_dict()

if __name__ == "__main__":
    train_and_evaluate(
        target='spot_close_change',
        time_column='trade_date',
        model_list=['XGBoost'],
        train_years=3,
        test_months=3,
        n_splits=3,
        use_bagging=True,
        bagging_n=10,
        bagging_fraction=1.0,
        optuna_trials=100
    )
    #假設今天:Y250321
    #模型最終目的:用Y2502資料預測Y2503 (y:next_spot_close_change，***y shift 1個月***)
    #測試區間:Y2411~Y2501(X:Y2411、y:Y2412)~(X:Y2501、y:Y2502)
    #訓練區間:Y2111~Y2410(X:Y2111、y:Y2212)~(X:Y2410、y:Y2411)
    #實際需要資料區間:Y2010~Y2502

    # nohup mlflow server \
    # --backend-store-uri mysql+pymysql://root:root@crypto_proj_mysql_container:3306/mlflow_metadata \
    # --default-artifact-root s3://mlflow-artifacts \
    # --host 0.0.0.0 \
    # --port 5000 > mlflow.log 2>&1 &