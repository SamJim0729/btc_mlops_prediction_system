import mlflow
import json
import time
from mlflow.models import infer_signature

def init_mlflow(experiment_name: str, tracking_uri: str = "http://localhost:5000"):
    """初始化 MLflow 設定與實驗名稱"""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"✅ MLflow 初始化完成 (Experiment: {experiment_name})")

def log_time_range(train_interval: dict, test_interval: dict, train_years: int, test_months: int):
    mlflow.log_params({
        "train_start": train_interval['start'],
        "train_end": train_interval['end'],
        "test_start": test_interval['start'],
        "test_end": test_interval['end'],
        "train_years": train_years,
        "test_months": test_months
    })

def log_dataset_info(df_train, df_test):
    mlflow.log_params({
        "train_sample_size": len(df_train),
        "test_sample_size": len(df_test),
        "train_X_shape": df_train.shape,
        "test_X_shape": df_test.shape
    })

def log_selected_features(columns: list):
    mlflow.log_dict({"selected_features": columns}, "selected_features.json")

def log_model_params(params: dict):
    # 注意：mlflow.log_params 的 value 只能是 str or float or int
    clean_params = {k: (",".join(v) if isinstance(v, list) else v) for k, v in params.items()}
    mlflow.log_params(clean_params)

def log_data_drift_report(drift_report, drift_results):
    drift_report.save_html("evidently_drift_report.html")
    mlflow.log_artifact("evidently_drift_report.html")
    mlflow.log_dict(drift_results, "evidently_drift_results.json")

    # Dataset 是否整體發生 drift（metrics[0] 為整體資料集漂移判定）
    drift_detected = drift_results["metrics"][0]["result"]["dataset_drift"]
    mlflow.log_metric("data_drift_detected", int(drift_detected))

def log_evaluation_metrics(results: dict):
    mlflow.log_metrics({
        "train_rmse": results["train"]["rmse"],
        "train_r2": results["train"]["r2"],
        "train_smape": results["train"]["smape"],
        "test_rmse": results["test"]["rmse"],
        "test_r2": results["test"]["r2"],
        "test_smape": results["test"]["smape"]
    })

def log_training_time(start_time: float):
    mlflow.log_metric("training_time", round(time.time() - start_time, 2))

def log_best_params(params: dict):
    mlflow.log_dict(params, "best_params.json")

def log_model_to_mlflow(model, model_name="model", signature=None, input_example=None):
    """
    根據模型類型選擇對應的 mlflow log_model 方法
    
    Args:
        model: 訓練好的模型實例
        model_name: 儲存名稱（artifact path）
        signature: 輸入輸出資料結構
        input_example: 範例輸入資料
    """
    import xgboost as xgb
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
    from sklearn.ensemble import BaggingRegressor

    if isinstance(model, xgb.XGBRegressor):
        mlflow.xgboost.log_model(model, artifact_path=model_name,
                                 signature=signature, input_example=input_example)
    elif isinstance(model, LGBMRegressor):
        import mlflow.lightgbm
        mlflow.lightgbm.log_model(model, artifact_path=model_name,
                                  signature=signature, input_example=input_example)
    elif isinstance(model, CatBoostRegressor):
        import mlflow.catboost
        mlflow.catboost.log_model(model, artifact_path=model_name,
                                  signature=signature, input_example=input_example)
    elif isinstance(model, BaggingRegressor):
        import mlflow.sklearn
        mlflow.sklearn.log_model(model, artifact_path=model_name,
                                 signature=signature, input_example=input_example)
    else:
        raise ValueError("不支援的模型類型")

def log_predictions(y_true, y_pred):
    """儲存預測結果 (JSON 格式)"""
    mlflow.log_text(json.dumps([float(y) for y in y_true]), "y_true.json")
    mlflow.log_text(json.dumps([float(y) for y in y_pred]), "y_pred.json")

def log_training_metrics(train_metrics: dict, test_metrics: dict):
    """記錄訓練與測試集的 RMSE, R2, MAPE"""
    mlflow.log_metrics({
        "train_rmse": train_metrics["rmse"],
        "train_r2": train_metrics["r2"],
        "train_smape": train_metrics["smape"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "test_smape": test_metrics["smape"]
    })

def log_model_signature(df_X, model):
    """建立並回傳模型簽名與範例輸入"""
    input_example = df_X.iloc[:1].to_dict(orient="records")[0]
    signature = infer_signature(df_X, model.predict(df_X[:5]))
    return signature, input_example

def log_model_final_prediction(predicted_change: float, prediction_month: str):
    """
    Log the model's final prediction result for a future month (no ground truth available yet).
    
    Parameters:
        predicted_change (float): The predicted value of next month's change.
        prediction_month (str): The month being predicted, in format YYYY-MM.
    """
    mlflow.log_metric("predicted_next_month_change", predicted_change)
    mlflow.set_tag("prediction_month", prediction_month)
