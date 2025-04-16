import pandas as pd
import numpy as np
import optuna
import logging

from sklearn.ensemble import BaggingRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import xgboost as xgb

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from optuna.exceptions import TrialPruned

import os
import sys
import random 
from typing import Optional, List, Tuple, Dict, Any

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

random.seed(42)
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = str(42)

# 設定日誌
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Regressor:
    def __init__(self, 
                 df_train:pd.DataFrame, 
                 df_test:pd.DataFrame, 
                 target_col:str, 
                 drop_cols:Optional[List[str]]=None,
                 scaler:Any = StandardScaler(), 
                 model_list:Optional[List[str]]=None, 
                 validation_method:str="TimeSeriesSplit",
                 bagging_n:int=10, 
                 bagging_fraction:float=1.0, 
                 n_splits:int=3):
        """
        初始化回歸器物件，設定基本參數與模型訓練設定。

        Args:
        - df_train: 訓練資料（需包含目標欄位）
        - df_test: 測試資料（需包含目標欄位）
        - target_col: 預測目標欄位名稱
        - drop_cols: 不參與訓練的欄位（如識別碼、時間戳等）
        - scaler: 資料標準化方法（預設使用 StandardScaler）
        - model_list: 可選用的模型名稱清單
        - validation_method: 驗證方式（預設為 TimeSeriesSplit）
        - bagging_n: Bagging 的模型數量
        - bagging_fraction: 每次 Bagging 抽樣的資料比例
        - n_splits: 用於交叉驗證的資料切分數量（僅適用於 TimeSeriesSplit）
        """
        self.SEED = 42
        np.random.seed(self.SEED)
        random.seed(self.SEED)
        os.environ["PYTHONHASHSEED"] = str(self.SEED)
        
        self.df_train = df_train
        self.df_test = df_test
        self.target_col = target_col
        self.drop_cols = drop_cols or []

        self.model_list = ["SVR", "XGBoost", "CatBoost", "LightGBM"] if model_list is None else model_list
        self.validation_method = validation_method
        self.bagging_n = bagging_n
        self.bagging_fraction = bagging_fraction
        self.n_splits = n_splits
        self.best_model = None
        self.best_params = None
        self.scaler = scaler

    def prepare_data(self, scale_data: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        準備訓練與測試資料，包含欄位過濾與標準化處理。

        Args：
        - scale_data: 是否對特徵進行標準化（True 表示使用訓練資料 fit，再對 train/test transform）

        Returns:
        - df_train_X: 處理後的訓練特徵
        - df_train_y: 訓練目標變數
        - df_test_X: 處理後的測試特徵
        - df_test_y: 測試目標變數
        """

        df_train_X = self.df_train.drop(columns=[self.target_col] + self.drop_cols)
        df_train_y = self.df_train[self.target_col]
        df_test_X = self.df_test.drop(columns=[self.target_col] + self.drop_cols)
        df_test_y = self.df_test[self.target_col]
        
        if scale_data:
            self.scaler.fit(df_train_X)
            df_train_X = pd.DataFrame(self.scaler.transform(df_train_X), columns=df_train_X.columns)
            df_test_X = pd.DataFrame(self.scaler.transform(df_test_X), columns=df_test_X.columns)

        return df_train_X, df_train_y, df_test_X, df_test_y
    
    def train_best_model(self, X_train:pd.DataFrame, y_train:pd.Series, trials:int=50, use_bagging:bool=True) -> None:
        """
        使用 Optuna 進行自動化超參數搜尋，並訓練出最佳模型，可選擇是否加上 Bagging。

        Args：
        - X_train: 訓練特徵資料
        - y_train: 訓練目標資料
        - trials: Optuna 搜尋超參數的試驗次數
        - use_bagging: 是否使用 Bagging 包裝最佳模型（預設為 True）
        """
        logging.info("開始 Optuna 超參數調整...")

        # 設定 Optuna logging 層級
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # optuna > n_startup_trials會決定探索階段嘗試幾次trail的隨機參數，沒設的話是總執行(n_trails次數的10%)
        # n_trails是總執行次數 = 探索階段(隨機參數) trails + TPE策略優化 trails次數
        # n_warmup_steps會決定前n次的trails有沒有剪枝
        #Optuna 預設會用 10% 的 n_trials 來探索(隨機參數數量)
        #trial.should_prune()有設此參數才會提前終止較差的trail
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),  # 先試 5 次，再開始剪枝
            sampler=optuna.samplers.TPESampler(seed=self.SEED)
        )

        #study.optimize()執行搜索最佳參數
        #Optuna 內部的 study.optimize() 會 自動產生 trial物件，並呼叫 lambda 函數，把 trial 當作參數傳入
        study.optimize(lambda trial: self._objective(trial, X_train, y_train), n_trials=trials, n_jobs=-1)

        best_trial = study.best_trial
        self.best_params = best_trial.params
        model_name = self.best_params.pop("model")

        print(f"Optuna 最佳模型: {model_name}, 參數: {self.best_params}")
        logging.info(f"Optuna 最佳模型: {model_name}, 參數: {self.best_params}")

        base_model = self._get_model(model_name, self.best_params)
        
        if use_bagging:
            self.best_model = self._get_bagging_model(base_model)
        else:
            self.best_model = base_model

        self.best_model.fit(X_train, y_train)

        logging.info("最佳模型訓練完成！")

    def evaluate_model(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        評估訓練後模型的效能，分別針對訓練集與測試集計算常用評估指標。

        Returns：
        包含訓練與測試資料的：
        - RMSE（均方根誤差）
        - R²（決定係數）
        - SMAPE（對稱平均絕對百分比誤差）
        - y_true: 真實值
        - y_pred: 預測值

        這些資訊可作為模型表現比較與調參依據。
        """

        if self.best_model is None:
            raise ValueError("請先訓練模型！")

        # 計算訓練集預測結果
        y_train_pred = self.best_model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        train_smape = np.mean(2 * np.abs(y_train - y_train_pred) / (np.abs(y_train) + np.abs(y_train_pred) + 1e-10)) * 100
        # 計算測試集預測結果
        y_test_pred = self.best_model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        test_smape = np.mean(2 * np.abs(y_test - y_test_pred) / (np.abs(y_test) + np.abs(y_test_pred) + 1e-10)) * 100

        # 記錄日誌
        logging.info(f"訓練集 RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}, SMAPE: {train_smape:.2f}%")
        logging.info(f"測試集 RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}, SMAPE: {test_smape:.2f}%")

        return {
            "train": {
                "rmse": train_rmse,
                "r2": train_r2,
                "smape": train_smape,
                "y_true": list(y_train),
                "y_pred": list(y_train_pred)
            },
            "test": {
                "rmse": test_rmse,
                "r2": test_r2,
                "smape": test_smape,
                "y_true": list(y_test),
                "y_pred": list(y_test_pred)
            }
        }

    def predict(self, X):
        """
        針對新資料進行預測。

        Args：
        - X: 欲預測的特徵資料

        Returns：
        - 模型預測結果（array）
        """

        if self.best_model is None:
            raise ValueError("請先訓練模型！")
        return self.best_model.predict(X)
    
    def _suggest_params(self, trial: optuna.trial.Trial, model_name: str) -> Dict[str, Any]:
        """
        根據選擇的模型類型，自動提供對應的超參數搜尋空間給 Optuna 使用。

        Args：
        - trial: Optuna 的 trial 實例
        - model_name: 模型名稱（SVR、XGBoost、CatBoost、LightGBM）

        Returns：
        - 超參數字典（dict）供模型初始化使用
        """
        
        if model_name == "XGBoost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            }
        
        elif model_name == "CatBoost":
            return {
                "iterations": trial.suggest_int("iterations", 50, 500, step=50),
                "depth": trial.suggest_int("depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),  # L2 正則化
                "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"])  # Bagging 類型
            }
        
        elif model_name == "LightGBM":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),  # 每棵樹的葉子數
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),  # 特徵抽樣比例
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),  # Bagging 抽樣比例
                "lambda_l1": trial.suggest_float("lambda_l1", 0, 10),  # L1 正則化
                "lambda_l2": trial.suggest_float("lambda_l2", 0, 10)  # L2 正則化
            }

    def _get_model(self, model_name:str, params:Dict[str, Any]) -> Any:
        """
        根據模型名稱與指定參數回傳對應的模型實例。

        Args：
        - model_name: 模型名稱字串
        - params: 經過 Optuna 搜尋出的最佳參數（不含 model_name）

        Returns：
        - 已初始化的模型物件
        """
        if model_name == "XGBoost":
            return xgb.XGBRegressor(**params, random_state=self.SEED)
        elif model_name == "CatBoost":
            return CatBoostRegressor(**params, verbose=0, random_seed=self.SEED)
        elif model_name == "LightGBM":
            return LGBMRegressor(**params, random_seed=self.SEED)
        else:
            raise ValueError(f"未支援的模型類型: {model_name}")

    def _get_bagging_model(self, base_model: Any) -> Any:
        """
        使用 sklearn 的 BaggingRegressor 對指定模型進行集成包裝。

        Args：
        - base_model: 單一回歸模型實例

        Returns：
        - BaggingRegressor 模型實例（若 n=1 則回傳原模型）
        """

        if self.bagging_n == 1:
            return base_model

        return BaggingRegressor(
            estimator=base_model,
            n_estimators=self.bagging_n,
            max_samples=self.bagging_fraction,
            random_state=self.SEED,
            n_jobs=-1
        )

    def _objective(self, trial: optuna.trial.Trial, X_train: pd.DataFrame, y_train: pd.Series) -> float:
        """
        Optuna 最核心的目標函數，決定每一次 trial 試驗的流程。

        流程：
        1. 根據 trial 隨機挑選模型類型
        2. 建立該模型的超參數組合
        3. 使用交叉驗證計算負 RMSE（越小越好）
        4. 回傳交叉驗證結果給 Optuna 進行參數更新

        參數：
        - trial: Optuna trial 實例
        - X_train: 訓練特徵
        - y_train: 訓練目標

        回傳：
        - 平均 RMSE（供 Optuna 用來選擇最佳參數）
        """
        #每次trail都提供不同的model，隨機挑選，前幾次是隨機，後面optuna會判斷哪些model比較好
        np.random.seed(self.SEED)  # <--- 每次 trial 內部也重新設置種子
        random.seed(self.SEED)
        model_name = trial.suggest_categorical("model", self.model_list)
        
        #前面隨機挑選參數
        params = self._suggest_params(trial, model_name)

        #建立model，帶入本次參數
        model = self._get_model(model_name, params)

        #self.validation_method決定進行哪種方式的CV
        if self.validation_method == "TimeSeriesSplit":
            tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=1)
            scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=-1)
        else:
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)

        trial.report(-np.mean(scores), step=0)
        
        #這邊有點沒意義，因為cv把每次結果計算完才回傳，所以不會提前剪枝
        #除非像深度學習每個epoch記錄一次，或是用for計算cv在迴圈中紀錄，這樣should_prune()才有意義
        if trial.should_prune():
            raise TrialPruned()

        return -np.mean(scores)
    
    def predict_from_raw(self, df_input: pd.DataFrame, scale_data: bool = True) -> np.ndarray:
        """
        對尚未經處理的新資料（df_input）進行預測。

        Args:
        - df_input: 包含所有欄位的新資料（可能包含 drop_cols 與 target_col）
        - scale_data: 是否執行標準化（預設會根據訓練資料的 scaler 執行 transform）

        Returns:
        - 預測結果（float）
        """
        if self.best_model is None:
            raise ValueError("請先訓練模型！")

        # 保留欄位過濾與標準化邏輯
        df_features = df_input.drop(columns=self.drop_cols + [self.target_col], errors='ignore')

        if scale_data:
            df_features = pd.DataFrame(
                self.scaler.transform(df_features),
                columns=df_features.columns
            )

        predictions = self.best_model.predict(df_features)
        return float(predictions)



if __name__ == '__main__':
    pass

