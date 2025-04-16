import os
import json
import pandas as pd
from typing import Tuple, Dict
from utils.file_utils import get_latest_feature_config
from utils.date_utils import get_time_intervals

def load_feature_config(config_dir: str, prefix: str = "selected_features_") -> Dict:
    """讀取最新的特徵篩選設定檔"""
    path = get_latest_feature_config(config_dir=config_dir, prefix=prefix)
    with open(path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def build_feature_dataframe_shift_and_rename_y(df_all:pd.DataFrame, feature_config:Dict, target:str, dropna:bool=True) -> pd.DataFrame:
    """根據特徵設定擷取欄位與處理 y shift"""
    X = feature_config['X_cols']
    y = feature_config['y_col']
    time_col = feature_config['time_col']

    df = df_all[X + [y, time_col]].copy()
    df[target] = df[target].shift(-1)
    if dropna:
        df.dropna(inplace=True)
    df.rename(columns={target: f"{target}_next"}, inplace=True)
    return df

def split_train_test(df: pd.DataFrame, time_col: str, train_years: int, test_months: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict, Dict]:
    """根據時間欄位切分訓練與測試資料"""
    train_interval, test_interval = get_time_intervals(train_value=train_years, train_unit="year", test_value=test_months, test_unit="month", delay_one_month=True)
    df_train = df[(df[time_col] >= train_interval['start']) & (df[time_col] <= train_interval['end'])]
    df_test = df[(df[time_col] >= test_interval['start']) & (df[time_col] <= test_interval['end'])]
    return df_train, df_test, train_interval, test_interval