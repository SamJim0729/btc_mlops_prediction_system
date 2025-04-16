import os
import sys
import pandas as pd
from dotenv import load_dotenv
from dateutil.relativedelta import relativedelta
from datetime import timedelta
from typing import Dict, Union

# === 路徑設定 ===
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
# === 模組匯入 ===
from utils.date_utils import get_time_intervals
from models.database.mysql_handler import Mysql
from models.data.data_preprocessor import DataPreprocessor
from models.data.feature_engineering import select_features

def get_db_connection() -> Mysql:
    """ 初始化 MySQL 連線 """
    load_dotenv()
    return Mysql(
        host=os.getenv("SQL_HOST"),
        user=os.getenv("SQL_USER"),
        password=os.getenv("SQL_PASSWORD"),
        database=os.getenv("SQL_DB")
    )

# 因為lag_features，跟Y shift關係，返回資料區間會比較大
def fetch_big_table_from_db(time_col:str='trade_date', train_years:int=3, test_months:int=3, delay_one_month:bool=True, lag_features:bool=True) -> pd.DataFrame:
    """
    計算 **訓練** & **測試區間** 的時間範圍
    - 訓練資料範圍: `train_years` 年
    - 測試資料範圍: `test_months` 個月
    - `delay_one_month=True` 避免當月數據影響
    """
    # 1. 載入環境變數並建立 MySQL 連線
    db = get_db_connection()

     #固定用最後2 months資料驗證，訓練資料固定固定5 years(測試資料再往前)，且當月預測資料結果無法驗證，故，故delay_one_month=True，讓所有資料往前一個月
    train_interval, test_interval = get_time_intervals(train_value=train_years, train_unit="year", test_value=test_months, test_unit="month", delay_one_month=delay_one_month)

    # 2️⃣ 計算特徵選擇的時間範圍 (考慮滯後變數)
    #因為變數中有lag_features(12)所以選變數時要多往前一年的資料，且Y會shift所以資料選要多加一個月(每個月預測未來1個月數據)
    #計算change變數假設start是2021-01-01，但change匯市Nan，所以需要往前一個月讓2021-01-01change不為0
    start = (pd.to_datetime(train_interval['start'])-relativedelta(years=1, months=1)).strftime("%Y-%m-%d")
    end = ((pd.to_datetime(test_interval['end'])).replace(day=1)+relativedelta(months=2)-timedelta(days=1)).strftime("%Y-%m-%d")


    # # 3. 讀取資料表
    query = f"SELECT * FROM btc_price_history WHERE LEFT({time_col}, 7)>='{start}' AND LEFT({time_col}, 7)<='{end}'"
    btc_df = pd.DataFrame(db.execute_query(query))
    macro_df = pd.DataFrame(db.execute_query(query.replace("btc_price_history", "macro_market_metrics")))

    # # 4. 數據合併與聚合
    df_merged = DataPreprocessor.merge_ffill_tables(price_df=btc_df, metrics_df=macro_df)
    df_processed = DataPreprocessor.aggregate_data(df=df_merged, time_col=time_col, freq='ME')

    # # 5. 特徵工程：高低價差、溢價率
    df_processed['spot_high_low_range'] = df_processed['spot_high'] - df_processed['spot_low']
    df_processed['futures_high_low_range'] = df_processed['futures_high'] - df_processed['futures_low']
    df_processed = DataPreprocessor.calculate_btc_premium(df_processed)

    # # 6. 資料整理
    df_processed.set_index(time_col, inplace=True)
    df_processed = df_processed.astype(float)
    df_processed.reset_index(inplace=True)

    # # 7. 月變化率計算
    transform_cols = df_processed.columns.drop(time_col)
    df_processed = DataPreprocessor.calculate_data_change(df=df_processed, time_col=time_col, target_cols=transform_cols)

    if lag_features:
        transform_cols = df_processed.columns.drop(time_col)
        df_processed = DataPreprocessor.add_lagged_features(df=df_processed, time_col=time_col,
                                                        target_cols=transform_cols, lags=[6, 12])
        df_processed.dropna(inplace=True)
    
    return df_processed

if __name__ == '__main__':
    df = fetch_big_table_from_db(time_col='trade_date', train_years=1, test_months=1, delay_one_month=True, lag_features=True)
    
