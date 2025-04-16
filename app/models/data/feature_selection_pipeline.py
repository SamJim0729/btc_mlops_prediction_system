import os
import sys
import json
from datetime import datetime
from typing import List, Dict


# === 路徑設定 ===
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

# === 模組匯入 ===
from feature_engineering import select_features
from feature_dataset_builder import fetch_big_table_from_db
from data_preprocessor import DataPreprocessor
from utils.date_utils import get_time_intervals


def feature_selection_pipeline(time_col: str = 'trade_date', 
                               target_col: str = 'spot_close_change', 
                               train_years: int = 3, 
                               test_months: int = 3) -> Dict[str, List[str]]:
    """
    執行特徵選擇 Pipeline：
    1. 初步變數篩選 (XGBoost + 相關性)
    2. 加入滯後變數 (Lag Features)
    3. 最終變數篩選 (XGBoost)
    
    Returns：
    - X_cols: 選出的特徵
    - y_col: 目標變數
    - time_col: 時間變數
    """
    
    # 1️.讀取大表(大表回傳範圍會預留shift跟lag feature所需的資料區間，所以需再調整)
    df_big_table = fetch_big_table_from_db(time_col=time_col, train_years=train_years, test_months=test_months, delay_one_month=True, lag_features=False)
    df_big_table[target_col] = df_big_table[target_col].shift(-1)
    df_big_table.dropna(inplace=True) 

    train_interval, test_interval = get_time_intervals(train_value=train_years, train_unit='year', test_value=test_months, test_unit='month', delay_one_month=True) 
    df_for_first_selection = df_big_table[(df_big_table[time_col] >= train_interval['start']) & (df_big_table[time_col] <= test_interval['end'])]

   #2️.初步變數篩選(第一次資料範圍要正確因為不需要lag feature)
    initial_selected_features = select_features(
        df=df_for_first_selection,
        target=target_col,
        drop_cols=[time_col],
        corr_threshold=0.05,
        xgb_threshold=0.001
    )

    if not initial_selected_features:
        raise ValueError("❌ 初步變數篩選失敗，請確認資料或調整閾值")
    
    # 3️.建立 DataFrame，保留選出的特徵 + 目標變數 + 時間變數
    df_filtered = df_big_table[initial_selected_features + [time_col, target_col]]

    # # 4️.加入滯後變數
    df_with_lag_features = DataPreprocessor.add_lagged_features(df=df_filtered, time_col=time_col,
                                                     target_cols=initial_selected_features, lags=[6, 12])
    df_with_lag_features.dropna(inplace=True) #區間正確符合訓練測試範圍
    
    # 5️.最終變數篩選
    final_features = select_features(
        df=df_with_lag_features,
        target=target_col,
        drop_cols=[time_col],
        corr_threshold=0.05,
        xgb_threshold=0.001,
        plot=False
    )

    if not final_features:
        raise ValueError("❌ 最終變數篩選失敗，請確認資料或調整閾值")
    
    print("✅ 相關係數、XGboost最終篩選後的變數:", final_features)

    result = {
                'X_cols':final_features,
                'y_col':target_col,
                'time_col':time_col
            }

    # === 設定儲存路徑 ===
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'configs')
    os.makedirs(save_dir, exist_ok=True)

    today_str = datetime.today().strftime('%Y-%m-%d')
    filename = f"selected_features_{today_str}.json"
    save_path = os.path.join(save_dir, filename)

    # # === 儲存 JSON ===
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    
    print(f"✅ 變數篩選結果已儲存至: {save_path}")
    
    return result

if __name__=="__main__":
    feature_selection_pipeline()