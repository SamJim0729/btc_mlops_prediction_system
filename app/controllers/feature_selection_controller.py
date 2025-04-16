import os
import sys

# === 路徑設定 ===
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# === 模組匯入 ===
from models.data.feature_selection_pipeline import feature_selection_pipeline


def run_feature_selection():
    """
    主控制流程：
    1. 執行特徵選擇
    2. 儲存特徵 JSON 至 configs/
    """
    feature_selection_pipeline(time_col='trade_date', target_col='spot_close_change', train_years=3, test_months=4)

if __name__ == "__main__":
    #假設今天:Y250321，test區間:3個月、train區間:3年
    #模型最終目的:用Y2502資料預測Y2503 (y:next_spot_close_change，***y shift 1個月***)
    #測試區間:Y2411~Y2501(X:Y2411、y:Y2412)~(X:Y2501、y:Y2502) #如果X設到Y2502，會沒有Y2503的y來驗證
    #訓練區間:Y2111~Y2410(X:Y2111、y:Y2212)~(X:Y2410、y:Y2411) 
    #實際需要資料區間:Y2010~Y2502 #Y2111在計算chang的lag時Y2011會需要Y2010算change
    run_feature_selection()


    



