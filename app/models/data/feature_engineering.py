import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List


def correlation_feature_selection(df: pd.DataFrame, target: str, threshold: float = 0.05, plot: bool = False) -> List[str]:
    """
    相關係數篩選變數

    Args:
        df (pd.DataFrame): 數據集
        target (str): 目標變數
        threshold (float): 相關係數閾值（絕對值大於此閾值的變數才保留）
        plot (bool): 是否繪製篩選結果

    Returns:
        List[str]: 選出的變數列表
    """
    correlations = df.corr()[target].abs().sort_values(ascending=False)
    selected_features = correlations[correlations > threshold].index.tolist()

    if plot:
        plt.figure(figsize=(12, 6))
        sns.heatmap(df.corr()[[target]].sort_values(by=target, ascending=False), 
                    annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title("🔍 相關係數變數篩選")
        plt.show()

    # print("✅ 相關係數 選出的變數:", selected_features)
    return selected_features

def xgboost_feature_selection(df: pd.DataFrame, target: str, top_k: int = 20, threshold: float = 0.001, plot: bool = False) -> List[str]:
    """
    XGBoost 特徵選擇 (使用梯度提升決策樹)

    Args:
        df (pd.DataFrame): 數據集
        target (str): 目標變數
        top_k (int): 選擇最重要的前 k 個特徵
        threshold (float): 重要性閾值
        plot (bool): 是否繪製變數重要性圖

    Returns:
        List[str]: 選出的變數列表
    """
    
    df = df.copy()
    X = df.drop(columns=[target])
    y = df[target]
    
    #一般模型先調 n-estimetor
    #subsample 有放回抽樣，樣本量大才調整，boostrap，樣本少，讓他正常抽樣放回
    #eta通常不調整，寄望剪枝調整
    
    model = xgb.XGBRegressor(
        booster="gbtree",
        n_estimators=100,
        max_depth=5,
        seed=42,  
        subsample=1.0,  
        random_state=42,  
    )
    model.fit(X, y)

    # 取得 XGBoost 特徵重要性
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    # 過濾掉影響力低於 `threshold` 的變數
    feature_importance = feature_importance[feature_importance["importance"] >= threshold]

    # 再從剩餘的變數中，選擇最重要的 `top_k`
    selected_features = feature_importance.head(top_k)["feature"].tolist()
    
    if plot:
        # 繪製變數重要性圖
        plt.figure(figsize=(10, 5))
        plt.barh(feature_importance.head(top_k)["feature"], feature_importance.head(top_k)["importance"])
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature Name")
        plt.title("🔍 XGBoost 重要變數")
        plt.gca().invert_yaxis()
        plt.show()

    # print(f"✅ XGBoost 選出的變數 (threshold={threshold}):", selected_features)
    return selected_features

def select_features(df: pd.DataFrame, target: str, drop_cols: List[str] = None, corr_threshold: float = 0.05, xgb_threshold: float = 0.001, plot: bool = False) -> List[str]:    
    """
    綜合特徵選擇：
    - 相關係數篩選
    - XGBoost 變數選擇

    Args:
        df (pd.DataFrame): 數據集
        target (str): 目標變數，shift但沒改變數名稱
        drop_cols (List[str]): 要移除的非特徵欄位
        corr_threshold (float): 相關係數閾值
        xgb_threshold (float): XGBoost 重要性閾值
        plot (bool): 是否顯示相關係數熱圖

    Returns:
        List[str]: 最終選出的變數列表
    """

    df = df.copy()

    # 2️⃣ 清理數據
    df.drop(columns=drop_cols, inplace=True, errors="ignore")  # 移除非特徵列
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 3️⃣ 相關係數篩選
    selected_features_corr = correlation_feature_selection(df, target, threshold=corr_threshold, plot=plot)

    # 5️⃣ XGBoost 選變數
    selected_features_xgb = xgboost_feature_selection(df, target, top_k=20, threshold=xgb_threshold, plot=plot)

    # 6️⃣ 最終變數
    final_selected_features = list(set(selected_features_corr) & set(selected_features_xgb))
    # print("✅ 最終篩選出的變數:", final_selected_features)

    return final_selected_features

if __name__=='__main__':
    pass


 