import pandas as pd
from typing import List


class DataPreprocessor:
    """
    數據預處理類別，包含數據合併、填充、滯後變數、特徵計算等功能。
    """

    @staticmethod
    def merge_ffill_tables(price_df: pd.DataFrame, metrics_df: pd.DataFrame, time_col: str = "trade_date") -> pd.DataFrame:
        """
        合併數據表並執行前向填充（Forward Fill）。

        Args:
            price_df (pd.DataFrame): 包含價格數據的 DataFrame。
            metrics_df (pd.DataFrame): 包含總經數據的 DataFrame。
            time_col (str, optional): 兩張表合併的時間欄位名稱，預設為 'trade_date'。

        Returns:
            pd.DataFrame: 合併並前向填充後的 DataFrame。
        """
        return price_df.merge(metrics_df, on=time_col, how="left").ffill().sort_values(by=time_col)
    
    #要注意資料是年、月、日單位，lags是資料之間相差筆數
    @staticmethod
    def add_lagged_features(df: pd.DataFrame, time_col: str, target_cols: List[str], lags: List[int] = [3, 6, 9, 12]) -> pd.DataFrame:
        """
        為指定變數添加滯後特徵。

        Args:
            df (pd.DataFrame): 原始數據集。
            time_col (str): 時間欄位名稱。
            target_cols (List[str]): 需要計算滯後變數的特徵名稱列表。
            lags (List[int], optional): 滯後時間步長，例如 [3, 6, 9, 12]。預設值為這些月份。

        Returns:
            pd.DataFrame: 包含滯後變數的新數據集。
        """
        df = df.copy().sort_values(by=time_col)
        for col in target_cols: 
            for lag in lags:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)
            
        return df
    
    @staticmethod
    def aggregate_data(df: pd.DataFrame, time_col: str = 'trade_date', freq: str = "ME") -> pd.DataFrame:
        """
        依指定時間頻率對數據進行聚合。

        Args:
            df (pd.DataFrame): 原始數據集。
            time_col (str): 時間欄位名稱，預設為 'trade_date'。
            freq (str, optional): 聚合頻率，'M' 表示按月，'Y' 表示按年。預設為 'M'。

        Returns:
            pd.DataFrame: 依時間聚合後的 DataFrame。
        """
        if freq not in ["ME", "YE"]:
            raise ValueError("freq 參數必須是 'M'（按月）或 'Y'（按年）")
        
        df[time_col] = pd.to_datetime(df[time_col]) # 確保時間欄位為索引
        df = df.set_index(time_col)

        # 轉換所有數據為 float
        df = df.astype(float)

        # 按時間頻率重新取樣，對數值欄位取平均
        df_agg = df.resample(freq).agg("mean").reset_index()

        return df_agg


    @staticmethod
    def calculate_btc_premium(df: pd.DataFrame, time_col: str = "trade_date") -> pd.DataFrame:
        """
        計算 BTC 溢價率 (Premium Rate)。

        公式: (期貨收盤價 - 現貨收盤價) / 現貨收盤價 * 100

        Args:
            df (pd.DataFrame): 包含現貨與期貨價格的 DataFrame。
            time_col (str, optional): 時間欄位名稱，預設為 'trade_date'。

        Returns:
            pd.DataFrame: 添加 `premium_rate` 欄位的 DataFrame。
        """
        required_cols = ["spot_close", "futures_close"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"❌ DataFrame 必須包含 {required_cols} 欄位！")

        df[time_col] = pd.to_datetime(df[time_col])  # 確保日期格式
        df = df.set_index(time_col)

        df["premium_rate"] = ((df["futures_close"] - df["spot_close"]) / df["spot_close"]) * 100

        return df.reset_index()

    @staticmethod
    def calculate_data_change(df: pd.DataFrame, time_col: str, target_cols: List[str]) -> pd.DataFrame:
        """
        計算指定變數的變化率 (%)

        公式: (當期值 - 前期值) / 前期值 * 100

        Args:
            df (pd.DataFrame): 原始數據集。
            time_col (str): 時間欄位名稱。
            target_cols (List[str]): 需要計算變化率的特徵名稱列表。

        Returns:
            pd.DataFrame: 添加變化率欄位的新 DataFrame。
        """
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])  # 確保日期格式
        df = df.set_index(time_col)

        # 計算變化率 (%)
        df_pct_change = df[target_cols].pct_change() * 100
        df_pct_change = df_pct_change.fillna(0)

        # 重新命名變化率欄位名稱 (加 `_change` 後綴)
        df_pct_change = df_pct_change.rename(columns={col: f"{col}_change" for col in target_cols})

        # 將變化率合併回原始月資料表
        df_final = df.merge(df_pct_change, how="left", left_index=True, right_index=True)

        return df_final.reset_index()


if __name__ == '__main__':
    pass




    

