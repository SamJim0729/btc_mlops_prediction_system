import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pandas_datareader.data as pdr

class BTCDataLocalFetcher:
    """
    BTC 數據抓取類別，包含歷史現貨與期貨價格，並計算溢價率，適配資料庫存儲。
    """
    def __init__(self, start=None, end=None):
        self.start = start
        self.end = end
        if self.start is None or self.end is None:
            today = datetime.today()
            #yf.download API 區間，end範圍會-1
            first_day_last_month = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
            #calendar.monthrange給(年、月)參數，回傳最後一天星期幾&這個月天數
            yesterday = today - timedelta(days=1)
            self.start = first_day_last_month.strftime("%Y-%m-%d")
            self.end = yesterday.strftime("%Y-%m-%d")
        

    def fetch_historical_prices(self, interval="1d"):
        """
        抓取 BTC 的歷史現貨與期貨價格，並計算每日溢價率，回傳處理後的 DataFrame。
        """
        try:
            # 1️⃣ 抓取現貨價格
            btc_spot = yf.download("BTC-USD", start=self.start, end=self.end, interval=interval)

            # 🔹 移除 MultiIndex（如果存在）
            if isinstance(btc_spot.columns, pd.MultiIndex):
                btc_spot.columns = btc_spot.columns.get_level_values(0)
            btc_spot = btc_spot[["Open", "High", "Low", "Close", "Volume"]]


            btc_spot.columns = ["spot_open", "spot_high", "spot_low", "spot_close", "spot_volume"]
            btc_spot.index = pd.to_datetime(btc_spot.index)
            btc_spot.reset_index(inplace=True)
            btc_spot.rename(columns={"Date": "trade_date"}, inplace=True)
            

            # 2️⃣ 抓取期貨價格（CME 期貨市場）
            btc_futures = yf.download("BTC=F", start=self.start, end=self.end, interval=interval)


            if isinstance(btc_spot.columns, pd.MultiIndex):
                btc_spot.columns = btc_spot.columns.get_level_values(0)

            if not btc_futures.empty:
                btc_futures = btc_futures[["Open", "High", "Low", "Close", "Volume"]]
                btc_futures.columns = ["futures_open", "futures_high", "futures_low", "futures_close", "futures_volume"]
                btc_futures.reset_index(inplace=True)
                btc_futures.rename(columns={"Date": "trade_date"}, inplace=True)
            else:
                btc_futures = pd.DataFrame(columns=["trade_date", "futures_open", "futures_high", "futures_low", "futures_close", "futures_volume"])
            
            # adjusted_end目的是抓取實際擷取到日期最大值，以免end條件超出範圍
            adjusted_end = btc_spot["trade_date"].max().strftime("%Y-%m-%d")

            # # 3️⃣ 合併數據，計算溢價率
            return self._calculate_basis_premium(btc_spot, btc_futures, self.start, adjusted_end)

        except Exception as e:
            pass
            print(f"❌ 歷史數據抓取失敗: {e}")
            return None

    def _calculate_basis_premium(self, btc_spot, btc_futures, start, end):
        """
        計算 BTC 期貨溢價率，確保所有日期都有數據，即使缺失也補 NULL。
        """
        try:
            # 1️⃣ 生成完整的 `trade_date` 日期範圍
            date_range = pd.date_range(start=start, end=end, freq="D").to_frame(index=False, name="trade_date")
            # # 2️⃣ 合併數據，確保所有日期都有值
            df = date_range.merge(btc_spot, on="trade_date", how="left")
            df = df.merge(btc_futures, on="trade_date", how="left")


            # # 3️⃣ 計算溢價率（注意後續 NaN 影響計算）
            df["premium_rate"] = ((df["futures_close"] - df["spot_close"]) / df["spot_close"]) * 100
            return df

        except Exception as e:
            print(f"❌ 溢價率計算失敗: {e}")
            return None


class MacroDataFetcher:
    def __init__(self, start=None, end=None):
        self.start = start
        self.end = end
        if start is None or end is None:
            today = datetime.today()
            first_day_last_month = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
            yesterday = today - timedelta(days=1)
            self.start = first_day_last_month.strftime("%Y-%m-%d")
            self.end = yesterday.strftime("%Y-%m-%d")

    def fetch_fed_rate(self):
        """ 獲取美聯儲利率 (Fed Rate) """
        try:
            fed_rate = pdr.get_data_fred('DFF', self.start, self.end)
            fed_rate.rename(columns={'DFF': 'fed_rate'}, inplace=True)

            return fed_rate
        except Exception as e:
            print(f"Error fetching Fed Rate: {e}")
            return None

    def fetch_sp500(self):
        """ 獲取標普500指數 (S&P 500) """
        try:
            sp500 = yf.Ticker('^GSPC').history(start=self.start, end=self.end)
            sp500 = sp500[['Close']].rename(columns={'Close': 'sp500_index'})
            return sp500
        except Exception as e:
            print(f"Error fetching S&P 500: {e}")
            return None

    def fetch_vix(self):
        """ 獲取 VIX 恐慌指數 """
        try:
            vix = yf.Ticker('^VIX').history(start=self.start, end=self.end)
            vix = vix[['Close']].rename(columns={'Close': 'vix_index'})
            return vix
        except Exception as e:
            print(f"Error fetching VIX: {e}")
            return None

    def fetch_dxy(self):
        """ 獲取美元指數 (DXY) """
        try:
            dxy = yf.Ticker('DX-Y.NYB').history(start=self.start, end=self.end)
            dxy = dxy[['Close']].rename(columns={'Close': 'dxy_index'})
            return dxy
        except Exception as e:
            print(f"Error fetching DXY: {e}")
            return None

    def _make_tz_naive(self, df):
        """移除 DataFrame 的時區，使索引統一為 tz-naive"""
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df = df.tz_convert(None)  # 轉換為無時區
            df.index = pd.to_datetime(df.index).normalize()
        return df

    def fetch_all_macro_data(self):
        """ 獲取所有總經數據，並合併成單一 DataFrame，使用 fed_rate 作為左表 """
        
        # 先獲取各項總經數據
        fed_rate = self._make_tz_naive(self.fetch_fed_rate())  # 作為左表
        sp500 = self._make_tz_naive(self.fetch_sp500())
        vix = self._make_tz_naive(self.fetch_vix())
        dxy = self._make_tz_naive(self.fetch_dxy())

        # 依據 fed_rate 為主表做 left join
        macro_df = fed_rate \
            .merge(sp500, how="left", left_index=True, right_index=True) \
            .merge(vix, how="left", left_index=True, right_index=True) \
            .merge(dxy, how="left", left_index=True, right_index=True)
        macro_df.reset_index(inplace=True)
        macro_df.rename(columns={"DATE": "trade_date"}, inplace=True)

        # # # 週末或其他缺失數據使用前值填充 (ffill)
        return macro_df



# 測試用例
if __name__ == "__main__":
    fetcher = BTCDataLocalFetcher()
    btc_data = fetcher.fetch_historical_prices()
    # fetcher = MacroDataFetcher()
    # macro_data = fetcher.fetch_all_macro_data()
    print(btc_data)



