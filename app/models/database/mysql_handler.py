import pymysql
import pandas as pd

class Mysql:
    def __init__(self, host, user, password, database):
        """初始化 MySQL 連線"""
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None

    def connect(self):
        """建立資料庫連線"""
        if self.connection is None or not self.connection.open:
            self.connection = pymysql.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                cursorclass=pymysql.cursors.DictCursor  # 🔹 返回 dict 而不是 tuple
            )
        return self.connection

    def execute_query(self, query, params=None, fetch_one=False):
        """執行 SQL 查詢，支援 SELECT / UPDATE / DELETE"""
        with self.connect().cursor() as cursor:
            cursor.execute(query, params or ())
            if query.strip().upper().startswith(("SELECT", "SHOW")):
                return cursor.fetchone() if fetch_one else cursor.fetchall()
            self.connection.commit()
        return None

    def insert_data(self, table, data):
        """插入單筆資料，若主鍵重複則更新"""
        columns = ", ".join(data.keys())
        
        values = ", ".join(["%s"] * len(data))
        
        updates = ", ".join([f"{col} = VALUES({col})" for col in data.keys()])  # 產生 ON DUPLICATE KEY UPDATE 條件
        sql = f"INSERT INTO {table} ({columns}) VALUES ({values}) ON DUPLICATE KEY UPDATE {updates}"
        values_tuple = tuple(None if pd.isna(v) else v for v in data.values())  # 確保 NaN 轉為 None

        with self.connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, values_tuple)
            conn.commit()

    def insert_bulk_data(self, table, data_list):
        """插入多筆資料，批量寫入提高效率"""
        if not data_list:
            return

        columns = ", ".join(data_list[0].keys())
        values_placeholder = ", ".join(["%s"] * len(data_list[0]))
        sql = f"INSERT INTO {table} ({columns}) VALUES ({values_placeholder}) ON DUPLICATE KEY UPDATE {', '.join([f'{col}=VALUES({col})' for col in data_list[0].keys()])}"

        values = [tuple(data.values()) for data in data_list]
        with self.connect().cursor() as cursor:
            cursor.executemany(sql, values)
            self.connection.commit()

    def fetch_latest_data(self, table, date_column="trade_date"):
        """取得資料庫最新的一筆資料"""
        sql = f"SELECT * FROM {table} ORDER BY {date_column} DESC LIMIT 1"
        return self.execute_query(sql, fetch_one=True)

    def close_connection(self):
        """關閉 MySQL 連線"""
        if self.connection:
            self.connection.close()
            self.connection = None

if __name__ == '__main__':

    from dotenv import load_dotenv
    import os
    load_dotenv()
    
    db = Mysql(
        host=os.getenv("SQL_HOST"),
        user=os.getenv("SQL_USER"),
        password=os.getenv("SQL_PASSWORD"),
        database=os.getenv("SQL_DB")
    )
    result = db.execute_query("show tables")
    print(result)

    result = db.execute_query("select * from btc_price_history where left(trade_date, 7)='2025-04'")
    print(result)


    
