FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app

# 複製 requirements 並安裝
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製所有程式碼
COPY app /app

COPY .env .env

# 切換進程執行資料夾
WORKDIR /app

# 設定啟動指令
CMD ["python", "main.py"]