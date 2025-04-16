# mlops.yml 的 btc-api 區段模板

FROM python:3.11-bullseye

WORKDIR /app

RUN apt-get update && apt-get install -y docker.io

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


COPY app/ ./


EXPOSE 8081

CMD ["python", "crypto_api_app.py"]

#configs、controllers、models、services、monitoring\prometheus_handler.py、utils
#requirements > 跟舊得確認
