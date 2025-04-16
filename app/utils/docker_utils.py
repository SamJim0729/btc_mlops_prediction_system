import os
import subprocess
from dotenv import load_dotenv

load_dotenv()

def get_container_ip(container_name: str) -> str:
    """透過 docker inspect 指令取得 container 的 IP"""
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", container_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        ip = result.stdout.strip()
        return ip
    except subprocess.CalledProcessError as e:
        print(f"❌ 無法取得 {container_name} 的 IP，錯誤：{e.stderr}")
        return None


if __name__=='__main__':
    # 取得 MinIO container 的 IP
    minio_container_name = "crypto_proj_minio_container"
    minio_ip = get_container_ip(minio_container_name)

    if minio_ip:
        os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("MINIO_ACCESS_KEY_ID")
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("MINIO_SECRET_ACCESS_KEY")
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{minio_ip}:9000"
        print("✅ MLFLOW_S3_ENDPOINT_URL 已設定為:", os.environ["MLFLOW_S3_ENDPOINT_URL"])
    else:
        print("⚠️ 使用預設的 MinIO endpoint，請確認是否正確連線")


