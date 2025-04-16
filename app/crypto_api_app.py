# btc_controller.py
from flask import Flask, jsonify
from controllers.update_data_controller import run_data_update
from controllers.model_training_controller import run_model_training
from controllers.feature_selection_controller import run_feature_selection
from dotenv import load_dotenv

# MinIO 啟動時可能會自動設置 AWS_ACCESS_KEY_ID 等環境變數，導致預設覆蓋我們自己的設定(強制使用 .env 中的參數覆蓋當前環境)
load_dotenv(dotenv_path=".env", override=True)
app = Flask(__name__)

@app.route("/run_data_update", methods=["POST"])
def trigger_monthly_data_update():
    try:
        run_data_update()
        return jsonify({
            "status": "success",
            "message": "Data updated."
        }), 200
    except Exception as e:
        # 記錄 log
        app.logger.error(f"❌ Data update failed: {e}")
        # 讓 requests.post() 收到 HTTP 500 回傳
        raise e  # ✅ 關鍵，Airflow 才會判斷成錯誤
    
@app.route("/run_model_training", methods=["POST"])
def trigger_monthly_model_training():
    try:
        run_model_training()
        return jsonify({
            "status": "success",
            "message": "Model trained."
        }), 200
    except Exception as e:
        # 記錄 log
        app.logger.error(f"❌ Model_training failed: {e}")
        # 讓 requests.post() 收到 HTTP 500 回傳
        raise e  # ✅ 關鍵，Airflow 才會判斷成錯誤

@app.route('/run_feature_selection', methods=['POST'])
def trigger_feature_selection():
    try:
        run_feature_selection()  # 從 feature_selection_controller 呼叫
        return jsonify({
            "status": "success",
            "message": "Feature selection re-computed."
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081)
