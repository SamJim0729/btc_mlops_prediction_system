import os
import re
from datetime import datetime

def get_latest_feature_config(config_dir: str, prefix: str = "selected_features_") -> str:
    pattern = re.compile(rf"{prefix}(\d{{4}}-\d{{2}}-\d{{2}})\.json")
    latest_date = None
    latest_file = None

    for filename in os.listdir(config_dir):
        match = pattern.match(filename)
        if match:
            file_date = datetime.strptime(match.group(1), "%Y-%m-%d")
            if not latest_date or file_date > latest_date:
                latest_date = file_date
                latest_file = filename

    if latest_file:
        return os.path.join(config_dir, latest_file)
    else:
        raise FileNotFoundError(f"❌ 找不到符合格式 {prefix}YYYY-MM-DD.json 的檔案")
