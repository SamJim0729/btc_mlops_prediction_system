import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from models.training.model_training_pipeline import train_and_evaluate

def run_model_training():
    train_and_evaluate(
        target='spot_close_change',
        time_column='trade_date',
        model_list=['XGBoost'],
        train_years=3,
        test_months=4,
        n_splits=4,
        use_bagging=True,
        bagging_n=10,
        bagging_fraction=1.0,
        optuna_trials=100
    )

if __name__ == "__main__":
    run_model_training()

#Pushgateway
#http://localhost:9091/ 

#Prometheus
#http://localhost:9090/

#Alertmanager
#http://localhost:9093/
