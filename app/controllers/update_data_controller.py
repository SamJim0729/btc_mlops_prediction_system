import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from services.monthly_update_service import update_all_data

def run_data_update():
    # api 資料更新至2天前(ex:今日4/4、今日數字更新至4/2)
    # S3 資料更新至昨日(ex:今日4/4、今日數字更新至4/3)
    update_all_data() 
    # update_all_data(start='2025-03-01', end='2025-04-03')

if __name__ == "__main__":
    run_data_update()
