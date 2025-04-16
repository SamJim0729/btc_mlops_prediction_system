from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def get_time_intervals(train_value=3, train_unit="year", test_value=3, test_unit="month", delay_one_month=True):
    """
    自動計算訓練與測試的時間區間，可選擇年、月、季等單位。
    
    :param train_value: 訓練集時間長度數值（如 5 年 或 12 月）
    :param train_unit: 訓練集時間單位（year / month）
    :param test_value: 測試集時間長度數值（如 3 個月）
    :param test_unit: 測試集時間單位（year / month）
    :return: 訓練與測試的時間區間字典
    """
    today = datetime.today()
    #(0404)

    # 計算測試集結束時間
    test_end = today.replace(day=1) - timedelta(days=1)
    if test_unit == "year":
        test_start = today.replace(day=1) - relativedelta(years=test_value)
    elif test_unit == "month":
        test_start = today.replace(day=1) - relativedelta(months=test_value)
    else:
        raise ValueError("test_unit 只接受 'year', 'month'")

    # 計算訓練集開始時間
    train_end = test_start - timedelta(days=1) #test_end往前一天
    if train_unit == "year":
        train_start = train_end.replace(day=1) - relativedelta(years=train_value) + relativedelta(months=1)
    elif train_unit == "month":
        train_start = train_end.replace(day=1) - relativedelta(months=train_value-1)

    else:
        raise ValueError("train_unit 只接受 'year', 'month'")

    
    if delay_one_month:
        train_start = train_start - relativedelta(months=1)
        train_end = train_end.replace(day=1) - timedelta(days=1)
        test_start = test_start - relativedelta(months=1)
        test_end = test_end.replace(day=1) - timedelta(days=1)


    return {"start": train_start.strftime("%Y-%m-%d"), "end": train_end.strftime("%Y-%m-%d")}, \
           {"start": test_start.strftime("%Y-%m-%d"), "end": test_end.strftime("%Y-%m-%d")}

if __name__=='__main__':
    pass
    # print(get_time_intervals(train_value=1, train_unit="year", test_value=1, test_unit="month"))
    



# from datetime import datetime, timedelta
# from dateutil.relativedelta import relativedelta

# def get_time_intervals(train_value=3, train_unit="year", test_value=3, test_unit="month", delay_one_month=True):
#     """
#     自動計算訓練與測試的時間區間，可選擇年、月、季等單位。
    
#     :param train_value: 訓練集時間長度數值（如 5 年 或 12 月）
#     :param train_unit: 訓練集時間單位（year / month）
#     :param test_value: 測試集時間長度數值（如 3 個月）
#     :param test_unit: 測試集時間單位（year / month）
#     :return: 訓練與測試的時間區間字典
#     """
#     today = datetime.today()
#     #(0404)

#     # 計算測試集結束時間
#     test_end = today.replace(day=1) - timedelta(days=1)
#     if test_unit == "year":
#         test_start = today.replace(day=1) - relativedelta(years=test_value)
#     elif test_unit == "month":
#         test_start = today.replace(day=1) - relativedelta(months=test_value)
#         print(test_start)
#     else:
#         raise ValueError("test_unit 只接受 'year', 'month'")

#     # 計算訓練集開始時間
#     train_end = test_start - timedelta(days=1) #test_end往前一天
#     print(train_end)
#     if train_unit == "year":
#         train_start = train_end - relativedelta(years=train_value) + timedelta(days=1)
#     elif train_unit == "month":
#         train_start = train_end - relativedelta(months=train_value) + timedelta(days=1)
#         print(train_start)
#     else:
#         raise ValueError("train_unit 只接受 'year', 'month'")
    
#     if delay_one_month:
#         train_start = train_start - relativedelta(months=1)
#         train_end = train_end.replace(day=1) - timedelta(days=1)
#         test_start = test_start - relativedelta(months=1)
#         test_end = test_end.replace(day=1) - timedelta(days=1)

#     return {"start": train_start.strftime("%Y-%m-%d"), "end": train_end.strftime("%Y-%m-%d")}, \
#            {"start": test_start.strftime("%Y-%m-%d"), "end": test_end.strftime("%Y-%m-%d")}

# if __name__=='__main__':
#     get_time_intervals(train_value=1, train_unit="month", test_value=1, test_unit="month", delay_one_month=False)
#     # print(get_time_intervals(train_value=1, train_unit="year", test_value=1, test_unit="month"))
    