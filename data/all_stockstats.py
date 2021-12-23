import pandas as pd
import re
import time

S_files = ['stockstats价格数据集/AAPL.xlsx',
         'stockstats价格数据集/CSCO.xlsx',
         'stockstats价格数据集/INTC.xlsx',
         'stockstats价格数据集/MSFT.xlsx']

Mc_files = ['MC/AAPL_Final.csv',
            'MC/CSCO_Final.csv',
            'MC/INTC_Final.csv',
            'MC/MSFT_Final.csv']

S_file_name = ['AAPL', 'CSCO', 'INTC', 'MSFT']


# 将datetime.date(year, month, day) 转化为时间戳
def time_chg(date_str):
    date_str = re.findall(r'[(](.*?)[)]', date_str)

    timeArray = time.strptime(date_str[0], "%Y, %m, %d")
    timestamp = time.mktime(timeArray)
    return int(timestamp)

# 列的新顺序
new_col = ['firm', 'date', 'TR', 'MACD', 'CR', 'ATR', 'close_5_sma', 'rsi_14',
           'boll', 'wr_10', 'kdjk', 'kdjd', 'kdjj', 'tema', 'Positive', 'Negative', 'Subjectivity']

# 合并所有的stockstats生成的数据集
data = None
for ind in range(len(S_files)):
    data_s = pd.read_excel(S_files[ind])
    data_mc = pd.read_csv(Mc_files[ind])
    data_s['date'] = data_s['date'].apply(time_chg)
    data_s['firm'] = S_file_name[ind]

    data_i = pd.merge(data_s,data_mc, on='date')
    if data is None:
        data = data_i
    else:
        data = pd.concat([data, data_i], axis=0)

data = data.reset_index(drop=True)
data = data[new_col]
data = data.dropna(axis=0)

data.to_csv(r'MC_data.csv', index=False)
