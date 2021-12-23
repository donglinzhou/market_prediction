import pandas as pd
import re
import time

# 原始股票指标数据集
RS_files = ['stockstats价格数据集/AAPL.xlsx',
            'stockstats价格数据集/CSCO.xlsx',
            'stockstats价格数据集/INTC.xlsx',
            'stockstats价格数据集/MSFT.xlsx']

# 降噪后的股票指标数据集
S_files = ['PCA降维后数据/AAPL_PCA.xlsx',
           'PCA降维后数据/CSCO_PCA.xlsx',
           'PCA降维后数据/INTC_PCA.xlsx',
           'PCA降维后数据/MSFT_PCA.xlsx']

# 情感数据集
St_files = ['stanford/sentiment_AAPL.csv',
            'stanford/sentiment_csco.csv',
            'stanford/sentiment_INTC.csv',
            'stanford/sentiment_MSFT.csv']

S_file_name = ['AAPL', 'CSCO', 'INTC', 'MSFT']


# 将datetime.date(year, month, day) 转化为时间戳
def time_chg(date_str):
    date_str = re.findall(r'[(](.*?)[)]', date_str)

    timeArray = time.strptime(date_str[0], "%Y, %m, %d")
    timestamp = time.mktime(timeArray)
    return int(timestamp)


# 列的新顺序
new_col = ['firm', 'date', 'feature1', 'feature2', 'feature3',
           'senti_0', 'senti_1', 'senti_2', 'senti_3', 'senti_4']

# 合并所有的stockstats生成的数据集
data = None
for ind in range(len(S_files)):
    data_rs = pd.read_excel(RS_files[ind])
    data_s = pd.read_excel(S_files[ind])
    data_st = pd.read_csv(St_files[ind])
    data_s['date'] = data_rs['date'].apply(time_chg)
    data_s['firm'] = S_file_name[ind]

    data_i = pd.merge(data_s, data_st, on='date')
    if data is None:
        data = data_i
    else:
        data = pd.concat([data, data_i], axis=0)

data = data.reset_index(drop=True)
data = data[new_col]
data = data.dropna(axis=0)

data.to_csv(r'stanford_data_PCA.csv', index=False)
