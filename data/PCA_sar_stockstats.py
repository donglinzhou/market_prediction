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
sar_files = ['senticnet_average_result/AAPL_result.csv',
             'senticnet_average_result/CSCO_result.csv',
             'senticnet_average_result/INTC_result.csv',
             'senticnet_average_result/MSFT_result.csv']

# 情感数据集
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
new_col = ['firm', 'date', 'feature1', 'feature2', 'feature3',
           'publication_date', 'negative', 'positive']

# 合并所有的stockstats生成的数据集
data = None
for ind in range(len(S_files)):
    data_rs = pd.read_excel(RS_files[ind])
    data_s = pd.read_excel(S_files[ind])
    data_st = pd.read_csv(sar_files[ind])
    data_mc = pd.read_csv(Mc_files[ind])

    data_st['date'] = data_mc['date']

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

data.to_csv(r'SenticentAR_data_PCA.csv', index=False)
