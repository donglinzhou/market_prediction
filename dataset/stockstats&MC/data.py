import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def get_data_att(filename,type='MC',look_back=30):
    # Load the data：加载数据
    df = pd.read_csv(filename, delimiter=',', header=0)
    df.head()

    data = df.copy()
    if(type=='MC'):
        data = data.iloc[:, [1, 2, 3, 4, 5, 6, 7]]
    if(type=='stanford'):
        data = data.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9]]
    if(type=='senticnetAR'):
        data = data.iloc[:, [1, 2, 3, 5, 6, 7]]

    num_features = data.shape[1]        # 特征+close的个数
    train_size = 0.7                    # 切分比例

    # 这里得到的数据用于训练模型
    X_train, y_train, X_test, y_test = df_to_cnn_att_format(df=data, train_size=train_size, look_back=look_back,
                                                            scale_X=True)
    return data,X_train, y_train, X_test, y_test, look_back, num_features, train_size

def df_to_cnn_att_format(df, train_size=0.5, look_back=5,scale_X=True):
    """
    TODO: output train and test datetime
    Input is a Pandas DataFrame.
    Output is a np array in the format of (samples, timesteps, features).
    Currently this function only accepts one target variable.

    Usage example:

    # variables
    df = data           # should be a pandas dataframe
    test_size = 0.5     # percentage to use for training
    target_column = 'c' # target column name, all other columns are taken as features
    scale_X = False
    look_back = 5       # Amount of previous X values to look at when predicting the current y value
    """
    target_location = df.shape[1] - 1  # close的位置，最后一列
    # 划分训练集和测试集
    # ...train
    X = df.values[:, :target_location]
    y = df.values[:, target_location]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    samples = len(X_train)
    num_features = target_location  # 自变量特征的个数

    samples_train = X_train.shape[0] - look_back  # 切分后训练数据集的个数
    X_train_reshaped = np.zeros((samples_train, look_back, num_features))
    y_train_reshaped = np.zeros((samples_train, look_back, 1))

    for i in range(samples_train):
        y_position = i + look_back
        X_train_reshaped[i] = X_train[i:y_position]
        for j in range(look_back):
            y_train_reshaped[i, j, :] = y_train[i+j]

    samples_test = X_test.shape[0] - look_back
    X_test_reshaped = np.zeros((samples_test, look_back, num_features))
    y_test_reshaped = np.zeros((samples_test, look_back, 1))

    for i in range(samples_test):
        y_position = i + look_back
        X_test_reshaped[i] = X_test[i:y_position]
        for j in range(look_back):
            y_test_reshaped[i, j, :] = y_test[i+j]

    return X_train_reshaped, y_train_reshaped, X_test_reshaped, y_test_reshaped

get_data_att("MC_MSFT_data_PCA.csv")