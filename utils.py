'''
author:Donglin Zhou
time:2021.12.4
工具包：数据集切分、获取数据、评估模型的方法
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import keras.backend as K
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 获取数据
def get_data(filename,type,look_back,test_size):
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

    # 将数据转化成三维张量的形态
    num_features = data.shape[1]        # 特征+close的个数

    # 这里得到的数据用于训练模型
    X_train, y_train, X_test, y_test = df_to_cnn_rnn_format(df=data, test_size=test_size, look_back=look_back,
                                                            scale_X=True)
    return X_train, y_train, X_test, y_test, look_back, num_features


# 切分数据集成三维形状
def df_to_cnn_rnn_format(df, test_size=0.5, look_back=5, scale_X=True):

    target_location = df.shape[1] - 1               # close的位置，最后一列

    # 划分训练集和测试集
    X = df.values[:, :target_location]
    y = df.values[:, target_location]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

    # 标准化
    if scale_X:
        scalerX = StandardScaler()
        X_train = scalerX.fit_transform(X_train)
        X_test = scalerX.transform(X_test)

    # 提取需要预测的train和test
    num_features = target_location  # 自变量特征的个数

    samples_train = X_train.shape[0] - look_back  # 切分后训练数据集的个数
    X_train_reshaped = np.zeros((samples_train, look_back, num_features))
    y_train_reshaped = np.zeros((samples_train))

    for i in range(samples_train):
        y_position = i + look_back
        X_train_reshaped[i] = X_train[i:y_position]
        y_train_reshaped[i] = y_train[y_position]

    samples_test = X_test.shape[0] - look_back
    X_test_reshaped = np.zeros((samples_test, look_back, num_features))
    y_test_reshaped = np.zeros((samples_test))

    for i in range(samples_test):
        y_position = i + look_back
        X_test_reshaped[i] = X_test[i:y_position]
        y_test_reshaped[i] = y_test[y_position]

    return X_train_reshaped, y_train_reshaped, X_test_reshaped, y_test_reshaped

# 评估指标:mape
def mape(y_true, y_pred):
    import keras.backend as K
    return (K.abs(y_true - y_pred) / K.abs(y_pred)) * 100

# 评估指标:smape
def smape(y_true, y_pred):
    return (K.abs(y_pred - y_true) / ((K.abs(y_true) + K.abs(y_pred))))*100

# 绘制结果图
def downsample_results(y_pred, y_true, model_name, path, savefig=False):
    y_pred = y_pred.reshape(y_pred.shape[0])
    y_true = y_true.reshape(y_true.shape[0])

    results = pd.DataFrame(y_true, y_pred)
    result = results.reset_index()
    result.columns = ['y_pred', 'y_true']
    result.to_csv(path + model_name + '_' + '_predictions.csv')

    ytrue = result['y_true']
    ypred = result['y_pred']
    n = len(result)

    # 计算评估指标
    mse_result = (1 / n) * np.sum((ypred - ytrue) ** 2)
    mape_result = (100 / n) * np.sum(np.abs((ytrue - ypred) / ypred))
    smape_result = (100 / n) * np.sum(np.abs((ytrue - ypred)) / (np.abs(ytrue) + np.abs(ypred)))

    # 画图
    plt.figure(figsize=(20, 10))
    plt.plot(result.index, result['y_true'], '.-', color='red', label='Real values', alpha=0.5,
             ms=10)  # ms is markersize
    plt.plot(result.index, result['y_pred'], '.-', color='blue', label='Predicted values', ms=10)

    plt.ylabel(r'price', fontsize=14)
    plt.xlabel('datetime [-]', fontsize=14)  # TODO: set x values as actual dates

    plt.xticks(fontsize=14, rotation=45)
    plt.yticks(fontsize=14)

    plt.legend(loc='upper left', borderaxespad=0, frameon=False, fontsize=14, markerscale=3)
    plt.title(
        model_name + 'predictions \n MSE = %.2f \n MAPE = %.1f [%%] \n SMAPE = %.1f [%%]' % (
        mse_result, mape_result, smape_result), fontsize=14)

    if savefig:
        plt.savefig(path + model_name + '_predict.png', dpi=1300)
    plt.close()

 # 训练过程中的loss的变化过程
def loss_plot(result,loss_png_path,history_path):
    plt.plot(result.history['loss'])
    plt.plot(result.history['val_loss'])
    plt.title('INTC_LSTM_stanford_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(loss_png_path)
    plt.show()
    plt.close()
    pd.DataFrame(result.history).to_csv(history_path)