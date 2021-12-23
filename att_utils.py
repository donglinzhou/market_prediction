'''
author:Donglin Zhou
time:2021.12.4
工具包：数据集切分、获取数据、评估模型的方法
'''
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from sklearn.model_selection import train_test_split

# mape
def mape_f(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

# smape
def smape_f(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

# 评估test集
def evaluate_test(model, X_test, y_test, type):
    global predictions,seqs
    if(type=='MC'):
        feature1 = X_test[:, :, 0]
        feature2 = X_test[:, :, 1]
        feature3 = X_test[:, :, 2]
        Positive = X_test[:, :, 3]
        Negative = X_test[:, :, 4]
        Subjectivity = X_test[:, :, 5]
        y_seq = y_test[:, :, :]
        feature1 = feature1.reshape(feature1.shape[0], feature1.shape[1], 1)
        feature2 = feature2.reshape(feature2.shape[0], feature2.shape[1], 1)
        feature3 = feature3.reshape(feature3.shape[0], feature3.shape[1], 1)
        Positive = Positive.reshape(Positive.shape[0], Positive.shape[1], 1)
        Negative = Negative.reshape(Negative.shape[0], Negative.shape[1], 1)
        Subjectivity = Subjectivity.reshape(Subjectivity.shape[0], Subjectivity.shape[1], 1)
        predictions, seqs = model.predict(y_seq, feature1, feature2, feature3, Positive, Negative, Subjectivity)

    if(type=='senticnetAR'):
        feature1 = X_test[:, :, 0]
        feature2 = X_test[:, :, 1]
        feature3 = X_test[:, :, 2]
        Positive = X_test[:, :, 3]
        Negative = X_test[:, :, 4]
        y_seq = y_test[:, :, :]
        feature1 = feature1.reshape(feature1.shape[0], feature1.shape[1], 1)
        feature2 = feature2.reshape(feature2.shape[0], feature2.shape[1], 1)
        feature3 = feature3.reshape(feature3.shape[0], feature3.shape[1], 1)
        Positive = Positive.reshape(Positive.shape[0], Positive.shape[1], 1)
        Negative = Negative.reshape(Negative.shape[0], Negative.shape[1], 1)
        predictions, seqs = model.predict(y_seq, feature1, feature2, feature3, Positive, Negative)

    if(type=='stanford'):
        feature1 = X_test[:, :, 0]
        feature2 = X_test[:, :, 1]
        feature3 = X_test[:, :, 2]
        v_Positive = X_test[:, :, 3]
        Positive = X_test[:, :, 4]
        confidence = X_test[:, :, 5]
        Negative = X_test[:, :, 6]
        v_Negative = X_test[:, :, 7]
        y_seq = y_test[:, :, :]
        feature1 = feature1.reshape(feature1.shape[0], feature1.shape[1], 1)
        feature2 = feature2.reshape(feature2.shape[0], feature2.shape[1], 1)
        feature3 = feature3.reshape(feature3.shape[0], feature3.shape[1], 1)
        positive = Positive.reshape(Positive.shape[0], Positive.shape[1], 1)
        negative = Negative.reshape(Negative.shape[0], Negative.shape[1], 1)
        v_positive = v_Positive.reshape(v_Positive.shape[0], v_Positive.shape[1], 1)
        confidence = confidence.reshape(confidence.shape[0], confidence.shape[1], 1)
        v_negative = v_Negative.reshape(v_Negative.shape[0], v_Negative.shape[1], 1)
        predictions, seqs = model.predict(y_seq, feature1, feature2, feature3, v_positive, positive, confidence, negative,v_negative)

    mse = torch.mean(torch.pow((predictions - seqs), 2))
    seqs = seqs.cpu()
    predictions = predictions.cpu()
    seqs = seqs.detach().numpy()
    predictions = predictions.detach().numpy()
    mape = mape_f(seqs, predictions)
    smape = smape_f(seqs, predictions)
    return mse, mape, smape

# 获取数据
def get_data_att(filename,type,look_back,test_size):
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

    # 这里得到的数据用于训练模型
    X_train, y_train, X_test, y_test = df_to_cnn_att_format(df=data, test_size=test_size, look_back=look_back,
                                                            scale_X=True)
    return X_train, y_train, X_test, y_test

# 将数据切分成合适的形状
def df_to_cnn_att_format(df, test_size=0.3, look_back=5,scale_X=True):

    target_location = df.shape[1] - 1                # close的位置，最后一列

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




