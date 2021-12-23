'''
author:Donglin Zhou
time:2021.12.4
function:基于市场情绪和技术面分析的股票价格预测GRU
'''
from utils import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model
import warnings
warnings.filterwarnings("ignore")

def LSTM_model(look_back, num_features):
    model = Sequential()
    model.add(GRU(16, input_shape=(look_back, num_features-1), return_sequences=False, kernel_initializer='TruncatedNormal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    # 5
    for _ in range(0):
        model.add(Dense(8, kernel_initializer='TruncatedNormal'))  # 512
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.3))  # 0.476
    # 6
    for _ in range(0):
        model.add(Dense(8, kernel_initializer='TruncatedNormal'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.3))  # 0.041
    # 7
    model.add(Dense(8, kernel_initializer='TruncatedNormal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.3))  # 0.688
    model.add(Dense(1))

    return model

if __name__ == '__main__':
    data_name = 'ST_MSFT_data_PCA'
    file_name = data_name + '.csv'
    file_name = './dataset/stockstats&stanford/' + file_name
    datatype = 'stanford'
    val_mape_path = "./result/GRU/" + datatype + "/" + data_name + "/" + data_name + ".mape.hdf5"
    val_loss_path = "./result/GRU/" + datatype + "/" + data_name + "/" + data_name + ".val_loss.hdf5"
    loss_png_path = "./result/GRU/" + datatype + "/" + data_name + "/" + data_name + "GRU_loss.png"
    history_path = "./result/GRU/" + datatype + "/" + data_name + "/" + data_name + "GRU_fit_history.csv"
    path = './result/GRU/' + datatype + '/' + data_name + "/"
    data_path = './result//GRU/' + datatype + "/" + data_name + '/data.png'
    test_size = 0.3
    epochs = 500
    bs = 8
    lr = 1e-3
    look_back = 5  # 5,30
    X_train, y_train, X_test, y_test, look_back, num_features = get_data(file_name, datatype, look_back, test_size)

    adam = Adam(lr=lr)
    model = LSTM_model(look_back, num_features)

    # compile & fit
    model.compile(optimizer='adam', loss=['mse'], metrics=[mape, smape, 'mse'])

    # # Fit the model
    early_stopping_monitor = EarlyStopping(patience=500)
    checkpoint = ModelCheckpoint(val_mape_path, monitor='val_mape', verbose=1, save_best_only=True, mode='min')
    checkpoint1 = ModelCheckpoint(val_loss_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    result = model.fit(X_train, y_train, epochs=epochs, batch_size=bs, validation_split=0.2,
                       verbose=1, callbacks=[early_stopping_monitor, checkpoint, checkpoint1])
    print(model.summary())

    # 绘制训练过程的loss图
    loss_plot(result, loss_png_path, history_path)

    # Load the architecture
    model = load_model(val_loss_path, custom_objects={'smape': smape, 'mape': mape})
    model.compile(loss='mse', metrics=[mape, smape], optimizer=adam)
    print('FINISHED')

    y_pred = model.predict(X_test)
    y_true = y_test.reshape(y_test.shape[0], 1)
    # 评估
    downsample_results(y_pred, y_true, model_name='GRU', path=path, savefig=True)