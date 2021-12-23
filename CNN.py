'''
author:Donglin Zhou
time:2021.12.4
function:基于市场情绪和技术面分析的股票价格预测CNN
'''
from utils import *
import warnings
from keras.models import Sequential
from keras.layers import Dense,  Flatten, Dropout,  Conv2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
magnitude = 1
warnings.filterwarnings("ignore")

def CNN_model(X_train):
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = Sequential()

    ks1_first = 3
    ks1_second = 3
    ks2_first = 3
    ks2_second = 3
    model.add(Conv2D(filters=(3),
                     kernel_size=(ks1_first, ks1_second),
                     input_shape=input_shape,
                     padding='same',
                     kernel_initializer='TruncatedNormal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.1))

    for _ in range(1):
        model.add(Conv2D(filters=(3),
                         kernel_size=(ks2_first, ks2_second),
                         padding='same',
                         kernel_initializer='TruncatedNormal'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.1))

    model.add(Flatten())

    for _ in range(4):
        model.add(Dense(64, kernel_initializer='TruncatedNormal'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.1))

    for _ in range(3):
        model.add(Dense(64, kernel_initializer='TruncatedNormal'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.1))

    model.add(Dense(64, kernel_initializer='TruncatedNormal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.1))

    model.add(Dense(1))
    return model

if __name__ == '__main__':
    data_name = 'ST_MSFT_data_PCA'
    file_name = data_name + '.csv'
    file_name = './dataset/stockstats&stanford/' + file_name
    datatype = 'stanford'
    val_mape_path = "./result/CNN/" + datatype + "/" + data_name + "/" + data_name + ".mape.hdf5"
    val_loss_path = "./result/CNN/" + datatype + "/" + data_name + "/" + data_name + ".val_loss.hdf5"
    loss_png_path = "./result/CNN/" + datatype + "/" + data_name + "/" + data_name + "CNN_loss.png"
    history_path = "./result/CNN/" + datatype + "/" + data_name + "/" + data_name + "CNN_fit_history.csv"
    path = './result/CNN/' + datatype + '/' + data_name + "/"
    data_path = './result//CNN/' + datatype + "/" + data_name + '/data.png'

    test_size = 0.3
    epochs = 500
    bs = 8
    lr = 1e-3
    look_back = 5           # 5,30
    X_train, y_train, X_test, y_test, look_back, num_features = get_data(file_name, datatype, look_back, test_size)
    # CNN的数据需要做转变
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

    adam = Adam(lr=lr)
    model = CNN_model(X_train)

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
    downsample_results(y_pred, y_true, model_name='CNN', path=path, savefig=True)