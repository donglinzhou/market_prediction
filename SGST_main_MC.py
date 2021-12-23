'''
author:Donglin Zhou
time:2021.12.15
model:基于市场情绪和技术面分析的股票价格预测:
MC字典做情感处理数据
'''
import os
import time
import argparse
from att_utils import *
from SGST_model_MC import SGST
import torch
from torch import nn

# 定义MSE作为损失函数
class MSE_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.mean(torch.pow((x - y), 2))

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

if __name__ == '__main__':
    # 读取数据的路径-每次使用不同的数据集时需要修改
    datafile_name = 'MC_INTC_data_PCA.csv'
    file_name = './dataset/stockstats&MC/' + datafile_name
    datatype = 'MC'

    parser = argparse.ArgumentParser()
    parser.add_argument('--sparse_rate', default=3, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--look_back', default=5, type=int)         # 5代表短期预测，30代表长期预测
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--num_epochs', default=1000, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--dropout_rate', default=0.1, type=float)
    parser.add_argument('--test_size', default=0.3, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    args = parser.parse_args()
    # 切分数据集
    X_train, y_train, X_test, y_test = get_data_att(file_name, datatype, args.look_back, args.test_size)

    model = SGST(args).to(args.device)
    num_batch = int(len(X_train) / args.batch_size)
    # 参数初始化
    for name, param in model.named_parameters():
        try:
            # 使得初始参数服从均匀分布
            torch.nn.init.xavier_uniform_(param.data)
        except:
            pass

    model.train()
    epoch_start_idx = 1
    criterion = MSE_loss()                                                                   # MSE
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))    # 亚当优化器

    T = 0.0
    t0 = time.time()
    all_data = []               # 存放训练结果

    # 进入迭代训练
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        for step in range(num_batch):
            # 获取数据并转化维度
            feature1 = X_train[step:step+args.batch_size, :, 0]
            feature2 = X_train[step:step+args.batch_size, :, 1]
            feature3 = X_train[step:step+args.batch_size, :, 2]
            Positive = X_train[step:step+args.batch_size, :, 3]
            Negative = X_train[step:step+args.batch_size, :, 4]
            Subjectivity = X_train[step:step+args.batch_size, :, 5]
            y_seq = y_train[step:step+args.batch_size,:,:]
            feature1 = feature1.reshape(feature1.shape[0], feature1.shape[1], 1)
            feature2 = feature2.reshape(feature2.shape[0], feature2.shape[1], 1)
            feature3 = feature3.reshape(feature3.shape[0], feature3.shape[1], 1)
            Positive = Positive.reshape(Positive.shape[0], Positive.shape[1], 1)
            Negative = Negative.reshape(Negative.shape[0], Negative.shape[1], 1)
            Subjectivity = Subjectivity.reshape(Subjectivity.shape[0],Subjectivity.shape[1],1)

            # 训练模型
            pos_logits, seqs = model(y_seq, feature1, feature2, feature3, Positive, Negative, Subjectivity)

            # 反向传播
            adam_optimizer.zero_grad()
            loss = criterion(pos_logits, seqs)
            loss.backward()
            adam_optimizer.step()
            print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item()))

        # 评估模型
        if epoch % 20 == 0:
            data_list = []
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            mse, mape, smape = evaluate_test(model, X_test, y_test, datatype)    # 评估测试集
            print('epoch:%d, time: %f(s),  test (mse: %.4f, mape:%.4f, smape:%.4f)'% (epoch, T, mse, mape,smape))

            data_list.append(float(mse))
            data_list.append(float(mape))
            data_list.append(float(smape))
            all_data.append(data_list)
            t0 = time.time()
            model.train()

        # 训练完成
        if epoch == args.num_epochs:
            # 结果保存的路径
            savepath = './result/SGST/' + datatype + '/' + datafile_name.split('.')[0] + '/'
            if not os.path.isdir(savepath):
                os.makedirs(savepath)
            fname = savepath + 'epoch={}.lr={}.layer={}.head={}.hidden={}.lookback={}.sparserate={}.batchsize={}.test_size={}'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units,
                                 args.look_back, args.sparse_rate, args.batch_size, args.test_size)
            all_data = pd.DataFrame(all_data)
            all_data.to_csv(fname + '_mse_result.csv', index=None, header=['mse', 'mape', 'smape'])

    print("Done")