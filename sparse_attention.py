'''
author:DongLin Zhou
time:2021/8/20
description: the python-coding file achieve the multi-head sparse-attention function with pytorch-v1.6
'''

import torch
from torch import nn

def temporal_padding(x, padding=(1, 1)):

    assert len(padding) == 2

    # pattern = [[0, 0], [padding[0], padding[1]], [0, 0]]
    # return tf.pad(x, pattern)
    # https://blog.csdn.net/yy_diego/article/details/81563160
    # https://blog.csdn.net/sinat_36618660/article/details/100122745

    pattern = [0, 0, padding[0], padding[1], 0, 0]
    return torch.nn.functional.pad(x, pattern)

def extract_seq_patches(x, kernel_size, rate):
    """
    x.shape = [batch, seq_len, seq_dim]
    滑动地把每个窗口的x取出来，为做局部attention作准备。
    """
    seq_dim = x.shape[-1]
    seq_len = x.shape[1]
    k_size = kernel_size + (rate - 1) * (kernel_size - 1)
    p_right = (k_size - 1) // 2     # 整数除法
    p_left = k_size - 1 - p_right
    x = temporal_padding(x, (p_left, p_right))
    xs = [x[:, i: i + seq_len] for i in range(0, k_size, rate)]
    x = torch.cat(xs, dim=2)

    # https://blog.csdn.net/qq_40178291/article/details/100298791
    return torch.reshape(x, (-1, seq_len, kernel_size, seq_dim))

class sparseAttention(nn.Module):
    def __init__(self, nb_head, size_per_head, rate=2, key_size=None):
        super().__init__()
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.rate = rate
        self.key_size = key_size if key_size else size_per_head

    def forward(self, x):
        # 对Q、K、V分别作线性映射
        seq_dim = x.shape[-1]  # 返回[batch,max_len,hidden_units*4]中的hidden_units*4=200
        seq_len = x.shape[1]   # 50
        pad_len = self.rate - seq_len % self.rate  # 1

        x = temporal_padding(x, (0, pad_len))  # (128,51,200)
        new_seq_len = x.shape[1]  # 51
        x = torch.reshape(x, (-1, new_seq_len, seq_dim))  # (128,51,200)

        dense = nn.Linear(x.shape[-1], self.nb_head * self.size_per_head)
        dense.cuda()
        Q = dense(x)    # (128,51,200)
        K = dense(x)    # (128,51,200)
        V = dense(x)    # (128,51,200)

        kernel_size = 1 + 2 * (self.rate - 1)  # 1
        kwp = extract_seq_patches(K, kernel_size, self.rate)  # (128,51,1,50)
        vwp = extract_seq_patches(V, kernel_size, self.rate)  # (128,51,1,50)

        heads = self.nb_head    # 1
        qw = torch.reshape(Q, (-1, new_seq_len // self.rate, self.rate, heads, self.key_size))  # (128,51,1,1,50)
        kw = torch.reshape(K, (-1, new_seq_len // self.rate, self.rate, heads, self.key_size))  # (128,51,1,1,50)
        vw = torch.reshape(V, (-1, new_seq_len // self.rate, self.rate, heads, self.size_per_head))  # (128,51,1,1,50)

        # (128,51,1,1,1,50)
        kwp = torch.reshape(kwp, (-1, new_seq_len // self.rate, self.rate, kernel_size, heads, self.key_size))
        # (128,51,1,1,1,50)
        vwp = torch.reshape(vwp, (-1, new_seq_len // self.rate, self.rate, kernel_size, heads, self.size_per_head))

        # https://zhuanlan.zhihu.com/p/76583143
        qw = qw.permute(0, 3, 2, 1, 4)      # (128,1,1,51,50)
        kw = kw.permute(0, 3, 2, 1, 4)      # (128,1,1,51,50)
        vw = vw.permute(0, 3, 2, 1, 4)      # (128,1,1,51,50)

        # https://blog.csdn.net/Strive_For_Future/article/details/109163682
        qwp = qw.unsqueeze(4)                # (128,1,1,51,1,50)
        kwp = kwp.permute(0, 4, 2, 1, 3, 5)  # (128,1,1,51,1,50)
        vwp = vwp.permute(0, 4, 2, 1, 3, 5)  # (128,1,1,51,1,50)

        # Attention1
        # https://blog.csdn.net/GhostintheCode/article/details/102556860
        # qw(128,1,1,51,50),kw(128,1,1,51,50)->(128,1,1,51,51)
        # a = tf.keras.backend.batch_dot(qw, kw, [4, 4]) /key_size ** 0.5
        a = torch.matmul(qw, kw.transpose(4, 3)) / self.key_size ** 0.5
        a = a.permute(0, 1, 2, 4, 3)

        # Attention2
        # ap = tf.keras.backend.batch_dot(qwp, kwp, [5, 5]) / key_size ** 0.5
        # qwp,kwp:(128,1,1,51,1,50)->(128,1,1,51,1,1)
        ap = torch.matmul(qwp, kwp.transpose(5, 4)) / self.key_size ** 0.5
        ap = ap.permute(0, 1, 2, 3, 5, 4)
        ap = ap.permute(0, 1, 2, 3, 5, 4)
        ap = ap[..., 0, :]  # (128,1,1,51,1)

        # 合并两个Attention
        A = torch.cat([a, ap], -1)
        softmax = nn.Softmax(dim=-1).cuda()
        A = softmax(A)

        a, ap = A[..., : a.shape[-1]], A[..., a.shape[-1]:]
        # print(a.shape)#(128,1,1,51,51)
        # print(ap.shape)#(128,1,1,51,1)
        # 完成输出1
        # print(a.shape) #(128,1,1,51,51)
        # print(vw.shape)# (128,1,1,51,50)
        o1 = torch.matmul(a, vw)    # (128,1,1,51,50)
        # print(o1.shape)
        # o1 = tf.keras.backend.batch_dot(a, vw, [4, 3])
        # 完成输出2
        # ap = tf.expand_dims(ap, -2)
        ap = ap.unsqueeze(-2)

        # o2 = tf.keras.backend.batch_dot(ap, vwp, [5, 4])
        # print(ap.shape) #(128, 1, 1, 51, 1, 1)
        # print(vwp.shape)  # (128, 1, 1, 51, 1, 50)
        o2 = torch.matmul(ap, vwp)  # (128,1,1,51,1,50)

        o2 = o2[..., 0, :]  # (128,1,1,51,50)
        # print(o2.shape)
        # 完成输出
        o = o1 + o2
        o = o.permute(0, 3, 2, 1, 4)
        o = torch.reshape(o, (-1, new_seq_len, self.nb_head * self.size_per_head))
        o = o[:, : - pad_len]
        return o

if  __name__=="__main__":
    # test for the multi-head sparse-attention
    qwp = torch.ones(128, 50, 50*4).cuda()
    att = sparseAttention(1, 50, 1)
    outputs = att(qwp)
    print(outputs)
    print(outputs.shape)