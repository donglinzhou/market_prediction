'''
author:DongLin Zhou
time:2021/8/19
description: the python-coding file achieve the multi-head self-attention function with pytorch-v1.6
'''
import torch
from torch import nn

class Multihead_attention(nn.Module):
    def __init__(self, num_units, num_heads, dropout_rate, causality, with_qk=False):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causality = causality
        self.with_qk = with_qk
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, queries, keys, value):
        # Linear projections
        Q_dense = torch.nn.Linear(queries.shape[-1], self.num_units).cuda()
        K_dense = torch.nn.Linear(keys.shape[-1], self.num_units).cuda()
        V_dense = torch.nn.Linear(value.shape[-1], self.num_units).cuda()

        Q = Q_dense(queries)
        K = K_dense(keys)
        V = V_dense(value)

        # Split and concat
        Q_ = torch.cat(torch.chunk(Q, dim=2, chunks=self.num_heads), dim=0).cuda()
        K_ = torch.cat(torch.chunk(K, dim=2, chunks=self.num_heads), dim=0).cuda()
        V_ = torch.cat(torch.chunk(V, dim=2, chunks=self.num_heads), dim=0).cuda()

        # Multiplication
        outputs = torch.matmul(Q_, K_.transpose(1, 2)).cuda()

        # Scale
        outputs = outputs / (K_.shape[-1] ** 0.5)

        # Key Masking
        key_masks = torch.sign(torch.abs(keys.sum(dim=-1))).cuda()
        key_masks = key_masks.repeat([self.num_heads, 1])
        key_masks = torch.unsqueeze(key_masks, 1)
        key_masks = key_masks.repeat([1, queries.shape[1], 1])

        paddings = torch.ones_like(outputs) * (-2 ** 32 + 1)
        paddings.cuda()
        zero_tensor = torch.zeros_like(key_masks).cuda()
        outputs = torch.where(torch.eq(key_masks, zero_tensor).cuda(), paddings, outputs).cuda()

        # Causality = Future blinding
        if self.causality:
            diag_vals = torch.ones_like(outputs[0, :, :]).cuda()
            tril = torch.tril(diag_vals, diagonal=0).cuda()
            tril = torch.unsqueeze(tril, 0)
            masks = tril.repeat([outputs.shape[0], 1, 1])

            paddings = torch.ones_like(masks) * (-2 ** 32 + 1)
            zero_tensor = torch.zeros_like(masks)
            outputs = torch.where(torch.eq(masks, zero_tensor), paddings, outputs)

        # Activation
        output = self.softmax(outputs)

        # Query Masking
        query_masks = torch.sign(torch.abs(queries.sum(dim=-1))).cuda()
        query_masks = query_masks.repeat([self.num_heads, 1])
        query_masks = torch.unsqueeze(query_masks, -1)
        query_masks = query_masks.repeat([1, 1, keys.shape[1]])
        outputs = output * query_masks      # broadcasting.

        # Dropouts
        outputs = self.dropout(outputs)

        # Weighted sum
        outputs = torch.matmul(outputs, V_).cuda()

        # Restore shape
        outputs = torch.cat(torch.chunk(outputs, dim=0, chunks=self.num_heads), dim=2)

        # Residual connection
        outputs += queries

        if self.with_qk:
            return Q, K
        else:
            return outputs

if  __name__=="__main__":
    # test for the multi-head self-attention
    q = torch.ones(128, 50, 50).cuda()
    k = torch.ones(128, 50, 50).cuda()
    v = torch.ones(128, 50, 50).cuda()

    att = Multihead_attention(50, 1, 0.5, True, False)
    outputs = att(q, k, v)
    print(outputs.device)
    print(outputs.shape)
    print(outputs)


