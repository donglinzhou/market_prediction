'''
author:Donglin Zhou
time:2021.12.15
function:SGST-model
使用情感引导和价格序列引导的未来股市长期价格预测
'''
from new_attention import *
from sparse_attention import *
import numpy as np

# 前馈神经网络
class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        # 一维卷积，输入向量维度，输出向量维度kernel_size卷积步长
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        # 最后两维转置，然后进行一维卷积，然后dropout，然后relu激活函数，然后再进行一次一维卷积
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        # as Conv1D requires (N, C, Length)     转置回（128，200，50）
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


class SGST(torch.nn.Module):
    def __init__(self, args):
        super(SGST, self).__init__()
        self.dev = args.device
        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.feature1_attention_layernorms = torch.nn.ModuleList()
        self.feature1_attention_layers = torch.nn.ModuleList()
        self.feature1_forward_layernorms = torch.nn.ModuleList()
        self.feature1_forward_layers = torch.nn.ModuleList()

        self.feature2_attention_layernorms = torch.nn.ModuleList()
        self.feature2_attention_layers = torch.nn.ModuleList()
        self.feature2_forward_layernorms = torch.nn.ModuleList()
        self.feature2_forward_layers = torch.nn.ModuleList()

        self.feature3_attention_layernorms = torch.nn.ModuleList()
        self.feature3_attention_layers = torch.nn.ModuleList()
        self.feature3_forward_layernorms = torch.nn.ModuleList()
        self.feature3_forward_layers = torch.nn.ModuleList()

        self.pos_attention_layernorms = torch.nn.ModuleList()
        self.pos_attention_layers = torch.nn.ModuleList()
        self.pos_forward_layernorms = torch.nn.ModuleList()
        self.pos_forward_layers = torch.nn.ModuleList()

        self.neg_attention_layernorms = torch.nn.ModuleList()
        self.neg_attention_layers = torch.nn.ModuleList()
        self.neg_forward_layernorms = torch.nn.ModuleList()
        self.neg_forward_layers = torch.nn.ModuleList()

        self.v_pos_attention_layernorms = torch.nn.ModuleList()
        self.v_pos_attention_layers = torch.nn.ModuleList()
        self.v_pos_forward_layernorms = torch.nn.ModuleList()
        self.v_pos_forward_layers = torch.nn.ModuleList()

        self.v_neg_attention_layernorms = torch.nn.ModuleList()
        self.v_neg_attention_layers = torch.nn.ModuleList()
        self.v_neg_forward_layernorms = torch.nn.ModuleList()
        self.v_neg_forward_layers = torch.nn.ModuleList()

        self.con_attention_layernorms = torch.nn.ModuleList()
        self.con_attention_layers = torch.nn.ModuleList()
        self.con_forward_layernorms = torch.nn.ModuleList()
        self.con_forward_layers = torch.nn.ModuleList()

        self.sparse_attention_layers = torch.nn.ModuleList()

        self.seqs_embedd = torch.nn.Linear(1, args.hidden_units).cuda()
        self.feature1_embedd = torch.nn.Linear(1, args.hidden_units).cuda()
        self.feature2_embedd = torch.nn.Linear(1, args.hidden_units).cuda()
        self.feature3_embedd = torch.nn.Linear(1, args.hidden_units).cuda()
        self.pos_embedd = torch.nn.Linear(1, args.hidden_units).cuda()
        self.neg_embedd = torch.nn.Linear(1, args.hidden_units).cuda()
        self.v_pos_embedd = torch.nn.Linear(1, args.hidden_units).cuda()
        self.v_neg_embedd = torch.nn.Linear(1, args.hidden_units).cuda()
        self.con_embedd = torch.nn.Linear(1, args.hidden_units).cuda()

        self.seq_last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.feature1_last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.feature2_last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.feature3_last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        self.price_concat_layernorm = torch.nn.LayerNorm(args.hidden_units * 9, eps=1e-8).cuda()
        self.price_lstm_layer = torch.nn.LSTM(input_size=args.hidden_units * 9, hidden_size=args.hidden_units,
                                                  batch_first=True).cuda()

        self.price_final_emd = torch.nn.Linear(args.hidden_units*2, 1).cuda()
        self.final_norm = torch.nn.LayerNorm(args.hidden_units*2, eps=1e-8)

        self.sparse_lstm_layer = torch.nn.LSTM(input_size=args.hidden_units, hidden_size=args.hidden_units,
                                               batch_first=True).cuda()
        self.seqs_final = torch.nn.Linear(args.hidden_units, 1).cuda()
        # 2层
        for _ in range(args.num_blocks):
            # 归一化层
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.feature1_attention_layernorms.append(new_attn_layernorm)
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.feature2_attention_layernorms.append(new_attn_layernorm)
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.feature3_attention_layernorms.append(new_attn_layernorm)
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.pos_attention_layernorms.append(new_attn_layernorm)
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.neg_attention_layernorms.append(new_attn_layernorm)
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.v_pos_attention_layernorms.append(new_attn_layernorm)
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.v_neg_attention_layernorms.append(new_attn_layernorm)
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.con_attention_layernorms.append(new_attn_layernorm)

            # attention_layers添加多头注意力层
            new_attn_layer = Multihead_attention(num_units=args.hidden_units, num_heads=args.num_heads,
                                                 dropout_rate=args.dropout_rate, causality=True)
            self.attention_layers.append(new_attn_layer)
            new_attn_layer = Multihead_attention(num_units=args.hidden_units, num_heads=args.num_heads,
                                                 dropout_rate=args.dropout_rate, causality=True)
            self.feature1_attention_layers.append(new_attn_layer)
            new_attn_layer = Multihead_attention(num_units=args.hidden_units, num_heads=args.num_heads,
                                                 dropout_rate=args.dropout_rate, causality=True)
            self.feature2_attention_layers.append(new_attn_layer)
            new_attn_layer = Multihead_attention(num_units=args.hidden_units, num_heads=args.num_heads,
                                                 dropout_rate=args.dropout_rate, causality=True)
            self.feature3_attention_layers.append(new_attn_layer)
            new_attn_layer = Multihead_attention(num_units=args.hidden_units, num_heads=args.num_heads,
                                                 dropout_rate=args.dropout_rate, causality=True)
            self.pos_attention_layers.append(new_attn_layer)
            new_attn_layer = Multihead_attention(num_units=args.hidden_units, num_heads=args.num_heads,
                                                 dropout_rate=args.dropout_rate, causality=True)
            self.neg_attention_layers.append(new_attn_layer)
            new_attn_layer = Multihead_attention(num_units=args.hidden_units, num_heads=args.num_heads,
                                                 dropout_rate=args.dropout_rate, causality=True)
            self.v_pos_attention_layers.append(new_attn_layer)
            new_attn_layer = Multihead_attention(num_units=args.hidden_units, num_heads=args.num_heads,
                                                 dropout_rate=args.dropout_rate, causality=True)
            self.v_neg_attention_layers.append(new_attn_layer)
            new_attn_layer = Multihead_attention(num_units=args.hidden_units, num_heads=args.num_heads,
                                                 dropout_rate=args.dropout_rate, causality=True)
            self.con_attention_layers.append(new_attn_layer)

            # FNN
            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)
            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.feature1_forward_layernorms.append(new_fwd_layernorm)
            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.feature2_forward_layernorms.append(new_fwd_layernorm)
            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.feature3_forward_layernorms.append(new_fwd_layernorm)
            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.pos_forward_layernorms.append(new_fwd_layernorm)
            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.neg_forward_layernorms.append(new_fwd_layernorm)
            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.v_pos_forward_layernorms.append(new_fwd_layernorm)
            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.v_neg_forward_layernorms.append(new_fwd_layernorm)
            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.con_forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)
            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.feature1_forward_layers.append(new_fwd_layer)
            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.feature2_forward_layers.append(new_fwd_layer)
            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.feature3_forward_layers.append(new_fwd_layer)
            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.pos_forward_layers.append(new_fwd_layer)
            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.neg_forward_layers.append(new_fwd_layer)
            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.v_pos_forward_layers.append(new_fwd_layer)
            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.v_neg_forward_layers.append(new_fwd_layer)
            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.con_forward_layers.append(new_fwd_layer)

            # 稀疏注意力
            new_spare_att = sparseAttention(nb_head=args.num_heads, size_per_head=args.hidden_units // args.num_heads,
                                                           rate=args.sparse_rate, key_size=None)
            # 添加稀疏注意力
            self.sparse_attention_layers.append(new_spare_att)

    def log2feats(self, seqs, feature1, feature2, feature3, v_positive_seqs, positive_seqs, confidence_seqs, negative_seqs,v_negative_seqs):

        global sparse_attention

        seqs = self.seqs_embedd(seqs)
        feature1 = self.feature1_embedd(feature1)
        feature2 = self.feature2_embedd(feature2)
        feature3 = self.feature3_embedd(feature3)
        pos_seqs = self.pos_embedd(positive_seqs)
        neg_seqs = self.neg_embedd(negative_seqs)
        v_pos_seqs = self.pos_embedd(v_positive_seqs)
        v_neg_seqs = self.neg_embedd(v_negative_seqs)
        con_seqs = self.pos_embedd(confidence_seqs)

        senti_emb_concat = torch.cat((v_pos_seqs, pos_seqs, con_seqs, neg_seqs, v_neg_seqs), -1)
        for i in range(len(self.attention_layers)):
            seqs_Q = self.attention_layernorms[i](seqs)
            feature1_Q = self.feature1_attention_layernorms[i](feature1)
            feature2_Q = self.feature2_attention_layernorms[i](feature2)
            feature3_Q = self.feature3_attention_layernorms[i](feature3)
            pos_seqs_Q = self.pos_attention_layernorms[i](pos_seqs)
            neg_seqs_Q = self.neg_attention_layernorms[i](neg_seqs)
            v_pos_seqs_Q = self.pos_attention_layernorms[i](v_pos_seqs)
            v_neg_seqs_Q = self.neg_attention_layernorms[i](v_neg_seqs)
            con_seqs_Q = self.pos_attention_layernorms[i](con_seqs)

            seqs_ = self.attention_layers[i](seqs_Q, seqs, seqs)
            feature1 = self.feature1_attention_layers[i](feature1_Q, seqs, seqs)
            feature2 = self.feature1_attention_layers[i](feature2_Q, seqs, seqs)
            feature3 = self.feature1_attention_layers[i](feature3_Q, seqs, seqs)
            pos_seqs = self.pos_attention_layers[i](pos_seqs_Q, seqs, seqs)
            neg_seqs = self.neg_attention_layers[i](neg_seqs_Q, seqs, seqs)
            v_pos_seqs = self.pos_attention_layers[i](v_pos_seqs_Q, seqs, seqs)
            v_neg_seqs = self.neg_attention_layers[i](v_neg_seqs_Q, seqs, seqs)
            con_seqs = self.pos_attention_layers[i](con_seqs_Q, seqs, seqs)

            seqs = self.forward_layernorms[i](seqs_)
            seqs = self.forward_layers[i](seqs)
            feature1 = self.feature1_forward_layernorms[i](feature1)
            feature1 = self.feature1_forward_layers[i](feature1)

            feature2 = self.feature2_forward_layernorms[i](feature2)
            feature2 = self.feature2_forward_layers[i](feature2)

            feature3 = self.feature3_forward_layernorms[i](feature3)
            feature3 = self.feature3_forward_layers[i](feature3)

            pos_seqs = self.pos_forward_layernorms[i](pos_seqs)
            pos_seqs = self.pos_forward_layers[i](pos_seqs)

            neg_seqs = self.neg_forward_layernorms[i](neg_seqs)
            neg_seqs = self.neg_forward_layers[i](neg_seqs)

            v_pos_seqs = self.pos_forward_layernorms[i](v_pos_seqs)
            v_pos_seqs = self.pos_forward_layers[i](v_pos_seqs)

            v_neg_seqs = self.neg_forward_layernorms[i](v_neg_seqs)
            v_neg_seqs = self.neg_forward_layers[i](v_neg_seqs)

            con_seqs = self.pos_forward_layernorms[i](con_seqs)
            con_seqs = self.pos_forward_layers[i](con_seqs)

            sparse_attention = self.sparse_attention_layers[i](senti_emb_concat)

        log_feats = self.seq_last_layernorm(seqs)
        log_feature1 = self.feature1_last_layernorm(feature1)
        log_feature2 = self.feature1_last_layernorm(feature2)
        log_feature3 = self.feature1_last_layernorm(feature3)
        log_price = torch.cat((log_feats,log_feature1,log_feature2,log_feature3,v_pos_seqs,pos_seqs,con_seqs,v_neg_seqs,neg_seqs),-1)
        log_price = self.price_concat_layernorm(log_price)
        p_ge_price, (p_ge_ht, p_ge_ct) = self.price_lstm_layer(log_price)
        p_ge_states, (p_ge_ht, p_ge_ct) = self.sparse_lstm_layer(sparse_attention)
        log_feats = torch.cat((p_ge_price, p_ge_states), -1)
        log_feats = self.final_norm(log_feats)
        log_feats = self.price_final_emd(log_feats)
        seqs = self.seqs_final(seqs)
        return log_feats, seqs


    def forward(self, seqs, feature1, feature2, feature3, v_positive, positive, confidence, negative,v_negative):
        seqs = torch.Tensor(seqs).cuda()
        feature1, feature2, feature3 = torch.Tensor(feature1).cuda(),torch.Tensor(feature2).cuda(),torch.Tensor(feature3).cuda()
        v_positive_seqs, positive_seqs, confidence_seqs, negative_seqs,v_negative_seqs= torch.Tensor(v_positive).cuda(), \
                                                                                        torch.Tensor(positive).cuda(),torch.Tensor(confidence).cuda(), torch.Tensor(negative).cuda(),torch.Tensor(v_negative).cuda()

        log_feats, seqs = self.log2feats(seqs, feature1, feature2, feature3, v_positive_seqs, positive_seqs, confidence_seqs, negative_seqs,v_negative_seqs)

        return log_feats, seqs

    def predict(self, seqs, feature1, feature2, feature3,  v_positive, positive, confidence, negative, v_negative):

        seqs = torch.Tensor(seqs).cuda()
        feature1, feature2, feature3 = torch.Tensor(feature1).cuda(), torch.Tensor(feature2).cuda(), torch.Tensor(
            feature3).cuda()
        v_positive_seqs, positive_seqs, confidence_seqs, negative_seqs, v_negative_seqs = torch.Tensor(
            v_positive).cuda(), \
                                                                                          torch.Tensor(
                                                                                              positive).cuda(), torch.Tensor(
            confidence).cuda(), torch.Tensor(negative).cuda(), torch.Tensor(v_negative).cuda()

        log_feats, seqs = self.log2feats(seqs, feature1, feature2, feature3, v_positive_seqs, positive_seqs, confidence_seqs,
                                   negative_seqs, v_negative_seqs)

        return log_feats, seqs