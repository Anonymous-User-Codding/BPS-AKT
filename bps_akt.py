# Code reused from https://github.com/arghosh/AKT
import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class AKT(nn.Module):
    def __init__(self, n_skill, n_eid, n_tid, n_fid, n_xid, n_yid, d_model, n_blocks,
                 kq_same, dropout, model_type, final_fc_dim=512, n_heads=8, d_ff=2048,  l2=1e-5, separate_sa=False):
        super().__init__()
        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            n_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
        """
        self.n_skill = n_skill
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_eid = n_eid
        self.n_tid = n_tid
        self.n_fid = n_fid
        self.n_xid = n_xid
        self.n_yid = n_yid
        self.l2 = l2
        self.model_type = model_type
        self.separate_sa = separate_sa
        embed_l = d_model

        if self.n_eid > 0:
            self.difficult_param = nn.Embedding(self.n_eid + 1, 1)
            self.s_embed_diff = nn.Embedding(self.n_skill + 1, embed_l)
            self.sa_embed_diff = nn.Embedding(2 * self.n_skill + 1, embed_l)
            self.t_difficult_param = nn.Embedding(self.n_tid + 1, 1)
            self.f_difficult_param = nn.Embedding(self.n_fid + 1, 1)
        if self.n_xid > 0:
            self.x_difficult_param = nn.Embedding(self.n_xid + 1, 1)
        if self.n_yid > 0:
            self.y_difficult_param = nn.Embedding(self.n_yid + 1, 1)


        # n_skill + 1, d_model
        self.s_embed = nn.Embedding(self.n_skill + 1, embed_l)
        if self.separate_sa:
            self.sa_embed = nn.Embedding(2 * self.n_skill + 1, embed_l)
        else:
            self.sa_embed = nn.Embedding(2, embed_l)
        # Architecture Object. It contains stack of attention block
        self.model = Architecture(n_skill=n_skill, n_blocks=n_blocks, n_heads=n_heads, dropout=dropout,
                                    d_model=d_model, d_feature=d_model / n_heads, d_ff=d_ff,
                                kq_same=self.kq_same, model_type=self.model_type)

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )
        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_eid+1 and self.n_eid > 0:
                torch.nn.init.constant_(p, 0.)
            if p.size(0) == self.n_tid+1 and self.n_tid > 0:
                torch.nn.init.constant_(p, 0.)
            if p.size(0) == self.n_fid+1 and self.n_fid > 0:
                torch.nn.init.constant_(p, 0.)
            if p.size(0) == self.n_xid+1 and self.n_xid > 0:
                torch.nn.init.constant_(p, 0.)
            if p.size(0) == self.n_yid+1 and self.n_yid > 0:
                torch.nn.init.constant_(p, 0.)

    def forward(self, s_data, sa_data, target, eid_data=None, tid_data=None, fid_data=None, xid_data=None, yid_data=None):
        s_embed_data = self.s_embed(s_data) # k_s
        if self.separate_sa:
            sa_embed_data = self.sa_embed(sa_data) # v_{(s,r)}
        else:
            sa_data = (sa_data - s_data)//self.n_skill  # r
            sa_embed_data = self.sa_embed(sa_data) + s_embed_data # v_{(s,r)}

        if self.n_eid > 0:
            s_embed_diff_data = self.s_embed_diff(s_data)  # d_s
            eid_embed_data = self.difficult_param(eid_data)  # mu_e
            s_embed_data += eid_embed_data * s_embed_diff_data  # BE: k_s + mu_e*d_s
            sa_embed_diff_data = self.sa_embed_diff(sa_data)  # f_{(s,r)}

            # \mathbf{x}^{PE}
            if self.n_tid > 0:
                tid_embed_data = self.t_difficult_param(tid_data)  # uq
                fid_embed_data = self.f_difficult_param(fid_data)  # uq
                s_embed_data += tid_embed_data * s_embed_diff_data  # BE + tau_e*d_s
                s_embed_data += fid_embed_data * s_embed_diff_data  # BE+PE: BE + tau_e*d_s + phi_e*d_s

            #\mathbf{x}^{SE}
            if self.n_xid > 0:
                xid_embed_data = self.x_difficult_param(xid_data)  # xi_{f_e^1}
                s_embed_data += xid_embed_data * s_embed_diff_data  # BE + PE + xi_{f_e^1} * d_s
            if self.n_yid > 0:
                yid_embed_data = self.y_difficult_param(yid_data)  # xi_{f_e^2}
                s_embed_data += yid_embed_data * s_embed_diff_data  # BE+PE+SE: BE+PE + xi_{f_e^1}*d_s + xi_{f_e^2}*d_s

            if self.separate_sa:
                # \mathbf{y}^{BE}
                sa_embed_data += eid_embed_data * sa_embed_diff_data
                # \mathbf{y}^{PE}
                if self.n_tid > 0:
                    sa_embed_data += tid_embed_data * sa_embed_diff_data
                    sa_embed_data += fid_embed_data * sa_embed_diff_data
                #\mathbf{y}^{SE}
                if self.n_xid > 0:
                    sa_embed_data += xid_embed_data * sa_embed_diff_data
                if self.n_yid > 0:
                    sa_embed_data += yid_embed_data * sa_embed_diff_data
            else:
                # \mathbf{y}^{BE}
                sa_embed_data += eid_embed_data * (sa_embed_diff_data + s_embed_diff_data)
                # \mathbf{y}^{PE}
                if self.n_tid > 0:
                    sa_embed_data += tid_embed_data * (sa_embed_diff_data + s_embed_diff_data)
                    sa_embed_data += fid_embed_data * (sa_embed_diff_data + s_embed_diff_data)
                # \mathbf{y}^{SE}
                if self.n_xid > 0:
                    sa_embed_data += xid_embed_data * (sa_embed_diff_data + s_embed_diff_data)
                if self.n_yid > 0:
                    sa_embed_data += yid_embed_data * (sa_embed_diff_data + s_embed_diff_data)

        d_output = self.model(s_embed_data, sa_embed_data)  # 211x512

        concat_s = torch.cat([d_output, s_embed_data], dim=-1)
        output = self.out(concat_s)
        labels = target.reshape(-1)
        m = nn.Sigmoid()
        preds = (output.reshape(-1))  # logit
        mask = labels > -0.9
        masked_labels = labels[mask].float()
        masked_preds = preds[mask]
        loss = nn.BCEWithLogitsLoss(reduction='none')
        output = loss(masked_preds, masked_labels)
        return output.sum(), m(preds), mask.sum()


class Architecture(nn.Module):
    def __init__(self, n_skill,  n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model
        self.model_type = model_type

        if model_type in {'akt'}:
            self.blocks_1 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks)
            ])
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks*2)
            ])

    def forward(self, s_embed_data, sa_embed_data):
        # target shape  bs, seqlen
        seqlen, batch_size = s_embed_data.size(1), s_embed_data.size(0)

        sa_pos_embed = sa_embed_data
        s_pos_embed = s_embed_data

        y = sa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = s_pos_embed

        # encoder
        for block in self.blocks_1:  # encode qas
            y = block(mask=1, query=y, key=y, values=y)
        flag_first = True
        for block in self.blocks_2:
            if flag_first:  # peek current question
                x = block(mask=1, query=x, key=x,
                          values=x, apply_pos=False)
                flag_first = False
            else:  # dont peek current response
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True)
                flag_first = True
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout,  kq_same):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        """
        Input:
            block : object of type BasicBlock(nn.Module). It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
            mask : 0 means, it can peek only past values. 1 means, block can peek only current and pas values
            query : Query. In transformer paper it is the input for both encoder and decoder
            key : Keys. In transformer paper it is the input for both encoder and decoder
            Values. In transformer paper it is the input for encoder and  encoded output for decoder (in masked attention part)

        Output:
            query: Input gets changed over the layer and returned.

        """

        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True)
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1((query2))
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query)
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad):

        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        gammas = self.gammas
        scores = attention(q, k, v, self.d_k,
                           mask, self.dropout, zero_pad, gammas)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous()\
            .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None):
    """
    This is called by Multi-head atention object to find the values.
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
        math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
        scores_ = scores_ * mask.float().to(device)
        distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, 8, sl, sl
        disttotal_scores = torch.sum(
            scores_, dim=-1, keepdim=True)  # bs, 8, sl, 1
        position_effect = torch.abs(
            x1-x2)[None, None, :, :].type(torch.FloatTensor).to(device)  # 1, 1, seqlen, seqlen
        # bs, 8, sl, sl positive distance
        dist_scores = torch.clamp(
            (disttotal_scores-distcum_scores)*position_effect, min=0.)
        dist_scores = dist_scores.sqrt().detach()
    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0)  # 1,8,1,1
    # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
    total_effect = torch.clamp(torch.clamp(
        (dist_scores*gamma).exp(), min=1e-5), max=1e5)
    scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)
