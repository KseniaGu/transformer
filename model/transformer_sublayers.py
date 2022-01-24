import torch
from torch import nn
from math import sqrt
import math

torch.random.manual_seed(0)


class MHAttetion(nn.Module):
    def __init__(self, emb_size, nrof_heads, d_hidden):
        super(MHAttetion, self).__init__()
        self.WO = nn.Linear(nrof_heads * d_hidden, emb_size)
        self.WO.bias.data.zero_()
        self.WO.weight.data.normal_(0.0, sqrt(2. / (emb_size + 4 * emb_size)))

    def forward(self, heads):
        return self.WO(torch.cat(heads, dim=-1))


class SDPAttention(nn.Module):
    def __init__(self, emb_size, hidden_size):
        super(SDPAttention, self).__init__()
        self.scale_factor = 1. / sqrt(hidden_size)
        self.WQ = nn.Linear(emb_size, hidden_size)
        self.WK = nn.Linear(emb_size, hidden_size)
        self.WV = nn.Linear(emb_size, hidden_size)
        self.SftMx = nn.Softmax(dim=-1)

        self.WQ.bias.data.zero_()
        self.WK.bias.data.zero_()
        self.WV.bias.data.zero_()
        self.WQ.weight.data.normal_(0.0, sqrt(2. / (emb_size + 4 * hidden_size)))
        self.WK.weight.data.normal_(0.0, sqrt(2. / (emb_size + 4 * hidden_size)))
        self.WV.weight.data.normal_(0.0, sqrt(2. / (emb_size + 4 * hidden_size)))

    def forward(self, x_query, x_key, x_value, mask=None):
        Q, K, V = self.WQ(x_query), self.WK(x_key), self.WV(x_value)
        scores = torch.bmm(Q, K.permute(0, 2, 1)) * self.scale_factor

        if not mask is None:
            scores[mask] = float('-inf')

        scores = self.SftMx(scores)
        Z = torch.matmul(scores, V)
        return Z


class SAttention(nn.Module):
    def __init__(self, nrof_heads=8, emb_size=512, hidden_size=64):
        super(SAttention, self).__init__()
        self.sdpa_heads = nn.ModuleList([SDPAttention(emb_size, hidden_size) for _ in range(nrof_heads)])
        self.mha = MHAttetion(emb_size, nrof_heads, hidden_size)

    def forward(self, x_query, x_key, x_value, mask):
        heads = [head(x_query, x_key, x_value, mask) for head in self.sdpa_heads]
        out = self.mha(heads)
        return out


class AddNormFF(nn.Module):
    def __init__(self, input_size, dropout):
        super(AddNormFF, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.lnorm = nn.LayerNorm(input_size)

    def forward(self, x, ff_layer):
        return x + self.dropout(ff_layer(self.lnorm(x)))


class AddNormSA(nn.Module):
    def __init__(self, input_size, dropout):
        super(AddNormSA, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.lnorm = nn.LayerNorm(input_size)

    def forward(self, x_query, x_key, x_value, sa_layer, mask=None):
        return x_query + self.dropout(sa_layer(self.lnorm(x_query),
                                               self.lnorm(x_key),
                                               self.lnorm(x_value), mask))


class FF(nn.Module):
    def __init__(self, emb_size=512, f_hidden_size=2048):
        super(FF, self).__init__()
        self.fc_0 = nn.Linear(emb_size, f_hidden_size)
        self.relu = nn.ReLU()
        self.fc_1 = nn.Linear(f_hidden_size, emb_size)

        self.fc_0.bias.data.zero_()
        self.fc_0.weight.data.normal_(0.0, sqrt(2. / (emb_size + f_hidden_size)))
        self.fc_1.bias.data.zero_()
        self.fc_1.weight.data.normal_(0.0, sqrt(2. / (emb_size + f_hidden_size)))

    def forward(self, x):
        x = self.relu(self.fc_0(x))
        return self.fc_1(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, dropout=0.2, emb_size=512, hidden_size=64, f_hidden_size=2048, nrof_heads=8):
        super(TransformerEncoder, self).__init__()
        self.SAttention = SAttention(emb_size=emb_size, hidden_size=hidden_size, nrof_heads=nrof_heads)
        self.FF = FF(emb_size, f_hidden_size)
        self.AddNormSA = AddNormSA(emb_size, dropout)
        self.AddNormFF = AddNormFF(emb_size, dropout)

    def forward(self, x, mask):
        z = self.AddNormSA(x, x, x, self.SAttention, mask)
        out = self.AddNormFF(z, self.FF)
        return out


class TransformerDecoder(nn.Module):
    def __init__(self, dropout=0.2, emb_size=512, hidden_size=64, f_hidden_size=2048, nrof_heads=8):
        super(TransformerDecoder, self).__init__()
        self.SAttention = SAttention(emb_size=emb_size, hidden_size=hidden_size, nrof_heads=nrof_heads)
        self.EncDecAttention = SAttention(emb_size=emb_size, hidden_size=hidden_size, nrof_heads=nrof_heads)
        self.FF = FF(emb_size, f_hidden_size)
        self.AddNormSA = AddNormSA(emb_size, dropout)
        self.AddNormEDA = AddNormSA(emb_size, dropout)
        self.AddNormFF = AddNormFF(emb_size, dropout)

    def forward(self, x, encoder_out, mask):
        z = self.AddNormSA(x, x, x, self.SAttention, mask)
        z = self.AddNormEDA(z, encoder_out, encoder_out, self.EncDecAttention)
        out = self.AddNormFF(z, self.FF)
        return out


class TransformerOutput(nn.Module):
    def __init__(self, emb_size, vocab_size):
        super(TransformerOutput, self).__init__()
        self.out_ff = nn.Linear(emb_size, vocab_size)

    def forward(self, x):
        return self.out_ff(x)
