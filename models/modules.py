import torch
from torch import nn
import numpy as np
from torch.nn import init
class BasicConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,transpose=False,act_norm=False):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if not transpose:
            self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        else:
            self.conv = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,output_padding=stride//2)
        self.norm = nn.GroupNorm(2,out_channels)
        self.act = nn.LeakyReLU(0.2,inplace=True)

    def forward(self,x):
        x = self.conv(x)
        if self.act_norm:
            x=self.act(self.norm(x))
        return x
class ConvSC(nn.Module):
    def __init__(self,C_in,C_out,stride,transpose=False,act_norm=True):
        super(ConvSC, self).__init__()
        if stride==1:
            transpose = False
        self.conv = BasicConv2d(C_in,C_out,kernel_size=3,stride=stride,padding=1,
                                transpose=transpose,act_norm=act_norm)
    def forward(self,x):
        return self.conv(x)


class GroupConv2d(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride, padding, groups, act_norm=False):
        super(GroupConv2d,self).__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1   # 不能恰好分组

        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,
                              padding=padding,groups=groups)
        self.norm = nn.GroupNorm(groups,out_channels)
        self.act = nn.LeakyReLU(0.2,inplace=True)

    def forward(self,x):
        x = self.conv(x)
        if self.act_norm:
            x = self.act(self.norm(x))
        return x

class Inception(nn.Module):
    def __init__(self,in_C,hid_C,out_C,incep_ker=[3,5,7,11], groups=8):
        super(Inception,self).__init__()
        self.conv1 = nn.Conv2d(in_C,hid_C,kernel_size=1,stride=1,padding=0)
        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(hid_C,out_C,kernel_size=ker,stride=1,padding=ker//2,groups=groups,act_norm=True))

        self.layers = nn.Sequential(*layers)

    def forward(self,x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)

        return y


class SingleModeEncoder(nn.Module):
    def __init__(self, feature_dim=1024, num_layers=2, num_heads=8):
        super(SingleModeEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x 形状：(time, batch, feature_dim)
        out = self.transformer_encoder(x)
        return out

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(MultiheadAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out

class GRU2d(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=128):
        '''GRU2d

        :param int hidden_dim: 隐状态通道数, defaults to 128
        :param int input_dim: 输入通道数, defaults to 128
        '''
        super().__init__()

        self.hidden_dim, self.input_dim = hidden_dim, input_dim

        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))  # 保证特征图大小不变
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, x, h):
        # horizontal
        hx = torch.cat([h, x], dim=1)  # 通道维度拼接
        z = torch.sigmoid(self.convz1(hx))  # 更新门
        r = torch.sigmoid(self.convr1(hx))  # 重置门
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        h = torch.nan_to_num(h)
        return h

if __name__ == '__main__':
    import torch
    x = torch.randn(10,1,64,64)
    inception = Inception(1,64,64)
    print(inception)
    print(inception(x).shape)
    single_mode_encoder = SingleModeEncoder()
    print(single_mode_encoder)


