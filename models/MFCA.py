import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einops

# Inception模块
class InceptionBlock(nn.Module):
    def __init__(self, input_dim, output_dim=256):
        super(InceptionBlock, self).__init__()
        self.branch_dim = output_dim // 4
        self.hidden_dim = self.branch_dim//2
        self.branch1 = nn.Conv2d(input_dim, self.branch_dim, kernel_size=1)

        self.branch2 = nn.Sequential(
            nn.Conv2d(input_dim, self.hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, self.branch_dim, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(input_dim, self.hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, self.branch_dim, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(input_dim, self.branch_dim, kernel_size=1)
        )

    def forward(self, x):
        # x [b,input_dim,h,w]
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        # out [b,output_dim,h,w]
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        padding = kernel_size // 2
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=bias)

    def forward(self, x, hidden):
        h_cur, c_cur = hidden  # 每个形状：(batch, hidden_dim, H, W)
        combined = torch.cat([x, h_cur], dim=1)
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


# 多层堆叠的 ConvLSTM 模块
class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()
        self.num_layers = num_layers
        self.cell_list = nn.ModuleList()
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim
            self.cell_list.append(ConvLSTMCell(cur_input_dim, hidden_dim, kernel_size))

    def forward(self, x):
        # 输入 x 形状：(batch, time, channels, H, W)
        b, t, _, H, W = x.size()
        h, c = [], []
        for i in range(self.num_layers):
            h.append(torch.zeros(b, self.cell_list[i].hidden_dim, H, W, device=x.device))
            c.append(torch.zeros(b, self.cell_list[i].hidden_dim, H, W, device=x.device))
        outputs = []
        for time in range(t):
            input_t = x[:, time, ...]
            for i, cell in enumerate(self.cell_list):
                h[i], c[i] = cell(input_t, (h[i], c[i]))
                input_t = h[i]
            outputs.append(h[-1])
        outputs = torch.stack(outputs, dim=1)  # (b, t, hidden_dim, H, W)
        return outputs


# 3. 雷达编码器：Inception -> MaxPool -> ConvLSTM -> 通道调整、全局池化和 FC 得到时空特征向量
class SSTEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dim = 64,feature_dim=512):
        super(SSTEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.inception = InceptionBlock(in_channels, output_dim=self.hidden_dim)
        self.pool = nn.MaxPool2d(2)  # 下采样
        self.conv_lstm = ConvLSTM(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, kernel_size=3, num_layers=3)
        self.conv_adjust = nn.Conv2d(self.hidden_dim, feature_dim, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # x 形状：(batch, time, 1, H, W)  1 10 1 64 64
        b, t, _, H, W = x.size()
        features = []
        for i in range(t):
            xi = x[:, i, ...]  # (b, 1, 64, 64)
            xi = self.inception(xi)  # (b, 256, 64, 64)
            xi = self.pool(xi)  # (b, 256, 32, 32)
            features.append(xi)
        features = torch.stack(features, dim=1)  # (b, t, 256, 32, 32)
        convlstm_out = self.conv_lstm(features)  # (b, t, 256, 32, 32)
        out = []
        for i in range(t):
            feat = convlstm_out[:, i, ...]  # (b, 256, 32, 32)
            feat = self.conv_adjust(feat)  # (b, 1024, 32, 32)
            # feat = self.global_pool(feat)  # (b, 1024, 1, 1)
            # feat = feat.view(b, -1)  # (b, feature_dim)
            out.append(feat)
        out = torch.stack(out, dim=0)  # (t, b, feature_dim,h,w)
        return out

class SSHEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dim = 64,feature_dim=512):
        super(SSHEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.inception = InceptionBlock(in_channels, output_dim=self.hidden_dim)
        self.pool = nn.MaxPool2d(2)  # 下采样
        self.conv_lstm = ConvLSTM(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, kernel_size=3, num_layers=3)
        self.conv_adjust = nn.Conv2d(self.hidden_dim, feature_dim, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # x 形状：(batch, time, 1, H, W)  1 10 1 64 64
        b, t, _, H, W = x.size()
        features = []
        for i in range(t):
            xi = x[:, i, ...]  # (b, 1, 64, 64)
            xi = self.inception(xi)  # (b, 256, 64, 64)
            xi = self.pool(xi)  # (b, 256, 32, 32)
            features.append(xi)
        features = torch.stack(features, dim=1)  # (b, t, 256, 32, 32)
        convlstm_out = self.conv_lstm(features)  # (b, t, 256, 32, 32)
        out = []
        for i in range(t):
            feat = convlstm_out[:, i, ...]  # (b, 256, 32, 32)
            feat = self.conv_adjust(feat)  # (b, 512, 32, 32)
            out.append(feat)
        out = torch.stack(out, dim=0)  # (t, b, feature_dim,h,w)
        return out


# 5. 单模态编码器：利用 Transformer 编码器层对各自模态特征进行自注意力编码
class SingleModeEncoder(nn.Module):
    def __init__(self, feature_dim=512, num_layers=2, num_heads=4):
        super(SingleModeEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x 形状：(time, batch, feature_dim)
        T,B,C,H,W = x.shape
        x = einops.rearrange(x,'t b c h w -> (t b) (h w) c')
        out = self.transformer_encoder(x)
        out = einops.rearrange(out,'(t b) (h w) c -> t b c h w',t=T,b=B,c=C,h=H,w=W)
        return out

# 6. 跨模态编码器：通过交叉注意力对齐并交换两种模态的信息
class CrossAttentionBlock(nn.Module):
    def __init__(self,feature_dim=512,num_heads=4):
        super(CrossAttentionBlock,self).__init__()
        self.cross_attn_sst = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)
        self.cross_attn_ssh = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)
        self.self_attn_sst = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)
        self.self_attn_ssh = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)
        self.mlp_sst = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Linear(feature_dim * 4, feature_dim)
        )
        self.mlp_ssh = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Linear(feature_dim * 4, feature_dim)
        )
    def forward(self,sst_feat,ssh_feat):
        # 交叉注意力
        sst_attn, _ = self.cross_attn_sst(query=sst_feat, key=ssh_feat, value=ssh_feat)
        ssh_attn, _ = self.cross_attn_ssh(query=ssh_feat, key=sst_feat, value=sst_feat)
        # 残差连接
        sst_feat = sst_feat + sst_attn
        ssh_feat = ssh_feat + ssh_attn
        # 自注意力细化加残差连接
        sst_feat = sst_feat+self.self_attn_sst(query=sst_feat,key=sst_feat,value=sst_feat)[0]
        ssh_feat = ssh_feat+self.self_attn_ssh(query=ssh_feat,key=ssh_feat,value=ssh_feat)[0]
        # MLP加残差连接
        sst_feat = sst_feat + self.mlp_sst(sst_feat)
        ssh_feat = ssh_feat + self.mlp_ssh(ssh_feat)
        return sst_feat, ssh_feat

class CrossModalEncoder(nn.Module):
    def __init__(self, feature_dim=512, num_layers=2, num_heads=4):
        super(CrossModalEncoder, self).__init__()
        self.num_layers = num_layers
        cross_layers = []
        for i in range(num_layers):
            cross_layers.append(CrossAttentionBlock(feature_dim=feature_dim, num_heads=num_heads))
        self.layers = nn.Sequential(*cross_layers)


    def forward(self, sst_feat, ssh_feat):
        # sst_feat, ssh_feat 形状：(time, batch, feature_dim,h,w)
        T,B,C,H,W = sst_feat.shape
        sst_feat = einops.rearrange(sst_feat,'t b c h w -> (t b) (h w) c')
        ssh_feat = einops.rearrange(ssh_feat,'t b c h w -> (t b) (h w) c')

        for i in range(self.num_layers):
            sst_feat,ssh_feat = self.layers[i](sst_feat, ssh_feat)

        sst_feat = einops.rearrange(sst_feat,'(t b) (h w) c -> t b c h w',t=T,b=B,c=C,h=H,w=W)
        ssh_feat = einops.rearrange(ssh_feat,'(t b) (h w) c -> t b c h w',t=T,b=B,c=C,h=H,w=W)
        return sst_feat, ssh_feat


# 7. 解码器：利用 MLP 将处理后的站点模态特征映射到未来预测的降水值
# 解码器：使用融合后的 SST 特征来预测未来的 SST
class Decoder(nn.Module):
    def __init__(self,feature_dim=512):
        super(Decoder, self).__init__()
        # 假设输入每个时间步的特征维度为1024，
        self.feature_dim = feature_dim
        # 使用四个转置卷积逐步上采样到 64x64
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.feature_dim, self.feature_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.feature_dim // 2),
            nn.ReLU(inplace=True),
            # 使用1*1卷积降维
            nn.Conv2d(self.feature_dim//2, self.feature_dim // 4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feature_dim // 4, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x 的初始形状为 [T, B, feature_dim,h',w']
        T, B, C, H, W = x.shape
        x = einops.rearrange(x, 't b c h w -> (t b) c h w')
        x = self.deconv(x)
        x = einops.rearrange(x, '(t b) c h w -> b t c h w', b=B, t=T)
        return x

# 8. 整体 MFCA 模型
class MFCA(nn.Module):
    def __init__(self, in_channels=1, feature_dim=512, prediction_steps=10):
        super(MFCA, self).__init__()
        self.sst_encoder = SSTEncoder(in_channels=in_channels, feature_dim=feature_dim)
        self.ssh_encoder = SSHEncoder(in_channels=in_channels, feature_dim=feature_dim)
        self.single_mode_encoder_sst = SingleModeEncoder(feature_dim=feature_dim, num_layers=2, num_heads=4)
        self.single_mode_encoder_ssh = SingleModeEncoder(feature_dim=feature_dim, num_layers=2, num_heads=4)
        self.cross_modal_encoder = CrossModalEncoder(feature_dim=feature_dim, num_layers=2, num_heads=4)
        self.decoder_sst = Decoder(feature_dim=feature_dim)
        self.decoder_ssh = Decoder(feature_dim=feature_dim)

    def forward(self, sst_data, ssh_data):
        sst_feat = self.sst_encoder(sst_data)  # (time, batch, feature_dim)
        ssh_feat = self.ssh_encoder(ssh_data)  # (time, batch, feature_dim)
        sst_feat = self.single_mode_encoder_sst(sst_feat)
        ssh_feat = self.single_mode_encoder_ssh(ssh_feat)
        sst_feat, ssh_feat = self.cross_modal_encoder(sst_feat, ssh_feat)

        output_sst = self.decoder_sst(sst_feat)  # (batch, time,1 ,H, W)
        output_ssh = self.decoder_ssh(ssh_feat)
        return output_sst,output_ssh

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 使用示例
if __name__ == '__main__':
    mcfa = MFCA(in_channels=1, feature_dim=128, prediction_steps=10)
    print(f"mfca模型参数量: {count_parameters(mcfa):,}")  # 逗号表示千位分隔符
    print(mcfa)
    # decoder = Decoder(feature_dim=512);
    # print(f"decoder模型参数量: {count_parameters(decoder):,}")
    sst = torch.randn(1, 10, 1, 64, 64)
    ssh = torch.randn(1, 10, 1,64,64)
    out,_ = mcfa(sst, ssh)
    print(out.shape)