import torch
import torch.nn as nn
from models.modules import GRU2d,CrossAttentionBlock,SEBlock

class Encoder(nn.Module):
    def __init__(self,input_dim=1,hidden_dim=64,out_dim=256):
        super(Encoder,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.gru = GRU2d(hidden_dim, input_dim)
        self.conv_out = nn.Conv2d(hidden_dim, out_dim, 1)
    def forward(self,x):
        B, T, C, H, W = x.shape
        h = torch.zeros(B, self.hidden_dim, H, W, device=x.device)
        enc_feats = []
        for t in range(T):
            h = self.gru(x[:, t], h)
            enc_feats.append(h.unsqueeze(1))

        enc_feats = torch.cat(enc_feats, dim=1)
        # [B,T,d_hid,H,W] -> [B*T,d_hid,H,W] -> [B*T,d_model,H,W]
        enc_proj = self.conv_out(enc_feats.flatten(0, 1))
        return enc_proj

class Decoder(nn.Module):
    def __init__(self,input_dim,hidden_dim,d_model,pred_len=10):
        """

        :param input_dim: the output dims of the decoder and the input dims of encoder
        :param hidden_dim: the hidden dims of the gru2d
        :param d_model: the input dims of the decoder
        :param pred_len: the time steps of the outs
        """
        super(Decoder,self).__init__()
        self.pred_len = pred_len
        self.decode_conv = nn.Conv2d(d_model, hidden_dim, 1)
        self.decoder_gru = GRU2d(hidden_dim, hidden_dim)
        self.output_conv = nn.Conv2d(hidden_dim, input_dim, 3, padding=1)

    def forward(self,x):
        """
        :param x: a Tensor with shape [B,d_model,H,W]
        :return:  a Tensor with shape [B,pred_len,H,W]
        """
        hidden = self.decode_conv(x)
        outputs = []
        for _ in range(self.pred_len):
            hidden = self.decoder_gru(hidden, hidden)
            # [B,d_hid,H,W] -> [B,input_dim,H,W]
            out = self.output_conv(hidden)
            outputs.append(out.unsqueeze(1))
        # [B,pred_len,d_model,H,W]
        outputs = torch.cat(outputs, dim=1)
        return outputs


class CrossAttentionLayer(nn.Module):
    def __init__(self,d_model,nhead,in_len=10,fusion_mode="mlp"):
        """
        :param d_model: I/O dimensionality of the model
        :param nhead: Number of heads
        :param in_len time steps of inputs
        :param fusion_mode: How to fuse the sst_proj and the fusion
        """
        super(CrossAttentionLayer,self).__init__()
        self.fusion_mode = fusion_mode
        self.in_len = in_len
        self.cross_attn = CrossAttentionBlock(d_model, nhead)
        self.temporal_compress_fusion = nn.Conv3d(d_model, d_model, kernel_size=(self.in_len, 1, 1))
        self.temporal_compress_sst = nn.Conv3d(d_model, d_model, kernel_size=(self.in_len, 1, 1))

        if fusion_mode == "gated":
            # 采用门控机制控制融合
            self.fusion_gate = nn.Sequential(
                nn.Conv2d(d_model * 2, d_model, 1),
                nn.Sigmoid()
            )
        elif fusion_mode == "mlp":
            # 采用mlp融合
            self.fusion_mlp = nn.Sequential(
                nn.Conv2d(d_model * 2, d_model, 1),
                nn.ReLU(),
                nn.Conv2d(d_model, d_model, 1)
            )
        elif fusion_mode == "self_attn":
            # 采用自注意力机制融合
            self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        elif fusion_mode == "se":
            # 采用se注意力融合
            self.se_block = SEBlock(d_model * 2)
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")

    def forward(self,sst_proj, ssh_proj):
        """
        :param sst_proj: a Tensor after encoding with the shape [B*T,d_model,H,W]
        :param ssh_proj: a Tensor after encoding with the shape [B*T,d_model,H,W]
        :return: a Tensor after fusing with the shape [B,d_model,H,W]
        """
        BT, d_model, H, W = sst_proj.shape
        B = BT // self.in_len
        T = self.in_len
        # [B*T, d_model, H, W] -> [B,T, d_model, H*W] -> [B*H*W, T, d_model]
        sst_proj_seq = sst_proj.view(B, T, -1, H * W).permute(0, 3, 1, 2).reshape(B * H * W, T, -1)
        ssh_proj_seq = ssh_proj.view(B, T, -1, H * W).permute(0, 3, 1, 2).reshape(B * H * W, T, -1)

        # 计算交叉注意力
        fusion_seq = self.cross_attn(sst_proj_seq, ssh_proj_seq, ssh_proj_seq)

        # [B*H*W,T,d_model] -> [B, d_model, H*W, T] -> [B, d_model, T, H, W]
        fusion_seq = fusion_seq.view(B, H * W, T, -1).permute(0, 3, 2, 1).reshape(B, -1, T, H, W)

        # 动态学习压缩权重
        fusion_weighted = self.temporal_compress_fusion(fusion_seq).squeeze(2)
        sst_proj_seq_reshaped = sst_proj.view(B, T, -1, H, W).permute(0, 2, 1, 3, 4)
        sst_weighted = self.temporal_compress_sst(sst_proj_seq_reshaped).squeeze(2)

        # 将sst_proj 和 fusion 融合
        if self.fusion_mode == "gated":
            # 采用门控机制控制融合
            gate_input = torch.cat([fusion_weighted, sst_weighted], dim=1)
            gate = self.fusion_gate(gate_input)
            fusion = gate * fusion_weighted + (1 - gate) * sst_weighted

        elif self.fusion_mode == "mlp":
            # 采用mlp融合
            fusion = self.fusion_mlp(torch.cat([fusion_weighted, sst_weighted], dim=1))
        elif self.fusion_mode == "self_attn":
            # 采用自注意力机制融合
            q = fusion_weighted.flatten(2).permute(0, 2, 1)
            k = sst_weighted.flatten(2).permute(0, 2, 1)
            fusion_token, _ = self.self_attn(q, k, k)
            fusion = fusion_token.permute(0, 2, 1).view(B, -1, H, W)

        elif self.fusion_mode == "se":
            # 采用se注意力融合
            fusion = self.se_block(fusion_weighted, sst_weighted)

        elif self.fusion_mode == "cross_attn2":
            # 采用自注意力机制融合
            q = sst_weighted.flatten(2).permute(0, 2, 1)
            k = fusion_weighted.flatten(2).permute(0, 2, 1)
            fusion_token, _ = self.self_attn(q, k, k)
            fusion = fusion_token.permute(0, 2, 1).view(B, -1, H, W)

        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")

        return fusion


class SSTPredictor(nn.Module):
    def __init__(self,input_dim=1,hidden_dim=64,d_model=256,nhead=8,in_len=10,pred_len=10,fusion_mode="mlp"):
        super(SSTPredictor,self).__init__()
        self.in_len = in_len
        self.pred_len = pred_len
        self.encoder_sst = Encoder(input_dim, hidden_dim, d_model)
        self.encoder_ssh = Encoder(input_dim, hidden_dim, d_model)
        self.cross_attn_layer = CrossAttentionLayer(d_model, nhead,10,fusion_mode)
        self.decoder = Decoder(input_dim, hidden_dim, d_model, pred_len)

    def forward(self,sst_seq,ssh_seq):
        sst_feats = self.encoder_sst(sst_seq)
        ssh_feats = self.encoder_ssh(ssh_seq)
        fusion = self.cross_attn_layer(sst_feats, ssh_feats)
        outputs = self.decoder(fusion)
        return outputs


class SSTPredictorWithoutSSH(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, d_model=256, nhead=8, in_len=10, pred_len=10, fusion_mode="mlp"):
        super(SSTPredictorWithoutSSH, self).__init__()
        self.in_len = in_len
        self.pred_len = pred_len
        self.encoder_sst = Encoder(input_dim, hidden_dim, d_model)
        self.cross_attn_layer = CrossAttentionLayer(d_model, nhead, 10, fusion_mode)
        self.decoder = Decoder(input_dim, hidden_dim, d_model, pred_len)

    def forward(self, sst_seq):
        sst_feats = self.encoder_sst(sst_seq)
        fusion = self.cross_attn_layer(sst_feats, sst_feats)
        outputs = self.decoder(fusion)
        return outputs

class SSTPredictorNoCA(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, d_model=256, in_len=10, pred_len=10):
        super(SSTPredictorNoCA, self).__init__()
        self.in_len = in_len
        self.pred_len = pred_len
        self.encoder_sst = Encoder(input_dim, hidden_dim, d_model)
        self.encoder_ssh = Encoder(input_dim, hidden_dim, d_model)
        self.fusion_conv = nn.Conv2d(d_model * 2, d_model, kernel_size=1)
        self.decoder = Decoder(input_dim, hidden_dim, d_model, pred_len)

    def forward(self, sst_seq, ssh_seq):
        sst_feats = self.encoder_sst(sst_seq)  # [B*T, d_model, H, W]
        ssh_feats = self.encoder_ssh(ssh_seq)
        B = sst_seq.shape[0]
        H, W = sst_seq.shape[-2], sst_seq.shape[-1]

        sst_avg = sst_feats.view(B, self.in_len, -1, H, W).mean(dim=1)  # [B, d_model, H, W]
        ssh_avg = ssh_feats.view(B, self.in_len, -1, H, W).mean(dim=1)

        fusion = torch.cat([sst_avg, ssh_avg], dim=1)  # [B, 2*d_model, H, W]
        fusion = self.fusion_conv(fusion)  # [B, d_model, H, W]

        outputs = self.decoder(fusion)
        return outputs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
if __name__ == "__main__":
    B, T, C, H, W = 2, 10, 1, 64, 64
    sst = torch.randn(B, T, C, H, W)
    ssh = torch.randn(B, T, C, H, W)
    model = SSTPredictor(input_dim=1,hidden_dim=64,d_model=256,nhead=8,in_len=10,pred_len=10,fusion_mode="se")
    print("Number of parameters:", count_parameters(model))
    # print(model)
    out = model(sst, ssh)
    print("Output shape:", out.shape)  # [B, 10, 1, 64, 64]
    model2 = SSTPredictorWithoutSSH(input_dim=1,hidden_dim=64,d_model=256,nhead=8,in_len=10,pred_len=10,fusion_mode="mlp")
    # print(model2)
    out2 = model2(sst)
    print("Output shape:", out2.shape)

    model3 = SSTPredictorNoCA(input_dim=1,hidden_dim=64,d_model=256,in_len=10,pred_len=10)
    out3 = model3(sst,ssh)
    print("Output shape:", out3.shape)
