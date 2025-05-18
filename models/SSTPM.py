import torch
import torch.nn as nn
from .modules import GRU2d
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        weight = self.fc(torch.cat([x1, x2], dim=1))
        return x1 * weight + x2 * (1 - weight)

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value):
        attn_output, _ = self.attn(query, key, value)
        return self.norm(query + attn_output)


# class SSTPredictor(nn.Module):
#     def __init__(self, input_dim=1, hidden_dim=64, d_model=256, nhead=8, pred_len=10):
#         super().__init__()
#         self.pred_len = pred_len
#         self.encoder_sst = GRU2d(hidden_dim, input_dim)
#         self.encoder_ssh = GRU2d(hidden_dim, input_dim)
#
#         self.sst_proj = nn.Conv2d(hidden_dim, d_model, 1)
#         self.ssh_proj = nn.Conv2d(hidden_dim, d_model, 1)
#
#         self.cross_attn = CrossAttentionBlock(d_model, nhead)
#         self.decode_conv = nn.Conv2d(d_model, hidden_dim, 1)
#
#         self.decoder_gru = GRU2d(hidden_dim, hidden_dim)
#         self.output_conv = nn.Conv2d(hidden_dim, input_dim, 3, padding=1)
#
#
#     def forward(self, sst_seq, ssh_seq):
#         B, T, C, H, W = sst_seq.shape
#         h_sst = torch.zeros(B, 64, H, W, device=sst_seq.device)
#         h_ssh = torch.zeros(B, 64, H, W, device=ssh_seq.device)
#
#         sst_feats, ssh_feats = [], []
#         for t in range(T):
#             h_sst = self.encoder_sst(sst_seq[:, t], h_sst)
#             h_ssh = self.encoder_ssh(ssh_seq[:, t], h_ssh)
#             sst_feats.append(h_sst.unsqueeze(1))
#             ssh_feats.append(h_ssh.unsqueeze(1))
#
#         sst_feats = torch.cat(sst_feats, dim=1)  # [B, T, C, H, W]
#         ssh_feats = torch.cat(ssh_feats, dim=1)
#
#         sst_proj = self.sst_proj(sst_feats.flatten(0, 1))  # [B*T, d_model, H, W]
#         ssh_proj = self.ssh_proj(ssh_feats.flatten(0, 1))
#
#         sst_proj = sst_proj.view(B, T, -1, H * W).permute(0, 3, 1, 2).reshape(B * H * W, T, -1)
#         ssh_proj = ssh_proj.view(B, T, -1, H * W).permute(0, 3, 1, 2).reshape(B * H * W, T, -1)
#
#         fusion = self.cross_attn(sst_proj, ssh_proj, ssh_proj)  # [B*H*W, T, d_model]
#         fusion = fusion[:, -1].view(B, H * W, -1).permute(0, 2, 1).reshape(B, -1, H, W)  # [B, d_model, H, W]
#         # 控制多少信息从ssh中融合到sst中
#
#         hidden = self.decode_conv(fusion)
#
#         outputs = []
#         for _ in range(self.pred_len):
#             hidden = self.decoder_gru(hidden, hidden)
#             out = self.output_conv(hidden)
#             outputs.append(out.unsqueeze(1))
#
#         return torch.cat(outputs, dim=1)  # [B, pred_len, C, H, W]

class SSTPredictorV2(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, d_model=256, nhead=8, pred_len=10):
        super(SSTPredictorV2,self).__init__()
        self.pred_len = pred_len
        self.encoder_sst = GRU2d(hidden_dim, input_dim)
        self.encoder_ssh = GRU2d(hidden_dim, input_dim)

        self.sst_proj = nn.Conv2d(hidden_dim, d_model, 1)
        self.ssh_proj = nn.Conv2d(hidden_dim, d_model, 1)

        self.cross_attn = CrossAttentionBlock(d_model, nhead)
        self.decode_conv = nn.Conv2d(d_model, hidden_dim, 1)

        self.decoder_gru = GRU2d(hidden_dim, hidden_dim)
        self.output_conv = nn.Conv2d(hidden_dim, input_dim, 3, padding=1)

        self.compress1 = nn.Conv3d(d_model, d_model, kernel_size=(10, 1, 1))
        self.compress2 = nn.Conv3d(d_model, d_model, kernel_size=(10, 1, 1))

        # 控制将多少从ssh中提取的信息融合到sst中
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(2 * d_model, d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, sst_seq, ssh_seq):
        B, T, C, H, W = sst_seq.shape
        h_sst = torch.zeros(B, 64, H, W, device=sst_seq.device)
        h_ssh = torch.zeros(B, 64, H, W, device=ssh_seq.device)

        sst_feats, ssh_feats = [], []
        for t in range(T):
            h_sst = self.encoder_sst(sst_seq[:, t], h_sst)
            h_ssh = self.encoder_ssh(ssh_seq[:, t], h_ssh)
            sst_feats.append(h_sst.unsqueeze(1))
            ssh_feats.append(h_ssh.unsqueeze(1))

        sst_feats = torch.cat(sst_feats, dim=1)  # [B, T, C, H, W]
        ssh_feats = torch.cat(ssh_feats, dim=1)

        sst_proj = self.sst_proj(sst_feats.flatten(0, 1))  # [B*T, d_model, H, W]
        ssh_proj = self.ssh_proj(ssh_feats.flatten(0, 1))

        sst_proj = sst_proj.view(B, T, -1, H * W).permute(0, 3, 1, 2).reshape(B * H * W, T, -1)
        ssh_proj = ssh_proj.view(B, T, -1, H * W).permute(0, 3, 1, 2).reshape(B * H * W, T, -1)

        fusion = self.cross_attn(sst_proj, ssh_proj, ssh_proj)  # [B*H*W, T, d_model]
        # fusion = fusion[:, -1].view(B, H * W, -1).permute(0, 2, 1).reshape(B, -1, H, W)  # [B, d_model, H, W]
        fusion = fusion.view(B, H * W, T, -1).permute(0, 3, 2, 1).reshape(B, -1, T, H, W)  # [B, d_model, T, H, W]
        fusion_weighted = self.compress1(fusion).squeeze(2)
        sst_proj = sst_proj.view(B, T, -1, H, W).permute(0, 2, 1, 3, 4)  # [B, d_model, T, H, W]
        sst_weighted = self.compress2(sst_proj).squeeze(2)  # [B, d_model, H, W]
        # 控制多少信息从ssh中融合到sst中
        gate_input = torch.cat([fusion_weighted, sst_weighted], dim=1)
        gate = self.fusion_gate(gate_input)
        fusion = fusion_weighted * gate + sst_weighted*(1-gate)

        hidden = self.decode_conv(fusion)

        outputs = []
        for _ in range(self.pred_len):
            hidden = self.decoder_gru(hidden, hidden)
            out = self.output_conv(hidden)
            outputs.append(out.unsqueeze(1))

        return torch.cat(outputs, dim=1)  # [B, pred_len, C, H, W]


class SSTPredictor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, d_model=256, nhead=8, pred_len=10, fusion_mode="gated"):
        super().__init__()
        self.pred_len = pred_len
        self.fusion_mode = fusion_mode

        self.encoder_sst = GRU2d(hidden_dim, input_dim)
        self.encoder_ssh = GRU2d(hidden_dim, input_dim)
        #
        self.sst_proj = nn.Conv2d(hidden_dim, d_model, 1)
        self.ssh_proj = nn.Conv2d(hidden_dim, d_model, 1)
        # self.enc_sst = Encoder(input_dim, hidden_dim, d_model)
        # self.enc_ssh = Encoder(input_dim, hidden_dim, d_model)

        self.cross_attn = CrossAttentionBlock(d_model, nhead)
        self.temporal_compress_fusion = nn.Conv3d(d_model, d_model, kernel_size=(10, 1, 1))
        self.temporal_compress_sst = nn.Conv3d(d_model, d_model, kernel_size=(10, 1, 1))

        self.fusion_gate = nn.Sequential(
            nn.Conv2d(d_model * 2, d_model, 1),
            nn.Sigmoid()
        )

        self.fusion_mlp = nn.Sequential(
            nn.Conv2d(d_model * 2, d_model, 1),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, 1)
        )

        # 采用自注意力机制融合
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.se_block = SEBlock(d_model * 2)

        # self.decoder = Decoder(input_dim, hidden_dim, d_model)
        self.decode_conv = nn.Conv2d(d_model, hidden_dim, 1)
        self.decoder_gru = GRU2d(hidden_dim, hidden_dim)
        self.output_conv = nn.Conv2d(hidden_dim, input_dim, 3, padding=1)

    def forward(self, sst_seq, ssh_seq):
        B, T, C, H, W = sst_seq.shape
        h_sst = torch.zeros(B, 64, H, W, device=sst_seq.device)
        h_ssh = torch.zeros(B, 64, H, W, device=ssh_seq.device)

        sst_feats, ssh_feats = [], []
        for t in range(T):
            h_sst = self.encoder_sst(sst_seq[:, t], h_sst)
            h_ssh = self.encoder_ssh(ssh_seq[:, t], h_ssh)
            sst_feats.append(h_sst.unsqueeze(1))
            ssh_feats.append(h_ssh.unsqueeze(1))

        sst_feats = torch.cat(sst_feats, dim=1)
        ssh_feats = torch.cat(ssh_feats, dim=1) # [B, T, C, H, W]

        sst_proj = self.sst_proj(sst_feats.flatten(0, 1))
        ssh_proj = self.ssh_proj(ssh_feats.flatten(0, 1)) # [B*T, C, H, W]
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

        if self.fusion_mode == "gated":
            gate_input = torch.cat([fusion_weighted, sst_weighted], dim=1)
            gate = self.fusion_gate(gate_input)
            fusion = gate * fusion_weighted + (1 - gate) * sst_weighted

        elif self.fusion_mode == "mlp":
            fusion = self.fusion_mlp(torch.cat([fusion_weighted, sst_weighted], dim=1))

        elif self.fusion_mode == "self_attn":
            q = fusion_weighted.flatten(2).permute(0, 2, 1)
            k = sst_weighted.flatten(2).permute(0, 2, 1)
            fusion_token, _ = self.self_attn(q, k, k)
            fusion = fusion_token.permute(0, 2, 1).view(B, -1, H, W)

        elif self.fusion_mode == "se":
            fusion = self.se_block(fusion_weighted, sst_weighted)

        elif self.fusion_mode == "cross_attn2":
            q = sst_weighted.flatten(2).permute(0, 2, 1)
            k = fusion_weighted.flatten(2).permute(0, 2, 1)
            fusion_token, _ = self.self_attn(q, k, k)
            fusion = fusion_token.permute(0, 2, 1).view(B, -1, H, W)

        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")

        hidden = self.decode_conv(fusion)
        outputs = []
        for _ in range(self.pred_len):
            hidden = self.decoder_gru(hidden, hidden)
            out = self.output_conv(hidden)
            outputs.append(out.unsqueeze(1))

        return torch.cat(outputs, dim=1)
def compute_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# Example usage:
if __name__ == "__main__":
    B, T, C, H, W = 2, 10, 1, 64, 64
    sst = torch.randn(B, T, C, H, W)
    ssh = torch.randn(B, T, C, H, W)
    model = SSTPredictorV2()
    print("Number of parameters:", compute_parameters(model))
    out = model(sst, ssh)
    print("Output shape:", out.shape)  # [B, 10, 1, 64, 64]