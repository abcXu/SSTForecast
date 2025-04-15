import torch
import torch.nn as nn
from .modules import GRU2d

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value):
        attn_output, _ = self.attn(query, key, value)
        return self.norm(query + attn_output)


class SSTPredictor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, d_model=256, nhead=8, pred_len=10):
        super().__init__()
        self.pred_len = pred_len
        self.encoder_sst = GRU2d(hidden_dim, input_dim)
        self.encoder_ssh = GRU2d(hidden_dim, input_dim)

        self.sst_proj = nn.Conv2d(hidden_dim, d_model, 1)
        self.ssh_proj = nn.Conv2d(hidden_dim, d_model, 1)

        self.cross_attn = CrossAttentionBlock(d_model, nhead)
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

        sst_feats = torch.cat(sst_feats, dim=1)  # [B, T, C, H, W]
        ssh_feats = torch.cat(ssh_feats, dim=1)

        sst_proj = self.sst_proj(sst_feats.flatten(0, 1))  # [B*T, d_model, H, W]
        ssh_proj = self.ssh_proj(ssh_feats.flatten(0, 1))

        sst_proj = sst_proj.view(B, T, -1, H * W).permute(0, 3, 1, 2).reshape(B * H * W, T, -1)
        ssh_proj = ssh_proj.view(B, T, -1, H * W).permute(0, 3, 1, 2).reshape(B * H * W, T, -1)

        fusion = self.cross_attn(sst_proj, ssh_proj, ssh_proj)  # [B*H*W, T, d_model]
        fusion = fusion[:, -1].view(B, H * W, -1).permute(0, 2, 1).reshape(B, -1, H, W)

        hidden = self.decode_conv(fusion)

        outputs = []
        for _ in range(self.pred_len):
            hidden = self.decoder_gru(hidden, hidden)
            out = self.output_conv(hidden)
            outputs.append(out.unsqueeze(1))

        return torch.cat(outputs, dim=1)  # [B, pred_len, C, H, W]

def compute_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# Example usage:
if __name__ == "__main__":
    B, T, C, H, W = 2, 10, 1, 64, 64
    sst = torch.randn(B, T, C, H, W)
    ssh = torch.randn(B, T, C, H, W)
    model = SSTPredictor()
    print("Number of parameters:", compute_parameters(model))
    out = model(sst, ssh)
    print("Output shape:", out.shape)  # [B, 10, 1, 64, 64]