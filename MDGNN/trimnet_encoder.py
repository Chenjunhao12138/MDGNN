import torch
import torch.nn as nn
import torch.nn.functional as F

class TrimNetEncoder(nn.Module):
    """
    针对 SMILES 结构的编码器。
    论文中 TrimNet 用于提取原子级和分子级特征。
    """
    def __init__(self, in_dim, hidden_dim, num_heads=4):
        super(TrimNetEncoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim) 
            for i in range(3)
        ])
        # 三元组注意力机制的简化实现
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: [batch_size, feature_dim]
        for layer in self.layers:
            x = F.leaky_relu(layer(x), negative_slope=0.2)
        
        # 模拟分子内部结构的交互
        x_unsqueezed = x.unsqueeze(1)
        attn_out, _ = self.self_attn(x_unsqueezed, x_unsqueezed, x_unsqueezed)
        x = self.norm(x + attn_out.squeeze(1))
        return x