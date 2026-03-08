import torch
import torch.nn as nn
import torch.nn.functional as F

class HANEncoder(nn.Module):
    """
    针对异构网络元路径的编码器。
    通过语义级注意力融合不同的元路径特征。
    """
    def __init__(self, in_dim, hidden_dim):
        super(HANEncoder, self).__init__()
        # 语义注意力层
        self.semantic_attn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False)
        )
        self.proj = nn.Linear(in_dim, hidden_dim)

    def forward(self, path_feats):
        # path_feats: [batch_size, num_metapaths, in_dim]
        # 计算每个元路径的权重 beta
        weights = self.semantic_attn(path_feats) # [B, M, 1]
        beta = F.softmax(weights, dim=1)
        
        # 融合元路径特征
        z = torch.sum(beta * path_feats, dim=1) # [B, in_dim]
        return self.proj(z)