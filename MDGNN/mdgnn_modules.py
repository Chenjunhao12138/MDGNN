import torch
import torch.nn as nn
import torch.nn.functional as F

class StructureAwareAttention(nn.Module):
    """
    结构感知跨药物注意力 (Step 2)
    """
    def __init__(self, dim):
        super(StructureAwareAttention, self).__init__()
        self.q_linear = nn.Linear(dim * 2, dim)
        self.k_linear = nn.Linear(dim * 2, dim)
        self.v_linear = nn.Linear(dim * 2, dim)

    def forward(self, u_i, u_j, adj_bias):
        # adj_bias 为 DDI 图的拓扑偏置
        q = self.q_linear(u_i)
        k = self.k_linear(u_j)
        v = self.v_linear(u_j)
        
        dot_product = torch.sum(q * k, dim=-1, keepdim=True) / (q.size(-1)**0.5)
        # 注入图拓扑结构先验
        attn_weights = F.softmax(dot_product + adj_bias, dim=-1)
        return attn_weights * v

class TopologyAdaptiveGating(nn.Module):
    """
    拓扑自适应门控 (Step 3)
    """
    def __init__(self, input_dim, context_dim):
        super(TopologyAdaptiveGating, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim + context_dim, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, C_ij, s_ij):
        # C_ij: 拼接后的注意力输出; s_ij: 拓扑上下文
        combined = torch.cat([C_ij, s_ij], dim=-1)
        g_ij = self.gate(combined)
        return g_ij