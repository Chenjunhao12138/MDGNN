import torch
import torch.nn as nn
from trimnet_encoder import TrimNetEncoder
from biobert_encoder import BioBERTEncoder
from han_encoder import HANEncoder
from mdgnn_modules import StructureAwareAttention, TopologyAdaptiveGating

class MDGNN(nn.Module):
    def __init__(self, config):
        super(MDGNN, self).__init__()
        h_dim = config['hidden_dim']
        
        # 1. 初始化多模态编码器
        self.struct_enc = TrimNetEncoder(config['s_in'], h_dim)
        self.text_enc = BioBERTEncoder(config['t_in'], h_dim)
        self.path_enc = HANEncoder(config['p_in'], h_dim)
        
        # 2. 交互模块
        self.cross_attn = StructureAwareAttention(h_dim)
        
        # 3. 门控模块
        # context_dim = h_dim * 2 + 1 (对应论文中 s_ij 的计算方式)
        context_dim = h_dim * 2 + 1
        self.gating = TopologyAdaptiveGating(h_dim * 3, context_dim)
        
        # 4. 预测层
        self.classifier = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.ReLU(),
            nn.Linear(h_dim // 2, config['num_classes'])
        )

    def forward(self, drug_i, drug_j, adj_info):
        # 解包特征
        s_i, t_i, p_i = drug_i
        s_j, t_j, p_j = drug_j
        adj_bias, s_ij = adj_info

        # Step 1: 多模态特征提取
        h_si, h_ti, h_pi = self.struct_enc(s_i), self.text_enc(t_i), self.path_enc(p_i)
        h_sj, h_tj, h_pj = self.struct_enc(s_j), self.text_enc(t_j), self.path_enc(p_j)

        # Step 2: 跨药物模态对融合 (S-T, S-P, T-P)
        # 构造对：u = [h1 || h2]
        c_st = self.cross_attn(torch.cat([h_si, h_ti], -1), torch.cat([h_sj, h_tj], -1), adj_bias)
        c_sp = self.cross_attn(torch.cat([h_si, h_pi], -1), torch.cat([h_sj, h_pj], -1), adj_bias)
        c_tp = self.cross_attn(torch.cat([h_ti, h_pi], -1), torch.cat([h_tj, h_pj], -1), adj_bias)

        # Step 3: 拓扑自适应融合
        C_ij = torch.cat([c_st, c_sp, c_tp], dim=-1)
        g = self.gating(C_ij, s_ij)
        
        # 最终药物对表征 z_ij
        z_ij = g[:, 0:1] * c_st + g[:, 1:2] * c_sp + g[:, 2:3] * c_tp

        # Step 4: DDI 预测
        return self.classifier(z_ij)

# --- 补全执行逻辑 ---
if __name__ == "__main__":
    # 配置参数
    config = {
        's_in': 128,      # SMILES 初始维度
        't_in': 768,      # BioBERT 初始维度 
        'p_in': 64,       # 元路径特征维度
        'hidden_dim': 64, # 模型隐层维度
        'num_classes': 167 # DDI 类型数量
    }

    # 模拟数据输入
    batch_size = 4
    # 药物 i 的三种模态特征
    d_i = (torch.randn(batch_size, 128), torch.randn(batch_size, 768), torch.randn(batch_size, 5, 64))
    # 药物 j 的三种模态特征
    d_j = (torch.randn(batch_size, 128), torch.randn(batch_size, 768), torch.randn(batch_size, 5, 64))
    # 拓扑信息 (adj_bias 和 context vector)
    adj_info = (torch.zeros(batch_size, 1), torch.randn(batch_size, 64 * 2 + 1))

    # 运行模型
    model = MDGNN(config)
    output = model(d_i, d_j, adj_info)

    print(f"MD-GNN 运行成功！")
    print(f"输入 Batch Size: {batch_size}")
    print(f"输出 DDI 预测概率形状: {output.shape} (对应 {config['num_classes']} 个类别)")