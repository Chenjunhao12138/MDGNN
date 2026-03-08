import torch
import torch.nn as nn

class BioBERTEncoder(nn.Module):
    """
    针对药物文本描述的编码器。
    通常输入是 BioBERT 提取的 768 维特征向量。
    """
    def __init__(self, bert_dim, hidden_dim):
        super(BioBERTEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, text_emb):
        # text_emb: [batch_size, bert_dim]
        return self.net(text_emb)