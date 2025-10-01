import torch
import torch.nn as nn
import torch.nn.functional as F

class GateFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 门控权重生成器
        self.gate_linear = nn.Linear(2 * d_model, d_model)
        self.sigmoid = nn.Sigmoid()

        # 初始化参数确保训练稳定性
        nn.init.xavier_uniform_(self.gate_linear.weight)
        nn.init.constant_(self.gate_linear.bias, 0.1)

    def forward(self, x_attn, x_conv):
        """
        输入形状:
        x_attn: (B, T, d_model) 自注意力输出
        x_gtu:  (B, T, d_model) 门控卷积输出
        返回:
        fused:  (B, T, d_model) 融合后特征
        """
        # 动态门控权重生成
        combined = torch.cat([x_attn, x_conv], dim=-1)  # (B, T, 2*d_model)
        gate = self.sigmoid(self.gate_linear(combined))  # (B, T, d_model)

        # 加权融合
        fused = gate * x_attn + (1 - gate) * x_conv  # (B, T, d_model)
        return fused