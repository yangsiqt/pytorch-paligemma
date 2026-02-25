import torch
import torch.nn as nn

# 方法1：直接传入模块列表
model = nn.Sequential(
    nn.Linear(784, 256),   # 全连接层1
    nn.ReLU(),              # 激活函数
    nn.Linear(256, 128),    # 全连接层2
    nn.ReLU(),              # 激活函数
    nn.Linear(128, 10),     # 输出层
    nn.Softmax(dim=1)       # 输出概率
)

# 使用模型
x = torch.randn(32, 784)  # 批量大小32，特征784
y = model(x)  # 这就是你说的 y = class(x)
print(f"输入形状: {x.shape}")
print(f"输出形状: {y.shape}")