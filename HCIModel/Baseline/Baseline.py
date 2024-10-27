import torch
import torch.nn as nn

class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = True  # 启用双向 LSTM
        self.num_directions = 2 if self.bidirectional else 1  # 2表示双向

        # 双向 LSTM 层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=self.bidirectional)

        # 全连接层的输入尺寸应考虑双向的隐藏层数量
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)

    def forward(self, x):
        # 初始化隐藏层和细胞状态
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(x.device)

        # LSTM 前向传播
        out, _ = self.lstm(x, (h0, c0))

        # 取最后一个时间步的输出，并通过全连接层
        out = self.fc(out[:, -1, :])
        return out
