import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from Baseline.Baseline import BiLSTMModel


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        return (self.data[idx:idx + self.seq_length], self.data[idx + self.seq_length])


# 超参数
input_size = 4
hidden_size = 64
num_layers = 2
output_size = 1
num_epochs = 100
learning_rate = 0.001
seq_length = 10

# 数据和模型
data = torch.sin(torch.linspace(0, 100, steps=500))  # 示例数据
dataset = TimeSeriesDataset(data, seq_length)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
model = BiLSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(num_epochs):
    for seq, labels in dataloader:
        seq = seq.unsqueeze(-1).float()  # 添加通道维度
        labels = labels.float()

        # 前向传播
        outputs = model(seq)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
model.eval()  # 切换到评估模式
with torch.no_grad():
    test_seq = data[:seq_length].unsqueeze(0).unsqueeze(-1).float()  # 测试序列
    predicted = []

    for _ in range(50):  # 预测未来50步
        output = model(test_seq)
        predicted.append(output.item())

        # 更新输入序列
        test_seq = torch.cat((test_seq[:, 1:, :], output.unsqueeze(0).unsqueeze(-1)), dim=1)

# 绘制预测结果
import matplotlib.pyplot as plt

plt.plot(range(len(data)), data.numpy(), label="True Data")
plt.plot(range(seq_length, seq_length + len(predicted)), predicted, label="Predicted Data")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.show()
