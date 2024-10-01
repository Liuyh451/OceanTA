import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class BasicUnit(nn.Module):
    """
        提取长短期特征
        输入通道数：32
        输出通道数：8
        参数：
        input_channels, output_channels,data
        返回：
        特征图 Tensor (N*C*L)
    """

    def __init__(self, input_channels, output_channels):
        super(BasicUnit, self).__init__()
        # 使用3x1的卷积核
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size=3, padding=1)  # 第一组卷积
        self.conv2 = nn.Conv1d(input_channels, output_channels, kernel_size=3, padding=1)  # 第二组卷积
        self.conv3 = nn.Conv1d(input_channels, output_channels, kernel_size=3, padding=1)  # 第三组卷积
        self.conv4 = nn.Conv1d(input_channels, output_channels, kernel_size=3, padding=1)  # 第四组卷积

    def forward(self, x):
        A = torch.tanh(self.conv1(x))  # A = tanh(x * w1 + b1)
        B = torch.sigmoid(self.conv2(x))  # B = σ(x * w2 + b2)
        C = torch.tanh(self.conv3(x))  # C = tanh(x * w3 + b3)
        D = torch.sigmoid(self.conv4(x))  # D = σ(x * w4 + b4)

        E = A * B + C * D  # E = A ⊗ B + C ⊗ D
        return E


class BrainAnalysisModule(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(BrainAnalysisModule, self).__init__()
        self.basic_unit1 = BasicUnit(input_channels, output_channels)  # 第一个基本单元
        self.basic_unit2 = BasicUnit(input_channels, output_channels)  # 第二个基本单元

    def forward(self, x):
        output1 = self.basic_unit1(x)  # 经过第一个基本单元
        output2 = self.basic_unit2(x)  # 经过第二个基本单元
        combined_output = output1 + output2  # 将两个输出相加
        return combined_output


import torch
import torch.nn as nn


class AnticipationModule(nn.Module):
    """
    Anticipation Module: Integrates deep-level features from the brain analysis module
    to produce final forecasts based on the outputs from the brain analysis module.
    """

    def __init__(self):
        super(AnticipationModule, self).__init__()  # 调用父类构造函数

    def forward(self, predicted_modes):
        """
        计算重建预测值
        输入通道数：8
        输出通道数：1
        参数：
        predicted_modes: List[torch.Tensor] - 各个分解模式的预测值，形状为 (N, C, L)
        返回：
        torch.Tensor - 重建预测值，形状为 (N, 1)
        """
        # 将所有模式的预测值求和sum_result = torch.sum(tensor, dim=1)  # 结果形状为 (N, L)
        reconstructed_prediction = torch.sum(predicted_modes, dim=1)
        # print('reconstructed_prediction',reconstructed_prediction.shape)
        final_forecast = torch.sum(reconstructed_prediction, dim=1)  # 时间步长求和，结果现在是 (N,)
        # print('final_forecast', final_forecast.shape)
        # 在 C 维度上进行平均，确保最后的输出为 (N, 1)
        # todo 要不要求平均

        # final_forecast = final_forecast.mean(dim=1, keepdim=True)  # (N, 1)

        return final_forecast


# 示例：使用 AnticipationModule
# if __name__ == "__main__":
#     # 假设有8个模式的预测值
#     predicted_modes = [torch.rand(1) for _ in range(8)]  # 生成随机预测值作为示例
#
#     anticipation_module = AnticipationModule()
#     reconstructed_prediction = anticipation_module.forward(predicted_modes)
#
#     print("重建预测值：", reconstructed_prediction.item())
#     print("预测值的维度：", reconstructed_prediction.size())  # 或者使用 reconstructed_prediction.shape
class ModelEvaluator:
    def __init__(self, true_values, predicted_values):
        self.true_values = np.array(true_values)
        self.predicted_values = np.array(predicted_values)

    def rmse(self):
        """计算均方根误差 (RMSE)"""
        return np.sqrt(np.mean((self.true_values - self.predicted_values) ** 2))

    def mae(self):
        """计算平均绝对误差 (MAE)"""
        return np.mean(np.abs(self.true_values - self.predicted_values))

    def sse(self):
        """计算误差平方和 (SSE)"""
        return np.sum((self.true_values - self.predicted_values) ** 2)

    def mape(self):
        """计算平均绝对百分比误差 (MAPE)"""
        return np.mean(np.abs((self.true_values - self.predicted_values) / self.true_values)) * 100

    def tic(self):
        """计算Theil不平等系数 (TIC)"""
        return np.sqrt(np.sum((self.true_values - self.predicted_values) ** 2) / np.sum(self.true_values ** 2))

    def pmape(self):
        """计算带偏差的平均绝对百分比误差 (PMAPE)"""
        return np.mean(np.abs((self.predicted_values - self.true_values) / self.true_values)) * 100

    def dm_test(self):
        """计算Diebold-Mariano (DM) 测试统计量"""
        d = self.predicted_values - self.true_values
        d_squared = d ** 2
        n = len(d)

        mean_d_squared = np.mean(d_squared)
        var_d_squared = np.var(d_squared)

        dm_statistic = np.sqrt(n) * mean_d_squared / np.sqrt(var_d_squared)
        return dm_statistic

    def cid(self):
        """计算CID (Cumulative Information Distribution)"""
        errors = self.true_values - self.predicted_values
        return np.sum(np.abs(errors)) / np.sum(np.abs(self.true_values))


from torch.utils.data import Dataset


class cmip_dataset(Dataset):
    def __init__(self, datax, datay):
        super().__init__()
        self.input_wh = datax
        self.target_wh = datay

    def GetDataShape(self):
        return {'wh input': self.input_wh.shape,
                'wh target': self.target_wh.shape}

    def __len__(self, ):
        return self.input_wh.shape[0]

    def __getitem__(self, idx):
        # print(f"Accessing index: {idx}")
        return self.input_wh[idx], self.target_wh[idx]


from torch.utils.data import DataLoader
from torch import optim


class ModelTrainer:
    def __init__(self, train_data, val_data, test_data, input_channels, output_channels,
                 batch_size=40, num_epochs=200, initial_lr=0.01, min_lr=0.00001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.initial_lr = initial_lr
        self.min_lr = min_lr

        # 初始化模型
        self.brain_analysis_module = BrainAnalysisModule(input_channels, output_channels).to(self.device)
        self.anticipation_module = AnticipationModule().to(self.device)

        # 定义损失函数和优化器
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(list(self.brain_analysis_module.parameters()) +
                                    list(self.anticipation_module.parameters()),
                                    lr=self.initial_lr, weight_decay=0.01)

    # def save_model(self, path):
    #     # torch.save({'net': self.network.state_dict(),
    #     #             'optimizer': self.opt.optimizer.state_dict()}, path)
    def train(self):
        print('loading train dataloader')
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        print('loading eval dataloader')
        val_loader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)
        learning_rate = self.initial_lr
        decay_steps = self.num_epochs // 10

        for epoch in range(self.num_epochs):
            print('\nepoch: {0}'.format(epoch + 1))
            # 训练模式
            self.brain_analysis_module.train()
            self.anticipation_module.train()
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                # 每10个批次打印一次
                if (batch_idx + 1) % 10 == 0:
                    print(f"第 {batch_idx + 1} 批次已训练完成")
                inputs = inputs.to(self.device).float()  # 获取输入数据并转移到 GPU
                targets = targets.to(self.device).float()  # 获取目标数据并转移到 GPU

                # 前向传播
                y_hat = self.brain_analysis_module(inputs)
                # print("y_hat shape",y_hat.shape)
                final_output = self.anticipation_module(y_hat)
                # print("final_output shape", final_output.shape,"target shape",targets.shape)
                # 计算损失
                loss = self.criterion(final_output, targets)  # 使用 targets 计算损失
                # print(final_output[0],targets[0])

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # 学习率衰减
            if epoch % decay_steps == 0 and learning_rate > self.min_lr:
                learning_rate *= 0.5
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = learning_rate

            # 验证模式
            self.validate(val_loader)

            # 打印损失
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f}')

    def validate(self, val_loader):
        self.brain_analysis_module.eval()  # 设置评估模式
        self.anticipation_module.eval()  # 设置评估模式
        val_loss = 0

        with torch.no_grad():  # 禁用梯度计算，节省内存和加速验证
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device).float()  # 获取输入数据并转移到 GPU
                targets = targets.to(self.device).float()  # 获取标签并转移到 GPU

                # 前向传播
                y_hat = self.brain_analysis_module(inputs)
                final_output = self.anticipation_module(y_hat)

                # 计算损失
                loss = self.criterion(final_output, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f'Validation Loss: {avg_val_loss:.4f}')
