import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import torch
import torch.nn as nn


class BasicUnit(nn.Module):
    """
        提取长短期特征
        输入通道数：32
        输出通道数：32
        参数：
        input_channels, output_channels, data
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
    def __init__(self, input_channels, output_channels, time_step):
        super(BrainAnalysisModule, self).__init__()
        self.fc_input_channels = output_channels * time_step
        self.fc_output_channels = 1
        self.basic_unit1 = BasicUnit(input_channels, output_channels)  # 第一个基本单元
        self.basic_unit2 = BasicUnit(output_channels, output_channels)  # 第二个基本单元
        self.fc = nn.Linear(self.fc_input_channels, self.fc_output_channels)  # 添加全连接层
        # self.fc = nn.Conv1d(output_channels, 8, kernel_size=1)  # 1x1卷积层替代全连接层

    def forward(self, x):
        output1 = self.basic_unit1(x)  # 经过第一个基本单元
        print("output1", output1.shape)
        output2 = self.basic_unit2(output1)  # 以output1作为第二个单元的输入
        print("output2", output2.shape)
        output2_flat = output2.view(output2.size(0), -1)  # 将输出展平成2D形状，适用于全连接层
        print("output2_flat", output2_flat.shape)
        final_output = self.fc(output2_flat)  # 经过全连接层
        # final_output = self.fc(output2)  # 经过全连接层
        return final_output


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
        # 在时间维度求和，获得各模式的预测值
        forecast_mode = torch.sum(predicted_modes, dim=2)
        # 将所有模式的预测值求和sum_result = torch.sum(tensor, dim=1)  # 结果形状为 (N, L)
        reconstructed_prediction = torch.sum(predicted_modes, dim=1)
        # print('reconstructed_prediction',reconstructed_prediction.shape)
        final_forecast = torch.sum(reconstructed_prediction, dim=1)  # 时间步长求和，结果现在是 (N,)
        # print('final_forecast', final_forecast.shape)
        # 在 C 维度上进行平均，确保最后的输出为 (N, 1)
        # todo 要不要求平均
        # final_forecast = final_forecast.mean(dim=1, keepdim=True)  # (N, 1)

        return final_forecast, forecast_mode


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

    def save_model(self, path):
        torch.save({'net': self.brain_analysis_module.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, path)

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
                if not isinstance(inputs, torch.Tensor):
                    raise ValueError(f"Expected inputs to be a torch.Tensor but got {type(inputs)} instead.")
                inputs = inputs.to(self.device).float()  # 获取输入数据并转移到 GPU
                targets = targets.to(self.device).float()  # 获取目标数据并转移到 GPU

                # 前向传播
                y_hat = self.brain_analysis_module(inputs)
                # print("y_hat shape",y_hat.shape)
                final_output, _ = self.anticipation_module(y_hat)
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
                final_output, _ = self.anticipation_module(y_hat)

                # 计算损失
                loss = self.criterion(final_output, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f'Validation Loss: {avg_val_loss:.4f}')

    def test_model(self):
        """
        测试模型并将预测值和真实值保存到 .npy 文件中

        参数：
        - model: 已训练的模型
        - test_loader: 测试集的数据加载器
        - device: 运行模型的设备（'cpu' 或 'cuda'）
        - save_path: 保存预测结果的文件路径，默认为 'predictions.npy'
        """
        print('loading eval dataloader')
        test_loader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
        self.brain_analysis_module.eval()  # 设置评估模式
        self.anticipation_module.eval()  # 设置评估模式
        all_predictions = []
        all_targets = []
        all_predictions_modes = []

        with torch.no_grad():  # 禁用梯度计算，节省内存
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device).float()  # 获取输入数据并转移到 GPU
                targets = targets.to(self.device).float()  # 获取标签并转移到 GPU
                # 模型进行前向传播，得到预测值
                y_hat = self.brain_analysis_module(inputs)
                predictions, predictions_mode = self.anticipation_module(y_hat)

                # 将预测值和真实值保存
                all_predictions.append(predictions.cpu().numpy())  # 将结果从GPU移到CPU，并转换为numpy格式
                all_targets.append(targets.cpu().numpy())
                all_predictions_modes.append(predictions_mode.cpu().numpy())

        # 转换为 numpy 数组
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_predictions_modes = np.concatenate(all_predictions_modes, axis=0)
        # 保存到 .npy 文件，确保预测值和真实值一一对应
        # 分别保存预测值和真实值
        pred_path = './data/predictions.npy'
        target_path = './data/targets.npy'
        pre_mode_path = './data/pre_mode.npy'
        np.save(pred_path, all_predictions)
        np.save(target_path, all_targets)
        np.save(pre_mode_path, all_predictions_modes)

        print(f'预测值已保存到 {pred_path}')
        print(f'真实值已保存到 {target_path}')
