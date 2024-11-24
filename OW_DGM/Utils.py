import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


class ContextualEncoder(nn.Module):
    def __init__(self, input_dim=4, embedding_dim=16, gru_hidden_size=512):
        """
        上下文编码器。

        Args:
            input_dim: 每个时间步的输入特征维度（例如 4）。
            embedding_dim: 浮标 ID 的嵌入维度。
            gru_hidden_size: GRU 隐藏层大小。
        """
        super(ContextualEncoder, self).__init__()

        # 浮标 ID 嵌入层
        self.embedding = nn.Embedding(num_embeddings=100, embedding_dim=embedding_dim)

        # GRU 编码器（共享的）输入参数：(batch_size, sequence_length, input_size)）sequence_length表示序列的长度 input_size是每个时间步输入特征的维度
        self.gru = nn.GRU(input_dim + embedding_dim, gru_hidden_size, batch_first=True)

        # 最大池化层（用于聚合浮标的上下文向量）
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, obs_data):
        """
        前向传播。

        Args:
            obs_data: 观测数据，形状为 (batch_size, num_buoys, window_size, input_dim)。
            buoy_ids: 浮标 ID，形状为 (num_buoys,)，用于嵌入。

        Returns:
            c: 全局上下文向量，形状为 (gru_hidden_size, 1, 1)。
        """
        # 确保 buoy_ids 和 obs_data 在相同的设备上
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        buoy_ids = torch.tensor([0, 1, 2, 3, 4])  # 注意方括号
        buoy_ids = buoy_ids.to(device)
        obs_data = obs_data.to(device)

        num_buoys = buoy_ids.shape[0]
        window_size = obs_data.size(2)

        # Step 1: 浮标 ID 嵌入
        # Step 1.1 构建 ID 嵌入
        buoy_embeddings = self.embedding(buoy_ids)  # (num_buoys, embedding_dim)
        buoy_embeddings = buoy_embeddings.unsqueeze(1).expand(-1, window_size,
                                                              -1)  # (num_buoys, window_size, embedding_dim)，即(5, 3, 16)
        # Step 1.2 添加batch_size维度
        batch_size = obs_data.shape[0]
        buoy_embeddings = buoy_embeddings.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
        # 此时形状变为(batch_size, num_buoys, window_size, embedding_dim)，即(32, 5, 3, 16)
        # Step 1.3 合并维度
        # 形状变为(batch_size * num_buoys, window_size, embedding_dim)
        buoy_embeddings = buoy_embeddings.reshape(batch_size * num_buoys, window_size, -1)
        # 此时形状变为(32 * 5, 3, 16)，符合(batch_size , window_size* num_buoys, embedding_dim)的形状要求

        # Step 2: 将浮标嵌入与窗口数据拼接
        # 确保拼接时数据的维度对齐
        obs_data_reshaped = obs_data.reshape(batch_size * num_buoys, window_size, -1)
        concatenated_data = torch.cat([obs_data_reshaped, buoy_embeddings],
                                      dim=-1)  # (batch_size * num_buoys, window_size, input_dim + embedding_dim)即(32*5,3, 20)
        # Step 3: 通过 GRU 编码每个浮标的窗口片段
        # 输出隐藏状态 output (batch_size, sequence_length, num_directions * hidden_size)，num_directions表示方向数，单向时为1，双向时为2
        # 最终隐藏状态 h_n (num_layers * num_directions, batch_size, hidden_size) num_layers是GRU的层数
        _, gru_hidden_states = self.gru(concatenated_data)  # (batch_size * num_buoys, window_size, gru_hidden_size)
        # gru_hidden_states torch.Size([1, 160, 512])
        buoy_embeddings = gru_hidden_states[-1, :, :]  # 取最后一个时间步的隐藏状态 (batch_size * num_buoys, gru_hidden_size)
        # Step 4: 恢复浮标维度
        buoy_embeddings = buoy_embeddings.reshape(batch_size, num_buoys, 512)  # (batch_size，num_buoys, gru_hidden_size)
        #  Step 5: 聚合所有浮标，并调整为 (batch_size，gru_hidden_size, 1, 1)
        context_vector = self.max_pool(buoy_embeddings.transpose(1, 2)).unsqueeze(-1)
        return context_vector


class WaveFieldEncoderWithBuoy(nn.Module):
    def __init__(self, input_channels=4, context_dim=512, latent_dim=512):
        """
        波场编码器，融合浮标上下文向量
        :param input_channels: 波场输入的通道数（默认为4）
        :param context_dim: 浮标上下文向量的维度（默认为512）
        :param latent_dim: 输出的潜在向量（μ和σ）的维度（默认为512）
        """
        super(WaveFieldEncoderWithBuoy, self).__init__()

        # 小模块 A：卷积 + BatchNorm + ReLU + MaxPooling
        self.module_a = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 下采样
        )

        self.module_a1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.module_a2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.module_a3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 小模块 B：卷积 + BatchNorm + ReLU + 全局平均池化
        self.module_b = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))  # 全局平均池化
        )

        # 1x1 卷积生成 μ 和 σ
        self.conv_mu = nn.Conv2d(in_channels=1024, out_channels=latent_dim, kernel_size=1)
        self.conv_sigma = nn.Conv2d(in_channels=1024, out_channels=latent_dim, kernel_size=1)

    def forward(self, wave_field, context_vector):
        """
        前向传播
        :param wave_field: 波场输入 (batch_size, 4, 128, 128)
        :param context_vector: 浮标上下文向量 (batch_size, context_dim)
        :return: μ 和 σ (batch_size, latent_dim, 1, 1)
        """
        # 1. 通过小模块 A 提取波场的卷积特征
        wave_features = self.module_a(wave_field)  # (batch_size, 64, 64, 64)
        wave_features = self.module_a1(wave_features)  # (batch_size, 128, 32, 32)
        wave_features = self.module_a2(wave_features)  # (batch_size, 256, 16, 16)
        wave_features = self.module_a3(wave_features)  # (batch_size, 512, 8, 8)

        # 2. 通过小模块 B 提取最终的特征向量
        wave_features = self.module_b(wave_features)  # (batch_size, 512, 1, 1)

        # 3. 将浮标上下文向量扩展到与波场特征相同的空间尺寸
        batch_size = wave_features.shape[0]
        context_vector = context_vector.view(batch_size, -1, 1, 1)  # (batch_size, context_dim, 1, 1)

        # 4. 融合波场特征和上下文向量
        combined_features = torch.cat([wave_features, context_vector], dim=1)  # (batch_size, 512 + context_dim, 1, 1)

        # 5. 通过两个1x1卷积生成 μ 和 σ
        mu = self.conv_mu(combined_features)  # (batch_size, latent_dim, 1, 1)
        sigma = self.conv_sigma(combined_features)  # (batch_size, latent_dim, 1, 1)

        return mu, sigma


class WaveFieldDecoder(nn.Module):
    def __init__(self, latent_dim=512, output_channels=4):
        """
        波场解码器，将潜在向量解码为海浪场
        :param latent_dim: 潜在向量的维度（默认为512）
        :param output_channels: 输出波场的通道数（默认为4）
        """
        super(WaveFieldDecoder, self).__init__()

        # 小模块 C：反卷积 + BatchNorm + ReLU
        self.module_c1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_dim * 2, out_channels=1024, kernel_size=4, stride=1, padding=0),
            # (1024, 1, 1) -> (1024, 4, 4)
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        self.module_c2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            # (1024, 4, 4) -> (512, 8, 8)
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.module_c3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            # (512, 8, 8) -> (256, 16, 16)
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.module_c4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            # (256, 16, 16) -> (128, 32, 32)
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.module_c5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            # (128, 32, 32) -> (64, 64, 64)
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # 小模块 D：反卷积 + Tanh
        self.module_d = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=output_channels, kernel_size=4, stride=2, padding=1),
            # (64, 64, 64) -> (4, 128, 128)
            nn.Tanh()
        )

    def forward(self, mu, sigma):
        """
        前向传播
        :param mu: 编码器生成的均值向量 (batch_size, latent_dim, 1, 1)
        :param sigma: 编码器生成的方差向量 (batch_size, latent_dim, 1, 1)
        :return: 解码后的波场 (batch_size, output_channels, 128, 128)
        """
        # 1. 将 μ 和 σ 融合为一个向量
        combined_vector = torch.cat([mu, sigma], dim=1)  # (batch_size, latent_dim * 2, 1, 1)

        # 2. 通过小模块 C 和小模块 D 逐步解码
        x = self.module_c1(combined_vector)  # (batch_size, 1024, 4, 4)
        x = self.module_c2(x)  # (batch_size, 512, 8, 8)
        x = self.module_c3(x)  # (batch_size, 256, 16, 16)
        x = self.module_c4(x)  # (batch_size, 128, 32, 32)
        x = self.module_c5(x)  # (batch_size, 64, 64, 64)
        output = self.module_d(x)  # (batch_size, 4, 128, 128)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # 模块A
        self.A = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 第二个模块A
        self.A2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 第三个模块A
        self.A3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 第四个模块A
        self.A4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 全局平均池化层
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # 输出层
        self.fc = nn.Linear(512, 1)
        # todo 激活层 确保结果在0到1之间
        self.sg = nn.Sigmoid()  # 确保输出在 [0, 1]

    def forward(self, x):
        x = self.A(x)  # 通过第一个模块A
        x = self.A2(x)  # 通过第二个模块A
        x = self.A3(x)  # 通过第三个模块A
        x = self.A4(x)  # 通过第四个模块A
        x = self.global_avg_pool(x)  # 全局平均池化
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)  # 通过全连接层输出
        x = self.sg(x)
        return x

class Loss:
    def reparameterize(self, mu, std):
        eps = torch.randn_like(std)
        return mu + eps * std

    # 损失函数：对抗损失、重建损失、KL损失
    def bce_loss(self,output, target):
        criterion = nn.BCELoss()
        return criterion(output, target)

    def mse_loss(self,X, X_hat):
        return nn.MSELoss()(X, X_hat)

    def kl_loss(self,mu, sigma):
        """
            计算 KL 损失。

            参数：
            - mu: torch.Tensor，编码器输出的均值 (batch_size, latent_dim)。
            - var: torch.Tensor，编码器输出的方差 (batch_size, latent_dim)。

            返回：
            - kl_loss: KL 损失，标量。
            """
        return 0.5 * torch.sum(mu.pow(2) + sigma.pow(2) - 1 - torch.log(sigma.pow(2))) / mu.size(1)
class Inference:
    def __init__(self, contextual_encoder, decoder, device):
        """
        初始化推理模块。
        :param contextual_encoder: 已训练的上下文编码器模型
        :param decoder: 已训练的解码器模型
        :param device: 计算设备 (cpu 或 cuda)
        """
        self.contextual_encoder = contextual_encoder.to(device)
        self.decoder = decoder.to(device)
        self.device = device

    def generate_wave_field(self, wave_buoy_data):
        """
        生成波场数据。
        :param wave_buoy_data: 浮标观测数据，形状为 (batch_size, channels, height, width)
        :param num_samples: 每个输入生成的波场样本数量
        :return: 生成的波场数据，形状为 (batch_size * num_samples, channels, height, width)
        """
        self.contextual_encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            # 获取上下文向量 c
            wave_buoy_data = wave_buoy_data.to(self.device)
            context = self.contextual_encoder(wave_buoy_data)  # 输出形状为 (batch_size, context_dim)

            # 生成 z 并解码
            batch_size = wave_buoy_data.size(0)
            wave_fields = []
            num_samples=1
            for _ in range(num_samples):
                # 从标准正态分布 N(0, I) 中采样 z
                z = torch.randn_like(context, device=self.device)# z 形状与 context 相同
                # 解码生成波场
                generated_wave_field = self.decoder(z, context)  # 形状为 (batch_size, channels, height, width)
                wave_fields.append(generated_wave_field)

            # 合并生成的波场样本
            wave_fields = torch.cat(wave_fields, dim=0)  # 合并生成的样本
            return wave_fields

import numpy as np


class EvaluationMetrics:
    def __init__(self, predictions, ground_truth):
        """
        初始化函数，接收预测值和真实值数据。

        参数：
        - predictions: numpy数组，模型的预测值，形状假设为 [num_examples, feature_dim, grid_height, grid_width]，这里简化先以二维理解为 [num_examples, num_grid_cells]。
        - ground_truth: numpy数组，对应的真实值，形状与预测值相同。
        """
        self.predictions = predictions
        self.ground_truth = ground_truth
        self.num_examples, self.num_grid_cells = self.predictions.shape[:2]

    def _flatten_data(self, data):
        """
        将多维数据展平，方便后续统一计算（假设最后两维看作网格单元维度，展平这些维度）。

        参数：
        - data: numpy数组，要展平的数据。

        返回：
        - 展平后的一维数组。
        """
        return data.reshape(self.num_examples * self.num_grid_cells)

    def root_mean_square_error(self):
        """
        计算均方根误差（RMSE）。

        返回：
        - RMSE值。
        """
        flattened_predictions = self._flatten_data(self.predictions)
        flattened_ground_truth = self._flatten_data(self.ground_truth)
        diff = flattened_predictions - flattened_ground_truth
        squared_diff = diff ** 2
        mean_squared_diff = np.mean(squared_diff)
        return np.sqrt(mean_squared_diff)

    def _compute_crps_per_grid_cell(self, predictions_per_cell, ground_truth_per_cell):
        """
        计算单个网格单元的连续排序概率评分（CRPS）。

        参数：
        - predictions_per_cell: 对应单个网格单元的预测值数组（这里假设已经是从预测分布中抽取的样本形式）。
        - ground_truth_per_cell: 对应单个网格单元的真实值。

        返回：
        - 单个网格单元的CRPS值。
        """
        num_samples = predictions_per_cell.shape[0]
        # 计算 E|P - Oi|
        abs_diff = np.abs(predictions_per_cell - ground_truth_per_cell)
        expected_abs_diff = np.mean(abs_diff)
        # 计算 E|P - P'|
        diff_between_samples = np.abs(predictions_per_cell[:, np.newaxis] - predictions_per_cell[np.newaxis, :])
        expected_diff_between_samples = np.mean(diff_between_samples) / 2
        return expected_abs_diff - expected_diff_between_samples

    def continuous_ranked_probability_score(self):
        """
        计算连续排序概率评分（CRPS）。

        返回：
        - CRPS值。
        """
        flattened_predictions = self._flatten_data(self.predictions)
        flattened_ground_truth = self._flatten_data(self.ground_truth)
        predictions_reshaped = flattened_predictions.reshape(-1, 10)  # 假设使用10个集合成员作为样本，按样本维度重塑
        ground_truth_reshaped = flattened_ground_truth.reshape(-1, 1)
        crps_per_cell = np.array([self._compute_crps_per_grid_cell(predictions_reshaped[i], ground_truth_reshaped[i])
                                 for i in range(self.num_examples * self.num_grid_cells)])
        return np.mean(crps_per_cell)
class TimeSeriesDataset(Dataset):
    def __init__(self, A, B, batch_size, time_step_A=3, time_step_B_ratio=3):
        """
        自定义数据集。

        参数：
        - A: torch.Tensor，形状为 (5, T, 4) 的时间序列 A。
        - B: torch.Tensor，形状为 (T, 4, 128, 128) 的时间序列 B。
        - batch_size: int，批次大小。
        - time_step_A: int，每次从 A 读取的时间步长。
        - time_step_B_ratio: int，从 B 读取的时间步与 A 的时间步长比率。
        """
        self.A = A
        self.B = B
        self.batch_size = batch_size
        self.time_step_A = time_step_A
        self.time_step_B_ratio = time_step_B_ratio
        self.num_samples = A.shape[1]  # A 的时间步数

    def __len__(self):
        # 计算总批次数
        total_batches = (self.num_samples + self.time_step_A - 1) // self.time_step_A
        return (total_batches + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        # 每个批次包含的时间段索引范围
        batch_start_idx = idx * self.batch_size * self.time_step_A
        batch_end_idx = min((idx + 1) * self.batch_size * self.time_step_A, self.num_samples)

        # 按时间步分段获取 A 和 B 的时间片
        A_batches = []
        B_batches = []

        for start_idx in range(batch_start_idx, batch_end_idx, self.time_step_A):
            end_idx = min(start_idx + self.time_step_A, self.num_samples)
            A_segment = self.A[:, start_idx:end_idx, :]  # (5, time_step_A, 4)
            B_segment = self.B[start_idx // self.time_step_B_ratio:end_idx // self.time_step_B_ratio]  # 对应 B 的时间步
            A_batches.append(A_segment)
            B_batches.append(B_segment)

        # 合并为 batch 维度
        A_batches = torch.stack(A_batches, dim=0)  # (batch_size, 5, time_step_A, 4)
        B_batches = torch.stack(B_batches, dim=0).squeeze(1)  # (batch_size, 4, 128, 128)

        return A_batches, B_batches