import math
import torch
import torch.nn as nn


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
        print("buoy_embeddings", buoy_embeddings.shape)
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


# 定义CVAE模型（生成器部分）
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
