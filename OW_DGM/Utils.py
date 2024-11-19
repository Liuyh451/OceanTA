import torch
import torch.nn as nn


import torch
import torch.nn as nn

class ContextualEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, gru_hidden_size):
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

        # GRU 编码器（共享的）
        self.gru = nn.GRU(input_dim + embedding_dim, gru_hidden_size, batch_first=True)

        # 最大池化层（用于聚合浮标的上下文向量）
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, obs_data, buoy_ids):
        """
        前向传播。

        Args:
            obs_data: 滑动窗口后的观测数据，形状为 (num_buoys * num_windows, window_size, input_dim)。
            buoy_ids: 浮标 ID，形状为 (num_buoys,)。

        Returns:
            c: 全局上下文向量，形状为 (512, 1, 1)。
        """
        num_buoys = buoy_ids.shape[0]
        num_windows = obs_data.size(0) // num_buoys  # 每个浮标的窗口数
        window_size = obs_data.size(1)

        # Step 1: 浮标 ID 嵌入
        buoy_embeddings = self.embedding(buoy_ids)  # (num_buoys, embedding_dim)
        buoy_embeddings = buoy_embeddings.unsqueeze(1).expand(-1, num_windows, -1)  # (num_buoys, num_windows, embedding_dim)
        buoy_embeddings = buoy_embeddings.reshape(-1, buoy_embeddings.size(-1))  # (num_buoys * num_windows, embedding_dim)

        # Step 2: 将浮标嵌入与窗口数据拼接
        concatenated_data = torch.cat([obs_data, buoy_embeddings.unsqueeze(1).expand(-1, window_size, -1)], dim=-1)

        # Step 3: 通过 GRU 编码每个浮标的窗口片段
        _, gru_hidden_states = self.gru(concatenated_data)  # (1, num_buoys * num_windows, gru_hidden_size)
        buoy_embeddings = gru_hidden_states.squeeze(0)  # (num_buoys * num_windows, gru_hidden_size)

        # Step 4: 恢复浮标维度，并进行最大池化
        buoy_embeddings = buoy_embeddings.reshape(num_buoys, num_windows, -1)  # (num_buoys, num_windows, gru_hidden_size)
        max_pooled = self.max_pool(buoy_embeddings.transpose(1, 2)).squeeze(-1)  # (num_buoys, gru_hidden_size)

        # Step 5: 聚合所有浮标，并调整为 (512, 1, 1)
        c = max_pooled.mean(dim=0, keepdim=True).squeeze(0).unsqueeze(-1).unsqueeze(-1)  # (512, 1, 1)

        return c

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
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
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

    def forward(self, x):
        x = self.A(x)  # 通过第一个模块A
        x = self.A2(x)  # 通过第二个模块A
        x = self.A3(x)  # 通过第三个模块A
        x = self.A4(x)  # 通过第四个模块A
        x = self.global_avg_pool(x)  # 全局平均池化
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)  # 通过全连接层输出
        return x


# 定义CVAE模型（生成器部分）
class CVAE(nn.Module):
    def __init__(self, context_encoder, encoder, decoder):
        super(CVAE, self).__init__()
        self.context_encoder = context_encoder
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, X, c):
        # 从上下文编码器获得条件向量 c
        c_encoded = self.context_encoder(c)

        # 编码器输出潜在变量 z
        z_mean, z_log_var = self.encoder(X, c_encoded)

        # 使用重参数化技巧生成潜在变量 z
        z = self.reparameterize(z_mean, z_log_var)

        # 解码器生成重建的波浪场
        X_hat = self.decoder(z, c_encoded)

        return X_hat, z_mean, z_log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


# 定义鉴别器

# 损失函数：对抗损失、重建损失、KL损失
def bce_loss(output, target):
    criterion = nn.BCELoss()
    return criterion(output, target)


def mse_loss(X, X_hat):
    return nn.MSELoss()(X, X_hat)


def kl_loss(mu, log_var):
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

# # 测试模型的构建
# if __name__ == "__main__":
#     model = Discriminator()
#     input_tensor = torch.randn(1, 3, 128, 128)  # 输入张量（batch_size=1, channels=3, height=128, width=128）
#     output = model(input_tensor)
#     print(output.shape)  # 输出形状，应该是 (1, 1)

# 示例代码
# if __name__ == "__main__":
#     # 假设编码器输出的 μ 和 σ
#     batch_size = 8
#     latent_dim = 512
#     mu = torch.randn(batch_size, latent_dim, 1, 1)  # (batch_size, 512, 1, 1)
#     sigma = torch.randn(batch_size, latent_dim, 1, 1)  # (batch_size, 512, 1, 1)
#
#     # 初始化解码器并前向传播
#     decoder = WaveFieldDecoder(latent_dim=latent_dim, output_channels=4)
#     wave_field = decoder(mu, sigma)
#
#     print("Wave field shape:", wave_field.shape)  # 输出: (batch_size, 4, 128, 128)

# # 示例代码
# if __name__ == "__main__":
#     # 波场输入 (batch_size, 4, 128, 128)
#     batch_size = 8
#     wave_field = torch.randn(batch_size, 4, 128, 128)
#
#     # 浮标上下文向量输入 (batch_size, context_dim)
#     context_vector = torch.randn(batch_size, 512)
#
#     # 初始化模型并前向传播
#     encoder = WaveFieldEncoderWithBuoy(input_channels=4, context_dim=512, latent_dim=512)
#     mu, sigma = encoder(wave_field, context_vector)
#
#     print("Mu shape:", mu.shape)  # 输出: (batch_size, 512, 1, 1)
#     print("Sigma shape:", sigma.shape)  # 输出: (batch_size, 512, 1, 1)
