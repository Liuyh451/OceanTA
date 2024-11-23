import numpy as np
import wave_filed_data_prepare
import Utils
from Utils import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


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


def train_cvae_al(
        context_encoder, encoder, decoder, discriminator,
        dataloader, device, latent_dim=128,
        lambda_kl=0.1, lambda_adv=0.01, lr=1e-4,
        epochs=50, verbose=True
):
    """
    训练 CVAE-AL 模型的函数。

    参数：
    - context_encoder: nn.Module，上下文编码器。
    - encoder: nn.Module，编码器。
    - decoder: nn.Module，解码器。
    - discriminator: nn.Module，判别器。
    - dataloader: torch.utils.data.DataLoader，数据加载器。
    - device: torch.device，设备（CPU/GPU）。
    - latent_dim: int，潜在向量的维度。
    - lambda_kl: float，KL 损失的权重。
    - lambda_adv: float，对抗损失的权重。
    - lr: float，学习率。
    - epochs: int，训练轮数。
    - verbose: bool，是否打印训练日志。

    返回：
    - context_encoder, encoder, decoder, discriminator：训练后的模型。
    """
    # 损失函数
    loss = Loss()
    mse_loss = nn.MSELoss()  # 重构损失
    bce_loss = nn.BCELoss()  # 二分类交叉熵损失

    # 优化器
    generator_params = list(context_encoder.parameters()) + \
                       list(encoder.parameters()) + \
                       list(decoder.parameters())
    discriminator_params = discriminator.parameters()

    optimizer_G = optim.Adam(generator_params, lr=lr)
    optimizer_D = optim.Adam(discriminator_params, lr=lr)

    # 将模型和损失函数移动到设备
    context_encoder.to(device)
    encoder.to(device)
    decoder.to(device)
    discriminator.to(device)

    for epoch in range(epochs):
        context_encoder.train()
        encoder.train()
        decoder.train()
        discriminator.train()

        for batch_idx, (buoy_data, wave_field) in enumerate(dataloader):
            print(f"Batch {batch_idx + 1}:", f"buoy_data_batch shape: {buoy_data.shape}",
                  f"wave_field_batch shape: {wave_field.shape}")
            # 将数据移动到设备
            buoy_data = buoy_data.to(device)  # 浮标数据 (batch_size, 5, 3, 4)
            wave_field = wave_field.to(device)  # 波场数据 (batch_size, 3, 128, 128)

            # 生成上下文向量 c
            context_vector = context_encoder(buoy_data)  # (batch_size, latent_dim)

            # 编码器生成 z 的均值和标准差
            z_mean, z_var = encoder(wave_field, context_vector)  # (batch_size, latent_dim)
            z = loss.reparameterize(z_mean, z_var)  # 重参数化采样
            # 解码器生成波场
            reconstructed_wave_field = decoder(z, context_vector)  # (batch_size, 3, 128, 128)
            # 重构损失
            l_rec = loss.mse_loss(wave_field, reconstructed_wave_field)
            # KL损失
            l_kl = loss.kl_loss(z_mean, z_var)
            fake_preds = discriminator(reconstructed_wave_field)  # 判别器对生成波场的判别
            print("Fake predictions range:", fake_preds.min().item(), fake_preds.max().item())
            l_adv_G = loss.bce_loss(fake_preds, torch.ones_like(fake_preds))  # 对抗损失（生成器）
            # print("Real predictions range:", real_preds.min().item(), real_preds.max().item())


            l_G = l_rec + lambda_kl * l_kl + lambda_adv * l_adv_G  # 总生成器损失
            # 优化生成器
            optimizer_G.zero_grad()
            l_G.backward()
            optimizer_G.step()

            # 计算判别器损失
            real_preds = discriminator(wave_field)  # 判别器对真实波场的判别
            fake_preds = discriminator(reconstructed_wave_field.detach())  # 判别器对生成波场的判别
            print("Real predictions range:", real_preds.min().item(), real_preds.max().item())
            print("Fake predictions range:", fake_preds.min().item(), fake_preds.max().item())

            l_adv_D = 0.5 * (bce_loss(real_preds, torch.ones_like(real_preds)) +
                             bce_loss(fake_preds, torch.zeros_like(fake_preds)))  # 判别器总损失

            # 优化判别器
            optimizer_D.zero_grad()
            l_adv_D.backward()
            optimizer_D.step()

            if verbose:
                print(f"Epoch [{epoch + 1}/{epochs}] Batch [{batch_idx + 1}/{len(dataloader)}]: "
                      f"Loss_G: {l_G.item():.4f}, Loss_D: {l_adv_D.item():.4f}, "
                      f"L_rec: {l_rec.item():.4f}, KL: {l_kl.item():.4f}")

    return context_encoder, encoder, decoder, discriminator


def replace_invalid_values(data):
    """
    将数据中值为 -32768、9999.0 或 NaN 的位置替换为 0。

    参数：
    - data: torch.Tensor，输入数据张量。

    返回：
    - 处理后的 torch.Tensor。
    """
    # 替换 NaN 值为 0
    data = torch.nan_to_num(data, nan=0.0)
    # 替换 -32768 和 9999.0 为 0
    data = torch.where((data == -32768) | (data == 9999.0), 0.0, data)
    return data


def count_nan_values(data):
    """
    统计数据中 NaN 值的总数。

    参数：
    - data: torch.Tensor，输入数据张量。

    返回：
    - NaN 值的数量。
    """
    return torch.isnan(data).sum().item()

def normalize(data):
    x_min = torch.min(data)
    x_max = torch.max(data)
    data = (data - x_min) / (x_max - x_min)
    return data
# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("loading swan data and buoy data........")
swan_data= torch.randn(270, 4, 128, 128)  # (T, 4, 128, 128)
buoy_data= torch.randn( 5, 810, 4)
swan_data=normalize(swan_data)
buoy_data=normalize(buoy_data)
# swan_data = np.load("E:/Dataset/met_waves/swan_data.npy")
# buoy_data = np.load('data/buoy_obs.npy')
print(torch.min(buoy_data), torch.max(buoy_data))
buoy_data = torch.tensor(buoy_data)
# swan_data = wave_filed_data_prepare.combine_monthly_data("/home/hy4080/met_waves/Swan_cropped/swanSula", 2017, 2018)
# swan_data = torch.tensor(swan_data).permute(0, 3, 1, 2)
# 处理 Swan 数据
swan_data = replace_invalid_values(swan_data)
# 查看 Swan 数据中的 NaN 值数量
# 处理 Buoy 数据
buoy_data = replace_invalid_values(buoy_data)
num_nan_swan = count_nan_values(swan_data)
print(f"Swan 数据中 NaN 值的数量: {num_nan_swan}")
# 查看 Buoy 数据中的 NaN 值数量
num_nan_buoy = count_nan_values(buoy_data)
print(f"Buoy 数据中 NaN 值的数量: {num_nan_buoy}")
print("swan data and buoy data shape", buoy_data.shape, swan_data.shape)
# 创建数据集和数据加载器
dataset = TimeSeriesDataset(buoy_data, swan_data, batch_size=32)
dataloader = DataLoader(dataset, batch_size=None, shuffle=False)
# 训练模型
context_encoder = Utils.ContextualEncoder()
encoder = Utils.WaveFieldEncoderWithBuoy()
decoder = Utils.WaveFieldDecoder()
discriminator = Discriminator()
trained_context_encoder, trained_encoder, trained_decoder, trained_discriminator = train_cvae_al(
    context_encoder, encoder, decoder, discriminator,
    dataloader, device, latent_dim=128,
    lambda_kl=0.1, lambda_adv=0.01, lr=1e-4,
    epochs=50, verbose=True
)
