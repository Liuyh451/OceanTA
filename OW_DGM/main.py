import os
import torch

import numpy as np

buoy_obs=np.load('data/buoy_obs.npy')

import torch
from torch.utils.data import Dataset, DataLoader

class BuoyDataset(Dataset):
    """
    自定义数据集，用于加载浮标观测数据进行训练。

    参数：
    - data: np.ndarray，形状为 (5, 113615, 4) 的浮标数据。
    """
    def __init__(self, data):
        # 将 NumPy 数据转换为 PyTorch Tensor
        self.data = torch.tensor(data, dtype=torch.float32)  # 形状 (5, 113615, 4)
        # 展平第一个维度，将浮标和时间序列合并
        self.data = self.data.view(-1, self.data.shape[2])  # 形状 (5 * 113615, 4)

    def __len__(self):
        """
        返回数据集的总样本数。
        """
        return self.data.shape[0]

    def __getitem__(self, idx):
        """
        获取指定索引的样本。
        """
        return self.data[idx]  # 返回形状为 (4,) 的特征向量

# 定义函数用于创建 DataLoader
def create_dataloader(data, batch_size=32, shuffle=True):
    """
    创建 DataLoader。

    参数：
    - data: np.ndarray，形状为 (5, 113615, 4) 的浮标数据。
    - batch_size: int，每个批次的样本数量，默认值为 32。
    - shuffle: bool，是否打乱数据，默认值为 True。

    返回：
    - DataLoader 对象。
    """
    dataset = BuoyDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# 示例调用
if __name__ == "__main__":
    # 加载数据
    buoy_obs = np.load('data/buoy_obs.npy')  # 假设数据形状为 (5, 113615, 4)

    # 创建 DataLoader
    batch_size = 64
    dataloader = create_dataloader(buoy_obs, batch_size=batch_size)

    # 测试 DataLoader
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx}: Shape {batch.shape}")
        if batch_idx == 1:  # 仅打印前两个批次
            break

# 训练过程
def train_cvae_al(cvae, discriminator, dataloader, optimizer_g, optimizer_d, lambda_kl=1.0, lambda_adv=1.0):
    cvae.train()
    discriminator.train()

    for X, c in dataloader:
        # 将数据和条件传入网络
        X, c = Variable(X), Variable(c)

        # 生成器训练
        optimizer_g.zero_grad()
        X_hat, z_mean, z_log_var = cvae(X, c)

        # 计算损失
        rec_loss = mse_loss(X, X_hat)
        kl_loss_val = kl_loss(z_mean, z_log_var)
        adv_loss_g = bce_loss(discriminator(X_hat), torch.ones_like(discriminator(X_hat)))

        loss_g = rec_loss + lambda_kl * kl_loss_val + lambda_adv * adv_loss_g
        loss_g.backward()
        optimizer_g.step()

        # 鉴别器训练
        optimizer_d.zero_grad()
        adv_loss_d = bce_loss(discriminator(X_hat), torch.zeros_like(discriminator(X_hat))) + \
                     bce_loss(discriminator(X), torch.ones_like(discriminator(X)))

        adv_loss_d.backward()
        optimizer_d.step()

        print(f"Generator loss: {loss_g.item()}, Discriminator loss: {adv_loss_d.item()}")


# 假设我们已经构建了上下文编码器、编码器、解码器和鉴别器，并且有训练数据集dataloader
context_encoder = ...
encoder = ...
decoder = ...
discriminator = Discriminator()

cvae = CVAE(context_encoder, encoder, decoder)

optimizer_g = optim.Adam(cvae.parameters(), lr=0.001)
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.001)

# 训练循环
train_cvae_al(cvae, discriminator, dataloader, optimizer_g, optimizer_d)