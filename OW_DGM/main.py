import wave_filed_data_prepare
import Utils
from Utils import *
import torch
import torch.optim as optim


def count_invalid(data):
    """
    统计数据中 NaN 值、-32768 和 9999.0 的总数，并返回它们的总和。

    参数：
    - data: torch.Tensor，输入数据张量。

    返回：
    - 这些特殊值数量的总和。
    """
    nan_count = torch.isnan(data).sum().item()
    minus_32768_count = torch.sum(data == -32768).item()
    nine_nine_nine_nine_count = torch.sum(data == 9999.0).item()
    return nan_count + minus_32768_count + nine_nine_nine_nine_count


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
            # 将原始波场中值为 0 的位置重建波场同样设置为 0,0在归一化之后是-1
            mask = (wave_field == -1)
            wave_field_batch = reconstructed_wave_field.clone()
            wave_field_batch[mask] = -1
            reconstructed_wave_field = wave_field_batch
            fake_preds = discriminator(reconstructed_wave_field)  # 判别器对生成波场的判别
            l_adv_G = loss.bce_loss(fake_preds, torch.ones_like(fake_preds))  # 对抗损失（生成器）

            l_G = l_rec + lambda_kl * l_kl + lambda_adv * l_adv_G  # 总生成器损失
            # 优化生成器
            optimizer_G.zero_grad()
            l_G.backward()
            optimizer_G.step()

            # 计算判别器损失
            real_preds = discriminator(wave_field)  # 判别器对真实波场的判别
            fake_preds = discriminator(reconstructed_wave_field.detach())  # 判别器对生成波场的判别

            l_adv_D = 0.5 * (loss.bce_loss(real_preds, torch.ones_like(real_preds)) +
                             loss.bce_loss(fake_preds, torch.zeros_like(fake_preds)))  # 判别器总损失

            # 优化判别器
            optimizer_D.zero_grad()
            l_adv_D.backward()
            optimizer_D.step()

            if verbose:
                print(f"Epoch [{epoch + 1}/{epochs}] Batch [{batch_idx + 1}/{len(dataloader)}]: "
                      f"Loss_G: {l_G.item():.4f}, Loss_D: {l_adv_D.item():.4f}, "
                      f"L_rec: {l_rec.item():.4f}, KL: {l_kl.item():.4f}")

    return context_encoder, encoder, decoder, discriminator


# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("loading swan data and buoy data........")
buoy_data = np.load('data/buoy_data_train.npy')
buoy_data = torch.tensor(buoy_data)
swan_data = wave_filed_data_prepare.combine_monthly_data("/home/hy4080/met_waves/Swan_cropped/swanSula", 2017,
                                                         2019)
# 将双精度张量转为单精度
swan_data = torch.tensor(swan_data).float()
print(swan_data.dtype)  # 输出 torch.float32
# # 查看 Swan 数据中的 NaN 值数量
# num_nan_swan = count_invalid(swan_data)
# num_nan_buoy = count_invalid(buoy_data)
# print(f"Swan和Buoy数据中无效值的数量分别为: {num_nan_swan},{num_nan_buoy}")
print("swan data and buoy data shape", buoy_data.shape, swan_data.shape)
# 创建数据集和数据加载器
dataset = TimeSeriesDataset(buoy_data, swan_data, batch_size=128)
dataloader = DataLoader(dataset, batch_size=None, shuffle=False)
# 训练模型
context_encoder = Utils.ContextualEncoder()
encoder = Utils.WaveFieldEncoderWithBuoy()
decoder = Utils.WaveFieldDecoder()
discriminator = Discriminator()
trained_context_encoder, trained_encoder, trained_decoder, trained_discriminator = train_cvae_al(
    context_encoder, encoder, decoder, discriminator,
    dataloader, device, latent_dim=128,
    lambda_kl=0.8, lambda_adv=0.01, lr=1e-4,
    epochs=100, verbose=True
)
# 保存 Contextual Encoder 和 Decoder 的参数
torch.save(trained_context_encoder.state_dict(), "net/context_encoder_params.pth")
torch.save(trained_decoder.state_dict(), "net/decoder_params.pth")
print("Contextual Encoder 和 Decoder 的参数已保存至net。")
