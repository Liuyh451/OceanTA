import os
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt


def compute_metrics(y_true, y_pred):
    """
    计算多通道数据的 RMSE、R²（展平所有批次和时间步后计算）
    
    Args:
        y_true (ndarray): 真实值张量，形状为 (batch*t, C, H, W)
                          包含批次和时间步展平后的多通道数据
        y_pred (ndarray): 预测值张量，形状与 y_true 一致

    Returns:
        tuple: 包含指标的列表元组，每个列表长度与通道数 C 一致
              (rmse_list, r2_list)

    Note:
        - 指标按通道维度独立计算
        - 样本级计算方式可能影响性能，大规模数据建议改用批处理计算
    """
    num_channels = y_true.shape[3]
    rmse_list, r2_list = [], []

    # 按通道维度遍历计算指标
    for ch in range(num_channels):
        # 展平空间维度 (H,W)，保留样本维度 (batch*t)
        true_flat = y_true[..., ch].reshape(y_true.shape[0], -1)  # (batch*t, H*W)
        pred_flat = y_pred[..., ch].reshape(y_pred.shape[0], -1)  # (batch*t, H*W)

        rmse_values, r2_values = [], []
        
        # 逐样本计算指标（当前实现方式，适用于小规模数据）
        for i in range(true_flat.shape[0]):
            sample_true = true_flat[i]
            sample_pred = pred_flat[i]
            
            # 单样本RMSE计算
            mse = mean_squared_error(sample_true, sample_pred)
            rmse = np.sqrt(mse)
            rmse_values.append(rmse)
            
            # 单样本R²计算
            r2 = r2_score(sample_true, sample_pred)
            r2_values.append(r2)

        # 收集当前通道所有样本的指标结果
        rmse_list.append(rmse_values)
        r2_list.append(r2_values)

    # 转换结果为 numpy 数组格式返回
    rmse_list = np.array(rmse_list)
    r2_list = np.array(r2_list)
    return rmse_list, r2_list



def batch_psnr(gen_frames, gt_frames):
    if gen_frames.ndim == 3:
        axis = (1, 2)
    elif gen_frames.ndim == 4:
        axis = (1, 2, 3)
    x = np.int32(gen_frames)
    y = np.int32(gt_frames)
    num_pixels = float(np.size(gen_frames[0]))
    mse = np.sum((x - y) ** 2, axis=axis, dtype=np.float32) / num_pixels
    psnr = 20 * np.log10(255) - 10 * np.log10(mse)
    return np.mean(psnr)


def save_prediction_samples(batch_id, res_path, test_ims, img_out, configs):
    """
    保存 Swan 数据集的预测示例，按通道分别存储
    :param batch_id: 当前 batch ID
    :param res_path: 结果保存路径
    :param test_ims: 真实值 (batch, time, height, width, channels)
    :param img_out: 预测值 (batch, time, height, width, channels)
    :param configs: 运行参数
    """
    if batch_id > configs.num_save_samples:
        return

    path = os.path.join(res_path, str(batch_id))
    os.makedirs(path, exist_ok=True)  # 允许目录已存在

    # 获取数据形状
    batch, total_length, height, width, channels = test_ims.shape  # (batch, 20, 128, 128, 3)
    output_length = configs.total_length - configs.input_length  # 预测步长

    # 只保存 batch 内的第一组数据 (test_ims[0])
    for t in range(configs.input_length):  # 遍历输入帧
        for ch in range(channels):  # 遍历 3 个通道
            name = f'truth_t{t + 1}_ch{ch + 1}.png'  # 例如: gt_t1_ch1.png
            file_name = os.path.join(path, name)
            img_gt = np.uint8(test_ims[0, t, :, :, ch] * 255)  # 归一化到 [0, 255]
            cv2.imwrite(file_name, img_gt)

    for t in range(output_length):  # 遍历预测帧
        for ch in range(channels):  # 遍历 3 个通道
            name = f'pred_t{t + 1}_ch{ch + 1}.png'  # 例如: pd_t1_ch1.png
            file_name = os.path.join(path, name)
            img_pd = img_out[0, t, :, :, ch]
            img_pd = np.clip(img_pd, 0, 1)  # 限制范围 [0, 1]
            img_pd = np.uint8(img_pd * 255)
            cv2.imwrite(file_name, img_pd)


def prepare_image_for_lpips(img, batch_size, img_width, img_channel):
    """
    预处理图像，适配 LPIPS 输入（3 通道格式）。

    :param img: 输入图像 (batch_size, height, width, channels)
    :param batch_size: 批量大小
    :param img_width: 图像宽度（假设高宽相等）
    :param img_channel: 图像通道数
    :return: 适配 LPIPS 的图像 (batch_size, 3, height, width)
    """
    img_lpips = np.zeros([batch_size, 3, img_width, img_width])

    if img_channel == 3:
        img_lpips[:, 0, :, :] = img[:, :, :, 0]  # R
        img_lpips[:, 1, :, :] = img[:, :, :, 1]  # G
        img_lpips[:, 2, :, :] = img[:, :, :, 2]  # B
    else:  # 单通道
        img_lpips[:, 0, :, :] = img[:, :, :, 0]
        img_lpips[:, 1, :, :] = img[:, :, :, 0]
        img_lpips[:, 2, :, :] = img[:, :, :, 0]

    return torch.tensor(img_lpips, dtype=torch.float32)


def calculate_lpips(loss_fn_alex, x, gx, configs):
    """
    计算 LPIPS 误差并返回。

    :param loss_fn_alex: LPIPS 计算模型
    :param x: 真实图像 (batch_size, height, width, channels)
    :param gx: 预测图像 (batch_size, height, width, channels)
    :param configs: 配置参数对象
    :return: LPIPS 损失值
    """
    batch_size = configs.batch_size
    img_width = configs.img_width
    img_channel = configs.img_channel

    # 预处理 x 和 gx
    img_x = prepare_image_for_lpips(x, batch_size, img_width, img_channel)
    img_gx = prepare_image_for_lpips(gx, batch_size, img_width, img_channel)

    # 计算 LPIPS
    lp_loss = loss_fn_alex(img_x, img_gx)
    return torch.mean(lp_loss).item()


def plot_metrics(channel_rmse, channel_r2, output_length):
    """
    绘制 RMSE 和 R² 折线图（每个通道单独一张图）
    :param channel_rmse: (C, batch*t) 形状的 RMSE 数据
    :param channel_r2: (C, batch*t) 形状的 R² 数据
    :param output_length: 预测时间步数
    """
    # 假设 batch*t 作为时间步
    time_steps = np.arange(channel_rmse.shape[1])

    # 通道名称
    channel_names = ['Channel 1', 'Channel 2', 'Channel 3']

    # 创建保存图片的目录
    save_dir = './results/Plt/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 分通道绘制 RMSE 折线图
    for ch in range(channel_rmse.shape[0]):
        plt.figure(figsize=(12, 5))
        plt.plot(time_steps, channel_rmse[ch], label=channel_names[ch])
        plt.xlabel('Time Step')
        plt.ylabel('RMSE')
        plt.title(f'{channel_names[ch]} RMSE Over Time')
        plt.legend()
        plt.grid()
        save_path = os.path.join(save_dir, f"RMSE_Channel_{ch + 1}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"{channel_names[ch]} RMSE 图表已保存至: {save_path}")
        plt.show()

    # 分通道绘制 R² 折线图
    for ch in range(channel_r2.shape[0]):
        plt.figure(figsize=(12, 5))
        plt.plot(time_steps, channel_r2[ch], label=channel_names[ch])
        plt.xlabel('Time Step')
        plt.ylabel('R² Score')
        plt.title(f'{channel_names[ch]} R² Score Over Time')
        plt.legend()
        plt.grid()
        save_path = os.path.join(save_dir, f"R2_Channel_{ch + 1}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"{channel_names[ch]} R² 图表已保存至: {save_path}")
        plt.show()
