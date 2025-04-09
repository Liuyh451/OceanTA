import warnings
import numpy as np
import matplotlib.pyplot as plt

def plot_mean_hs(predictions, ground_truths, feature_name, channel_idx=0):
    """
    绘制预测和真实值的均值图以及它们的差异图。

    参数：
    - predictions: 4D NumPy 数组，形状为 [时间, 通道, X, Y]
    - ground_truths: 4D NumPy 数组，形状为 [时间, 通道, X, Y]
    - feature_name: 字符串，表示特征的名称
    - channel_idx: 整数，表示通道索引（默认值为0）

    输出：
    - 生成预测、真实值均值图以及差异图
    """
    # 提取指定通道的 hs 数据
    hs_predictions = predictions[:, channel_idx, :, :]
    hs_ground_truths = ground_truths[:, channel_idx, :, :]

    # 将 0 替换为 NaN
    hs_predictions[hs_predictions == 0] = np.nan
    hs_ground_truths[hs_ground_truths == 0] = np.nan

     # 定义安全的均值计算函数
    def safe_nanmean(data, axis):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(data, axis=axis)

    # 计算均值
    hs_mean_predictions = safe_nanmean(hs_predictions, axis=0)
    hs_mean_ground_truths = safe_nanmean(hs_ground_truths, axis=0)

    # 计算差异（取绝对值）
    hs_mean_difference = np.abs(hs_mean_predictions - hs_mean_ground_truths)

    # 绘制图像
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # 左图：预测均值
    im1 = axes[0].imshow(hs_mean_predictions, cmap='viridis', origin='lower', aspect='auto')
    axes[0].set_title(f"Mean {feature_name} - Predictions", fontsize=14)
    axes[0].set_xlabel("Grid X")
    axes[0].set_ylabel("Grid Y")
    fig.colorbar(im1, ax=axes[0], label=f"Mean {feature_name}")

    # 中图：真实值均值
    im2 = axes[1].imshow(hs_mean_ground_truths, cmap='viridis', origin='lower', aspect='auto')
    axes[1].set_title(f"Mean {feature_name} - Ground Truths", fontsize=14)
    axes[1].set_xlabel("Grid X")
    axes[1].set_ylabel("Grid Y")
    fig.colorbar(im2, ax=axes[1], label=f"Mean {feature_name}")

    # 右图：差异图
    im3 = axes[2].imshow(hs_mean_difference, cmap='viridis', origin='lower', aspect='auto')
    axes[2].set_title(f"Difference in {feature_name}", fontsize=14)
    axes[2].set_xlabel("Grid X")
    axes[2].set_ylabel("Grid Y")
    fig.colorbar(im3, ax=axes[2], label=f"Absolute Difference ({feature_name})")

    # 显示图像
    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt


def calculate_rmse_per_time(predictions, ground_truths, channel):
    """
    计算每个时间点的 RMSE（每个点在时间维度上计算）。

    参数：
    - predictions: 4D NumPy 数组，形状为 [时间, 通道, X, Y]
    - ground_truths: 4D NumPy 数组，形状为 [时间, 通道, X, Y]
    - channel: int，指定通道索引

    返回：
    - time_rmse: 一维数组，表示每个时间点的 RMSE 值
    """
    # 提取当前通道数据
    pred = predictions[:, channel, :]
    truth = ground_truths[:, channel, :, :]

    # 将 0 替换为 NaN，避免无效数据对计算的影响
    pred[pred == 0] = np.nan
    truth[truth == 0] = np.nan

    # 初始化用于存储每个时间点的 RMSE
    time_rmse = []
    # 对每个时间点计算 RMSE
    for t in range(pred.shape[0]):
        pred_c = pred[t, :, :]
        truth_c = truth[t, :, :]

        # 对于方向（dirm1, dirm2），需要计算周期差异 (0-360) 之间的最小差异
        if channel == 2:  # dirm1 或 dirm2
            diff = np.minimum(np.abs(pred_c - truth_c), 360 - np.abs(pred_c - truth_c))  # 计算周期最小差异
        else:
            diff = np.abs(pred_c - truth_c)  # 对于其他通道，直接计算差异

        squared_error = diff ** 2  # 计算平方误差

        total_valid_points = np.sum(~np.isnan(squared_error))  # 非 NaN 点数

        # 计算 RMSE 并归一化
        if total_valid_points > 0:
            mse = np.nansum(squared_error) / total_valid_points
            rmse = np.sqrt(mse)
        else:
            rmse = np.nan

        time_rmse.append(rmse)

    return np.array(time_rmse)


def plot_rmse_over_time(predictions, ground_truths):
    # 预定义颜色列表
    colors = ['b', 'g', 'r']  # 蓝色、绿色、红色、青色

    # 创建一个 4 行 1 列的子图，绘制 4 个通道的 RMSE
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))  # 调整高度，确保上下排列
    channel_name = ['Hs', 'Tm', 'dirm']
    for channel in range(3):  # 对每个通道绘制
        # 计算每个时间点的 RMSE
        time_rmse = calculate_rmse_per_time(predictions, ground_truths, channel)

        # 绘制 RMSE 折线图
        axes[channel].plot(time_rmse, label=f"RMSE - Channel {channel_name[channel]}", color=colors[channel])
        axes[channel].set_title(f"RMSE Over Time - Channel {channel} ({channel_name[channel]})", fontsize=14)
        axes[channel].set_xlabel("Time Step")
        axes[channel].set_ylabel("RMSE")
        axes[channel].legend()
        axes[channel].grid(True)
        # 设置 y 轴从 0 开始
        axes[channel].set_ylim(0, np.max(time_rmse) + 0.1)  # +0.1 给 y 轴上限留点空白
    plt.tight_layout()  # 自动调整布局，防止子图重叠
    plt.show()