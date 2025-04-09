import os.path
import datetime
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from torch import nn

from core.utils import preprocess, metrics
import lpips

from core.utils.metrics import save_prediction_samples, calculate_lpips, plot_metrics, compute_metrics

# 初始化 LPIPS 计算器
loss_fn_alex = lpips.LPIPS(net='alex')


def average_metrics(rmse_list, r2_list, window=10):
    """
    对形状为 (3, N) 的 RMSE 和 R² 列表按窗口大小进行滑动平均
    参数:
        rmse_list : np.ndarray, 形状 (3, N) - 各通道的 RMSE 值
        r2_list   : np.ndarray, 形状 (3, N) - 各通道的 R² 值
        window    : int - 分段窗口大小 (默认为10)
    返回:
        avg_rmse : np.ndarray, 形状 (3, M) - 分段平均后的 RMSE
        avg_r2   : np.ndarray, 形状 (3, M) - 分段平均后的 R²
    """
    # 确保输入为 NumPy 数组
    rmse_arr = np.array(rmse_list)
    r2_arr = np.array(r2_list)

    # 初始化结果容器
    avg_rmse, avg_r2 = [], []

    # 遍历每个通道 (3个)
    for i in range(3):
        # 提取当前通道数据
        channel_rmse = rmse_arr[i]
        channel_r2 = r2_arr[i]
        n_samples = len(channel_rmse)

        # 计算分段数
        n_groups = np.ceil(n_samples / window).astype(int)

        # 分段计算均值
        rmse_avgs = [np.mean(channel_rmse[j * window: (j + 1) * window])
                     for j in range(n_groups)]
        r2_avgs = [np.mean(channel_r2[j * window: (j + 1) * window])
                   for j in range(n_groups)]

        avg_rmse.append(rmse_avgs)
        avg_r2.append(r2_avgs)

    return np.array(avg_rmse), np.array(avg_r2)


def train(model, ims, real_input_flag, configs, itr):
    """
    训练模型。

    参数:
    - model: 模型对象，用于训练。
    - ims: 输入模型的图像数据。
    - real_input_flag: 标志位，指示输入数据是否为真实数据。
    - configs: 配置对象，包含训练配置信息。
    - itr: 当前迭代次数。

    返回值:
    无。
    """
    # 训练模型并计算损失
    cost = model.train(ims, real_input_flag)

    # 如果配置了输入反转，则执行数据反转训练
    if configs.reverse_input:
        # 沿轴1翻转图像数据
        ims_rev = np.flip(ims, axis=1).copy()
        # 使用反转的数据进行训练，并累加损失
        cost += model.train(ims_rev, real_input_flag)
        # 计算平均损失
        cost = cost / 2

    # 根据显示间隔配置，定期打印训练信息
    if itr % configs.display_interval == 0:
        # 打印当前时间和迭代次数
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'itr: ' + str(itr))
        # 打印当前训练损失
        print('training loss: ' + str(cost))


def test(model, test_input_handle, configs, itr):
    """
    测试模型性能。

    参数:
    - model: 已训练的模型对象。
    - test_input_handle: 测试数据处理器。
    - configs: 配置对象，包含测试配置信息。
    - itr: 当前迭代次数，用于保存结果。

    返回值:
    无。
    """
    # 打印测试开始时间
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'test...')
    # 初始化数据处理器，不进行数据打乱
    test_input_handle.begin(do_shuffle=False)
    # 创建目录以保存测试结果
    res_path = os.path.join(configs.gen_frm_dir, str(itr))
    os.mkdir(res_path)
    # 初始化批次计数器
    batch_id = 0
    avg_mse = 0
    # 初始化列表以存储评估指标
    img_mse,ssim, psnr, lp = [], [], [],[]
    # 初始化评估指标数据结构
    output_length = configs.total_length - configs.input_length
    num_channels = configs.img_channel  # 假设有3个通道（hs, tm02, theta0）

    # 初始化存储 RMSE 和 R² 的数组
    channel_rmse = np.zeros((output_length * 10, num_channels))
    channel_r2 = np.zeros((output_length * 10, num_channels))
    average_rmse = [0.0] * output_length  # 每个时间步的平均RMSE

    # 初始化每个帧的评估指标
    for i in range(configs.total_length - configs.input_length):
        img_mse.append(0)
        ssim.append(0)
        psnr.append(0)
        lp.append(0)

    # 设置输入掩码，根据反向调度采样配置
    if configs.reverse_scheduled_sampling == 1:
        mask_input = 1
    else:
        mask_input = configs.input_length

    # 准备真实的输入标志位
    real_input_flag = np.zeros(
        (configs.batch_size,
         configs.total_length - mask_input - 1,
         configs.img_width // configs.patch_size,
         configs.img_width // configs.patch_size,
         configs.patch_size ** 2 * configs.img_channel))

    # 调整输入标志位，适用于反向调度采样
    if configs.reverse_scheduled_sampling == 1:
        real_input_flag[:, :configs.input_length - 1, :, :] = 1.0

    # 开始测试
    # 用于存储所有批次的时间步展开数据
    all_true = []
    all_pred = []
    while not test_input_handle.no_batch_left():

        batch_id += 1
        # 获取一批测试数据
        test_ims = test_input_handle.get_batch()
        # 预处理数据
        test_dat = preprocess.reshape_patch(test_ims, configs.patch_size)
        test_ims = test_ims[:, :, :, :, :configs.img_channel]
        # 生成预测结果
        img_gen = model.test(test_dat, real_input_flag)

        # 将生成的图像重新整形
        img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)
        output_length = configs.total_length - configs.input_length
        img_out = img_gen[:, -output_length:]

        # 遍历每个预测时间步，计算RMSE，LPIPS，PSNR，R2
        for i in range(output_length):
            # 获取真实值和预测值
            x = test_ims[:, i + configs.input_length, :, :, :]  # [batch,  H, W,C]
            gx = img_out[:, i, :, :, :]  # [batch, H, W,C]
            gx = np.clip(gx, 0, 1)
            # 如果是测试阶段展开所有时间步
            if configs.is_training == 0:
                all_true.append(x)
                all_pred.append(gx)
            mse = np.square(x - gx).sum()
            img_mse[i] += mse
            avg_mse += mse
            print(f"第{i}帧的MSE: {mse.item()}")
            # 计算 LPIPS
            lp[i] += calculate_lpips(loss_fn_alex, x, gx, configs)
            # 计算 PSNR
            real_frm = np.uint8(x * 255)
            pred_frm = np.uint8(gx * 255)
            psnr[i] += metrics.batch_psnr(pred_frm, real_frm)
            # 计算 SSIM
            for b in range(configs.batch_size):
                score, _ = compare_ssim(pred_frm[b], real_frm[b], full=True, channel_axis=-1, win_size=7)
                ssim[i] += score

        # 保存预测结果
        save_prediction_samples(batch_id, res_path, test_ims, img_out, configs)
        test_input_handle.next()  # 继续下一个批次
    # 如果是测试阶段，计算平均 RMSE 和 R²
    if configs.is_training == 0:
        # 拼接所有批次的时间步，形状变为 (batch*t, H, W, C)
        all_true = np.concatenate(all_true, axis=0)
        all_pred = np.concatenate(all_pred, axis=0)
        print(all_true.shape)
        rmse_list, r2_list = compute_metrics(all_true, all_pred)
        # 每10个样本分段平均
        avg_rmse, avg_r2 = average_metrics(rmse_list, r2_list, window=10)
        # 创建目录
        save_dir = "./results/metrics"
        os.makedirs(save_dir, exist_ok=True)
        # 保存为 .npy
        npy_path = os.path.join(save_dir, "rmse.npy")
        np.save(npy_path, avg_rmse)
        npy_path = os.path.join(save_dir, "r2.npy")
        np.save(npy_path, avg_r2)
        print("Metrics saved to:", save_dir)
        save_dir = "./results/data"
        os.makedirs(save_dir, exist_ok=True)
        npy_path = os.path.join(save_dir, "true.npy")
        np.save(npy_path, all_true)
        npy_path = os.path.join(save_dir, "pred.npy")
        np.save(npy_path, all_pred)
        print("Data saved to:", save_dir)

    # 输出最终结果
    print("\n\n=== Final Statistics ===")
    avg_mse = avg_mse / (batch_id * configs.batch_size)
    print('mse per seq: ' + str(avg_mse))
    for i in range(configs.total_length - configs.input_length):
        print(img_mse[i] / (batch_id * configs.batch_size))
    # print(
    #     f"\n当前样本平均 R²   - HS: {avg_r2[0]:.4f}, TM02: {avg_r2[1]:.4f}, Theta0: {avg_r2[2]:.4f}")
    # 计算并打印每帧的平均结构相似性指数（SSIM）
    ssim = np.asarray(ssim, dtype=np.float32) / (configs.batch_size * batch_id)
    print('ssim per frame ——>1: ' + str(np.mean(ssim)))
    for i in range(configs.total_length - configs.input_length):
        print(ssim[i])

    # 计算并打印每帧的平均峰值信噪比（PSNR）
    psnr = np.asarray(psnr, dtype=np.float32) / batch_id
    print('psnr per frame ↑: ' + str(np.mean(psnr)))
    for i in range(configs.total_length - configs.input_length):
        print(psnr[i])

    # 计算并打印每帧的平均感知损失（LPIPS）
    lp = np.asarray(lp, dtype=np.float32) / batch_id
    print('lpips per frame ↓: ' + str(np.mean(lp)))
    for i in range(configs.total_length - configs.input_length):
        print(lp[i])
