import os.path
import datetime
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from core.utils import preprocess, metrics
import lpips
import torch

loss_fn_alex = lpips.LPIPS(net='alex')


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
def compute_metrics(true_data, pred_data):
    """
    输入形状: [batch, channels, height, width]
    返回:
        rmse_list - 各通道RMSE列表
        r2_list - 各通道R²列表
    """
    rmse_list = []
    r2_list = []

    for ch in range(true_data.shape[1]):
        # 展平所有样本和空间维度
        true_flat = true_data[:, ch, :, :].flatten()
        pred_flat = pred_data[:, ch, :, :].flatten()

        # 计算RMSE
        mse = np.mean((true_flat - pred_flat) ** 2)
        rmse = np.sqrt(mse)
        rmse_list.append(rmse)

        # 计算R²
        ss_total = np.var(true_flat) * len(true_flat)
        ss_res = np.sum((true_flat - pred_flat) ** 2)
        r2 = 1 - (ss_res / ss_total) if ss_total != 0 else 0
        r2_list.append(r2)

    return rmse_list, r2_list
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
    # 初始化列表以存储评估指标
    ssim, psnr,lp =  [],[],[]
    # 初始化评估指标数据结构
    output_length = configs.total_length - configs.input_length
    num_channels = configs.img_channel  # 假设有3个通道（hs, tm02, theta0）

    # 每个时间步的通道指标存储
    channel_rmse = [[0.0 for _ in range(num_channels)] for _ in range(output_length)]  # [时间步][通道]
    channel_r2 = [[0.0 for _ in range(num_channels)] for _ in range(output_length)]  # [时间步][通道]
    average_rmse = [0.0] * output_length  # 每个时间步的平均RMSE

    # 初始化每个帧的评估指标
    for i in range(configs.total_length - configs.input_length):
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
        # 遍历每个预测时间步
        for i in range(output_length):
            # 获取真实值和预测值
            x = test_ims[:, i + configs.input_length, :, :, :]  # [batch, C, H, W]
            gx = img_out[:, i, :, :, :]  # [batch, C, H, W]
            gx = np.clip(gx, 0, 1)

            # 计算指标（返回各通道的RMSE和R²）
            rmse_list, r2_list = compute_metrics(x, gx)  # 假设返回两个长度为3的列表

            # 存储通道级指标
            for ch_idx in range(num_channels):
                channel_rmse[i][ch_idx] = rmse_list[ch_idx]
                channel_r2[i][ch_idx] = r2_list[ch_idx]

            # 计算当前时间步的平均RMSE
            average_rmse[i] = sum(rmse_list) / num_channels

            # 打印当前时间步结果
            print(f"\nTime Step {i + 1}/{output_length}:")
            print(f"  Channel RMSE - HS: {rmse_list[0]:.4f}, TM02: {rmse_list[1]:.4f}, Theta0: {rmse_list[2]:.4f}")
            print(f"  Average RMSE: {average_rmse[i]:.4f}")
            print(f"  Channel R²   - HS: {r2_list[0]:.4f}, TM02: {r2_list[1]:.4f}, Theta0: {r2_list[2]:.4f}")

            # 计算 LPIPS
            img_x = np.zeros([configs.batch_size, 3, configs.img_width, configs.img_width])
            if configs.img_channel == 3:
                img_x[:, 0, :, :] = x[:, :, :, 0]
                img_x[:, 1, :, :] = x[:, :, :, 1]
                img_x[:, 2, :, :] = x[:, :, :, 2]
            else:
                img_x[:, 0, :, :] = x[:, :, :, 0]
                img_x[:, 1, :, :] = x[:, :, :, 0]
                img_x[:, 2, :, :] = x[:, :, :, 0]
            img_x = torch.FloatTensor(img_x)
            img_gx = np.zeros([configs.batch_size, 3, configs.img_width, configs.img_width])
            if configs.img_channel == 3:
                img_gx[:, 0, :, :] = gx[:, :, :, 0]
                img_gx[:, 1, :, :] = gx[:, :, :, 1]
                img_gx[:, 2, :, :] = gx[:, :, :, 2]
            else:
                img_gx[:, 0, :, :] = gx[:, :, :, 0]
                img_gx[:, 1, :, :] = gx[:, :, :, 0]
                img_gx[:, 2, :, :] = gx[:, :, :, 0]
            img_gx = torch.FloatTensor(img_gx)
            lp_loss = loss_fn_alex(img_x, img_gx)
            lp[i] += torch.mean(lp_loss).item()

            real_frm = np.uint8(x * 255)
            pred_frm = np.uint8(gx * 255)

            psnr[i] += metrics.batch_psnr(pred_frm, real_frm)
            for b in range(configs.batch_size):
                score, _ = compare_ssim(pred_frm[b], real_frm[b], full=True,  channel_axis=-1, win_size=7)
                ssim[i] += score

        # 保存预测示例
        if batch_id <= configs.num_save_samples:
            path = os.path.join(res_path, str(batch_id))
            os.mkdir(path)
            for i in range(configs.total_length):
                name = 'gt' + str(i + 1) + '.png'
                file_name = os.path.join(path, name)
                img_gt = np.uint8(test_ims[0, i, :, :, :] * 255)
                cv2.imwrite(file_name, img_gt)
            for i in range(output_length):
                name = 'pd' + str(i + 1 + configs.input_length) + '.png'
                file_name = os.path.join(path, name)
                img_pd = img_out[0, i, :, :, :]
                img_pd = np.clip(img_pd, 0, 1)  # 确保预测值在 [0, 1] 范围内
                img_pd = np.uint8(img_pd * 255)
                cv2.imwrite(file_name, img_pd)
        test_input_handle.next()

    # 计算全局统计
    # --------------------------------------------------
    # 各通道在所有时间步的平均RMSE
    global_channel_rmse = [
        sum(channel_rmse[i][ch] for i in range(output_length)) / output_length
        for ch in range(num_channels)
    ]

    # 所有时间步和通道的总平均RMSE
    global_avg_rmse = sum(sum(channel_rmse[i]) for i in range(output_length)) / (output_length * num_channels)

    # 各通道的全局平均R²
    global_channel_r2 = [
        sum(channel_r2[i][ch] for i in range(output_length)) / output_length
        for ch in range(num_channels)
    ]

    # 输出最终结果
    print("\n\n=== Final Statistics ===")
    print(
        f"Global Channel RMSE - HS: {global_channel_rmse[0]:.4f}, TM02: {global_channel_rmse[1]:.4f}, Theta0: {global_channel_rmse[2]:.4f}")
    print(f"Global Average RMSE: {global_avg_rmse:.4f}")
    print(
        f"\nGlobal Channel R²   - HS: {global_channel_r2[0]:.4f}, TM02: {global_channel_r2[1]:.4f}, Theta0: {global_channel_r2[2]:.4f}")
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
