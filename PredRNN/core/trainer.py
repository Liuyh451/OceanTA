# file:D:\WorkSpace\PycharmWorkSpace\OceanTA\PredRNN\core\trainer.py
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
    # 初始化平均均方误差
    avg_mse = 0
    # 初始化批次计数器
    batch_id = 0
    # 初始化列表以存储评估指标
    img_mse, ssim, psnr = [], [], []
    lp = []

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

        # 计算每帧的均方误差（MSE）
        for i in range(output_length):
            x = test_ims[:, i + configs.input_length, :, :, :]
            gx = img_out[:, i, :, :, :]
            gx = np.clip(gx, 0, 1)  # 确保预测值在 [0, 1] 范围内
            mse = np.square(x - gx).sum()
            img_mse[i] += mse
            avg_mse += mse
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

    # 计算并打印序列的平均均方误差（MSE）
    avg_mse = avg_mse / (batch_id * configs.batch_size)
    print('mse per seq: ' + str(avg_mse))
    for i in range(configs.total_length - configs.input_length):
        print(img_mse[i] / (batch_id * configs.batch_size))

    # 计算并打印每帧的平均结构相似性指数（SSIM）
    ssim = np.asarray(ssim, dtype=np.float32) / (configs.batch_size * batch_id)
    print('ssim per frame: ' + str(np.mean(ssim)))
    for i in range(configs.total_length - configs.input_length):
        print(ssim[i])

    # 计算并打印每帧的平均峰值信噪比（PSNR）
    psnr = np.asarray(psnr, dtype=np.float32) / batch_id
    print('psnr per frame: ' + str(np.mean(psnr)))
    for i in range(configs.total_length - configs.input_length):
        print(psnr[i])

    # 计算并打印每帧的平均感知损失（LPIPS）
    lp = np.asarray(lp, dtype=np.float32) / batch_id
    print('lpips per frame: ' + str(np.mean(lp)))
    for i in range(configs.total_length - configs.input_length):
        print(lp[i])
