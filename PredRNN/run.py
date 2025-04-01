import os
import shutil
import argparse
import numpy as np
import math
from core.data_provider import datasets_factory
from core.models.model_factory import Model
from core.utils import preprocess
import core.trainer as trainer

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='PyTorch video prediction model - PredRNN')

# 添加训练/测试相关的参数
parser.add_argument('--is_training', type=int, default=0)
parser.add_argument('--device', type=str, default='cuda')

# 添加数据集相关的参数
parser.add_argument('--dataset_name', type=str, default='swan')
#E:/Dataset/met_waves/Swan4predRNN/train.npy，E:/Dataset/met_waves/Swan4predRNN/val.npy
parser.add_argument('--train_data_paths', type=str, default='/root/autodl-fs/train.npy')
parser.add_argument('--valid_data_paths', type=str, default='/root/autodl-fs/val.npy')
parser.add_argument('--test_data_paths', type=str, default='/root/autodl-fs/test.npy')
parser.add_argument('--save_dir', type=str, default='checkpoints/swan_predrnn_v2')
parser.add_argument('--gen_frm_dir', type=str, default='results/swan_predrnn_v2')
parser.add_argument('--input_length', type=int, default=10)
parser.add_argument('--total_length', type=int, default=20)
parser.add_argument('--img_width', type=int, default=128)
parser.add_argument('--img_channel', type=int, default=3)

# 添加模型相关的参数
parser.add_argument('--model_name', type=str, default='predrnn_v2')
#checkpoints/model.ckpt
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('--num_hidden', type=str, default='128,128,128,128')
parser.add_argument('--filter_size', type=int, default=5)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--patch_size', type=int, default=4)
parser.add_argument('--layer_norm', type=int, default=0)
parser.add_argument('--decouple_beta', type=float, default=0.1)

# 添加逆向调度采样相关的参数
parser.add_argument('--reverse_scheduled_sampling', type=int, default=1)
parser.add_argument('--r_sampling_step_1', type=float, default=600)
parser.add_argument('--r_sampling_step_2', type=int, default=1100)
parser.add_argument('--r_exp_alpha', type=int, default=500)
# 添加调度采样相关的参数
parser.add_argument('--scheduled_sampling', type=int, default=1)
parser.add_argument('--sampling_stop_iter', type=int, default=1100)
parser.add_argument('--sampling_start_value', type=float, default=1.0)
parser.add_argument('--sampling_changing_rate', type=float, default=0.0002)

# 添加优化相关的参数
# 学习率
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--reverse_input', type=int, default=1)
# 批大小
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--max_iterations', type=int, default=2000)  # 最大迭代次数
parser.add_argument('--display_interval', type=int, default=50)  # 每X轮显示一次
parser.add_argument('--test_interval', type=int, default=200)  # 每X轮测试一次
parser.add_argument('--snapshot_interval', type=int, default=200)  # 每X轮保存一次模型
parser.add_argument('--num_save_samples', type=int, default=10)  # 保存的样本数
parser.add_argument('--n_gpu', type=int, default=1)

# 添加可视化相关的参数
parser.add_argument('--visual', type=int, default=0)
parser.add_argument('--visual_path', type=str, default='./decoupling_visual')

# 添加基于动作的PredRNN相关的参数
parser.add_argument('--injection_action', type=str, default='concat')
parser.add_argument('--conv_on_input', type=int, default=0, help='conv on input')
parser.add_argument('--res_on_conv', type=int, default=0, help='res on conv')
parser.add_argument('--num_action_ch', type=int, default=4, help='num action ch')
# 解析命令行参数
args, unknown = parser.parse_known_args()
# 打印解析后的参数
print(args)


# 定义逆向调度采样函数
def reserve_schedule_sampling_exp(itr):
    """
    根据当前迭代次数计算逆向调度采样的概率，并生成相应的采样标志。

    参数:
    itr (int): 当前迭代次数

    返回:
    real_input_flag (np.ndarray): 采样标志数组
    """
    # 根据当前迭代次数计算逆向调度采样的概率
    # r_eta表示逆向调度采样的概率值
    if itr < args.r_sampling_step_1:
        r_eta = 0.5
    elif itr < args.r_sampling_step_2:
        r_eta = 1.0 - 0.5 * math.exp(-float(itr - args.r_sampling_step_1) / args.r_exp_alpha)
    else:
        r_eta = 1.0
    # eta 表示正向调度采样的概率值
    if itr < args.r_sampling_step_1:
        eta = 0.5
    elif itr < args.r_sampling_step_2:
        eta = 0.5 - (0.5 / (args.r_sampling_step_2 - args.r_sampling_step_1)) * (itr - args.r_sampling_step_1)
    else:
        eta = 0.0

    # 生成逆向调度采样的标志
    r_random_flip = np.random.random_sample(
        (args.batch_size, args.input_length - 1))  # 生成一个元素的值在 [0, 1) 之间的数组用于后续判断是否进行逆向调度采样。
    """
    该行代码的功能是根据生成的随机数数组 `r_random_flip` 和逆向调度采样的概率值 `r_eta`，生成一个布尔数组 `r_true_token`。
    具体来说，如果 `r_random_flip` 中的元素小于 `r_eta`，则对应的 `r_true_token` 元素为 `True`，否则为 `False`。
    """
    r_true_token = (r_random_flip < r_eta)
    # 下面这段代码和上面的那个差不多，是正向调度算法
    random_flip = np.random.random_sample(
        (args.batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)
    # 创建两个三维数组 ones 和 zeros，用于后续生成采样标志，这两个数组的形状由图像宽度、高度和通道数决定。
    ones = np.ones((args.img_width // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))

    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - 2):
            if j < args.input_length - 1:
                if r_true_token[i, j]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)
            else:
                if true_token[i, j - (args.input_length - 1)]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)

    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (args.batch_size,
                                  args.total_length - 2,
                                  args.img_width // args.patch_size,
                                  args.img_width // args.patch_size,
                                  args.patch_size ** 2 * args.img_channel))
    return real_input_flag


# 定义调度采样函数
def schedule_sampling(eta, itr):
    """
    根据当前迭代次数和给定的eta值计算调度采样的概率，并生成相应的采样标志。

    参数:
    eta (float): 当前的eta值
    itr (int): 当前迭代次数

    返回:
    eta (float): 更新后的eta值
    real_input_flag (np.ndarray): 采样标志数组
    """
    zeros = np.zeros((args.batch_size,
                      args.total_length - args.input_length - 1,
                      args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (args.batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((args.img_width // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (args.batch_size,
                                  args.total_length - args.input_length - 1,
                                  args.img_width // args.patch_size,
                                  args.img_width // args.patch_size,
                                  args.patch_size ** 2 * args.img_channel))
    return eta, real_input_flag


# 定义训练包装函数
def train_wrapper(model):
    """
    包装训练过程，包括加载预训练模型、数据加载、训练和测试。

    参数:
    model (Model): 训练模型实例
    """
    if args.pretrained_model:
        model.load(args.pretrained_model)
    # 加载数据，args.injection_action：是否使用某种特定的数据增强或特性注入
    # 这个地方test_input_handle：实际上是验证集，下面的路径有写
    train_input_handle, test_input_handle = datasets_factory.data_provider(
        args.dataset_name, args.train_data_paths, args.valid_data_paths, args.batch_size, args.img_width,
        seq_length=args.total_length, injection_action=args.injection_action, is_training=True)

    eta = args.sampling_start_value

    for itr in range(1, args.max_iterations + 1):
        # 检查数据是否用完，如果当前 train_input_handle 数据批已经全部读取完毕
        if train_input_handle.no_batch_left():
            # 重新开始数据迭代 (begin() 方法)，do_shuffle=True 表示重新洗牌数据，提高模型泛化能力
            train_input_handle.begin(do_shuffle=True)
        # 读取一个批次的训练数据
        # 获取训练数据批次
        ims = train_input_handle.get_batch()
        # 对图像进行补丁重塑处理
        ims = preprocess.reshape_patch(ims, args.patch_size)

        # 根据参数决定是否使用反转调度采样
        if args.reverse_scheduled_sampling == 1:
            real_input_flag = reserve_schedule_sampling_exp(itr)
        else:
            # 使用调度采样技术
            eta, real_input_flag = schedule_sampling(eta, itr)

        # 使用获取的数据训练模型
        trainer.train(model, ims, real_input_flag, args, itr)

        # 定期保存模型快照
        if itr % args.snapshot_interval == 0:
            model.save(itr)

        # 定期进行模型测试
        if itr % args.test_interval == 0:
            trainer.test(model, test_input_handle, args, itr)

        # 准备下一批训练数据
        train_input_handle.next()


# 定义测试包装函数
def test_wrapper(model):
    """
    包装测试过程，包括加载预训练模型和数据加载。
    
    参数:
    model (Model): 测试模型实例
    """
    model.load(args.pretrained_model)
    test_input_handle = datasets_factory.data_provider(
        args.dataset_name, args.train_data_paths, args.test_data_paths, args.batch_size, args.img_width,
        seq_length=args.total_length, injection_action=args.injection_action, is_training=False)
    trainer.test(model, test_input_handle, args, 'test_result')


# 检查并删除旧的保存目录
if os.path.exists(args.save_dir):
    shutil.rmtree(args.save_dir)
os.makedirs(args.save_dir)
# 创建新的保存目录
if os.path.exists(args.gen_frm_dir):
    shutil.rmtree(args.gen_frm_dir)
os.makedirs(args.gen_frm_dir)

# 打印初始化模型的信息
print('Initializing models')

# 创建模型实例
model = Model(args)

# 根据命令行参数决定是训练还是测试模型
if args.is_training:
    train_wrapper(model)
else:
    test_wrapper(model)
