from utils import Trainer, cmip_dataset
from torch.utils.data import DataLoader
import torch
import numpy as np


def create_dataset(data, time_step):
    """
    将时间序列数据转换为滑动窗口形式的样本集
    
    参数：
    data : numpy.ndarray
        输入的二维时间序列数据，形状为(样本数, 特征数)
    time_step : int
        每个样本包含的连续时间步长（滑动窗口大小）
        
    返回：
    numpy.ndarray
        三维数组，形状为(生成的样本数, time_step, 特征数)
        每个样本包含time_step个连续时间步的数据
    """
    dataX = []
    # 遍历所有可能的滑动窗口起始位置
    # 每个窗口包含time_step个连续时间步的数据
    for i in range(data.shape[0] - time_step + 1):
        dataX.append(data[i:i + time_step])
    return np.array(dataX)



def data_load(path, time_step):
    """
    加载并预处理海洋波浪预测数据
    
    Args:
        path (str): .npy数据文件路径，包含波浪参数的字典数据
        time_step (int): 时间序列预测的时间步长
    
    Returns:
        X (ndarray): 输入数据张量，形状为(样本数, 时间步长, 3通道, 纬度, 经度)
        Y (ndarray): 目标数据张量，形状为(样本数, 3通道, 纬度, 经度)
    """
    # 加载原始数据字典
    data_dict = np.load(path, allow_pickle=True).item()
    hs = data_dict['hs']       # 有效波高数据
    tm02 = data_dict['tm02']   # 平均波周期数据
    theta0 = data_dict['theta0']  # 波向数据 (时间步, 经度, 纬度)

    # 数据合并与预处理
    # 将三个参数堆叠为3通道数据 (时间, 纬度, 经度, 通道)
    data = np.stack([hs, tm02, theta0], axis=-1).astype('float32')
    data = np.nan_to_num(data, nan=0.0)  # 处理缺失值
    
    # 归一化处理（各通道独立归一化）
    mins = np.min(data, axis=(0, 1, 2), keepdims=True)
    maxs = np.max(data, axis=(0, 1, 2), keepdims=True)
    data = (data - mins) / (maxs - mins + 1e-8)

    # 构建时间序列数据集
    width = data.shape[2]  # 经度维度长度
    lenth = data.shape[1]  # 纬度维度长度
    X = create_dataset(data, time_step)  # 创建滑动窗口数据集
    X = X.reshape(X.shape[0], time_step, lenth, width, 3)
    
    # 构建目标数据
    Y = data[time_step - 1: data.shape[0]]
    Y = Y.reshape(Y.shape[0], lenth, width, 3)

    # 调整维度顺序适配模型输入
    X = X.transpose(0, 1, 4, 2, 3)  # 将通道维度提前
    Y = Y.transpose(0, 3, 1, 2)     # 对齐通道维度
    
    print(f"Data shape: {data.shape}")
    return X, Y



train_X, train_Y = data_load('E:/Dataset/met_waves/Swan4predRNN/train.npy', time_step=3)
valid_X, valid_Y = data_load('E:/Dataset/met_waves/Swan4predRNN/val.npy', time_step=3)
test_X, test_Y = data_load('E:/Dataset/met_waves/Swan4predRNN/test.npy', time_step=3)


class Configs:
    def __init__(self):
        pass


configs = Configs()

# trainer related
configs.n_cpu = 0
configs.device = torch.device('cuda:0')
configs.batch_size_test = 8
configs.batch_size = 8
# configs.lr = 0.001
configs.weight_decay = 0
configs.display_interval = 20
configs.num_epochs = 400
# 这是早停的耐心参数。即使模型在900个epoch内没有改善性能，训练仍会继续。如果在900个epoch内性能没有改善，训练将停止
configs.early_stopping = True
configs.patience = 100
# 禁用梯度裁剪（Gradient Clipping）。梯度裁剪用于防止梯度爆炸问题，但在这里未启用
configs.gradient_clipping = False
# 设置梯度裁剪的阈值为1。如果梯度裁剪启用，梯度的最大值将被限制为1。不过在这种配置下，由于梯度裁剪被禁用，这个参数实际上不会生效
configs.clipping_threshold = 1.

# lr warmup
# 这是学习率预热的步数设置。在训练的前3000步内，学习率将逐渐从一个较小的值线性增加到预设的学习率。这种技术通常用于训练的初始阶段，以帮助模型更稳定地开始训练，减少初期的震荡。
configs.warmup = 50

# data related
# 这是输入数据的维度设置。这通常取决于你使用的数据的特征数或通道数
configs.input_dim = 1  # 4 #这里应该是5吧 但是写的1我总感觉是5
'''
人家这个1是对的这个模型就是要保证输入通道和输出通道得一样
默认为1
'''
configs.output_dim = 1
# 表示模型的输入序列长度为5，即模型在预测时会使用前5个时间步的数据作为输入
configs.input_length = 5
# 表示模型的输出长度为1，即模型预测一个时间步的值。通常用于单步预测
configs.output_length = 1
# 表示输入序列中的数据点之间的时间间隔为1。即数据是逐步连续的，没有跳跃
configs.input_gap = 1
# 表示预测的时间偏移量为24。这可能意味着模型的目标是预测未来24个时间步后的数据点
configs.pred_shift = 24
# model
# 表示模型的维度即每个输入数据在模型中的表示为256维
configs.d_model = 256
# 表示模型处理数据时的patch（小块）的大小为5×5。这通常用于图像或序列数据的分块处理
configs.patch_size = (5, 5)
# 表示嵌入的空间尺寸。这里12*16可能是表示最终嵌入的特征图的尺寸（例如视觉模型中的特征图大小）
configs.emb_spatial_size = 25 * 25
# 表示多头注意力机制中的头数为4。多头注意力允许模型从不同的角度“看”数据，从而捕捉不同的关系
configs.nheads = 4
# 表示前馈神经网络的维度用于增加模型的表达能力
configs.dim_feedforward = 512
# 表示在模型中使用的dropout率为0.3。Dropout是一种正则化技术，用于减少过拟合。
configs.dropout = 0.3
# 表示编码器的层数为4。这意味着模型有4个堆叠的编码器层
configs.num_encoder_layers = 4
configs.num_decoder_layers = 4
# 这可能是学习率的衰减率（scheduler decay rate），用来控制模型训练过程中学习率的递减速度，以便在训练的后期进行更细致的优化
configs.ssr_decay_rate = 3.e-3

is_training = 1
trainer = Trainer(configs)
if is_training:
    trainer.save_configs('config_train.pkl')
    dataset_train = cmip_dataset(train_X, train_Y)
    print(dataset_train.GetDataShape())
    dataset_eval = cmip_dataset(valid_X, valid_Y)
    print(dataset_eval.GetDataShape())
    trainer.train(dataset_train, dataset_eval, './../../checkpoints/checkpoint.chk')

else:
    dataset_test = cmip_dataset(test_X, test_Y)
    dataloader_test = DataLoader(dataset_test, batch_size=configs.batch_size_test, shuffle=False)
    chk = torch.load('./checkpoint.chk')
    trainer.network.load_state_dict(chk['net'])
    loss_test, test_pred, test_true = trainer.infer_test(dataset=dataset_test, dataloader=dataloader_test)
    print(loss_test)

    test_pred = test_pred.cpu().numpy()
    test_true = test_true.cpu().numpy()

    np.save("./data/test_pred.npy", test_pred)
    np.save("./data/test_true.npy", test_true)
