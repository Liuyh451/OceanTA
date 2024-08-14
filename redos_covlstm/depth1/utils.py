import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import numpy as np
import math
from torch.utils.data import Dataset
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error
class Configs:
    def __init__(self):
        pass


configs = Configs()

# trainer related
configs.vtype = 't'
# configs.depth = 11
# configs.time_step = 1
configs.n_cpu = 0
# configs.device = torch.device('cpu')
configs.device = torch.device('cuda:0')
configs.batch_size_test = 4
configs.batch_size = 2
#configs.lr = 0.001
configs.weight_decay = 0
configs.display_interval = 10
configs.num_epochs = 50
#这是早停的耐心参数。即使模型在900个epoch内没有改善性能，训练仍会继续。如果在900个epoch内性能没有改善，训练将停止
configs.early_stopping = True
configs.patience = 50
#禁用梯度裁剪（Gradient Clipping）。梯度裁剪用于防止梯度爆炸问题，但在这里未启用
configs.gradient_clipping = False
#设置梯度裁剪的阈值为1。如果梯度裁剪启用，梯度的最大值将被限制为1。不过在这种配置下，由于梯度裁剪被禁用，这个参数实际上不会生效
configs.clipping_threshold = 1.

# lr warmup
#这是学习率预热的步数设置。在训练的前3000步内，学习率将逐渐从一个较小的值线性增加到预设的学习率。这种技术通常用于训练的初始阶段，以帮助模型更稳定地开始训练，减少初期的震荡。
configs.warmup = 150

# data related
#这是输入数据的维度设置。这通常取决于你使用的数据的特征数或通道数
configs.input_dim = 1 # 4 #这里应该是5吧 但是写的1我总感觉是5
'''
人家这个1是对的这个模型就是要保证输入通道和输出通道得一样
默认为1
'''
configs.output_dim = 1
#表示模型的输入序列长度为5，即模型在预测时会使用前5个时间步的数据作为输入
configs.input_length = 5
#表示模型的输出长度为1，即模型预测一个时间步的值。通常用于单步预测
configs.output_length = 1
#表示输入序列中的数据点之间的时间间隔为1。即数据是逐步连续的，没有跳跃
configs.input_gap = 1
#表示预测的时间偏移量为24。这可能意味着模型的目标是预测未来24个时间步后的数据点
configs.pred_shift = 24
#这个列表包含了一系列的深度值，这可能与模型的层次结构或者不同深度的输入特征相关联
configs.depth = [5,6,11,16,20,25,30,34,36,38,40,42,44,46,48,50,51,52,53,54,55,57]
#这个列表可能对应于不同深度的索引或层次级别。每个索引可能用于定位或选择特定深度的特征或数据
configs.depthindex = [30,50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]

# model
#表示模型的维度即每个输入数据在模型中的表示为256维
configs.d_model = 256
#表示模型处理数据时的patch（小块）的大小为5×5。这通常用于图像或序列数据的分块处理
configs.patch_size = (5,5)
#表示嵌入的空间尺寸。这里12*16可能是表示最终嵌入的特征图的尺寸（例如视觉模型中的特征图大小）
configs.emb_spatial_size = 12*16
#表示多头注意力机制中的头数为4。多头注意力允许模型从不同的角度“看”数据，从而捕捉不同的关系
configs.nheads = 4
#表示前馈神经网络的维度用于增加模型的表达能力
configs.dim_feedforward =512
#表示在模型中使用的dropout率为0.3。Dropout是一种正则化技术，用于减少过拟合。
configs.dropout = 0.3
#表示编码器的层数为4。这意味着模型有4个堆叠的编码器层
configs.num_encoder_layers = 4
configs.num_decoder_layers = 4
#这可能是学习率的衰减率（scheduler decay rate），用来控制模型训练过程中学习率的递减速度，以便在训练的后期进行更细致的优化
configs.ssr_decay_rate = 3.e-6


# plot 表示绘图的分辨率为600 DPI
configs.plot_dpi = 600

class covlstmformer(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.d_model = 25
        self.device = configs.device
        # 5,输入通道数（或输入特征图的通道数）。通常对应于输入数据的深度。
        # 8：输出通道数（或输出特征图的通道数）。表示卷积操作后输出的特征图的深度。
        # 3：卷积核的大小，通常表示一个 3x3 的卷积核（filter）。
        # 5：步幅（stride）
        self.cov1 = Cov(5, 8, 3, 5)
        self.cov2 = Cov(5, 8, 3, 5)
        # 两个编码器
        self.encode1 = EncoderLayer(self.d_model, 1, configs.dim_feedforward, configs.dropout)
        self.encode2 = EncoderLayer(self.d_model, 1, configs.dim_feedforward, configs.dropout)
        self.cov_last = Cov_last(5, 8, 3, 1)

    def forward(self, x):
        resdual1 = self.cov1(x)
        # 将特征图按 (5, 5) 大小的块展开或重新排列
        resdual1 = unfold_StackOverChannel(resdual1, (5, 5))
        x = resdual1
        # 跳跃连接操作
        x = resdual1 + self.encode1(x)
        # Debug: Print shape after first addition
        # 函数将特征图 x 折叠成 (60, 80) 尺寸，可能对应于输入尺寸的恢复
        x = fold_tensor(x, (28, 52), (5, 5))

        resdual2 = x + self.cov2(x)  # xiu gai 的地方在这
        resdual2 = unfold_StackOverChannel(resdual2, (5, 5))
        x = resdual2
        x = resdual2 + self.encode2(x)
        # Debug: Print shape after second addition
        x = fold_tensor(x, (28, 52), (5, 5))
        x = self.cov_last(x)
        return x


# 编码器层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, nheads, dim_feedforward, dropout):
        super().__init__()
        # 实现了时间维度上的多头注意力机制
        self.time_attn = MultiHeadedAttention(d_model, nheads, TimeAttention, dropout)
        # 实现了空间维度上的多头注意力机制
        self.space_attn = MultiHeadedAttention(d_model, nheads, SpaceAttention, dropout)
        '''
        #一个更复杂的全连接网络（被注释掉），可以用于替代 feed_forward 部分
        self.net = nn.Sequential(
                  nn.Linear(256, 25),
                  nn.ReLU(),
                  nn.Linear(25, 256),
                   nn.ReLU(),
                   nn.Linear(256, 512),
                   nn.ReLU(),
                   nn.Linear(512,256)
                  )
        '''
        # 前馈神经网络，用于进一步处理通过注意力机制后的输出。该网络包括两个线性层和ReLU激活函数，用于非线性映射和特征提取
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )

    '''
    #一个分离空间和时间注意力的实现（被注释掉），可能用于更精细地控制注意力的应用顺序
    def divided_space_time_attn(self, query, key, value, mask):
        """
        Apply space and time attention sequentially
        Args:
            query (N, S, T, D)
            key (N, S, T, D)
            value (N, S, T, D)
        Returns:
            (N, S, T, D)
        """
        m = self.time_attn(query, key, value, mask)
        return self.space_attn(m, m, m, mask)
    '''

    def forward(self, x, mask=None):
        # x = self.sublayer[0](x, lambda x: self.divided_space_time_attn(x, x, x, mask))
        # x = x + self.net(x)
        # return self.sublayer[1](x, self.feed_forward)
        # 融合时间和空间注意力机制
        x = x + self.time_attn(x, x, x, mask)
        x = x + self.space_attn(x, x, x, mask)
        # 应用前馈神经网络处理，并将结果与经过空间注意力后的输出相加，生成最终的编码器输出。
        x = x + self.feed_forward(x)
        return x


# 卷积长短时记忆网络（ConvLSTM）的单元 ConvLSTMCell
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        # 根据卷积核的大小，自动计算填充大小，以确保输入和输出张量的空间尺寸（高度和宽度）一致。
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2  # 保证在传递过程中 （h,w）不变
        # 是否在卷积操作中添加偏置项
        self.bias = bias
        # 定义了一个二维卷积层，该卷积层接收输入张量和当前隐藏状态张量的拼接，并输出 4 倍的隐藏状态张量大小，用于计算 LSTM 的四个门（输入门 i、遗忘门 f、输出门 o 和候选状态 g）。
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,  # i门，f门，o门，g门放在一起计算，然后在split开
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state  # 每个timestamp包含两个状态张量：h和c

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis # 把输入张量与h状态张量沿通道维度串联

        combined_conv = self.conv(combined)  # i门，f门，o门，g门放在一起计算，然后在split开
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g  # c状态张量更新
        h_next = o * torch.tanh(c_next)  # h状态张量更新

        return h_next, c_next  # 输出当前timestamp的两个状态张量

    def init_hidden(self, batch_size, image_size):
        """
        初始状态张量初始化.第一个timestamp的状态张量0初始化
        :param batch_size:
        :param image_size:
        :return:
        """
        height, width = image_size
        init_h = torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        init_c = torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        return (init_h, init_c)


class ConvLSTM(nn.Module):
    """
    Parameters:参数介绍
        input_dim: Number of channels in input# 输入张量的通道数
        hidden_dim: Number of hidden channels # h,c两个状态张量的通道数，可以是一个列表
        kernel_size: Size of kernel in convolutions # 卷积核的尺寸，默认所有层的卷积核尺寸都是一样的,也可以设定不通lstm层的卷积核尺寸不同
        num_layers: Number of LSTM layers stacked on each other # 卷积层的层数，需要与len(hidden_dim)相等
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers # 是否返回所有lstm层的h状态
        Note: Will do same padding. # 相同的卷积核尺寸，相同的padding尺寸
    Input:输入介绍
        A tensor of size [B, T, C, H, W] or [T, B, C, H, W]# 需要是5维的
    Output:输出介绍
        返回的是两个列表：layer_output_list，last_state_list
        列表0：layer_output_list--单层列表，每个元素表示一层LSTM层的输出h状态,每个元素的size=[B,T,hidden_dim,H,W]
        列表1：last_state_list--双层列表，每个元素是一个二元列表[h,c],表示每一层的最后一个timestamp的输出状态[h,c],h.size=c.size = [B,hidden_dim,H,W]
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:使用示例
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)  # 转为列表
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)  # 转为列表
        if not len(kernel_size) == len(hidden_dim) == num_layers:  # 判断一致性
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):  # 多层LSTM设置
            # 当前LSTM层的输入维度
            # if i==0:
            #     cur_input_dim = self.input_dim
            # else:
            #     cur_input_dim = self.hidden_dim[i - 1]
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]  # 与上等价
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)  # 把定义的多个LSTM层串联成网络模型

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: 5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            b, _, _, h, w = input_tensor.size()  # 自动获取 b,h,w信息
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)  # 根据输入张量获取lstm的长度
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):  # 逐层计算

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):  # 逐个stamp计算
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)  # 第 layer_idx 层的第t个stamp的输出状态

            layer_output = torch.stack(output_inner, dim=1)  # 第 layer_idx 层的第所有stamp的输出状态串联
            cur_layer_input = layer_output  # 准备第layer_idx+1层的输入张量

            layer_output_list.append(layer_output)  # 当前层的所有timestamp的h状态的串联
            last_state_list.append([h, c])  # 当前层的最后一个stamp的输出状态的[h,c]

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        """
        所有lstm层的第一个timestamp的输入状态0初始化
        :param batch_size:
        :param image_size:
        :return:
        """
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        """
        检测输入的kernel_size是否符合要求，要求kernel_size的格式是list或tuple
        :param kernel_size:
        :return:
        """
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        """
        扩展到多层lstm情况
        :param param:
        :param num_layers:
        :return:
        """
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class Cov(nn.Module):
    def __init__(self, intput_dim, hidden_dim, bn_dim, output_dim):  # bn_dim是时间步
        super().__init__()
        self.cov1 = ConvLSTM(input_dim=intput_dim,
                             hidden_dim=hidden_dim,
                             kernel_size=(3, 3),
                             num_layers=1,
                             batch_first=True,
                             bias=True,
                             return_all_layers=True)
        self.bn1 = nn.BatchNorm3d(bn_dim)
        self.cov2 = ConvLSTM(input_dim=hidden_dim,
                             hidden_dim=hidden_dim,
                             kernel_size=(3, 3),
                             num_layers=1,
                             batch_first=True,
                             bias=True,
                             return_all_layers=True)
        self.bn2 = nn.BatchNorm3d(bn_dim)
        self.cov3 = ConvLSTM(input_dim=hidden_dim,
                             hidden_dim=output_dim,
                             kernel_size=(3, 3),
                             num_layers=1,
                             batch_first=True,
                             bias=True,
                             return_all_layers=True)

    def forward(self, x):
        x, _ = self.cov1(x)  # 因为上面的Covlstm返回两个值所以先用_接住第二个用不到的值
        x = self.bn1(x[0])
        x, _ = self.cov2(x)
        x = self.bn2(x[0])
        x, _ = self.cov3(x)
        x = x[0]
        return x


class Cov_last(nn.Module):
    def __init__(self, intput_dim, hidden_dim, bn_dim, output_dim):  # bn_dim是时间步
        super().__init__()
        self.cov1 = ConvLSTM(input_dim=intput_dim,
                             hidden_dim=hidden_dim,
                             kernel_size=(3, 3),
                             num_layers=1,
                             batch_first=True,
                             bias=True,
                             return_all_layers=True)
        self.bn1 = nn.BatchNorm3d(bn_dim)
        self.cov2 = ConvLSTM(input_dim=hidden_dim,
                             hidden_dim=hidden_dim,
                             kernel_size=(3, 3),
                             num_layers=1,
                             batch_first=True,
                             bias=True,
                             return_all_layers=True)
        self.bn2 = nn.BatchNorm3d(bn_dim)
        self.cov3 = ConvLSTM(input_dim=hidden_dim,
                             hidden_dim=output_dim,
                             kernel_size=(3, 3),
                             num_layers=1,
                             batch_first=True,
                             bias=True,
                             return_all_layers=True)

    def forward(self, x):
        x, _ = self.cov1(x)  # 因为上面的Covlstm返回两个值所以先用_接住第二个用不到的值
        x = self.bn1(x[0])
        x, _ = self.cov2(x)
        x = self.bn2(x[0])
        x, _ = self.cov3(x)
        x = x[0]
        return x[:, -1]


def unfold_StackOverChannel(img, kernel_size):
    """
    divide the original image to patches, then stack the grids in each patch along the channels
    Args:
        img (N, *, C, H, W): the last two dimensions must be the spatial dimension
        kernel_size: tuple of length 2
    Returns:
        output (N, *, C*H_k*N_k, H_output, W_output)
    """
    T = img.size(1)
    n_dim = len(img.size())
    assert n_dim == 4 or n_dim == 5

    pt = img.unfold(-2, size=kernel_size[0], step=kernel_size[0])
    pt = pt.unfold(-2, size=kernel_size[1], step=kernel_size[1]).flatten(-2)  # (N, *, C, n0, n1, k0*k1)
    if n_dim == 4:  # (N, C, H, W)
        pt = pt.permute(0, 1, 4, 2, 3).flatten(1, 2)
    elif n_dim == 5:  # (N, T, C, H, W)
        pt = pt.permute(0, 1, 2, 5, 3, 4).flatten(2, 3)
    assert pt.size(-3) == img.size(-3) * kernel_size[0] * kernel_size[1]
    pt = pt.reshape(pt.size(0), T, 25, -1).permute(0, 3, 1, 2)
    return pt


def fold_tensor(tensor, output_size, kernel_size):
    """
    reconstruct the image from its non-overlapping patches
    Args:
        input tensor of size (N, *, C*k_h*k_w, n_h, n_w)
        output_size of size(H, W), the size of the original image to be reconstructed
        kernel_size: (k_h, k_w)
        stride is usually equal to kernel_size for non-overlapping sliding window
    Returns:
        (N, *, C, H=n_h*k_h, W=n_w*k_w)
    """
    tensor = tensor.reshape(-1, 50, 3, 25)
    T = tensor.size(2)
    tensor = tensor.permute(0, 2, 3, 1)  # (N, T, C_, S)
    tensor = tensor.reshape(tensor.size(0), T, 25,
                            5, 10)
    tensor = tensor.float()
    n_dim = len(tensor.size())
    assert n_dim == 4 or n_dim == 5
    f = tensor.flatten(0, 1) if n_dim == 5 else tensor
    folded = F.fold(f.flatten(-2), output_size=output_size, kernel_size=kernel_size, stride=kernel_size)
    if n_dim == 5:
        folded = folded.reshape(tensor.size(0), tensor.size(1), *folded.size()[1:])
    return folded.reshape(-1, T, 5, 28, 52)


def TimeAttention(query, key, value, mask=None, dropout=None):
    """
    attention over the time axis
    Args:
        query, key, value: linearly-transformed query, key, value (N, h, S, T, D)
        mask: of size (T (query), T (key)) specifying locations (which key) the query can and cannot attend to
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)  # (N, h, S, T, T)
    if mask is not None:
        assert mask.dtype == torch.bool
        assert len(mask.size()) == 2
        scores = scores.masked_fill(mask[None, None, None], float("-inf"))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value)  # (N, h, S, T, D)


def SpaceAttention(query, key, value, mask=None, dropout=None):
    """
    attention over the two space axes
    Args:
        query, key, value: linearly-transformed query, key, value (N, h, S, T, D)
        mask: None (space attention does not need mask), this argument is intentionally set for consistency
    """
    d_k = query.size(-1)
    query = query.transpose(2, 3)  # (N, h, T, S, D)
    key = key.transpose(2, 3)  # (N, h, T, S, D)
    value = value.transpose(2, 3)  # (N, h, T, S, D)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)  # (N, h, T, S, S)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value).transpose(2, 3)  # (N, h, S, T_q, D)


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, nheads, attn, dropout):
        super().__init__()
        assert d_model % nheads == 0
        self.d_k = d_model // nheads
        self.nheads = nheads
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.dropout = nn.Dropout(p=dropout)
        self.attn = attn

    def forward(self, query, key, value, mask=None):
        """
        Transform the query, key, value into different heads, then apply the attention in parallel
        Args:
            query, key, value: size (N, S, T, D)
        Returns:
            (N, S, T, D)
        """
        nbatches = query.size(0)
        nspace = query.size(1)
        ntime = query.size(2)
        # (N, h, S, T, d_k)
        query, key, value = \
            [l(x).view(x.size(0), x.size(1), x.size(2), self.nheads, self.d_k).permute(0, 3, 1, 2, 4)
             for l, x in zip(self.linears, (query, key, value))]

        # (N, h, S, T, d_k)
        x = self.attn(query, key, value, mask=mask, dropout=self.dropout)

        # (N, S, T, D)
        x = x.permute(0, 2, 3, 1, 4).contiguous() \
            .view(nbatches, nspace, ntime, self.nheads * self.d_k)
        return self.linears[-1](x)


class Attention(nn.Module):
    def __init__(self, dropout, attn):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(50, 50) for _ in range(3)])
        self.attn = attn
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        query, key, value = \
            [l(x)
             for l, x in zip(self.linears, (query, key, value))]
        x = self.attn(query, key, value, mask=mask, dropout=self.dropout)
        return x


class NoamOpt:
    """
    learning rate warmup and decay
    """

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


class Trainer:
    def __init__(self, configs):
        self.configs = configs
        self.device = configs.device
        torch.manual_seed(5)
        self.network = covlstmformer(configs).to(configs.device)
        adam = torch.optim.Adam(self.network.parameters(), lr=0, weight_decay=configs.weight_decay)
        factor = math.sqrt(configs.d_model * configs.warmup) * 0.0014
        self.opt = NoamOpt(configs.d_model, factor, warmup=configs.warmup, optimizer=adam)

    def loss_sst(self, y_pred, y_true):
        # y_pred/y_true (N, 26, 24, 48)
        rmse = torch.mean((y_pred - y_true) ** 2, dim=[2, 3])
        rmse = torch.sum(rmse.sqrt().mean(dim=0))
        return rmse

    def train_once(self, input_sst, sst_true, ssr_ratio):
        sst_pred = self.network(input_sst.float().to(self.device))
        self.opt.optimizer.zero_grad()
        loss_sst = self.loss_sst(sst_pred, sst_true.float().to(self.device))
        # loss_nino = self.loss_nino(nino_pred, nino_true.float().to(self.device))
        loss_sst.backward()
        if configs.gradient_clipping:
            nn.utils.clip_grad_norm_(self.network.parameters(), configs.clipping_threshold)
        self.opt.step()
        return loss_sst.item()

    def test(self, dataloader_test):
        # nino_pred = []
        sst_pred = []
        with torch.no_grad():
            for input_sst, sst_true, in dataloader_test:
                sst = self.network(input_sst.float().to(self.device))
                # nino_pred.append(nino)
                sst_pred.append(sst)

        return torch.cat(sst_pred, dim=0)

    def infer(self, dataset, dataloader):
        self.network.eval()
        with torch.no_grad():
            sst_pred = self.test(dataloader)
            # nino_true = torch.from_numpy(dataset.target_nino).float().to(self.device)
            sst_true = torch.from_numpy(dataset.target_sst).float().to(self.device)
            # sc = self.score(nino_pred, nino_true)
            #             print(sst_pred.shape)
            #             print(sst_true.shape)
            loss_sst = self.loss_sst(sst_pred, sst_true).item()
            # loss_nino = self.loss_nino(nino_pred, nino_true).item()
        return loss_sst

    def infer_test(self, dataset, dataloader):
        self.network.eval()
        with torch.no_grad():
            sst_pred = self.test(dataloader)
            # nino_true = torch.from_numpy(dataset.target_nino).float().to(self.device)
            sst_true = torch.from_numpy(dataset.target_sst).float().to(self.device)
            # sc = self.score(nino_pred, nino_true)
            loss_sst = self.loss_sst(sst_pred, sst_true).item()
            # loss_nino = self.loss_nino(nino_pred, nino_true).item()
        return loss_sst, sst_pred, sst_true

    def train(self, dataset_train, dataset_eval, chk_path):
        torch.manual_seed(0)
        # print('loading train dataloader')
        dataloader_train = DataLoader(dataset_train, batch_size=self.configs.batch_size, shuffle=True)
        # print('loading eval dataloader')
        dataloader_eval = DataLoader(dataset_eval, batch_size=self.configs.batch_size_test, shuffle=False)

        count = 0
        best = math.inf
        ssr_ratio = 1
        for i in range(self.configs.num_epochs):
            print('\nepoch: {0}'.format(i + 1))
            # train
            self.network.train()
            for j, (input_sst, sst_true) in enumerate(dataloader_train):
                if ssr_ratio > 0:
                    ssr_ratio = max(ssr_ratio - self.configs.ssr_decay_rate, 0)
                loss_sst = self.train_once(input_sst, sst_true, ssr_ratio)  # y_pred for one batch\

                if j % self.configs.display_interval == 0:
                    print('batch training loss: {:.5f}, ssr: {:.5f}, lr: {:.5f}'.format(loss_sst, ssr_ratio,
                                                                                        self.opt.rate()))

                # increase the number of evaluations in order not to miss the optimal point
                # which is feasible because of the less training time of timesformer
                if (i + 1 >= 9) and (j + 1) % 300 == 0:
                    loss_sst_eval = self.infer(dataset=dataset_eval, dataloader=dataloader_eval)
                    print('epoch eval loss: sc: {:.4f}'.format(loss_sst_eval))
                    if loss_sst_eval < best:
                        self.save_model(chk_path)
                        best = loss_sst_eval
                        count = 0

            # evaluation
            loss_sst_eval = self.infer(dataset=dataset_eval, dataloader=dataloader_eval)
            print('epoch eval loss:\nsst: {:.2f}'.format(loss_sst_eval))
            if loss_sst_eval >= best:
                count += 1
                print('eval score is not improved for {} epoch'.format(count))
            else:
                count = 0
                print('eval score is improved from {:.5f} to {:.5f}, saving model'.format(best, loss_sst_eval))
                self.save_model(chk_path)
                best = loss_sst_eval

            if count == self.configs.patience:
                print('early stopping reached, best score is {:5f}'.format(best))
                break

    def save_configs(self, config_path):
        with open(config_path, 'wb') as path:
            pickle.dump(self.configs, path)

    def save_model(self, path):
        torch.save({'net': self.network.state_dict(),
                    'optimizer': self.opt.optimizer.state_dict()}, path)
class cmip_dataset(Dataset):
    def __init__(self, datax,datay):
        super().__init__()

        self.input_sst = datax
        self.target_sst = datay


    def GetDataShape(self):
        return {'sst input': self.input_sst.shape,
                'sst target': self.target_sst.shape}

    def __len__(self,):
        return self.input_sst.shape[0]

    def __getitem__(self, idx):
        return self.input_sst[idx], self.target_sst[idx]

def loss(data_mask, depth, test_pred, test_true):
    test_preds = np.array(test_pred, copy=True)
    test_trues = np.array(test_true, copy=True)


    test_preds = np.squeeze(test_preds)
    test_trues = np.squeeze(test_trues)

    test_preds[np.isnan(test_preds)] = 0
    test_trues[np.isnan(test_trues)] = 0
    mask = data_mask
    print(mask.shape,test_preds.shape, test_trues.shape)
    #     mask = np.squeeze(mask)
    mask = mask[0]
    mask=np.transpose(mask)

    total = mask.shape[0] * mask.shape[1]
    total_nan = len(mask[np.isnan(mask)])
    total_real = total - total_nan
    #     print('Total NaN:',total_nan)#统计数据中的nan值
    #     print('Total Real:',total_real)#统计数据中的nan值
    #     #nan：0 values ：1
    mask[~np.isnan(mask)] = 1
    mask[np.isnan(mask)] = 0
    rmse = []
    rmse_temp = []
    nrmse = []
    nrmse_temp = []
    mae = []
    mae_temp = []
    for i in range(0, test_preds.shape[0]):
        final_temp = mask * test_preds[i]
        test_temp = mask * test_trues[i]
        # np.sum((y_actual - y_predicted) ** 2)
        sse = np.sum((test_temp - final_temp) ** 2)
        mse_temp = sse / total_real
        rmse_temp = np.sqrt(mse_temp)
        nrmse_temp = rmse_temp / np.mean(test_temp)
        rmse.append(rmse_temp)
        nrmse.append(nrmse_temp)
        mae_temp = mean_absolute_error(test_temp, final_temp) * total / total_real

        mae.append(mae_temp)
    #     print('NAN:',len(test_pred[np.isnan(test_pred)]))
    #     print('TEST NANMIN',np.nanmin(test_pred))
    #     print('TEST MIN',test_pred.min())
    # print(str(depth)+'层')
    RMSE = np.sum(rmse) / len(rmse)
    MAE = np.sum(mae) / len(mae)
    NRMSE = np.sum(nrmse) / len(nrmse)
    # NRMSE = nrmse
    print(str(depth) + '层:' + 'NRMSE RESULT:\n', NRMSE)

    #     print('MAE RESULT:\n',MAE)

    return NRMSE