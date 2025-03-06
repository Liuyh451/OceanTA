import torch
import torch.nn as nn
from core.layers.SpatioTemporalLSTMCell_v2 import SpatioTemporalLSTMCell
import torch.nn.functional as F
from core.utils.tsne import visualization


class RNN(nn.Module):
    """
    一个基于SpatioTemporalLSTMCell实现的循环神经网络（RNN）模型。

    参数:
        num_layers (int): RNN的层数。
        num_hidden (list): 每层隐藏单元数量的列表。
        configs: 包含各种超参数和配置的配置对象。

    返回:
        next_frames (Tensor): 预测的下一帧序列。
        loss (Tensor): 总损失，包括MSE损失和解耦损失。
    """
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()

        # 初始化配置和可视化相关参数
        self.configs = configs
        self.visual = self.configs.visual  # 是否启用可视化
        self.visual_path = self.configs.visual_path  # 可视化结果保存路径

        # 计算输入帧的通道数
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden

        # 构建SpatioTemporalLSTMCell层列表
        cell_list = []
        width = configs.img_width // configs.patch_size
        self.MSE_criterion = nn.MSELoss()  # 定义MSE损失函数

        for i in range(num_layers):
            # 根据当前层索引确定输入通道数
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], width, configs.filter_size,
                                       configs.stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)  # 将所有LSTM单元存储为ModuleList

        # 定义最后一层卷积层，用于生成预测帧
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel, kernel_size=1, stride=1, padding=0,
                                   bias=False)

        # 共享适配器卷积层，用于特征归一化
        adapter_num_hidden = num_hidden[0]
        self.adapter = nn.Conv2d(adapter_num_hidden, adapter_num_hidden, 1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true):
        """
        前向传播函数，用于生成预测帧并计算损失。

        参数:
            frames_tensor (Tensor): 输入的帧序列，形状为[batch, length, height, width, channel]。
            mask_true (Tensor): 掩码张量，用于控制真实帧和预测帧的混合比例。

        返回:
            next_frames (Tensor): 预测的下一帧序列，形状为[batch, length, height, width, channel]。
            loss (Tensor): 总损失，包括MSE损失和解耦损失。
        """
        # 调整输入张量维度顺序以适应后续处理
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        # 获取批次大小、高度和宽度
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        # 初始化存储变量
        next_frames = []  # 存储预测帧
        h_t = []  # 存储每层的隐藏状态
        c_t = []  # 存储每层的细胞状态
        delta_c_list = []  # 存储每层的delta_c
        delta_m_list = []  # 存储每层的delta_m
        if self.visual:
            delta_c_visual = []  # 用于可视化的delta_c
            delta_m_visual = []  # 用于可视化的delta_m

        decouple_loss = []  # 解耦损失列表

        # 初始化隐藏状态、细胞状态和delta_c/delta_m
        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        # 初始化记忆单元
        memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.configs.device)

        # 遍历时间步，逐步生成预测帧
        for t in range(self.configs.total_length - 1):
            # 根据配置选择是否使用反向调度采样
            if self.configs.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]  # 第一帧直接使用真实帧
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < self.configs.input_length:
                    net = frames[:, t]  # 在输入长度内使用真实帧
                else:
                    net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                          (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            # 更新第一层的隐藏状态、细胞状态和记忆单元
            h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](net, h_t[0], c_t[0], memory)
            delta_c_list[0] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
            delta_m_list[0] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)
            if self.visual:
                delta_c_visual.append(delta_c.view(delta_c.shape[0], delta_c.shape[1], -1))
                delta_m_visual.append(delta_m.view(delta_m.shape[0], delta_m.shape[1], -1))

            # 更新后续层的隐藏状态、细胞状态和记忆单元
            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
                delta_c_list[i] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                delta_m_list[i] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)
                if self.visual:
                    delta_c_visual.append(delta_c.view(delta_c.shape[0], delta_c.shape[1], -1))
                    delta_m_visual.append(delta_m.view(delta_m.shape[0], delta_m.shape[1], -1))

            # 使用最后一层的隐藏状态生成预测帧
            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

            # 计算解耦损失
            for i in range(0, self.num_layers):
                decouple_loss.append(
                    torch.mean(torch.abs(torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=2))))

        # 如果启用了可视化，保存delta_c和delta_m的可视化结果
        if self.visual:
            delta_c_visual = torch.stack(delta_c_visual, dim=0)
            delta_m_visual = torch.stack(delta_m_visual, dim=0)
            visualization(self.configs.total_length, self.num_layers, delta_c_visual, delta_m_visual, self.visual_path)
            self.visual = 0

        # 计算总解耦损失
        decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))

        # 调整预测帧的维度顺序并计算总损失
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:]) + self.configs.decouple_beta * decouple_loss
        return next_frames, loss
