import numpy as np
import random


class InputHandle:
    """
    处理 SWAN 海浪数据集的输入
    """

    def __init__(self, input_param):
        """
        初始化输入处理器

        参数:
        - input_param: 字典，包含数据路径、minibatch大小、输入/输出时间步等配置信息
        """
        self.paths = input_param['paths']  # 数据路径
        self.minibatch_size = input_param['minibatch_size']  # batch size
        self.N = 10  # 输入时间步
        self.M = 10 # 预测时间步
        self.stride = input_param.get('stride', 3)  # 滑动窗口步长
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.output_data_type = input_param.get('output_data_type', 'float32')

        self.data = None  # 存储数据
        self.indices = []  # 存储样本索引
        self.current_position = 0
        self.current_batch_size = 0
        self.current_batch_indices = []

        self.load()
        self.create_indices()

    def load(self):
        """
        加载数据，并合并成 3 通道
        """
        # 加载 .npy 文件并提取字典
        data_dict = np.load(self.paths[0], allow_pickle=True).item()  # 使用 .item() 或 [()]
        hs = data_dict['hs']
        tm02 = data_dict['tm02']
        theta0 = data_dict['theta0']  # (T, W, H)

        # 合并成 3 通道数据 (T, W, H, C)
        self.data = np.stack([hs, tm02, theta0], axis=-1).astype(self.input_data_type)
        self.T, self.W, self.H, self.C = self.data.shape

        print(f"数据加载完成，形状: {self.data.shape}")  # 输出 (T, W, H, 3)

    def create_indices(self):
        """
        生成滑动窗口的索引列表
        """
        self.indices = []
        for i in range(0, self.T - self.N - self.M + 1, self.stride):
            self.indices.append(i)
        random.shuffle(self.indices)  # 打乱数据顺序
        print(f"总样本数: {len(self.indices)}")

    def total(self):
        """返回数据集中可用的样本数"""
        return len(self.indices)

    def begin(self):
        """初始化批次索引"""
        self.current_position = 0
        self.current_batch_size = min(self.minibatch_size, self.total())
        self.current_batch_indices = self.indices[:self.current_batch_size]

    def next(self):
        """获取下一个minibatch的索引"""
        self.current_position += self.current_batch_size
        if self.no_batch_left():
            return None
        self.current_batch_size = min(self.minibatch_size, self.total() - self.current_position)
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.current_batch_size]

    def no_batch_left(self):
        """检查是否还有剩余的 minibatch"""
        return self.current_position >= self.total()

    def input_batch(self):
        """
        获取当前minibatch的输入数据，形状 (batch_size, N, W, H, C)
        """
        if self.no_batch_left():
            return None

        input_batch = np.zeros((self.current_batch_size, self.N, self.W, self.H, self.C), dtype=self.input_data_type)

        for i, start_idx in enumerate(self.current_batch_indices):
            input_batch[i] = self.data[start_idx:start_idx + self.N]  # 取 N 帧作为输入

        return input_batch

    def output_batch(self):
        """
        获取当前minibatch的输出数据，形状 (batch_size, M, W, H, C)
        """
        if self.no_batch_left():
            return None

        output_batch = np.zeros((self.current_batch_size, self.M, self.W, self.H, self.C), dtype=self.output_data_type)

        for i, start_idx in enumerate(self.current_batch_indices):
            output_batch[i] = self.data[start_idx + self.N:start_idx + self.N + self.M]  # 取 M 帧作为预测目标

        return output_batch

    def get_batch(self):
        """获取当前minibatch的输入和输出数据"""
        input_seq = self.input_batch()
        output_seq = self.output_batch()
        return input_seq, output_seq
