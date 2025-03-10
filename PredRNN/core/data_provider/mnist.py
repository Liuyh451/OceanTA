import numpy as np
import random

class InputHandle:
    """
    输入数据处理器

    该类负责根据提供的参数加载数据，并对数据进行预处理，以便于后续的训练或测试使用
    它支持从多个路径加载数据，并可以处理不同类型的输入和输出数据
    """

    def __init__(self, input_param):
        """
        初始化输入处理器

        参数:
        - input_param: 字典，包含数据路径、名称、数据类型、minibatch大小等配置信息
        """
        self.paths = input_param['paths']
        self.num_paths = len(input_param['paths'])
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.output_data_type = input_param.get('output_data_type', 'float32')
        self.minibatch_size = input_param['minibatch_size']
        self.is_output_sequence = input_param['is_output_sequence']
        self.data = {}
        self.indices = {}
        self.current_position = 0
        self.current_batch_size = 0
        self.current_batch_indices = []
        self.current_input_length = 0
        self.current_output_length = 0
        self.load()

    def load(self):
        """
        加载数据

        该方法从指定的路径中加载数据，并根据路径的数量决定如何合并数据
        它还负责打印出加载的数据的关键字和形状，以便于调试和验证
        """
        dat_1 = np.load(self.paths[0])
        for key in dat_1.keys():
            self.data[key] = dat_1[key]
        if self.num_paths == 2:
            dat_2 = np.load(self.paths[1])
            num_clips_1 = dat_1['clips'].shape[1]
            dat_2['clips'][:,:,0] += num_clips_1
            self.data['clips'] = np.concatenate(
                (dat_1['clips'], dat_2['clips']), axis=1)
            self.data['input_raw_data'] = np.concatenate(
                (dat_1['input_raw_data'], dat_2['input_raw_data']), axis=0)
            self.data['output_raw_data'] = np.concatenate(
                (dat_1['output_raw_data'], dat_2['output_raw_data']), axis=0)
        for key in self.data.keys():
            print(key)
            print(self.data[key].shape)

    def total(self):
        """
        获取数据集大小

        返回:
        - 数据集中样本的总数
        """
        return self.data['clips'].shape[1]

    def begin(self, do_shuffle = True):
        """
        开始一个新的epoch

        参数:
        - do_shuffle: 布尔值，指示是否在开始之前打乱数据顺序
        该方法准备数据集，包括创建索引和确定第一个minibatch的大小和内容
        """
        self.indices = np.arange(self.total(),dtype="int32")
        if do_shuffle:
            random.shuffle(self.indices)
        self.current_position = 0
        if self.current_position + self.minibatch_size <= self.total():
            self.current_batch_size = self.minibatch_size
        else:
            self.current_batch_size = self.total() - self.current_position
        self.current_batch_indices = self.indices[
            self.current_position:self.current_position + self.current_batch_size]
        self.current_input_length = max(self.data['clips'][0, ind, 1] for ind
                                        in self.current_batch_indices)
        self.current_output_length = max(self.data['clips'][1, ind, 1] for ind
                                         in self.current_batch_indices)

    def next(self):
        """
        获取下一个minibatch的数据

        该方法更新当前的位置，并确定下一个minibatch的大小和内容
        如果没有更多的数据，则返回None
        """
        self.current_position += self.current_batch_size
        if self.no_batch_left():
            return None
        if self.current_position + self.minibatch_size <= self.total():
            self.current_batch_size = self.minibatch_size
        else:
            self.current_batch_size = self.total() - self.current_position
        self.current_batch_indices = self.indices[
            self.current_position:self.current_position + self.current_batch_size]
        self.current_input_length = max(self.data['clips'][0, ind, 1] for ind
                                        in self.current_batch_indices)
        self.current_output_length = max(self.data['clips'][1, ind, 1] for ind
                                         in self.current_batch_indices)

    def no_batch_left(self):
        """
        检查是否还有剩余的minibatch

        返回:
        - 布尔值，如果不再有剩余的minibatch，则为True，否则为False
        """
        if self.current_position >= self.total() - self.current_batch_size:
            return True
        else:
            return False

    def input_batch(self):
        """
        获取当前minibatch的输入数据

        返回:
        - 当前minibatch的输入数据数组，如果没有更多数据，则返回None
        """
        if self.no_batch_left():
            return None
        input_batch = np.zeros(
            (self.current_batch_size, self.current_input_length) +
            tuple(self.data['dims'][0])).astype(self.input_data_type)
        input_batch = np.transpose(input_batch,(0,1,3,4,2))
        for i in range(self.current_batch_size):
            batch_ind = self.current_batch_indices[i]
            begin = self.data['clips'][0, batch_ind, 0]
            end = self.data['clips'][0, batch_ind, 0] + \
                    self.data['clips'][0, batch_ind, 1]
            data_slice = self.data['input_raw_data'][begin:end, :, :, :]
            data_slice = np.transpose(data_slice,(0,2,3,1))
            input_batch[i, :self.current_input_length, :, :, :] = data_slice
        input_batch = input_batch.astype(self.input_data_type)
        return input_batch

    def output_batch(self):
        """
        获取当前minibatch的输出数据

        返回:
        - 当前minibatch的输出数据数组，如果没有更多数据，则返回None
        """
        if self.no_batch_left():
            return None
        if(2 ,3) == self.data['dims'].shape:
            raw_dat = self.data['output_raw_data']
        else:
            raw_dat = self.data['input_raw_data']
        if self.is_output_sequence:
            if (1, 3) == self.data['dims'].shape:
                output_dim = self.data['dims'][0]
            else:
                output_dim = self.data['dims'][1]
            output_batch = np.zeros(
                (self.current_batch_size,self.current_output_length) +
                tuple(output_dim))
        else:
            output_batch = np.zeros((self.current_batch_size, ) +
                                    tuple(self.data['dims'][1]))
        for i in range(self.current_batch_size):
            batch_ind = self.current_batch_indices[i]
            begin = self.data['clips'][1, batch_ind, 0]
            end = self.data['clips'][1, batch_ind, 0] + \
                    self.data['clips'][1, batch_ind, 1]
            if self.is_output_sequence:
                data_slice = raw_dat[begin:end, :, :, :]
                output_batch[i, : data_slice.shape[0], :, :, :] = data_slice
            else:
                data_slice = raw_dat[begin, :, :, :]
                output_batch[i,:, :, :] = data_slice
        output_batch = output_batch.astype(self.output_data_type)
        output_batch = np.transpose(output_batch, [0,1,3,4,2])
        return output_batch

    def get_batch(self):
        """
        获取当前minibatch的输入和输出数据

        返回:
        - 当前minibatch的数据数组，其中包括输入和输出数据，如果没有更多数据，则返回None
        """
        input_seq = self.input_batch()
        output_seq = self.output_batch()
        batch = np.concatenate((input_seq, output_seq), axis=1)
        return batch
