from core.data_provider import  mnist,swan

# 数据集名称与对应数据处理模块的映射
datasets_map = {
    'mnist': mnist,
    'swan':swan,
}

def data_provider(dataset_name, train_data_paths, valid_data_paths, batch_size,
                  input_length, seq_length, injection_action, is_training=True):
    """
    提供不同数据集的数据输入接口。

    参数:
        dataset_name: 数据集名称。
        train_data_paths: 训练数据路径，多个路径用逗号分隔。
        valid_data_paths: 验证数据路径，多个路径用逗号分隔。
        batch_size: 每个批次的数据大小。
        img_width: 图像宽度。
        seq_length: 序列长度。
        injection_action: 数据注入动作。
        is_training: 是否处于训练模式，默认为True。

    返回:
        如果是训练模式，返回训练和测试数据迭代器；否则只返回测试数据迭代器。
    """
    # 检查数据集名称是否在已知数据集中
    if dataset_name not in datasets_map:
        raise ValueError('未知的数据集名称 %s' % dataset_name)

    # 将训练和验证数据路径分割成列表
    train_data_list = train_data_paths.split(',')
    valid_data_list = valid_data_paths.split(',')
    if dataset_name=='swan':
        # 准备测试数据参数
        test_input_param = {
            'paths': valid_data_list,
            'minibatch_size': batch_size,
            'input_data_type': 'float32',
            'is_output_sequence': True,
            'name': 'swan test iterator',
            'input_length': input_length,
            'pred_length': seq_length - input_length
        }
        test_input_handle = datasets_map['swan'].InputHandle(test_input_param)
        test_input_handle.begin(do_shuffle=False)  # 不打乱顺序开始测试数据迭代

        # 如果是训练模式，准备训练数据参数
        if is_training:
            train_input_param = {
                'paths': train_data_list,
                'minibatch_size': batch_size,
                'input_data_type': 'float32',
                'is_output_sequence': True,
                'name': 'swan train iterator',
                'input_length': input_length,
                'pred_length':seq_length-input_length
            }
            train_input_handle = datasets_map['swan'].InputHandle(train_input_param)
            train_input_handle.begin(do_shuffle=True)  # 训练数据需要打乱
            return train_input_handle, test_input_handle
        else:
            return test_input_handle
    # 处理MNIST数据集
    if dataset_name == 'mnist':
        # 准备测试数据参数
        test_input_param = {'paths': valid_data_list,
                            'minibatch_size': batch_size,
                            'input_data_type': 'float32',
                            'is_output_sequence': True,
                            'name': dataset_name + 'test iterator'}
        test_input_handle = datasets_map[dataset_name].InputHandle(test_input_param)
        test_input_handle.begin(do_shuffle=False)  # 不打乱顺序开始测试数据迭代

        # 如果是训练模式，准备训练数据参数
        if is_training:
            train_input_param = {'paths': train_data_list,
                                 'minibatch_size': batch_size,
                                 'input_data_type': 'float32',
                                 'is_output_sequence': True,
                                 'name': dataset_name + ' train iterator'}
            train_input_handle = datasets_map[dataset_name].InputHandle(train_input_param)
            train_input_handle.begin(do_shuffle=True)  # 打乱顺序开始训练数据迭代
            return train_input_handle, test_input_handle
        else:
            return test_input_handle



