import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset

from Util import cmip_dataset, ModelTrainer
from Util import ModelEvaluator

file_path = 'E:/Dataset/waves/'
# 设置滑动窗口大小
window_size = 17
train_size = 1906
val_size = 477
test_size = 600


def create_dataset(data, time_step):
    dataX = []
    for i in range(data.shape[0] - time_step + 1):
        dataX.append(data[i:i + time_step])
    return np.array(dataX)


# def create_dataset2(data, window_size):
#     X, y = [], []
#     for i in range(len(data) - window_size):
#         X.append(data[i:i + window_size])
#         y.append(data[i + window_size])  # 预测下一个值
#     return np.array(X), np.array(y)

# todo 测试集不能用分解后的数据

def standardize(data):
    """
    按通道标准化多通道数据，返回标准化后的数据、均值和标准差。

    参数：
    data: np.ndarray - 输入数据，形状为 (样本数, 通道数)。

    返回：
    standardized_data: np.ndarray - 标准化后的数据。
    means: np.ndarray - 每个通道的均值。
    stds: np.ndarray - 每个通道的标准差。
    """
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)

    standardized_data = (data - means) / stds

    return standardized_data, means, stds


def denormalize(z, mu, sigma):
    return z * sigma + mu


def standardize_train():
    file_prefix = 'mode'
    file_suffix = '.csv'
    num_modes = 8
    standardized_data_list = []
    for i in range(1, num_modes + 1):
        # 读取数据
        file_path_mode = file_path + f'{file_prefix}{i}{file_suffix}'
        data = pd.read_csv(file_path_mode)
        # 标准化
        standardized_data, mean, std = standardize(data)
        columns = list(data.columns)
        train_data = standardized_data[columns]
        # 保存到临时变量
        standardized_data_list.append(train_data)
        print(f'File: {file_path} - Standardized data shape: {train_data.shape}')
        print(f'File: {file_path} - Standardized data:\n{train_data.head()}')
        # 按列（axis=1）拼接数据
    combined_data = pd.concat(standardized_data_list, axis=1)
    print(f'Combined data shape: {combined_data.shape}')
    return combined_data


def standardize_label(column):
    file_path_ori = file_path + 'dataset1.csv'
    data = pd.read_csv(file_path_ori)
    # 忽略 'Date/Time' 列
    if 'Date/Time' in data.columns:
        data = data.drop(columns=['Date/Time'])
    y = data
    # 切分数据集
    y_train = y[:train_size].reset_index(drop=True)  # 重置索引
    y_val = y[train_size:train_size + val_size].reset_index(drop=True)  # 重置索引
    y_test = y[train_size + val_size:train_size + val_size + test_size].reset_index(drop=True)  # 重置索引
    # 标准化
    standardized_data_test, mean, std = standardize(y_test)
    standardized_data_train, _, _ = standardize(y_train)
    standardized_data_val, _, _ = standardize(y_val)
    # 提取标签列
    label_data_train = standardized_data_train[column]
    label_data_test = standardized_data_test[column]
    label_data_val = standardized_data_val[column]
    return label_data_train, label_data_val, label_data_test, mean, std


def trans(data):
    data1 = create_dataset(data, window_size)
    transed_data = data1.transpose(0, 2, 1)
    return transed_data


# 划分训练集、验证集和测试集
X_data = standardize_train()
X_train = X_data[:train_size].reset_index(drop=True)
X_val = X_data[train_size:train_size + val_size].reset_index(drop=True)
X_test = X_data[train_size + val_size:train_size + val_size + test_size].reset_index(drop=True)
X_train = trans(X_train)
X_val = trans(X_val)
X_test = trans(X_test)
y_train, y_val, y_test, test_mean, test_std = standardize_label('MWH')
# todo 测试label是用原数据还是分解后的数据
# 打印数据集的形状
print("训练集 X 形状:", X_train.shape)
print("训练集 y 形状:", y_train.shape)
print("验证集 X 形状:", X_val.shape)
print("验证集 y 形状:", y_val.shape)
print("测试集 X 形状:", X_test.shape)
print("测试集 y 形状:", y_test.shape)
dataset_train = cmip_dataset(X_train, y_train)
print(dataset_train.GetDataShape())
dataset_eval = cmip_dataset(X_val, y_val)
print(dataset_eval.GetDataShape())
dataset_test = cmip_dataset(X_test, y_test)
print(dataset_test.GetDataShape())
# 创建模型训练器实例
trainer = ModelTrainer(train_data=dataset_train,
                       val_data=dataset_eval,
                       test_data=dataset_test,
                       input_channels=32,
                       output_channels=8,
                       batch_size=40,
                       num_epochs=200,
                       initial_lr=0.01,
                       min_lr=0.00001)

# 训练模型
trainer.train()
