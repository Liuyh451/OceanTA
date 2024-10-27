import numpy as np
import pandas as pd
import torch

from model import cmip_dataset, ModelTrainer, AnticipationModule

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


def standardize_dataset():
    file_prefix = 'mode'
    file_suffix = '.csv'
    num_modes = 8
    standardized_data_list = []
    standardized_label_list = []
    for i in range(1, num_modes + 1):
        # 读取数据
        file_path_mode = file_path + f'{file_prefix}{i}{file_suffix}'
        data = pd.read_csv(file_path_mode)
        # 标准化
        standardized_data, mean, std = standardize(data)
        columns = list(data.columns)
        train_data = standardized_data[columns]
        label_data = standardized_data['MWH']
        # 保存到临时变量
        standardized_data_list.append(train_data)
        standardized_label_list.append(label_data)
        print(f'File: {file_path} - Standardized data shape: {train_data.shape}')
        print(f'File: {file_path} - Standardized data:\n{train_data.head()}')
        # 按列（axis=1）拼接数据
    combined_data = pd.concat(standardized_data_list, axis=1)
    print(f'Combined data shape: {combined_data.shape}')
    return combined_data, standardized_label_list


def split_label(y):
    # 切分数据集
    y_train = y[17:17 + train_size].reset_index(drop=True)  # 从第18个数据开始
    y_val = y[17 + train_size:17 + train_size + val_size].reset_index(drop=True)  # 验证集
    y_test = y[17 + train_size + val_size:17 + train_size + val_size + test_size].reset_index(drop=True)  # 测试集
    return y_train, y_val, y_test


def trans(data):
    # data1 = create_dataset(data, window_size)
    transed_data = data.transpose(0, 2, 1)
    return transed_data


# 标准化数据集
X_data, y_data = standardize_dataset()
X_data = create_dataset(X_data, window_size)
# 划分训练集、验证集和测试集
# 初始化空列表保存结果
y_train_list = []
y_val_list = []
y_test_list = []
# 遍历standardized_label_list中的每个mode的数据，并调用split_label函数
for y in y_data:
    y_train, y_val, y_test = split_label(y)
    y_train_list.append(y_train)  # 将结果添加到y_train_list
    y_val_list.append(y_val)  # 将结果添加到y_val_list
    y_test_list.append(y_test)  # 将结果添加到y_test_list

X_train = X_data[:train_size]
X_val = X_data[train_size:train_size + val_size]
X_test = X_data[train_size + val_size:train_size + val_size + test_size]
X_train = trans(X_train)
X_val = trans(X_val)
X_test = trans(X_test)
print(X_train.shape, X_val.shape, X_test.shape)
# 打印数据集的形状
print("训练集 X 形状:", X_train.shape)
print("训练集 y 形状:", y_train_list[0].shape)
print("验证集 X 形状:", X_val.shape)
print("验证集 y 形状:", y_val_list[0].shape)
print("测试集 X 形状:", X_test.shape)
print("测试集 y 形状:", y_test_list[0].shape)

# 创建模型训练器实例
trainer = ModelTrainer(
    input_channels=32,
    output_channels=8,
    batch_size=40,
    num_epochs=200,
    initial_lr=0.01,
    min_lr=0.00001,
)

# for i in range(1,2):
#     dataset_train = cmip_dataset(X_train, y_train_list[i - 1])
#     print(f'dataset_mode{i} train shape', dataset_train.GetDataShape())
#     dataset_eval = cmip_dataset(X_val, y_val_list[i - 1])
#     print(f'dataset_mode{i} val shape', dataset_eval.GetDataShape())
#     # 训练模型
#     trainer.train(dataset_train, dataset_eval)
#     print(f'mode{i} train completed')
#     # 保存模型
#     trainer.save_model('./net/' + f'checkpoint{i}.chk')
#     print(f'model{i} save completed')
all_predictions_list = []
all_targets_list = []
for i in range(1, 9):
    # 加载模型
    chk = torch.load('./net/' + f'checkpoint{i}.chk')
    trainer.brain_analysis_module.load_state_dict(chk['net'])
    # 测试
    dataset_test = cmip_dataset(X_test, y_test_list[i - 1])
    print(dataset_test.GetDataShape())
    all_predictions, all_targets = trainer.test_model(dataset_test)
    # 将每个模式的预测值和真实值添加到列表中
    all_predictions_list.append(all_predictions.reshape(-1, 1))
    all_targets_list.append(all_targets.reshape(-1, 1))
# 将列表中的数据转换为 (600, 8) 的数组
predictions_combined = np.hstack(all_predictions_list)
targets_combined = np.hstack(all_targets_list)

# 保存最终的预测值和真实值数组
pred_path = 'data/single/predictions.npy'
target_path = 'data/single/targets.npy'
np.save(pred_path, predictions_combined)
np.save(target_path, targets_combined)
print(f'所有模式的预测值已保存到 {pred_path}')
print(f'所有模式的真实值已保存到 {target_path}')

