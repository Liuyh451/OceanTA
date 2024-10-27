import numpy as np
import pandas as pd
import torch

from Util import cmip_dataset, ModelTrainer, AnticipationModule

file_path = 'E:/Dataset/waves/mode_residual.csv'
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


def standardize_dataset(file_path):
    # 读取数据
    data = pd.read_csv(file_path)
    # 标准化
    standardized_data, mean, std = standardize(data)
    columns = list(data.columns)
    train_data = standardized_data[columns]
    label_data = standardized_data['MWH']
    print(f'File: {file_path} - Standardized data shape: {train_data.shape}')
    print(f'File: {file_path} - Standardized data:\n{train_data.head()}')
    return train_data, label_data


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
X_data, y_data = standardize_dataset(file_path)
X_data = create_dataset(X_data, window_size)
# 划分训练集、验证集和测试集
y_train, y_val, y_test = split_label(y_data)
X_train = X_data[:train_size]
X_val = X_data[train_size:train_size + val_size]
X_test = X_data[train_size + val_size:train_size + val_size + test_size]
X_train = trans(X_train)
X_val = trans(X_val)
X_test = trans(X_test)
print(X_train.shape, X_val.shape, X_test.shape)
# 打印数据集的形状
print("训练集 X 形状:", X_train.shape)
print("训练集 y 形状:", y_train.shape)
print("验证集 X 形状:", X_val.shape)
print("验证集 y 形状:", y_val.shape)
print("测试集 X 形状:", X_test.shape)
print("测试集 y 形状:", y_test.shape)

# 创建模型训练器实例
trainer = ModelTrainer(
    input_channels=4,
    output_channels=1,
    batch_size=40,
    num_epochs=200,
    initial_lr=0.01,
    min_lr=0.00001,
)


# dataset_train = cmip_dataset(X_train, y_train)
# print('dataset_mode train shape', dataset_train.GetDataShape())
# dataset_eval = cmip_dataset(X_val, y_val)
# print('dataset_mode val shape', dataset_eval.GetDataShape())
# # 训练模型
# trainer.train(dataset_train, dataset_eval)
# print('mode train completed')
# # 保存模型
# trainer.save_model('./checkpoint.chk')
# print('model save completed')
#加载模型
chk = torch.load('./' + 'checkpoint.chk')
trainer.brain_analysis_module.load_state_dict(chk['net'])
# 测试
dataset_test = cmip_dataset(X_test, y_test)
print(dataset_test.GetDataShape())
all_predictions, all_targets = trainer.test_model(dataset_test)
pred_path = './data'+ '/predictions.npy'
target_path = './data'+ '/targets.npy'
np.save(pred_path, all_predictions)
np.save(target_path, all_targets)
print(f'预测值已保存到 {pred_path}')
print(f'真实值已保存到 {target_path}')

# def anticipate_mode_ori(data_path):
#     predication = np.load(data_path)
#     anticipation_module = AnticipationModule()
#     prediction_data = anticipation_module(predication)
#     return prediction_data
# anticipate_data = anticipate_mode_ori('data/dt1/predictions_inverse.npy')
# print(anticipate_data.shape)
# np.save('data/dt1/pre_original_data.npy', anticipate_data)

