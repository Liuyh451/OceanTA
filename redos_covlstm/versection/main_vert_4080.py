# todo 用点进行训练 确定最佳卷积核，以及调参
""""
找一些点，比如3个，那么在构造训练集时，应该从这个点纵深进行，取这个点1-24层的数据，构成训练集，测试集同理
"""
import xarray as xr
from datetime import datetime, timedelta
import torch
from torch.utils.data import DataLoader
import numpy as np


class dataPreprocess:
    def __init__(self, path, time_step):
        """
        初始化方法，传入待处理的数据及其他参数。
        """
        self.path = path
        self.time_step = time_step

    def fill_depth_for_zeta(self, ds):
        """
        给zeta补深度
        """
        zeta = ds['zeta']

        # 创建新的深度坐标，24层深度
        new_depth = np.arange(24)

        # 获取 zeta 的形状（假设它是2D：lat, lon）
        lat_dim, lon_dim = zeta.shape

        # 创建新的 zeta 数据数组，并初始化为 0
        new_zeta_data = np.zeros((24, lat_dim, lon_dim))  # 新的 zeta 数据，24层深度

        # 将原来的 zeta 数据放入新的数组中的第一层
        new_zeta_data[0, :, :] = zeta.values

        # 创建新的 zeta DataArray，并添加深度维度
        new_zeta = xr.DataArray(
            data=new_zeta_data,
            dims=['lv', 'lat', 'lon'],  # 新增 'lv' 维度，并保留原有纬度和经度
            coords={'lv': new_depth, 'lat': zeta.coords['lat'], 'lon': zeta.coords['lon']}
        )

        # 将新的 zeta 数据加入原数据集
        ds['zeta'] = new_zeta

        return ds

    def extract_profile_from_points(self, ds):
        """
               根据指定的经纬度提取数据点，并返回新的 Dataset。
               Parameters:
               ds (xarray.Dataset): 输入的 xarray Dataset。
               lon (float): 经度值。
               lat (float): 纬度值。

               Returns:
               xarray.Dataset: 包含指定点数据的 Dataset。
               """
        profiles = {}
        lon = 115
        lat = 14
        # 处理 u 和 v 变量（使用 (lonu, latu) 坐标系统）
        for var_name in ['u', 'v']:
            if var_name in ds:
                if var_name == 'u':
                    aligned_var = ds[var_name].sel(lonu=lon, latu=lat, method='nearest')
                else:
                    aligned_var = ds[var_name].sel(lonv=lon, latv=lat, method='nearest')
                profiles[var_name] = aligned_var

        # 处理 s 和 t 变量（使用 (lon, lat) 坐标系统）
        for var_name in ['s', 't', 'zeta']:
            if var_name in ds:
                # 直接提取 (lon, lat) 坐标系统中的数据
                profiles[var_name] = ds[var_name].sel(lon=lon, lat=lat, method='nearest')

        # 将提取的数据转换为 Dataset 格式
        profile_ds = xr.Dataset(profiles)

        return profile_ds

    def expand_dimensions(self, data_dict):
        """
        对字典中的指定变量进行维度扩展。

        参数:
        data_dict (dict): 包含数据的字典，键为变量名，值为 numpy 数组。

        返回:
        dict: 对指定变量进行维度扩展后的字典。
        """
        expanded_dict = {}

        for key, value in data_dict.items():
            if key in ['u', 'v', 's', 'zeta', 't']:
                # 对变量进行维度扩展，增加最后一个维度
                expanded_dict[key] = value.reshape(value.shape[0], value.shape[1], 1)
            else:
                # 对其他变量保持原样
                expanded_dict[key] = value

        return expanded_dict

    # 使用新方式读取数据，避免重复打开文件
    def load_all_nc_data(self, start_year, end_year):
        path = self.path
        data_dict = {}
        data_dict_Jan = {}
        current_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year + 1, 1, 1)
        while current_date < end_date:
            date_str = current_date.strftime('%Y%m%d')
            nc_file = path + '/subset_' + date_str + '.nc'
            # 读取 NetCDF 文件
            dataset = xr.open_dataset(nc_file)
            dataset = self.fill_depth_for_zeta(dataset)
            dataset = self.extract_profile_from_points(dataset)
            for var_name in dataset.variables:
                data = dataset[var_name].values
                # 替换标记值 -32768 为 NaN
                if np.issubdtype(data.dtype, np.integer):
                    # 如果数据是整数类型，标记值为整数
                    marker_value = -32768
                    # 转换为浮点型
                    data = data.astype(float)
                else:
                    # 如果数据是浮点类型，标记值为浮点数
                    marker_value = -32768.0

                    # 替换标记值为 NaN
                data[data == marker_value] = np.nan
                if current_date.month == 1:
                    if var_name not in data_dict_Jan:
                        data_dict_Jan[var_name] = []  # 初始化为列表
                    # 处理1月份的数据
                    data_dict_Jan[var_name].append(data)
                else:
                    if var_name not in data_dict:
                        data_dict[var_name] = []  # 初始化为列表
                    # 处理其他月份的数据
                    data_dict[var_name].append(data)

            current_date += timedelta(days=1)

            # 将列表转换为数组
        for var_name in data_dict:
            data_dict[var_name] = np.array(data_dict[var_name])
            data_dict_Jan[var_name] = np.array(data_dict_Jan[var_name])
        return data_dict, data_dict_Jan

    def create_dataset(self, data):
        time_step = self.time_step
        dataX = []
        for i in range(data.shape[0] - time_step + 1):
            dataX.append(data[i:i + time_step])
        return np.array(dataX)

    def read_col_data(self, data_dict, vtype):
        # 提取层数据
        data_dict = self.expand_dimensions(data_dict)
        raw_data = data_dict[vtype]
        point = raw_data.shape[2]  # 点数
        depth = raw_data.shape[1]  # 纵深

        X = self.create_dataset(raw_data)
        X = X.reshape(X.shape[0], self.time_step, depth, point, 1)
        Y = raw_data[self.time_step - 1:]
        Y = Y.reshape(Y.shape[0], depth, point, 1)

        # 转置维度
        X = X.transpose(0, 1, 4, 2, 3)
        Y = Y.transpose(0, 3, 1, 2)
        return X, Y, raw_data


file_path = '/home/hy4080/redos/subset_file/'
data_pre = dataPreprocess(file_path, 3)
# 划分训练集
data_dict, data_dict_Jan = data_pre.load_all_nc_data(1992, 2006)
train_sssa, _, _ = data_pre.read_col_data(data_dict, 's')
train_ssha, _, _ = data_pre.read_col_data(data_dict, 'zeta')
train_sswu, _, _ = data_pre.read_col_data(data_dict, 'u')
train_sswv, _, _ = data_pre.read_col_data(data_dict, 'v')
train_argo, label_argo, data_mask_t = data_pre.read_col_data(data_dict, 't')

# 划分验证集
test_sssa, _, _ = data_pre.read_col_data(data_dict_Jan, 's')
test_ssha, _, _ = data_pre.read_col_data(data_dict_Jan, 'zeta')
test_sswu, _, _ = data_pre.read_col_data(data_dict_Jan, 'u')
test_sswv, _, _ = data_pre.read_col_data(data_dict_Jan, 'v')
test_argo, label_test_argo, _ = data_pre.read_col_data(data_dict_Jan, 't')
del data_dict, data_dict_Jan  # 删除字典对象


def scaler(data):
    # normalise [0,1]
    data_max = np.nanmax(data)
    data_min = np.nanmin(data)
    data_scale = data_max - data_min
    data_std = (data - data_min) / data_scale
    # data_std = data_std * (2)  -1
    data_std[np.isnan(data_std)] = 0
    print("data_max------", data_max, "data_min------", data_min)
    return data_std, data_min, data_scale


# 反归一化
def unscaler(data, data_min, data_scale):
    data_inv = (data * data_scale) + data_min
    return data_inv


# 对数据进行归一化
sta_train, _, _ = scaler(train_argo[:])
ssa_train, _, _ = scaler(train_sssa[:])
ssha_train, _, _ = scaler(train_ssha[:])
sswu_train, _, _ = scaler(train_sswu[:])
sswv_train, _, _ = scaler(train_sswv[:])
true_train, _, _ = scaler(label_argo[:])

# 用历年一月份数据作为验证集
sta_test, _, _ = scaler(test_argo[:])
ssa_test, _, _ = scaler(test_sssa[:])
ssha_test, _, _ = scaler(test_ssha[:])
sswu_test, _, _ = scaler(test_sswu[:])
sswv_test, _, _ = scaler(test_sswv[:])

# 将多个不同类型的训练数据和测试数据沿着指定轴进行拼接，axis=2即增加特征的数量（即通道或变量的数量）
sta_train = np.concatenate((sta_train, ssa_train, ssha_train, sswu_train, sswv_train), axis=2)
sta_test = np.concatenate((sta_test, ssa_test, ssha_test, sswu_test, sswv_test), axis=2)

# true_test是归一化后的 label_argo 数据
# test_min是 数据中的最小值，在归一化过程中用作偏移量。
# test_scale，即最大值与最小值的差值。在归一化过程中用于缩放数据
true_test, test_min, test_scale = scaler(label_test_argo[:])

# 将拼接后的数据作为训练集
X_train = sta_train
# 训练集的标签
true_train = true_train

# 训练集上用于评估
X_eval = sta_test
#
true_eval = true_test
X_test = sta_test
true_test = true_test
print("X_train.shape", X_train.shape, "true_test.shape", true_test.shape, "X_eval.shape", X_eval.shape)


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
configs.batch_size_test = 64
configs.batch_size = 128
# configs.lr = 0.001
configs.weight_decay = 0
configs.display_interval = 400
configs.num_epochs = 800
# 这是早停的耐心参数。如果在N个epoch内性能没有改善，训练将停止
configs.early_stopping = True
configs.patience = 800
# 梯度裁剪用于防止梯度爆炸问题，但在这里未启用
configs.gradient_clipping = False
# 设置梯度裁剪的阈值为1。如果梯度裁剪启用，梯度的最大值将被限制为1。不过在这种配置下，由于梯度裁剪被禁用，这个参数实际上不会生效
configs.clipping_threshold = 1.

# lr warmup
# 这是学习率预热的步数设置。在训练的前n步内，学习率将逐渐从一个较小的值线性增加到预设的学习率。  √
configs.warmup = 2500

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
# 这个列表包含了一系列的深度值，这可能与模型的层次结构或者不同深度的输入特征相关联
configs.depth = [5, 6, 11, 16, 20, 25, 30, 34, 36, 38, 40, 42, 44, 46, 48, 50, 51, 52, 53, 54, 55, 57]
# 这个列表可能对应于不同深度的索引或层次级别。每个索引可能用于定位或选择特定深度的特征或数据
configs.depthindex = [30, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500,
                      1600, 1700, 1800, 1900]

# model
# 表示模型的维度即每个输入数据在模型中的表示为256维    √
configs.d_model = 256
# 表示模型处理数据时的patch（小块）的大小为5×5。这通常用于图像或序列数据的分块处理
configs.patch_size = (5, 5)
# 表示嵌入的空间尺寸。这里12*16可能是表示最终嵌入的特征图的尺寸（例如视觉模型中的特征图大小）
# 这个地方测试是是不是改为2*4 这个地方找不到哪里用了
configs.emb_spatial_size = 5 * 10
# 表示多头注意力机制中的头数为4。多头注意力允许模型从不同的角度“看”数据，从而捕捉不同的关系
configs.nheads = 4
# 表示前馈神经网络的维度用于增加模型的表达能力,          √
configs.dim_feedforward = 512
# 表示在模型中使用的dropout率为0.3。Dropout是一种正则化技术，用于减少过拟合。   √
configs.dropout = 0.3
# 表示编码器的层数为4。这意味着模型有4个堆叠的编码器层
configs.num_encoder_layers = 4
configs.num_decoder_layers = 4
# 这可能是学习率的衰减率，用来控制学习率的递减速度     √
configs.ssr_decay_rate = 3.e-6

# plot 表示绘图的分辨率为600 DPI
configs.plot_dpi = 600

from utils_point import cmip_dataset, loss
from utils_point import Trainer

dataset_train = cmip_dataset(X_train, true_train)
print(dataset_train.GetDataShape())

dataset_eval = cmip_dataset(X_eval, true_eval)
print(dataset_eval.GetDataShape())

trainer = Trainer(configs)
trainer.save_configs('config_train.pkl')

trainer.train(dataset_train, dataset_eval, 'checkpoint.chk')

dataset_test = cmip_dataset(X_test, true_test)
dataloader_test = DataLoader(dataset_test, batch_size=configs.batch_size_test, shuffle=False)
print(dataset_test.GetDataShape())

chk = torch.load('./checkpoint.chk')

trainer.network.load_state_dict(chk['net'])

loss_test, test_pred, test_true = trainer.infer_test(dataset=dataset_test, dataloader=dataloader_test)

print(loss_test)

test_pred = test_pred.cpu().numpy()
test_true = test_true.cpu().numpy()

test_pred = unscaler(np.array(test_pred), test_min, test_scale)
test_true = unscaler(np.array(test_true), test_min, test_scale)

np.save("./data/test_pred.npy", test_pred)
np.save("./data/test_true.npy", test_true)

print(test_pred.shape)
print(test_true.shape)
# todo 这里他写的形状突然由(12, 1, 28, 52)变为了(12,28,52,1)没见到改变形状的操作啊
# 对数组重新塑形，这里是np不是张量
test_true = np.transpose(test_true, (0, 3, 1, 2))
test_pred = np.transpose(test_pred, (0, 3, 1, 2))
print(test_pred.shape)
print(test_true.shape)
test_pred = np.squeeze(test_pred)
test_true = np.squeeze(test_true)
cha = (test_true[0] - test_pred[0]) ** 2
test_pred[np.isnan(test_pred)] = 0
test_true[np.isnan(test_true)] = 0

rmse = []
corr = []
for i in range(test_pred.shape[0]):
    predict_result = test_pred[i]
    # print(predict_result)
    true_result = test_true[i]
    #由于我改成了1个点所以原来的第二维就是总大小
    total = predict_result.shape[0]
    #print(total)
    sse = np.sum((true_result - predict_result) ** 2)
    print(i,"_sse:",sse)
    rmse_temp = np.sqrt(sse / total)
    '''
    if i == 0:
        print(total)
        print(sse)
        print(rmse_temp)
    '''
    # print( np.sum(rmse_temp) / len(rmse_temp))
    rmse.append(rmse_temp)

    predict_result_f = predict_result.flatten()
    true_result_f = true_result.flatten()
    corr_temp = np.corrcoef(predict_result_f, true_result_f)[0, -1]
    corr.append(corr_temp)
RMSE = np.sum(rmse) / len(rmse)
CORR = np.sum(corr) / len(corr)
print("RMSE:",RMSE)
print("CORR",CORR)
nrmse = loss(data_mask_t, 1, test_pred, test_true)
