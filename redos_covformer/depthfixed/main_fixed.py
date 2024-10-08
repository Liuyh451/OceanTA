import numpy as np
import netCDF4 as nc
import xarray as xr
from datetime import datetime, timedelta

from scipy.interpolate import RegularGridInterpolator

""""
使用新方式读取数据，避免重复打开文件
"""
depth_globel=10

def resize_nc_variables(nc_file, new_shape=(60, 80)):
    dataset = xr.open_dataset(nc_file)
    resized_data = {}

    for var_name in dataset.variables:
        data = dataset[var_name].values
        data[data == -32768.0] = np.nan  # 替换标记值为 NaN

        if data.ndim == 3:  # 三维数据
            depth, lat, lon = dataset[var_name].dims
            if var_name in ['u', 'v']:
                if var_name == 'u':
                    lat, lon = 'latu', 'lonu'
                elif var_name == 'v':
                    lat, lon = 'latv', 'lonv'
            old_coords = (dataset[depth].values, dataset[lat].values, dataset[lon].values)
            new_coords = np.meshgrid(dataset[depth].values,
                                     np.linspace(dataset[lat].values.min(), dataset[lat].values.max(), new_shape[0]),
                                     np.linspace(dataset[lon].values.min(), dataset[lon].values.max(), new_shape[1]),
                                     indexing='ij')
            interpolator = RegularGridInterpolator(old_coords, data, bounds_error=False, fill_value=np.nan)
            resized_data[var_name] = interpolator(tuple(map(np.ravel, new_coords))).reshape(len(dataset[depth]),
                                                                                            *new_shape)

        elif data.ndim == 2:  # 二维数据
            lat, lon = dataset[var_name].dims
            old_coords = (dataset[lat].values, dataset[lon].values)
            new_coords = np.meshgrid(np.linspace(dataset[lat].values.min(), dataset[lat].values.max(), new_shape[0]),
                                     np.linspace(dataset[lon].values.min(), dataset[lon].values.max(), new_shape[1]),
                                     indexing='ij')
            interpolator = RegularGridInterpolator(old_coords, data, bounds_error=False, fill_value=np.nan)
            resized_data[var_name] = interpolator(tuple(map(np.ravel, new_coords))).reshape(new_shape)

    return resized_data
# 这样写可以减少重复打开文件
def load_all_nc_data(path, start_year, end_year):
    data_dict = {}
    data_dict_Jan={}
    current_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year + 1, 1, 1)
    while current_date < end_date:
        date_str = current_date.strftime('%Y%m%d')
        nc_file = path + '/subset_' + date_str + '.nc'
        # 当月份大于2时停止循环
        if current_date.month > 11:
            print("月份大于3，停止循环。")
            break
        # 读取 NetCDF 文件
        resized_data = resize_nc_variables(nc_file)
        for var_name in resized_data:
            data = resized_data[var_name]
            data[data == -32768.0] = np.nan  # 替换标记值
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
def extract_layer_data(data_dict, var_type, depth):
    if var_type in data_dict:
        data = data_dict[var_type]
        #zeta为海表没有深度，所以永远为0
        if var_type=='zeta':
            data_type = data[ :,:, :]
        else:
            data_type = data[:,depth, :, :]
        return data_type
    else:
        raise ValueError(f"Variable '{var_type}' not found in data dictionary.")

def create_dataset(data, time_step):
    dataX = []
    for i in range(data.shape[0] - time_step + 1):
        dataX.append(data[i:i + time_step])
    return np.array(dataX)

def read_raw_data(var_type, depth, time_step, data_dict):
    # 提取层训练数据
    raw_data_1 = extract_layer_data(data_dict, var_type, 23)
    # 提取层标签数据
    raw_data_2 = extract_layer_data(data_dict, var_type, depth)
    width = raw_data_1.shape[2]  # 经度
    length = raw_data_1.shape[1]  # 纬度

    X = create_dataset(raw_data_1, time_step)
    X = X.reshape(X.shape[0], time_step, length, width, 1)
    Y = raw_data_2[time_step - 1:]
    Y = Y.reshape(Y.shape[0], length, width, 1)

    # 转置维度
    X = X.transpose(0, 1, 4, 2, 3)
    Y = Y.transpose(0, 3, 1, 2)
    return X, Y, raw_data_2
# 使用示例
file_path = 'E:/DataSet/redos/Subset_1.0_1995'
data_dict,data_dict_Jan=load_all_nc_data(file_path,1995,1995)
#划分训练集
train_sssa, _, _ = read_raw_data('s', 23, 3, data_dict)
train_ssha, _, _ = read_raw_data('zeta', 23, 3, data_dict)
train_sswu, _, _ = read_raw_data('u', 23, 3, data_dict)
train_sswv, _, _ = read_raw_data('v', 23, 3, data_dict)
train_argo, label_argo, data_mask_t_1 = read_raw_data('t', depth_globel, 3, data_dict)
#划分验证集
test_sssa, _, _ = read_raw_data('s', 23, 3, data_dict_Jan)
test_ssha, _, _ = read_raw_data('zeta', 23, 3, data_dict_Jan)
test_sswu, _, _ = read_raw_data('u', 23, 3, data_dict_Jan)
test_sswv, _, _ = read_raw_data('v', 23, 3, data_dict_Jan)
test_argo, label_test_argo, data_mask_t_2 = read_raw_data('t', depth_globel, 3, data_dict_Jan)
del data_dict,data_dict_Jan# 删除字典对象
def scaler(data):
    #normalise [0,1]
    data_max = np.nanmax(data)
    data_min = np.nanmin(data)
    data_scale = data_max - data_min
    data_std = (data - data_min) / data_scale
    # data_std = data_std * (2)  -1
    data_std [np.isnan(data_std)] = 0
    print("data_max------",data_max,"data_min------", data_min)
    return data_std,data_min,data_scale

#反归一化
def unscaler(data, data_min, data_scale):
    data_inv = (data * data_scale) + data_min
    return data_inv

num_test=12
#对数据进行归一化
print("sta_train",end=' ')
sta_train,_,_ = scaler(train_argo[:])
print("ssa_train",end=' ')
ssa_train,_,_  = scaler(train_sssa[:])
print("ssha_train",end=' ')
ssha_train,_,_ = scaler(train_ssha[:])
print("sswu_train",end=' ')
sswu_train,_,_ = scaler(train_sswu[:])
print("sswv_train",end=' ')
sswv_train,_,_ = scaler(train_sswv[:])
print("train_train",end=' ')
true_train,_,_ = scaler(label_argo[:])

#用历年一月份数据作为验证集
sta_test,_,_ = scaler(test_argo[:])
ssa_test,_,_  = scaler(test_sssa[:])
ssha_test,_,_ = scaler(test_ssha[:])
sswu_test,_,_ = scaler(test_sswu[:])
sswv_test,_,_ = scaler(test_sswv[:])

#将多个不同类型的训练数据和测试数据沿着指定轴进行拼接，axis=2即增加特征的数量（即通道或变量的数量）
sta_train = np.concatenate((sta_train,ssa_train,ssha_train,sswu_train,sswv_train),axis = 2 )
sta_test = np.concatenate((sta_test,ssa_test,ssha_test,sswu_test,sswv_test),axis = 2)
data_mask_t=np.concatenate((data_mask_t_1,data_mask_t_2),axis = 0)
print("data_mask_t.shape",data_mask_t.shape)
true_test,test_min,test_scale = scaler(label_test_argo[:])
#true_test是归一化后的 label_argo 数据，对应于最后 12 个时间步的标签数据
#test_min是 label_argo[-12:] 数据中的最小值，在归一化过程中用作偏移量。
#test_scale是 label_argo[-12:] 数据的范围，即最大值与最小值的差值。在归一化过程中用于缩放数据


#将拼接后的数据作为训练集
X_train = sta_train
#训练集的标签
true_train = true_train

#训练集上用于评估
X_eval = sta_test
#
true_eval = true_test
X_test = sta_test
true_test = true_test
print("X_train.shape",X_train.shape,"true_test.shape",true_test.shape,"X_eval.shape",X_eval.shape)


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import numpy as np
import math
from torch.utils.data import Dataset

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
configs.batch_size_test = 10
configs.batch_size = 16
#configs.lr = 0.001
configs.weight_decay = 0
configs.display_interval = 20
configs.num_epochs = 300
#这是早停的耐心参数。即使模型在900个epoch内没有改善性能，训练仍会继续。如果在900个epoch内性能没有改善，训练将停止
configs.early_stopping = True
configs.patience = 150
#禁用梯度裁剪（Gradient Clipping）。梯度裁剪用于防止梯度爆炸问题，但在这里未启用
configs.gradient_clipping = False
#设置梯度裁剪的阈值为1。如果梯度裁剪启用，梯度的最大值将被限制为1。不过在这种配置下，由于梯度裁剪被禁用，这个参数实际上不会生效
configs.clipping_threshold = 1.

# lr warmup
#这是学习率预热的步数设置。在训练的前3000步内，学习率将逐渐从一个较小的值线性增加到预设的学习率。这种技术通常用于训练的初始阶段，以帮助模型更稳定地开始训练，减少初期的震荡。
configs.warmup = 500

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
configs.d_model = 128
#表示模型处理数据时的patch（小块）的大小为5×5。这通常用于图像或序列数据的分块处理
configs.patch_size = (5,5)
#表示嵌入的空间尺寸。这里12*16可能是表示最终嵌入的特征图的尺寸（例如视觉模型中的特征图大小）
#todo 这个地方测试是是不是改为5*10，或者7*8(planB)
configs.emb_spatial_size = 12*16
#表示多头注意力机制中的头数为4。多头注意力允许模型从不同的角度“看”数据，从而捕捉不同的关系
configs.nheads = 4
#表示前馈神经网络的维度用于增加模型的表达能力
configs.dim_feedforward =256
#表示在模型中使用的dropout率为0.3。Dropout是一种正则化技术，用于减少过拟合。
configs.dropout = 0.3
#表示编码器的层数为4。这意味着模型有4个堆叠的编码器层
configs.num_encoder_layers = 4
configs.num_decoder_layers = 4
#这可能是学习率的衰减率（scheduler decay rate），用来控制模型训练过程中学习率的递减速度，以便在训练的后期进行更细致的优化
configs.ssr_decay_rate = 3.e-4


# plot 表示绘图的分辨率为600 DPI
configs.plot_dpi = 600

from utils import  cmip_dataset
from utils import  Trainer

dataset_train = cmip_dataset(X_train,true_train)
print(dataset_train.GetDataShape())


dataset_eval = cmip_dataset(X_eval,true_eval)
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

test_pred = unscaler(np.array(test_pred),test_min,test_scale)
test_true = unscaler(np.array(test_true),test_min,test_scale)
# 应该不用加，推测是用来加新数据的

# test_pred = add('temp', 1, test_pred)
# test_true = add('temp', 1, test_true)

np.save("./data/test_pred.npy",test_pred)
np.save("./data/test_true.npy",test_true)

print(test_pred.shape)
print(test_true.shape)
#todo 这里他写的形状突然由(12, 1, 28, 52)变为了(12,28,52,1)没见到改变形状的操作啊
#对数组重新塑形，这里是np不是张量
test_true = np.transpose(test_true, (0, 3, 1, 2))
test_pred = np.transpose(test_pred, (0, 3, 1, 2))
print(test_pred.shape)
print(test_true.shape)
test_pred = np.squeeze(test_pred)
test_true = np.squeeze(test_true)
cha = (test_true[0] - test_pred[0]) ** 2
test_pred[np.isnan(test_pred)] = 0
test_true[np.isnan(test_true)] = 0
print("test_pred",test_pred[0, 0, :10])
print("test_true",test_true[0, 0, :10])
rmse = []
corr = []
print(test_pred.shape)
for i in range(test_pred.shape[0]):
    predict_result = test_pred[i]
    #print(predict_result)
    true_result = test_true[i]
    total = predict_result.shape[0] * predict_result.shape[1]
    print(total)
    sse = np.sum((true_result - predict_result) ** 2)
    print(sse)
    rmse_temp = np.sqrt(sse / total)
    '''
    if i == 0:
        print(total)
        print(sse)
        print(rmse_temp)
    '''
    #print( np.sum(rmse_temp) / len(rmse_temp))
    rmse.append(rmse_temp)

    predict_result_f = predict_result.flatten()
    true_result_f = true_result.flatten()
    corr_temp = np.corrcoef(predict_result_f, true_result_f)[0, -1]
    corr.append(corr_temp)
RMSE = np.sum(rmse) / len(rmse)
CORR = np.sum(corr) / len(corr)

print(RMSE)
print(CORR)

from sklearn.metrics import mean_absolute_error


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
nrmse = loss(data_mask_t, depth_globel, test_pred, test_true)

