import numpy as np
import netCDF4 as nc
import xarray as xr
from datetime import datetime, timedelta
""""
使用新方式读取数据，避免重复打开文件
"""


# todo 这样写可以减少重复打开文件
def load_all_nc_data(path, start_year, end_year):
    data_dict = {}
    current_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year + 1, 1, 1)
    while current_date < end_date:
        date_str = current_date.strftime('%Y%m%d')
        nc_file = path + '/subset_' + date_str + '.nc'
        # 当月份大于2时停止循环
        if current_date.month > 2:
            print("月份大于2，停止循环。")
            break
        # 读取 NetCDF 文件
        dataset = xr.open_dataset(nc_file)

        for var_name in dataset.variables:
            if var_name not in data_dict:
                data_dict[var_name] = []
            data = dataset[var_name].values
            data[data == -32768.0] = np.nan  # 替换标记值
            data_dict[var_name].append(data)

        current_date += timedelta(days=1)

    # 将列表转换为数组
    for var_name in data_dict:
        data_dict[var_name] = np.array(data_dict[var_name])

    return data_dict
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
    # 提取层数据
    raw_data = extract_layer_data(data_dict, var_type, depth)
    print(raw_data[0])
    width = raw_data.shape[2]  # 经度
    length = raw_data.shape[1]  # 纬度

    X = create_dataset(raw_data, time_step)
    X = X.reshape(X.shape[0], time_step, length, width, 1)
    Y = raw_data[time_step - 1:]
    Y = Y.reshape(Y.shape[0], length, width, 1)

    # 转置维度
    X = X.transpose(0, 1, 4, 2, 3)
    Y = Y.transpose(0, 3, 1, 2)

    return X, Y, raw_data
# 使用示例
file_path = 'E:/DataSet/redos/Subset_1.0_1995'
data_dict=load_all_nc_data(file_path,1995,1995)
train_sssa, _, _ = read_raw_data('s', 0, 3, data_dict)
train_ssha, _, _ = read_raw_data('zeta', 0, 3, data_dict)
train_sswu, _, _ = read_raw_data('u', 0, 3, data_dict)
train_sswv, _, _ = read_raw_data('v', 0, 3, data_dict)
train_argo, label_argo, data_mask_t = read_raw_data('t', 1, 3, data_dict)
del data_dict  # 删除字典对象
# def extract_nc_layer_data(path,type,depth,start_year, end_year):
#     daily_data = []
#     # 从指定年份的1月1日开始
#     current_date = datetime(start_year, 1, 1)
#
#     # 指定结束日期
#     end_date = datetime(end_year + 1, 1, 1)
#
#     # 循环自增日期
#     while current_date < end_date:
#         # 当月份大于2时停止循环
#         if current_date.month > 2:
#             print("月份大于2，停止循环。")
#             break
#         # 转为8位字符串格式 YYYYMMDD
#         date_str = current_date.strftime('%Y%m%d')
#         nc_file=path+'/subset_'+date_str+'.nc'
#         # print(nc_file)
#         # 日期自增1天
#         current_date += timedelta(days=1)
#         dataset = xr.open_dataset(nc_file)
#         # 选择数据变量
#         data = dataset[type].values
#
#         # 替换标记值
#         data[data == -32768.0] = np.nan
#         # 添加到 daily_data
#         if type == 'zeta':
#             # zeta 数据不涉及深度，直接添加替换后的数据
#             daily_data.append(data)
#         else:
#             # 其他数据需要根据深度进行处理
#             temp_lv0 = data[depth, :, :]
#             daily_data.append(temp_lv0)
#     day_lon_lat = np.array(daily_data)
#     return day_lon_lat
#
# def create_dataset(data, time_step):
#     dataX = []
#     for i in range(data.shape[0] - time_step + 1):
#         dataX.append(data[i:i + time_step])
#     return np.array(dataX)
#
# def read_raw_data(vtype, depth, time_step,nc_file):
#     #训练用的数据是第0层，也就是海表，原来那个是按照深度进行划分的，这个nc文件是按天数进行划分的，这里只有一天，所以shape[0]=1
#     train_argo = extract_nc_layer_data(nc_file,vtype,depth,1995,1995)
#     data_mask=train_argo
#     label_argo = extract_nc_layer_data(nc_file,vtype,depth,1995,1995)
#     width = train_argo.shape[2] #对应经度
#     lenth = train_argo.shape[1] #对应纬度
#     X = create_dataset(train_argo, time_step)
#     X = X.reshape(X.shape[0],time_step,lenth,width,1)
#     Y = label_argo[time_step-1 : label_argo.shape[0]]
#     Y =Y.reshape(Y.shape[0],lenth,width,1)
#     #X 转置维度，变为 (样本数, 时间步长, 通道数, 纬度, 经度)。
#     #Y 转置维度，变为 (样本数, 时间步长， 经度, 纬度)。
#     X = X.transpose(0,1,4,2,3)
#     Y = Y.transpose(0,3,1,2)
#     return X, Y,data_mask
#
# #这几个数据格式一样，但是内容不一样，读的分别是不同的列
#
# file_path = 'E:/DataSet/redos/Subset_1.0_1995'
# train_sssa,_,_=read_raw_data('s',0,3,file_path)
# train_ssha,_,_ = read_raw_data('zeta',0,3,file_path) #海面高度异常（Sea Level Anomaly）,他写的是sla，但是这里是zeta
# train_sswu,_,_ = read_raw_data('u',0,3,file_path)#U vwnd分量的风速（即沿经度方向的风速）,这里是u
# train_sswv,_,_ = read_raw_data('v',0,3,file_path)#V vwnd分量的风速（即沿纬度方向的风速），这里是v
# train_argo, label_argo,data_mask_t = read_raw_data('t', 1, 3,file_path)#temp 代表温度数据,预测深度为1时的海温

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
sta_train,_,_ = scaler(train_argo[:-num_test,:])
ssa_train,_,_  = scaler(train_sssa[:-num_test,:])
ssha_train,_,_ = scaler(train_ssha[:-num_test,:])
sswu_train,_,_ = scaler(train_sswu[:-num_test,:])
sswv_train,_,_ = scaler(train_sswv[:-num_test,:])
true_train,_,_ = scaler(label_argo[:-num_test,:])

#用倒数12个数据作为验证集
sta_test,_,_ = scaler(train_argo[-num_test:])
ssa_test,_,_  = scaler(train_sssa[-num_test:])
ssha_test,_,_ = scaler(train_ssha[-num_test:])
sswu_test,_,_ = scaler(train_sswu[-num_test:])
sswv_test,_,_ = scaler(train_sswv[-num_test:])

#将多个不同类型的训练数据和测试数据沿着指定轴进行拼接，axis=2即增加特征的数量（即通道或变量的数量）
sta_train = np.concatenate((sta_train,ssa_train,ssha_train,sswu_train,sswv_train),axis = 2 )
sta_test = np.concatenate((sta_test,ssa_test,ssha_test,sswu_test,sswv_test),axis = 2)

true_test,test_min,test_scale = scaler(label_argo[-num_test:])
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
print("true_test.shape",true_test.shape,"X_eval.shape",X_eval.shape)


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
configs.display_interval = 10
configs.num_epochs = 100
#这是早停的耐心参数。即使模型在900个epoch内没有改善性能，训练仍会继续。如果在900个epoch内性能没有改善，训练将停止
configs.early_stopping = True
configs.patience = 100
#禁用梯度裁剪（Gradient Clipping）。梯度裁剪用于防止梯度爆炸问题，但在这里未启用
configs.gradient_clipping = False
#设置梯度裁剪的阈值为1。如果梯度裁剪启用，梯度的最大值将被限制为1。不过在这种配置下，由于梯度裁剪被禁用，这个参数实际上不会生效
configs.clipping_threshold = 1.

# lr warmup
#这是学习率预热的步数设置。在训练的前3000步内，学习率将逐渐从一个较小的值线性增加到预设的学习率。这种技术通常用于训练的初始阶段，以帮助模型更稳定地开始训练，减少初期的震荡。
configs.warmup = 250

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
#todo 这个地方测试是是不是改为5*10，或者7*8(planB)
configs.emb_spatial_size = 5*10
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
#todo 应该不用加，推测是用来加新数据的

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

rmse = []
corr = []
test_pred.shape[0]
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
nrmse = loss(data_mask_t, 1, test_pred, test_true)

