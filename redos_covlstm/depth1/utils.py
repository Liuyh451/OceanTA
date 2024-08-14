import numpy as np

import xarray as xr
from pathlib import Path
import random

from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeat
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from sklearn.metrics import r2_score

# im1 = SimpleImputer(missing_values=np.nan,strategy='constant',fill_value = 0 )
im1 = SimpleImputer(missing_values=np.nan,strategy='mean')


def create_dataset(data, time_step):
    dataX = []
    for i in range(data.shape[0] - time_step + 1):
        dataX.append(data[i:i + time_step])
    return np.array(dataX)




def read_raw_data(vtype, depth, time_step):
    train_argo = np.load('./data/'+vtype+'_0_ano.npy')#读取数据，指定维度标签
    label_argo = np.load('./data/'+vtype+'_'+str(depth)+'_ano.npy')#206,60,80
    width = train_argo.shape[2] #对应经度
    lenth = train_argo.shape[1] #对应纬度
    X = create_dataset(train_argo, time_step)
    X = X.reshape(X.shape[0],time_step,lenth,width,1)
    Y = label_argo[time_step-1 : label_argo.shape[0]] 
    Y =Y.reshape(Y.shape[0],lenth,width,1)
    return X, Y


# def scaler(data):
#     #normalise [-1, 1]
#     data_max = np.nanmax(data)
#     data_min = np.nanmin(data)
#     data_scale = 2 / (data_max - data_min)
#     data_std = ((data - data_min) * data_scale) -1
#     # data_std = (data_std * 2)  -1
#     data_std [np.isnan(data_std)] = 0
#     return data_std,data_min,data_scale

# def unscaler(data, data_min, data_scale):
#     data_inv = (data + 1)/data_scale + data_min
#     return data_inv

def scaler(data):
    #normalise [0,1]
    data_max = np.nanmax(data)
    data_min = np.nanmin(data)
    data_scale = data_max - data_min
    data_std = (data - data_min) / data_scale
    # data_std = data_std * (2)  -1
    data_std [np.isnan(data_std)] = 0
    return data_std,data_min,data_scale


def unscaler(data, data_min, data_scale):
    data_inv = (data * data_scale) + data_min
    return data_inv

    


    
def add(vtype,depth,data):
    
    ds=xr.open_dataset('./data/BOA_Argo_annual.nc',decode_times=False)
    annual = ds[vtype].data[0,depth,49:109,159:239]
    for i in range(0,data.shape[0]):
        data[i,:,:,0] = data[i,:,:,0]+ annual
    return data

                                
# def loss(vtype,depth,test_pred,test_true):
#     test_preds = np.array(test_pred,copy=True)
#     test_trues = np.array(test_true,copy=True)
    
#     test_preds = np.squeeze(test_preds)
#     test_trues = np.squeeze(test_trues)

#     test_preds[np.isnan(test_preds)] = 0
#     test_trues[np.isnan(test_trues)] = 0
#     mask = np.load('./data/'+vtype+'_'+str(depth)+'_ano.npy')
# #     mask = np.squeeze(mask)
#     mask = mask[0]

#     total = mask.shape[0]* mask.shape[1]
#     total_nan = len(mask[np.isnan(mask)])
#     total_real = total - total_nan
# #     print('Total NaN:',total_nan)#统计数据中的nan值
# #     print('Total Real:',total_real)#统计数据中的nan值
# #     #nan：0 values ：1
#     mask[~np.isnan(mask)] = 1
#     mask[np.isnan(mask)] = 0
#     rmse = []
#     rmse_temp = []
#     nrmse = []
#     nrmse_temp = []
#     mae = []
#     mae_temp = []
#     for i in range(0,test_preds.shape[0]):

#         final_temp = mask * test_preds[i]
#         test_temp = mask * test_trues[i]
#         # np.sum((y_actual - y_predicted) ** 2)
#         sse = np.sum((test_temp - final_temp) ** 2)
#         mse_temp = sse/total_real
#         rmse_temp = np.sqrt(mse_temp)
#         nrmse_temp = rmse_temp/np.mean(test_temp)
#         rmse.append(rmse_temp)
#         nrmse.append(nrmse_temp)
#         mae_temp = mean_absolute_error(test_temp,final_temp)*total/total_real

#         mae.append(mae_temp)
# #     print('NAN:',len(test_pred[np.isnan(test_pred)]))
# #     print('TEST NANMIN',np.nanmin(test_pred))
# #     print('TEST MIN',test_pred.min())
#     # print(str(depth)+'层')
#     # RMSE = np.sum(rmse)/len(rmse)
#     # MAE = np.sum(mae)/len(mae)
#     NRMSE= np.sum(nrmse)/len(nrmse)
#     # NRMSE = nrmse
#     # print(str(depth)+'层:'+'NRMSE RESULT:\n',mean(NRMSE))

# #     print('MAE RESULT:\n',MAE)

#     return NRMSE

def loss(vtype,depth,test_pred,test_true):
    test_preds = np.array(test_pred,copy=True)
    test_trues = np.array(test_true,copy=True)
    
    test_preds = np.squeeze(test_preds)
    test_trues = np.squeeze(test_trues)

    test_preds[np.isnan(test_preds)] = 0
    test_trues[np.isnan(test_trues)] = 0
    mask = np.load('./data/'+vtype+'_'+str(depth)+'_ano.npy')
#     mask = np.squeeze(mask)
    mask = mask[0]

    total = mask.shape[0]* mask.shape[1]
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
    for i in range(0,test_preds.shape[0]):

        final_temp = mask * test_preds[i]
        test_temp = mask * test_trues[i]
        # np.sum((y_actual - y_predicted) ** 2)
        sse = np.sum((test_temp - final_temp) ** 2)
        mse_temp = sse/total_real
        rmse_temp = np.sqrt(mse_temp)
        nrmse_temp = rmse_temp/np.mean(test_temp)
        rmse.append(rmse_temp)
        nrmse.append(nrmse_temp)
        mae_temp = mean_absolute_error(test_temp,final_temp)*total/total_real

        mae.append(mae_temp)
#     print('NAN:',len(test_pred[np.isnan(test_pred)]))
#     print('TEST NANMIN',np.nanmin(test_pred))
#     print('TEST MIN',test_pred.min())
    # print(str(depth)+'层')
    RMSE = np.sum(rmse)/len(rmse)
    MAE = np.sum(mae)/len(mae)
    NRMSE= np.sum(nrmse)/len(nrmse)
    # NRMSE = nrmse
    print(str(depth)+'层:'+'NRMSE RESULT:\n',NRMSE)

#     print('MAE RESULT:\n',MAE)

    return NRMSE

def corr(vtype,depth,test_pred,test_true):
    test_preds = np.array(test_pred,copy=True)
    test_trues = np.array(test_true,copy=True)
    
    test_preds = np.squeeze(test_preds)
    test_trues = np.squeeze(test_trues)

    test_preds[np.isnan(test_preds)] = 0
    test_trues[np.isnan(test_trues)] = 0
    mask = np.load('./data/'+vtype+'_'+str(depth)+'_ano.npy')
#     mask = np.squeeze(mask)
    mask = mask[0]

    total = mask.shape[0]* mask.shape[1]
    total_nan = len(mask[np.isnan(mask)])
    total_real = total - total_nan
#     print('Total NaN:',total_nan)#统计数据中的nan值
#     print('Total Real:',total_real)#统计数据中的nan值
#     #nan：0 values ：1
    mask[~np.isnan(mask)] = 1
    mask[np.isnan(mask)] = 0
    CORR = []
    corr = []
    corr_temp = []
    for i in range(0,test_preds.shape[0]):

        final_temp = mask * test_preds[i]
        final_temp_f = final_temp.flatten()
        test_temp = mask * test_trues[i]
        test_temp_f = test_temp.flatten()
        corr_temp = np.corrcoef(final_temp_f,test_temp_f)[0,-1]
        # print(corr_temp)
        corr.append(corr_temp)
#     print('NAN:',len(test_pred[np.isnan(test_pred)]))
#     print('TEST NANMIN',np.nanmin(test_pred))
#     print('TEST MIN',test_pred.min())
    # print(str(depth)+'层')
    CORR = np.sum(corr)/len(corr)
    # CORR = corr
    print(str(depth)+'层:'+'CORR RESULT:\n',CORR)

#     print('MAE RESULT:\n',MAE)

    return CORR

# def corr(vtype,depth,test_pred,test_true):
#     test_preds = np.array(test_pred,copy=True)
#     test_trues = np.array(test_true,copy=True)
    
#     test_preds = np.squeeze(test_preds)
#     test_trues = np.squeeze(test_trues)

#     test_preds[np.isnan(test_preds)] = 0
#     test_trues[np.isnan(test_trues)] = 0
#     mask = np.load('./data/'+vtype+'_'+str(depth)+'_ano.npy')
# #     mask = np.squeeze(mask)
#     mask = mask[0]

#     total = mask.shape[0]* mask.shape[1]
#     total_nan = len(mask[np.isnan(mask)])
#     total_real = total - total_nan
# #     print('Total NaN:',total_nan)#统计数据中的nan值
# #     print('Total Real:',total_real)#统计数据中的nan值
# #     #nan：0 values ：1
#     mask[~np.isnan(mask)] = 1
#     mask[np.isnan(mask)] = 0
#     CORR = []
#     corr = []
#     corr_temp = []
#     for i in range(0,test_preds.shape[0]):

#         final_temp = mask * test_preds[i]
#         final_temp_f = final_temp.flatten()
#         test_temp = mask * test_trues[i]
#         test_temp_f = test_temp.flatten()
#         corr_temp = r2_score(test_temp_f,final_temp_f)
#         # print(corr_temp)
#         corr.append(corr_temp)
# #     print('NAN:',len(test_pred[np.isnan(test_pred)]))
# #     print('TEST NANMIN',np.nanmin(test_pred))
# #     print('TEST MIN',test_pred.min())
#     # print(str(depth)+'层')
#     # CORR = np.sum(corr)/len(corr)

    CORR = corr
    print(str(depth)+'层:'+'CORR RESULT:\n',CORR)

#     print('MAE RESULT:\n',MAE)

    return CORR


def plot(test_pred, test_true):
    ###依然没有解决xy坐标轴的问题
    plt.rcParams['figure.dpi'] = 600
    ds=xr.open_dataset('./data/BOA_Argo_2004_01.nc',decode_times=False)
    lons = ds.lon[159:239]
    lats = ds.lat[49:109]
    offset = 180
    lons,lats=np.meshgrid(lons-offset,lats)
    error = test_pred - test_true
    np.squeeze(test_pred)
    np.squeeze(test_pred)
    np.squeeze(error)
    data = error
    # data = np.concatenate((test_pred, test_true, error),axis=-1)
    projection = ccrs.PlateCarree(central_longitude=180)  # 创建投影
    # projection = ccrs.PlateCarree()  # 创建投影

    axes_class = (GeoAxes,
                  dict(map_projection=projection))
    fig = plt.figure(figsize=(10,3))
    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(1, 1),
                    axes_pad=0.6,
                    cbar_location='right',
                    cbar_mode='single',
                    cbar_pad=0.2,
                    cbar_size='3%',
                    label_mode='')  # note the empty label_mode
    color_min = int(np.nanmin(data))-1
    color_max = int(np.nanmax(data))+1
    n_gap = (color_max-color_min)/30
    levels = np.arange(color_min, color_max+n_gap, n_gap)
    for i, ax in enumerate(axgr):
        ax.coastlines()
        ax.add_feature(cfeat.COASTLINE.with_scale('50m'), linewidth=0.6, zorder=1)
    #     ax.set_xticks(np.linspace(-180, 180, 5), crs=projection)
    #     ax.set_yticks(np.linspace(-90, 90, 5), crs=projection)
    #     ax.set_extent(extent,crs=ccrs.PlateCarree())
    #     lon_formatter = LongitudeFormatter(zero_direction_label=True)
    #     lat_formatter = LatitudeFormatter()

    #     ax.xaxis.set_major_formatter(lon_formatter)
    #     ax.yaxis.set_major_formatter(lat_formatter)
    #     gl = ax.gridlines(crs=proj,draw_labels=True,
    #   linewidth=1.2, color='k', alpha=0.5, linestyle='--')
    #     gl.xlabels_top = False  #关闭顶端标签
    #     gl.ylabels_right = False  #关闭右侧标签
    #     gl.xformatter = LONGITUDE_FORMATTER  #x轴设为经度格式
    #     gl.yformatter = LATITUDE_FORMATTER  #y轴设为纬度格式

        p = ax.contourf(lons, lats, data[:,:,0],
                        cmap='RdBu_r', levels=levels)

    axgr.cbar_axes[0].colorbar(p)

    # plt.show()
    # plt.colse()
    return plt


