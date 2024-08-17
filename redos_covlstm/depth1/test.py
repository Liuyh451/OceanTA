import numpy as np
import netCDF4 as nc
import xarray as xr
from datetime import datetime, timedelta
class dataPreprocess:
    def __init__(self, path, time_step):
        """
        初始化方法，传入待处理的数据及其他参数。

        参数:
        data (any): 需要处理的数据。
        normalize (bool): 是否对数据进行归一化。
        fill_value (any): 用于填充缺失值的值。
        """
        self.path = path
        self.time_step = time_step

    def fill_depth_for_zeta(self,ds):
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
               对于不同坐标系统的变量，使用插值对齐到统一的坐标系统。

               Parameters:
               ds (xarray.Dataset): 输入的 xarray Dataset。
               lon (float): 经度值。
               lat (float): 纬度值。

               Returns:
               xarray.Dataset: 包含指定点数据的 Dataset。
               """
        profiles = {}
        lon=115
        lat=14
        # 处理 u 和 v 变量（使用 (lonu, latu) 坐标系统）
        for var_name in ['u', 'v']:
            if var_name in ds:
                if var_name == 'u':
                    aligned_var = ds[var_name].sel(lonu=lon, latu=lat, method='nearest')
                else:
                    aligned_var = ds[var_name].sel(lonv=lon, latv=lat, method='nearest')
                profiles[var_name] = aligned_var

        # 处理 s 和 t 变量（使用 (lon, lat) 坐标系统）
        for var_name in ['s', 't','zeta']:
            if var_name in ds:
                # 直接提取 (lon, lat) 坐标系统中的数据
                profiles[var_name] = ds[var_name].sel(lon=lon, lat=lat, method='nearest')

        # 将提取的数据转换为 Dataset 格式
        profile_ds = xr.Dataset(profiles)

        return profile_ds

    def expand_dimensions(self,data_dict):
        """
        对字典中的指定变量进行维度扩展。

        参数:
        data_dict (dict): 包含数据的字典，键为变量名，值为 numpy 数组。

        返回:
        dict: 对指定变量进行维度扩展后的字典。
        """
        expanded_dict = {}

        for key, value in data_dict.items():
            if key in ['u','v','s','zeta', 't']:
                # 对 'u' 和 't' 变量进行维度扩展，增加最后一个维度
                expanded_dict[key] = value.reshape(value.shape[0], value.shape[1], 1)
            else:
                # 对其他变量保持原样
                expanded_dict[key] = value

        return expanded_dict

    # 使用新方式读取数据，避免重复打开文件
    # todo 1.加载数据 2.形状 3.特征图大小
    def load_all_nc_data(self, start_year, end_year):
        global dataset_tmp
        path = self.path
        data_dict = {}
        data_dict_Jan = {}
        current_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year + 1, 1, 1)
        while current_date < end_date:
            date_str = current_date.strftime('%Y%m%d')
            nc_file = path + '/subset_' + date_str + '.nc'
            # 当月份大于2时停止循环
            if current_date.month > 2:
                print("月份大于3，停止循环。")
                break
            # 读取 NetCDF 文件
            dataset = xr.open_dataset(nc_file)
            dataset=self.fill_depth_for_zeta(dataset)
            dataset=self.extract_profile_from_points(dataset)
            dataset_tmp=dataset
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

    def read_col_data(self, data_dict,vtype):
        # 提取层数据
        data_dict = self.expand_dimensions(data_dict)
        raw_data=data_dict[vtype]
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


# 使用示例
file_path = 'E:/DataSet/redos/Subset_1.0_1995'
data_pre=dataPreprocess(file_path,3)
data_dict, data_dict_Jan=data_pre.load_all_nc_data(1995,1995)
data_s,_,_=data_pre.read_col_data(data_dict,'s')
data_t,_,_=data_pre.read_col_data(data_dict,'t')
data_zeta,_,_=data_pre.read_col_data(data_dict,'zeta')
data_u,_,_=data_pre.read_col_data(data_dict,'u')
data_v,_,_=data_pre.read_col_data(data_dict,'v')
print(data_s.shape,data_t.shape,data_zeta.shape,data_u.shape,data_v.shape)


# 打印所有键
# def print_data_shapes(data_dict, data_dict_Jan, keys_to_check):
#     print("Shapes of selected keys in data_dict:")
#     for key in keys_to_check:
#         if key in data_dict:
#             print(f"{key}: {data_dict[key].shape}")
#         else:
#             print(f"{key} not found in data_dict")
#
#     print("\nShapes of selected keys in data_dict_Jan:")
#     for key in keys_to_check:
#         if key in data_dict_Jan:
#             print(f"{key}: {data_dict_Jan[key].shape}")
#         else:
#             print(f"{key} not found in data_dict_Jan")
# # 指定要检查的键
# keys_to_check = ['u','v','s','zeta', 't']  # 替换为实际的变量名
#
# # 打印数据形状
# print_data_shapes(data_dict, data_dict_Jan, keys_to_check)