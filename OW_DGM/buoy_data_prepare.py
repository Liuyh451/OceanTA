import netCDF4 as nc
import numpy as np
import os
import xarray as xr
import torch

def process_wave_data(dirm):
    """
    处理波浪数据，规范化 Hs 和 Tm 到 [-1, 1]，并将 dirm 转换为 (cos(dirm), sin(dirm))

    参数：
    - Hs: 显著波高
    - Tm: 平均波周期
    - dirm: 平均波方向（单位：度）

    返回：
    - 处理后的波浪特征 (Hs, Tm, cos(dirm), sin(dirm))
    """
    # 将 dirm 转换为 (cos(dirm), sin(dirm))
    dirm_rad = np.deg2rad(dirm)  # 将度转换为弧度
    cos_dirm = np.cos(dirm_rad)
    sin_dirm = np.sin(dirm_rad)
    return cos_dirm, sin_dirm


# 定义归一化函数


# 定义反归一化函数
def denormalize(data_norm, x_min, x_max):
    """
    自动反归一化到原始范围
    :param data_norm: 归一化后的数据 (numpy array)
    :param x_min: 原始数据的最小值
    :param x_max: 原始数据的最大值
    :return: 反归一化后的数据
    """
    return data_norm * (x_max - x_min) / 2 + (x_max + x_min) / 2


def read_nc_files_and_extract_features(base_path, year_month):
    """
    从指定路径读取 .nc 文件，提取所需特征并拼接为目标形状。对时间序列进行对齐，不足的用 NaN 填充。

    参数：
    - base_path: str, 存放 .nc 文件的目录路径
    - year_month: str, 需要读取的年月标识（如 '201701'）

    返回：
    - data: np.ndarray, 形状为 (5, max_timestep, 3)，浮标个数、时间步、特征数
    """
    buoy_count = 5  # 固定浮标数量
    features = ['tm02', 'mdir', 'Hm0']  # 所需提取的特征
    all_data = []
    max_timestep = 0  # 用于记录最大时间步数

    # 找到指定年月的 5 个文件
    files = [f for f in os.listdir(base_path) if f.startswith(year_month) and f.endswith('.nc')]
    files = sorted(files)[:buoy_count]  # 按字母排序并选取前 5 个文件

    if len(files) != buoy_count:
        raise ValueError(f"文件数量不足！需要 5 个文件，但仅找到 {len(files)} 个。")

    # 读取每个文件的数据并获取最大时间步
    for file in files:
        file_path = os.path.join(base_path, file)

        # 打开 .nc 文件
        with xr.open_dataset(file_path) as ds:
            # 提取所需特征
            Hs = ds['Hm0'].values
            Tm = ds['tm02'].values
            dirm = ds['mdir'].values

            # 处理波浪数据
            cos_dirm, sin_dirm = process_wave_data(dirm)

            # 将处理后的数据拼接为 (时间步, 特征数)
            data = np.stack([Hs, Tm, cos_dirm, sin_dirm], axis=-1)
            all_data.append(data)

            print(f"文件 {file} 处理后形状: {data.shape}")
            max_timestep = max(max_timestep, data.shape[0])  # 更新最大时间步

    # 对齐时间步，不足的部分用 NaN 填充
    aligned_data = []
    for data in all_data:
        # 计算当前文件的时间步差
        timestep_diff = max_timestep - data.shape[0]
        if timestep_diff > 0:
            # 使用 NaN 填充不足的时间步
            padding = np.full((timestep_diff, data.shape[1]), np.nan)
            data = np.vstack([data, padding])  # 拼接填充的 NaN
        aligned_data.append(data)

    # 拼接为目标形状 (5, max_timestep, 3)
    data_array = np.stack(aligned_data, axis=0)
    return data_array


def combine_monthly_data(base_path, start_year, end_year):
    """
    读取每个月的数据并拼接成一个总的数据张量。

    参数：
    - base_path (str): 数据文件的根路径。
    - start_year (int): 起始年份。
    - end_year (int): 结束年份。

    返回：
    - combined_data (torch.Tensor): 拼接后的数据张量，形状为 (num_buoys, total_time_steps, feature_dim)。
    """
    monthly_data = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):  # 遍历1到12月
            year_month = f"{year}{month:02d}"  # 格式化为"YYYYMM"
            if (year == 2019 and month > 2):
                break
            # 检查文件是否存在
            # 调用函数读取数据
            print(f"Processing file for {year_month}...")
            monthly_obs = read_nc_files_and_extract_features(base_path, year_month)
            # 确保数据是torch.Tensor类型
            if isinstance(monthly_obs, np.ndarray):
                monthly_obs = torch.tensor(monthly_obs)
            monthly_data.append(monthly_obs)
    # 检查是否有数据
    if not monthly_data:
        raise ValueError("No valid data files found for the specified range.")

    # 拼接所有月份的数据
    combined_data = torch.cat(monthly_data, dim=1)  # 按时间步拼接 (num_buoys, total_time_steps, feature_dim)
    return combined_data


# 提取滑动窗口数据
def create_sliding_windows(data, window_size, step_size):
    """
    使用滑动窗口生成子序列
    Args:
        data: numpy 数组，形状为 (num_samples, seq_length, features)
        window_size: 滑动窗口大小
        step_size: 滑动步长
    Returns:
        result: numpy 数组，形状为 (num_samples, num_windows, window_size, features)
    """
    num_samples, seq_length, features = data.shape
    num_windows = (seq_length - window_size) // step_size + 1  # 计算窗口数量

    # 初始化结果数组
    result = np.zeros((num_samples, num_windows, window_size, features))

    # 滑动窗口提取
    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        result[:, i, :, :] = data[:, start_idx:end_idx, :]

    return result


def normalize_and_save_numpy(data, save_path):
    """
    使用 NumPy 对输入数据的前两维特征进行归一化，并记录最大值和最小值保存为 .npy 文件。

    参数：
    - data: torch.Tensor，形状为 (浮标数, 时间步数, 特征数)。
    - save_path: str，用于保存最大最小值的文件夹路径。

    返回：
    - normalized_data: torch.Tensor，归一化后的数据，与输入形状相同。
    """
    if not isinstance(data, torch.Tensor):
        raise ValueError("输入数据必须是 torch.Tensor 类型")

    # 转为 NumPy
    data_np = data.numpy()

    # 初始化最大值和最小值存储
    min_vals = []
    max_vals = []

    # 归一化前两维特征
    normalized_data_np = np.zeros_like(data_np)
    for i in range(2):  # 遍历每个特征
        print(i)
        feature = data_np[:, :, i]  # 提取当前特征

        # 忽略 NaN 值，计算最小值和最大值
        min_val = np.nanmin(feature)
        max_val = np.nanmax(feature)

        # 存储最大最小值
        min_vals.append(min_val)
        max_vals.append(max_val)

        # 归一化公式
        normalized_feature = 2 * (feature - min_val) / (max_val - min_val) - 1
        normalized_data_np[:, :, i] = normalized_feature

    # 转回 PyTorch
    normalized_data = torch.from_numpy(normalized_data_np)

    # 将最大最小值保存为 .npy 文件
    min_vals = np.array(min_vals)
    max_vals = np.array(max_vals)

    np.save(f"{save_path}/min_values.npy", min_vals)
    np.save(f"{save_path}/max_values.npy", max_vals)

    print(f"最大最小值已保存到 {save_path}")

    return normalized_data


def create_missing_nc(template_file, output_dir, start_month, end_month, file_prefix):
    """
    根据模板文件生成缺失的 NetCDF 文件，并用 NaN 填充。

    参数：
        - template_file: str，模板 nc 文件路径
        - output_dir: str，生成文件的保存目录
        - start_month: str，起始月份，格式为 YYYYMM
        - end_month: str，结束月份，格式为 YYYYMM
        - file_prefix: str，文件名前缀，如 "E39_C_Sulafjorden_wave" 或 "E39_F_Vartdalsfjorden_wave"
    """
    # 加载模板文件
    ds_template = nc.Dataset(template_file, 'r')

    # 获取经纬度和时间维度
    lat = ds_template.variables['latitude'][:]
    lon = ds_template.variables['longitude'][:]
    time_len = len(ds_template.dimensions['time'])  # 假设时间维度存在

    # 确定时间范围
    year_start, month_start = int(start_month[:4]), int(start_month[4:])
    year_end, month_end = int(end_month[:4]), int(end_month[4:])

    # 遍历所有需要生成的月份
    for year in range(year_start, year_end + 1):
        for month in range(1, 13):
            # 如果不在指定范围内，则跳过
            if (year == year_start and month < month_start) or (year == year_end and month > month_end):
                continue

            # 生成文件名
            file_name = f"{year}{str(month).zfill(2)}_{file_prefix}.nc"
            output_path = os.path.join(output_dir, file_name)

            # 创建新 NetCDF 文件
            with nc.Dataset(output_path, 'w', format='NETCDF4') as new_ds:
                # 创建维度
                new_ds.createDimension('lat', len(lat))
                new_ds.createDimension('lon', len(lon))
                new_ds.createDimension('time', time_len)

                # 创建变量并用 NaN 填充
                lat_var = new_ds.createVariable('lat', 'f4', ('lat',))
                lon_var = new_ds.createVariable('lon', 'f4', ('lon',))
                time_var = new_ds.createVariable('time', 'f4', ('time',))

                # 拷贝变量数据
                lat_var[:] = lat
                lon_var[:] = lon
                time_var[:] = ds_template.variables['time'][:]

                # 处理所有其他变量
                for var_name in ds_template.variables:
                    if var_name not in ('lat', 'lon', 'time'):
                        var_template = ds_template.variables[var_name]
                        var_dims = var_template.dimensions
                        var_dtype = var_template.datatype
                        new_var = new_ds.createVariable(var_name, var_dtype, var_dims)
                        new_var[:] = np.full(var_template.shape, np.nan)  # 用 NaN 填充

                print(f"生成文件: {output_path}")

    # 关闭模板文件
    ds_template.close()


base_path = r"E:\Dataset\met_waves\buoy"
start_year = 2017
end_year = 2019
combined_data = combine_monthly_data(base_path, start_year, end_year)
print(f"Combined data shape: {combined_data.shape}")
save_path = "./data"
# 创建保存路径
os.makedirs(save_path, exist_ok=True)
# 调用函数
normalized_data = normalize_and_save_numpy(combined_data, save_path)
print("归一化后的数据形状:", normalized_data.shape)
# 滑动窗口参数
window_size = 3  # 滑动窗口大小（过去 20 分钟，3 条观测数据）
step_size = 1  # 滑动步长

#使用滑动窗口提取数据
sliding_window_data = create_sliding_windows(normalized_data, window_size, step_size)

# 查看滑动窗口后的数据形状
print(f"原始数据形状: {normalized_data.shape}")
print(f"滑动窗口后的数据形状: {sliding_window_data.shape}")
np.save(f"{save_path}/buoy_obs.npy", sliding_window_data)
print(f"浮标观察值已保存到 {save_path}")



# 创建 Sulafjorden 的 2017 年 1 月到 3 月的文件
# create_missing_nc(
#     template_file=r'E:\Dataset\met_waves\buoy\201704_E39_C_Sulafjorden_wave.nc',
#     output_dir=r'E:\Dataset\met_waves\fill',
#     start_month='201701',
#     end_month='201703',
#     file_prefix='E39_C_Sulafjorden_wave'
# )

# 创建 Vartdalsfjorden 的 2017 年 1 月到 10 月的文件
# create_missing_nc(
#     template_file=r'E:\Dataset\met_waves\buoy\201711_E39_F_Vartdalsfjorden_wave.nc',
#     output_dir=r'E:\Dataset\met_waves\fill',
#     start_month='201701',
#     end_month='201710',
#     file_prefix='E39_F_Vartdalsfjorden_wave'
# )
