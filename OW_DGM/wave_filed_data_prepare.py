import numpy as np
import torch
import xarray as xr


def normalize(data):
    """
    使用 NumPy 对输入数据的前两维特征进行归一化

    参数：
    - data: npy，形状为 (浮标数, 时间步数, 特征数)。
    返回：
    - normalized_data: npy，归一化后的数据，与输入形状相同。
    """
    # 归一化前两维特征
    normalized_data_np = np.zeros_like(data)
    for i in range(4):  # 遍历每个特征
        feature = data[:, i, :, :]  # 提取当前特征

        # 忽略 NaN 值，计算最小值和最大值
        min_val = np.nanmin(feature)
        max_val = np.nanmax(feature)
        print(min_val, max_val)
        # 归一化公式
        normalized_feature = 2 * (feature - min_val) / (max_val - min_val) - 1
        if (i > 2):
            normalized_data_np[:, i, :, :] = feature
        else:
            normalized_data_np[:, i, :, :] = normalized_feature
    return normalized_data_np


def replace_invalid_values(data):
    """
    将数据中值为 -32768、9999.0 或 NaN 的位置替换为 0。

    参数：
    - data: npy，输入数据npy

    返回：
    - 处理后的 npy。
    """
    # 条件值
    condition = (data == -32768) | (data == 9999.0)
    # 将 -32768 和 9999.0 替换为 np.nan
    data = np.where(condition, np.nan, data)
    # 替换 NaN 为 0
    data[np.isnan(data)] = 0
    return data


# def normalize(data):
#     # 归一化到[-1,1]
#     x_min = np.nanmin(data)
#     x_max = np.nanmax(data)
#     data = 2 * (data - x_min) / (x_max - x_min) - 1
#     return data


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


def read_nc_files_and_extract_features(base_path, year_month):
    """
    从指定路径读取 .nc 文件，提取所需特征。

    参数：
    - base_path: str, 存放 .nc 文件的目录路径
    - year_month: str, 需要读取的年月标识（如 '201701'）

    返回：
    - data: np.ndarray, 形状为 (5, max_timestep, 3)，浮标个数、时间步、特征数
    """
    file_path = base_path + year_month + '_cropped.nc'
    # 打开 .nc 文件
    with xr.open_dataset(file_path) as ds:
        # 提取所需特征
        Hs = ds['hs'].values
        Tm = ds['tm02'].values
        dirm = ds['theta0'].values
        # 这里就很奇怪，数据保存的时候确实是float32，读取后就变为timedelta64了
        Tm = np.array(Tm, dtype='float32')
        # 处理波浪数据
        cos_dirm, sin_dirm = process_wave_data(dirm)
        # 将处理后的数据拼接为 (时间步, 特征数)
        data = np.stack([Hs, Tm, cos_dirm, sin_dirm], axis=-1)
        print(f"文件 {file_path} 处理后形状: {data.shape}")
        if (year_month == '201701'):
            # 对时间维度进行切片，去掉第一个时间点的数据，对齐时间
            data = data[1:]
        if (year_month == '201812'):
            # 创建全0的数据，形状与 data 除时间维度外一致，保证时间对齐
            zeros_to_append = np.zeros((1, *data.shape[1:]), dtype=data.dtype)
            data = np.concatenate([data, zeros_to_append], axis=0)
    return data


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

    for year in range(start_year, end_year):
        for month in range(1, 13):  # 遍历1到12月
            year_month = f"{year}{month:02d}"  # 格式化为"YYYYMM"
            if (year == 2020 and month > 2):
                break
            # 检查文件是否存在
            # 调用函数读取数据
            print(f"Processing file for {year_month}...")
            monthly_swan = read_nc_files_and_extract_features(base_path, year_month)
            monthly_swan = monthly_swan.transpose(0, 3, 1, 2)
            # 确保数据是torch.Tensor类型
            monthly_data.append(monthly_swan)
    # 检查是否有数据
    if not monthly_data:
        raise ValueError("No valid data files found for the specified range.")

    # 拼接所有月份的数据
    combined_data = np.concatenate(monthly_data, axis=0)  # 按时间步拼接 (num_buoys, total_time_steps, feature_dim)
    # 处理 Swan数据中的无效值
    swan_data = replace_invalid_values(combined_data)
    # 对数据进行归一化
    swan_data = normalize(swan_data)
    return swan_data
# swan_data = combine_monthly_data("/home/hy4080/met_waves/Swan_cropped/swanSula", 2017, 2019)
