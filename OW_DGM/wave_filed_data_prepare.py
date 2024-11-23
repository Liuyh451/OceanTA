import numpy as np
import torch
import xarray as xr

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
    file_path = base_path+year_month+'_cropped.nc'
    # 打开 .nc 文件
    with xr.open_dataset(file_path) as ds:
        # 提取所需特征
        Hs = ds['hs'].values
        Tm = ds['tm'].values
        dirm = ds['dirm'].values
        #这里就很奇怪，数据保存的时候确实是float32，读取后就变为timedelta64了
        Tm = np.array(Tm, dtype='float32')
        # 处理波浪数据
        cos_dirm, sin_dirm = process_wave_data(dirm)
        # 将处理后的数据拼接为 (时间步, 特征数)
        data = np.stack([Hs, Tm, cos_dirm, sin_dirm], axis=-1)
        print(f"文件 {file_path} 处理后形状: {data.shape}")
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
            monthly_obs = read_nc_files_and_extract_features(base_path, year_month)
            # 确保数据是torch.Tensor类型
            if isinstance(monthly_obs, np.ndarray):
                monthly_obs = torch.tensor(monthly_obs)
            monthly_data.append(monthly_obs)
    # 检查是否有数据
    if not monthly_data:
        raise ValueError("No valid data files found for the specified range.")

    # 拼接所有月份的数据
    combined_data = torch.cat(monthly_data, dim=0)  # 按时间步拼接 (num_buoys, total_time_steps, feature_dim)
    return combined_data
