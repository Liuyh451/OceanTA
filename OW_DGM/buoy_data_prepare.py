import netCDF4 as nc
import os
import pandas as pd
import numpy as np
import wave_filed_data_prepare


def fill_missing_time_points(file_path):
    # 从文件名中提取年和月
    file_name = file_path.split('\\')[-1]
    year = int(file_name.split('_')[0][:4])
    month = int(file_name.split('_')[0][4:6])

    # 确定每个月的天数
    days_in_month = pd.Timestamp(year=year, month=month, day=1).days_in_month

    # 创建完整的时间序列
    time_freq = '10min'
    full_time_range = pd.date_range(
        start=f'{year}-{month:02d}-01 00:00:00',
        end=f'{year}-{month:02d}-{days_in_month} 23:50:00',
        freq=time_freq
    )

    # 读取文件中的时间数据和其他数据
    with nc.Dataset(file_path, 'r') as ds:
        # 获取时间变量
        time_var = ds.variables['time'][:]  # 假设时间变量名为 'time'
        time_units = ds.variables['time'].units  # 获取时间单位
        time_calendar = ds.variables['time'].calendar if hasattr(ds.variables['time'], 'calendar') else 'standard'
        # 转换为标准的 Python datetime
        file_time = nc.num2date(time_var, units=time_units, calendar=time_calendar)
        file_time = pd.to_datetime([t.isoformat() for t in file_time])  # 转换为 ISO 格式后解析为 pandas 时间戳

        Hs = ds['Hm0'][:]  # 假设波高数据变量名为 'Hm0'
        Tm = ds['tm02'][:]  # 假设平均周期数据变量名为 'tm02'
        dirm = ds['mdir'][:]  # 假设波向数据变量名为 'mdir'
    # 创建填充后的时间对齐数据框架
    filled_data = pd.DataFrame(index=full_time_range)

    # 将每个数据对齐到完整时间序列，并填充缺失数据为 NaN
    hs_df = pd.DataFrame({'Hs': Hs}, index=file_time)
    tm_df = pd.DataFrame({'Tm': Tm}, index=file_time)
    dirm_df = pd.DataFrame({'dirm': dirm}, index=file_time)

    # 使用 join 对齐数据
    merged_data = filled_data.join(hs_df).join(tm_df).join(dirm_df)

    # 打印填充后的形状
    print("填充后的形状:", merged_data.shape)

    # 返回填充后的数据
    return merged_data


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
        ds = fill_missing_time_points(file_path)
        # 打开 .nc 文件
        Hs = ds['Hs'].values  # 提取波高数据
        Tm = ds['Tm'].values  # 提取平均周期数据
        dirm = ds['dirm'].values  # 提取波向数据

        # 处理波浪数据
        cos_dirm, sin_dirm = process_wave_data(dirm)

        # 将处理后的数据拼接为 (时间步, 特征数)
        data = np.stack([Hs, Tm, cos_dirm, sin_dirm], axis=-1)
        all_data.append(data)

        print(f"文件 {file} 特征维度处理后形状: {data.shape}")
        max_timestep = max(max_timestep, data.shape[0])  # 更新最大时间步

    # 拼接为目标形状 (5, max_timestep, 4)
    data_array = np.stack(all_data, axis=0)
    return data_array


def combine_monthly_data(base_path, start_year, end_year):
    """
    读取每个月的数据并拼接成一个总的数据张量。

    参数：
    - base_path (str): 数据文件的根路径。
    - start_year (int): 起始年份。
    - end_year (int): 结束年份。

    返回：
    - combined_data (torch.Tensor): 拼接后的数据tensor，形状为 (num_buoys, total_time_steps, feature_dim)。
    """
    monthly_data_train = []
    monthly_data_test = []

    for year in range(start_year, end_year):
        for month in range(1, 13):  # 遍历1到12月
            year_month = f"{year}{month:02d}"  # 格式化为"YYYYMM"
            if (year == 2020 and month > 2):
                break
            # 检查文件是否存在
            # 调用函数读取数据
            print(f"Processing file for {year_month}...")
            monthly_obs = read_nc_files_and_extract_features(base_path, year_month)
            if (year < 2019):
                monthly_data_train.append(monthly_obs)
            else:
                monthly_data_test.append(monthly_obs)
    # 检查是否有数据
    if not monthly_data_train and not monthly_data_test:
        raise ValueError("No valid data files found for the specified range.")

    # 拼接所有月份的数据
    combined_data_train = np.concatenate(monthly_data_train,
                                         axis=1)  # 按时间步拼接 (num_buoys, total_time_steps, feature_dim)
    combined_data_test = np.concatenate(monthly_data_test, axis=1)
    return combined_data_train, combined_data_test


def normalize_and_save_numpy(data, save_path):
    """
    使用 NumPy 对输入数据的前两维特征进行归一化，并记录最大值和最小值保存为 .npy 文件。

    参数：
    - data: torch.Tensor，形状为 (浮标数, 时间步数, 特征数)。
    - save_path: str，用于保存最大最小值的文件夹路径。

    返回：
    - normalized_data: torch.Tensor，归一化后的数据，与输入形状相同。
    """

    # 初始化最大值和最小值存储
    min_vals = []
    max_vals = []

    # 归一化前两维特征
    normalized_data_np = np.zeros_like(data)
    for i in range(4):  # 遍历每个特征
        feature = data[:, :, i]  # 提取当前特征

        # 忽略 NaN 值，计算最小值和最大值
        min_val = np.nanmin(feature)
        max_val = np.nanmax(feature)

        # 存储最大最小值
        min_vals.append(min_val)
        max_vals.append(max_val)

        # 归一化公式
        normalized_feature = 2 * (feature - min_val) / (max_val - min_val) - 1
        if (i > 2):
            normalized_data_np[:, :, i] = feature
        else:
            normalized_data_np[:, :, i] = normalized_feature
    # 将最大最小值保存为 .npy 文件
    min_vals = np.array(min_vals)
    max_vals = np.array(max_vals)
    return normalized_data_np, min_vals, max_vals


def generate_indices(sequence_length, group_size, skip_size):
    """
    动态生成需要提取的索引，第一个索引值从3开始。

    参数:
        sequence_length: int，序列的总长度。
        group_size: int，每次提取的连续点数。
        skip_size: int，每组之间跳过的点数。

    返回:
        indices: list，生成的索引列表（从1开始计数）。
    """
    indices = []
    start = 2  # 从2开始，这样下一次循环添加的第一个索引就是3
    while start < sequence_length:
        # 添加连续的group_size个点
        indices.extend(range(start + 1, start + group_size + 1))
        # 更新起点，跳过skip_size个点
        start += group_size + skip_size
    # 确保索引不超过序列长度
    return [i for i in indices if i <= sequence_length]


def extract_elements(data, indices):
    """
    从多维数组的b维度中提取指定索引的元素。

    参数:
        data: ndarray，多维数组，时间序列位于b维度。
        indices: list，指定b维度中需要提取的索引列表（从1开始计数）。

    返回:
        result: ndarray，包含提取的目标数据。
    """
    zero_based_indices = [i - 1 for i in indices]  # 转换为0基索引
    return data[:, zero_based_indices, :]


base_path = r"E:\Dataset\met_waves\buoy"
save_path = "./data"
start_year = 2017
end_year = 2021
combined_data_train, combined_data_test = combine_monthly_data(base_path, start_year, end_year)
print(f"Combined data shape: {combined_data_train.shape},{combined_data_test.shape}")
# 动态生成索引：每次提取连续 3 个点，跳过 3 个点
indices_to_extract_train = generate_indices(combined_data_train.shape[1], group_size=3, skip_size=3)
indices_to_extract_test = generate_indices(combined_data_test.shape[1], group_size=3, skip_size=3)
# 提取指定索引的数据
extracted_data_train = extract_elements(combined_data_train, indices_to_extract_train)
extracted_data_test = extract_elements(combined_data_test, indices_to_extract_test)
# 替换数据中的填充值
valid_data_train = wave_filed_data_prepare.replace_invalid_values(extracted_data_train)
valid_data_test = wave_filed_data_prepare.replace_invalid_values(extracted_data_test)
# 归一化并保存数据
buoy_data_train, max_vals, min_vals = normalize_and_save_numpy(valid_data_train, save_path)
np.save(f"{save_path}/buoy_data_train.npy", buoy_data_train)
np.save(f"{save_path}/min_values_train.npy", min_vals)
np.save(f"{save_path}/max_values_train.npy", max_vals)
buoy_data_test, max_vals, min_vals = normalize_and_save_numpy(valid_data_test, save_path)
np.save(f"{save_path}/buoy_data_test.npy", buoy_data_test)
np.save(f"{save_path}/min_values_test.npy", min_vals)
np.save(f"{save_path}/max_values_test.npy", max_vals)
print(f"浮标值、最大最小值已保存到 {save_path}")
