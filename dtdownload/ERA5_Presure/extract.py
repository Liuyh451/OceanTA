import pandas as pd
import os

def parse_cma_typhoon_file(filepath):
    """
    解析 CMA Micaps 第6类台风路径文件，提取中心点经纬度和时间

    参数:
        filepath: CMA台风文件路径（如 CH2001BST.txt）

    返回:
        DataFrame，包含 ['time', 'lat', 'lon']
    """
    records = []

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line == "" or line.startswith("66666") or len(line) < 20:
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        try:
            yyyymmddhh = parts[0]  # 如 1949011300
            time = pd.to_datetime(yyyymmddhh, format='%Y%m%d%H')
            lat = float(parts[2]) / 10  # 纬度
            lon = float(parts[3]) / 10  # 经度

            records.append((time, lat, lon))
        except Exception as e:
            continue  # 忽略异常行

    df = pd.DataFrame(records, columns=['time', 'lat', 'lon'])
    return df


def extract_era5_from_cma_batch(cma_folder, nc_path, years, var_name='divergence',level_index=0, save_path=None):
    """
    从多个CMA台风路径文件提取中心点，并从ERA5中提取对应的10x10度网格

    参数：
        cma_folder: 存放 CMA 台风路径的文件夹（如 E:/Dataset/CMA/CMABSTdata）
        nc_path: ERA5 数据路径（一个完整的.nc文件，时间共7304个步长）
        years: 年份列表，如 [2001, 2002, 2003, 2004, 2005]
        var_name: 提取的变量名
        save_path: 可选，保存为.npy的路径

    返回:
        patches: list，每个是 [lat, lon] 网格数据
    """
    from netCDF4 import Dataset, num2date
    import numpy as np

    ds = Dataset(nc_path)
    var = ds.variables[var_name]  # e.g., (time, lat, lon)
    lats = ds.variables['latitude'][:]
    lons = ds.variables['longitude'][:]
    times = num2date(ds.variables['valid_time'][:], ds.variables['valid_time'].units)
    # 判断变量是否含 pressure_level
    has_level = len(var.dimensions) == 4

    # 构造时间索引表
    time_dict = {t: i for i, t in enumerate(times)}

    patches = []

    for year in years:
        filename = os.path.join(cma_folder, f'CH{year}BST.txt')
        if not os.path.exists(filename):
            print(f"文件缺失：{filename}")
            continue

        typhoon_df = parse_cma_typhoon_file(filename)

        for _, row in typhoon_df.iterrows():
            time = row['time']
            print(time)
            lat_c = row['lat']
            lon_c = row['lon']

            if time not in time_dict:
                continue  # ERA5中没有这个时间，跳过

            time_idx = time_dict[time]

            # 提取10°x10°区域
            lat_mask = (lats >= lat_c - 5) & (lats <= lat_c + 5)
            lon_mask = (lons >= lon_c - 5) & (lons <= lon_c + 5)

            lat_indices = np.where(lat_mask)[0]
            lon_indices = np.where(lon_mask)[0]

            if len(lat_indices) == 0 or len(lon_indices) == 0:
                print(f"跳过边界外点：{lat_c}, {lon_c}")
                continue

            # patch = var[time_idx, 0, lat_indices[0]:lat_indices[-1] + 1, lon_indices[0]:lon_indices[-1] + 1]
            try:
                if has_level:
                    patch = var[time_idx, level_index,
                            lat_indices[0]:lat_indices[-1] + 1,
                            lon_indices[0]:lon_indices[-1] + 1]
                else:
                    patch = var[time_idx,
                            lat_indices[0]:lat_indices[-1] + 1,
                            lon_indices[0]:lon_indices[-1] + 1]
            except IndexError as e:
                print(f"索引错误: {e}, 跳过")
                continue

            if patch.size == 0:
                print(f"空patch跳过: year={year}, time={time}, lat={lat_c}, lon={lon_c}")
                continue

            patches.append(patch)

    ds.close()

    if save_path:
        shapes = [p.shape for p in patches]
        unique_shapes = set(shapes)
        print(f"总共 patch 数量: {len(patches)}，唯一形状数量: {len(unique_shapes)}")
        print("前几个 patch 的形状：", shapes[:5])

        np.save(save_path, np.array(patches, dtype=object))

        print(f"已保存到：{save_path}")

    return patches
cma_folder = r"E:\Dataset\CMA\CMABSTdata"

# for i in range(2004, 2006):
#     nc_path = r"E:\Dataset\ERA5\Presure\WindSpeed\U-component of wind\2001-2005 200hPa.nc"
#     save_path = r"E:\Dataset\ERA5\Extracted\200hPa\UWind" + str(i) + ".npy"
#     patches = extract_era5_from_cma_batch(cma_folder, nc_path, years=[i],
#                                           var_name='u', save_path=save_path)
# for i in range(2001, 2006):
#     nc_path = r"E:\Dataset\ERA5\Presure\WindSpeed\V-component of wind\VWind2001_2005.nc"
#     save_path = r"E:\Dataset\ERA5\Extracted\200hPa\VWind" + str(i) + ".npy"
#     patches = extract_era5_from_cma_batch(cma_folder, nc_path, years=[i],
#                                           var_name='v', save_path=save_path)
# for i in range(2001, 2006):
#     nc_path = r"E:\Dataset\ERA5\Presure\WindSpeed\700hPa.nc"
#     save_path = r"E:\Dataset\ERA5\Extracted\700hPa\UWind" + str(i) + ".npy"
#     patches_u = extract_era5_from_cma_batch(cma_folder, nc_path, years=[i],
#                                           var_name='u', save_path=save_path)
#     save_path = r"E:\Dataset\ERA5\Extracted\700hPa\VWind" + str(i) + ".npy"
#     patches_v = extract_era5_from_cma_batch(cma_folder, nc_path, years=[i],
#                                           var_name='v', save_path=save_path)
# for i in range(2001, 2006):
#     nc_path = r'E:\Dataset\ERA5\Presure\Geopotential\2001_05.nc'
#     save_path = r"E:\Dataset\ERA5\Extracted\225hPa\Geopotential" + str(i) + ".npy"
#     patches_u = extract_era5_from_cma_batch(cma_folder, nc_path, years=[i],
#                                           var_name='z', level_index=0,save_path=save_path)
#     save_path = r"E:\Dataset\ERA5\Extracted\500hPa\Geopotential" + str(i) + ".npy"
#     patches_v = extract_era5_from_cma_batch(cma_folder, nc_path, years=[i],
#                                           var_name='z',level_index=1, save_path=save_path)
for i in range(2001, 2006):
    nc_path = r'E:\Dataset\ERA5\Presure\Geopotential\2001_05.nc'
    save_path = r"E:\Dataset\ERA5\Extracted\225hPa\Geopotential" + str(i) + ".npy"
    patches_u = extract_era5_from_cma_batch(cma_folder, nc_path, years=[i],
                                          var_name='z', level_index=0,save_path=save_path)
    save_path = r"E:\Dataset\ERA5\Extracted\500hPa\Geopotential" + str(i) + ".npy"
    patches_v = extract_era5_from_cma_batch(cma_folder, nc_path, years=[i],
                                          var_name='z',level_index=1, save_path=save_path)