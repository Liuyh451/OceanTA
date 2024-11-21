import netCDF4 as nc
import numpy as np
import os

# 定义裁剪区域
lon_min, lon_max = 5.6, 6.2
lat_min, lat_max = 62.2, 62.5
target_grid_points = 128

# 定义文件路径
input_dir = "E:/Dataset/met_waves/swan"  # 输入文件所在目录
output_dir = "E:/Dataset/met_waves/Swan_cropped/"  # 输出文件保存目录

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def crop_and_save(input_file, output_file):
    # 打开原始文件
    with nc.Dataset(input_file, mode='r') as src:
        # 获取变量和维度数据
        lons = src.variables['longitude'][:]
        lats = src.variables['latitude'][:]

        # 找到符合裁剪条件的经纬度索引
        current_lon_idx = np.where((lons >= lon_min) & (lons <= lon_max))[0]
        current_lat_idx = np.where((lats >= lat_min) & (lats <= lat_max))[0]

        # 获取当前裁剪区域的经纬度范围
        current_cropped_lons = lons[current_lon_idx]
        current_cropped_lats = lats[current_lat_idx]

        # 计算当前经纬度范围的步长
        lon_step = (current_cropped_lons[-1] - current_cropped_lons[0]) / (len(current_cropped_lons) - 1)
        lat_step = (current_cropped_lats[-1] - current_cropped_lats[0]) / (len(current_cropped_lats) - 1)

        # 重新计算裁剪区域的经纬度范围以满足128网格
        new_lon_min = current_cropped_lons[0]  # 保持左边界不变
        new_lon_max = new_lon_min + lon_step * (target_grid_points - 1)

        new_lat_min = current_cropped_lats[0]  # 保持下边界不变
        new_lat_max = new_lat_min + lat_step * (target_grid_points - 1)

        # 找到新的经纬度索引
        new_lon_idx = np.where((lons >= new_lon_min) & (lons <= new_lon_max))[0]
        new_lat_idx = np.where((lats >= new_lat_min) & (lats <= new_lat_max))[0]

        # 获取裁剪后的经纬度数据
        cropped_lons = lons[new_lon_idx]
        cropped_lats = lats[new_lat_idx]

        # 提取目标变量的数据
        hs_data = src.variables['hs'][:, new_lat_idx, new_lon_idx]
        tm02_data = src.variables['tm02'][:, new_lat_idx, new_lon_idx]
        theta0_data = src.variables['theta0'][:, new_lat_idx, new_lon_idx]

        # 时间变量（如果有时间维度）
        if 'time' in src.variables:
            time_data = src.variables['time'][:]
            time_units = src.variables['time'].units
            time_calendar = src.variables['time'].calendar

    # 创建新的裁剪后的文件
    with nc.Dataset(output_file, mode='w', format='NETCDF4') as dst:
        # 定义维度
        if 'time' in src.variables:
            dst.createDimension('time', None)  # 可变时间维度
        dst.createDimension('latitude', len(cropped_lats))
        dst.createDimension('longitude', len(cropped_lons))

        # 创建并写入变量
        # 时间
        if 'time' in src.variables:
            time_var = dst.createVariable('time', 'f8', ('time',))
            time_var[:] = time_data
            time_var.units = time_units
            time_var.calendar = time_calendar

        # 纬度
        lat_var = dst.createVariable('latitude', 'f4', ('latitude',))
        lat_var[:] = cropped_lats
        lat_var.units = 'degrees_north'

        # 经度
        lon_var = dst.createVariable('longitude', 'f4', ('longitude',))
        lon_var[:] = cropped_lons
        lon_var.units = 'degrees_east'

        # hs
        hs_var = dst.createVariable('hs', 'f4', ('time', 'latitude', 'longitude'))
        hs_var[:] = hs_data
        hs_var.units = 'meters'
        hs_var.long_name = 'Significant Wave Height'

        # tm
        tm_var = dst.createVariable('tm', 'f4', ('time', 'latitude', 'longitude'))
        tm_var[:] = tm02_data
        tm_var.units = 'seconds'
        tm_var.long_name = 'Mean Wave Period'

        # dirm
        dirm_var = dst.createVariable('dirm', 'f4', ('time', 'latitude', 'longitude'))
        dirm_var[:] = theta0_data
        dirm_var.units = 'degrees'
        dirm_var.long_name = 'Mean Wave Direction'

    print(f"裁剪完成并保存文件: {output_file}")
crop_and_save("E:/Dataset/met_waves/swan/swanSula202001.nc","E:/Dataset/met_waves/Swan_cropped/swanSula202001_cropped.nc")

def process_all_files():
    # 遍历文件名，并对每个文件进行裁剪
    for year in range(2017, 2021):
        for month in range(2, 13):  # 从2月到12月
            # 构建文件名
            file_name = f"swanSula{year}{month:02d}.nc"
            input_file = os.path.join(input_dir, file_name)

            # 检查文件是否存在
            if os.path.exists(input_file):
                # 输出文件路径
                output_file = os.path.join(output_dir, f"swanSula{year}{month:02d}_cropped.nc")

                # 裁剪并保存
                crop_and_save(input_file, output_file)
            else:
                print(f"文件 {input_file} 不存在")


# 执行批量处理
# process_all_files()
