import netCDF4 as nc
import numpy as np
import os
import xarray as xr
# 定义文件路径
input_dir = "E:/Dataset/met_waves/swan"  # 输入文件所在目录
output_dir = "E:/Dataset/met_waves/Swan_cropped/"  # 输出文件保存目录

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def crop_and_save(input_nc_file, output_nc_file):
    # 目标裁剪区域
    lon_min, lon_max = 5.6, 6.2
    lat_min, lat_max = 62.2, 62.5

    # 目标网格大小
    target_grid_size = (128, 128)

    # 打开原始 nc 文件
    ds = xr.open_dataset(input_nc_file)

    # 裁剪经纬度范围
    cropped_ds = ds.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_min, lat_max))

    # 提取裁剪后的经纬度和时间
    lons = cropped_ds.longitude.values
    lats = cropped_ds.latitude.values
    time = cropped_ds.time.values

    # 创建目标网格
    new_lon = np.linspace(lon_min, lon_max, target_grid_size[1])
    new_lat = np.linspace(lat_min, lat_max, target_grid_size[0])

    # 插值到新网格
    resampled_ds = cropped_ds.interp(longitude=new_lon, latitude=new_lat, method="linear")

    # 保留指定变量
    selected_vars = ["hs", "tm02", "theta0"]
    final_ds = resampled_ds[selected_vars]

    # 保存到新的 nc 文件
    final_ds.to_netcdf(output_nc_file)

    print(f"裁剪并重采样后的数据已保存到: {output_nc_file}")

def process_all_files():
    # 遍历文件名，并对每个文件进行裁剪
    for year in range(2017, 2021):
        for month in range(1, 13):  # 从2月到12月
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
# process_all_files()
crop_and_save(input_dir+'/swanSula202001.nc', output_dir+'/swanSula202001_cropped.nc')