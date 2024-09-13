import numpy as np
import xarray as xr
from datetime import datetime, timedelta


def clip_nc_region(nc_file, date):
    # 打开 NetCDF 文件
    data = xr.open_dataset(nc_file)
    # 确定经纬度范围
    lon_range = np.linspace(112.0, 117.1, data['u'].shape[2])
    lat_range = np.linspace(12.9, 15.7, data['u'].shape[1])
    u = xr.DataArray(data['u'].values, dims=['lv', 'latu', 'lonu'], coords={'lonu': lon_range, 'latu': lat_range})

    # 创建 DataArray 对象并赋予坐标
    lon_range = np.linspace(112.0, 117.1, data['v'].shape[2])
    lat_range = np.linspace(12.9, 15.7, data['v'].shape[1])
    v = xr.DataArray(data['v'].values, dims=['lv', 'latv', 'lonv'], coords={'lonv': lon_range, 'latv': lat_range})

    # 确定目标经纬度范围
    target_lon = data['lon'].values
    target_lat = data['lat'].values

    # 进行插值
    regrid_u = u.interp(lonu=target_lon, latu=target_lat, method='linear')
    regrid_v = v.interp(lonv=target_lon, latv=target_lat, method='linear')

    # 截取 s, t, zeta 使用的 lon 和 lat
    # 指定要截取的经纬度范围
    lon_min, lon_max = 112, 117.1  # 经度范围
    lat_min, lat_max = 12.9, 15.7  # 纬度范围
    subset_t = data['t'].sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
    subset_s = data['s'].sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
    subset_zeta = data['zeta'].sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
    subset_u = regrid_u.sel(lonu=slice(lon_min, lon_max), latu=slice(lat_min, lat_max))
    subset_v = regrid_v.sel(lonv=slice(lon_min, lon_max), latv=slice(lat_min, lat_max))
    # 创建新的数据集
    subset = xr.Dataset({
        't': subset_t,
        's': subset_s,
        'zeta': subset_zeta,
        'u': subset_u,
        'v': subset_v
    })
    # 保存为新的 NetCDF 文件
    subset.to_netcdf('/home/hy4080/redos/subset_file/subset_' + date + '.nc')

def date_incrementer(start_year, end_year):
    # 从指定年份的1月1日开始
    current_date = datetime(start_year, 1, 1)
    start_date = datetime(2001, 10, 31)
    current_date
    # 指定结束日期
    end_date = datetime(end_year + 1, 1, 1)

    # 循环自增日期
    while current_date < end_date:
        # 转为8位字符串格式 YYYYMMDD
        date_str = current_date.strftime('%Y%m%d')
        nc_file = '/home/hy4080/redos/decomprefile/REDOS_1.0_' + date_str + '.nc'
        clip_nc_region(nc_file, date_str)
        print(nc_file,"cilp done")
        # 日期自增1天
        current_date += timedelta(days=1)

# 示例用法：从2024年1月1日开始，到2024年12月31日
date_incrementer(1992, 2006)
