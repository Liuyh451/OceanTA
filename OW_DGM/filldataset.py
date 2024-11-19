import netCDF4 as nc
import numpy as np
import os


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


# 使用示例
# 创建 Sulafjorden 的 2017 年 1 月到 3 月的文件
create_missing_nc(
    template_file=r'E:\Dataset\met_waves\buoy\201704_E39_C_Sulafjorden_wave.nc',
    output_dir=r'E:\Dataset\met_waves\fill',
    start_month='201701',
    end_month='201703',
    file_prefix='E39_C_Sulafjorden_wave'
)

# 创建 Vartdalsfjorden 的 2017 年 1 月到 10 月的文件
create_missing_nc(
    template_file=r'E:\Dataset\met_waves\buoy\201711_E39_F_Vartdalsfjorden_wave.nc',
    output_dir=r'E:\Dataset\met_waves\fill',
    start_month='201701',
    end_month='201710',
    file_prefix='E39_F_Vartdalsfjorden_wave'
)
