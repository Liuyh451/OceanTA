from netCDF4 import Dataset
def data_information(file_path):

    # 打开文件
    dataset = Dataset(file_path, mode='r')

    # 打印全部变量名
    print("变量列表：", dataset.variables.keys())

    # 读取经纬度
    lats = dataset.variables['latitude'][:]
    lons = dataset.variables['longitude'][:]

    # 打印经纬度范围和分辨率
    print(f"纬度范围: {lats.min()} 到 {lats.max()}，共 {len(lats)} 个纬度点")
    print(f"经度范围: {lons.min()} 到 {lons.max()}，共 {len(lons)} 个经度点")

    # 网格形状
    print(f"网格总形状: {len(lats)} × {len(lons)}，总网格数: {len(lats) * len(lons)}")

    # 打印维度信息
    print("\n维度信息:")
    for dim in dataset.dimensions.values():
        print(f"{dim.name}: {len(dim)}")

    # 打印变量的基本信息（维度、单位等）
    print("\n变量详细信息:")
    for var_name, var in dataset.variables.items():
        print(f"{var_name}: 维度 {var.dimensions}, 形状 {var.shape}, 单位: {getattr(var, 'units', '无单位')}")

    # 关闭文件
    dataset.close()

