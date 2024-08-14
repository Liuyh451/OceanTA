# redos

## 1.数据处理

### 1.1 数据提取

从文件夹中提取相应的文件

```python
for _ in range(31):  # 模拟从 "01" 自增到 "31"
    number_str = increment_number(number_str)
    num2=number_str
    nc_file = 'E:/DataSet/redos/REDOS_1.0_1994/'+num1+'/REDOS_1.0_199401'+num2+'.nc/REDOS_1.0_199401'+num2+'.nc'
```

### 1.2 数据裁剪

把数据裁剪到以下范围内，并存为新的文件

```python
lon_min, lon_max = 112, 117.1  # 经度范围
lat_min, lat_max = 12.9, 15.7  # 纬度范围
```

### 1.3 重新绘制u，v图像

由于u，v变量使用的是lonu和latu，和s，t等使用的lon，lat坐标不同，要对其进行重新绘制