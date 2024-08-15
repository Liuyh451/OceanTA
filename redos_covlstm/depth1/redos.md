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
subset_t = data['t'].sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
```

问题起因：cuda内存溢出，只用十几天的数据，批次设置很低，也是一样，在4080跑也是溢出，问了师兄师姐得知没有进行裁剪，分辨率太高。

### 1.3 重新绘制u，v图像

由于u，v变量使用的是lonu和latu，和s，t等使用的lon，lat坐标不同，要对其进行重新绘制

1.确定目标数据经纬度范围

```python
lon_range = np.linspace(112.0, 117.1, data['u'].shape[2])
lat_range = np.linspace(12.9, 15.7, data['u'].shape[1])
u = xr.DataArray(data['u'].values, dims=['lv', 'latu', 'lonu'], coords={'lonu': lon_range, 'latu': lat_range})
```

2.创建dataarray对象并赋予坐标

```python
lon_range = np.linspace(112.0, 117.1, data['v'].shape[2])
lat_range = np.linspace(12.9, 15.7, data['v'].shape[1])
v = xr.DataArray(data['v'].values, dims=['lv', 'latv', 'lonv'], coords={'lonv': lon_range, 'latv': lat_range})
```

3.确定源数据经纬度范围

```python
target_lon = data['lon'].values
target_lat = data['lat'].values
```

4.进行插值

```python
regrid_u = u.interp(lonu=target_lon, latu=target_lat, method='linear')
regrid_v = v.interp(lonv=target_lon, latv=target_lat, method='linear')
```

### 1.4 替换缺省值

数据集的默认缺省值为-32768.0，在进行归一化的时候，np.nanmax不能识别这个数，必须把它转为nan才可以

#### 1.遇到的问题：

之前没注意到这个地方，导致数据集max=4.x，min=-32768，归一化后数据集数据均为0.9999x，这样的np数组在进行转张量时，会因为数据精度丢失，导致数据集全为1.0，在进行反归一化和rmse计算时会出现大量负值，rmse可能达到几千万

```python
torch.from_numpy(dataset.target_sst).float().to(self.device)
```

#### 2.排查思路：

首先发现了test_true全为1.0的问题，伴随rmse异常，发现转tensor前数据正常，于是发现了上面函数的问题，但是将float换为double仍不能解决问题。

然后想到为什么数据集归一化后都是0.9999x，于是排查归一化函数，调试过程中发现max正常，min值为-32768，查资料得知数据缺失时用这个值填充。

#### 3.对数据进行替换

```python
# 选择数据变量
data = dataset['variable'].values
# 替换标记值
data[data == -32768] = np.nan
```

解决问题

## 2.网络

### 2.1 covlstmformer

该类下x = fold_tensor(x, (28, 52), (5, 5))，要把28*52，尺寸算对

### 2.2 fold_tensor

改完尺寸后遇到报错

```
RuntimeError: shape '[-1, 192, 3, 25]' is invalid for input of size 37500
```

一层层找，最终锁定了`tensor = tensor.reshape(-1, 192, 3, 25)`首先是-1，`-1` 是一个特殊的值，表示该维度的大小由其他维度的大小推断出来。在这种情况下，`-1` 让 PyTorch 自动计算出这个维度的大小，以确保张量的总元素数不变。，所以推测37500这个数可能是由这个形状的乘积得到的，37500/75=500, 而500/192不能除尽，所以才会有shape[0] invalid（37500这个数忘了咋得到的了，就是调参中无意得到的），而原图像是60*80=4800，4800/3/25=64刚好是192的三分之，所以推测，第二维可以通过28×52（即分辨率大小）/25 （补丁块的大小）得到，而3代表的是卷积核大小，并未参与运算。所以最终确定第二维为50（<58，56可能更好）

## 3.对比实验

### 3.1 实验a

**采用老方法 100轮，尺寸12*16**

```python
toal:1456
sse:5.6228657
RMSE：0.057160055630032104
CORR：0.9694028981996533
size:(59, 28, 52) (12, 52, 28) (12, 52, 28)
1层:NRMSE RESULT:
 0.010819319167659274
```

### 3.2 实验b

**采用新方法 100轮，尺寸12*16**

```python
toal:1456
sse:6.6913757
RMSE:0.05769584256973941
CORR：0.9686116887876041
size:(59, 28, 52) (12, 52, 28) (12, 52, 28)
1层:NRMSE RESULT:
 0.01086920567416595
```

### 3.3 实验c

**采用新方法 100轮，尺寸5*10**

#### C_1

```python
total: 1456
sse: 5.7022204
RMSE: 0.057232943476270164
CORR: 0.9692575649772497
size: (59, 28, 52) (12, 52, 28) (12, 52, 28)
1层: NRMSE RESULT:
 0.010835468714278389
```

#### C_2

```python
total: 5.6253853
sse: 0.05705396438060146
RMSE: 0.9699527028668825
CORR: 0.010798477388576214
size: (59, 28, 52) (12, 52, 28) (12, 52, 28)
1层: NRMSE RESULT:
 0.010798477388576214
```

### 3. 4 实验d

报错，遇到2.2中的错误

### 结论

通过实验a，b确定新方法读入数据是正确的；通过b，c确定特征图尺寸影响不大

## 4. 实机训练

### 4.1 depth 1

#### 1. 实验a

**total data epoch=800**

