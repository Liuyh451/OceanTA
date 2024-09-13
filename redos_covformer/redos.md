# 1.数据处理

## 1.1 数据提取

从文件夹中提取相应的文件

```python
#使用datetime可以方便的按日期获得文件名
current_date = datetime(start_year, 1, 1)
end_date = datetime(end_year + 1, 1, 1)
while current_date < end_date:
    date_str = current_date.strftime('%Y%m%d')
    nc_file = path + '/subset_' + date_str + '.nc'
    current_date += timedelta(days=1)
    ......
    #用字典保存数据，可以避免重复打开文件
    data_dict[var_name].append(data)
    for var_name in data_dict:
        #从列表转为numpy
        data_dict[var_name] = np.array(data_dict[var_name])
```

## 1.2 数据裁剪

把数据裁剪到以下范围内，并存为新的文件

```python
lon_min, lon_max = 112, 117.1  # 经度范围
lat_min, lat_max = 12.9, 15.7  # 纬度范围
subset_t = data['t'].sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
```

问题起因：cuda内存溢出，只用十几天的数据，批次设置很低，也是一样，在4080跑也是溢出，问了师兄师姐得知没有进行裁剪，分辨率太高。

## 1.3 对齐坐标

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

## 1.4 替换缺省值

数据集的默认缺省值为-32768.0，在进行归一化的时候，np.nanmax不能识别这个数，必须把它转为nan才可以

### 1.遇到的问题：

之前没注意到这个地方，导致数据集max=4.x，min=-32768，归一化后数据集数据均为0.9999x，这样的np数组在进行转张量时，会因为数据精度丢失，导致数据集全为1.0，在进行反归一化和rmse计算时会出现大量负值，rmse可能达到几千万

```python
torch.from_numpy(dataset.target_sst).float().to(self.device)
```

### 2.排查思路：

首先发现了test_true全为1.0的问题，伴随rmse异常，发现转tensor前数据正常，于是发现了上面函数的问题，但是将float换为double仍不能解决问题。

然后想到为什么数据集归一化后都是0.9999x，于是排查归一化函数，调试过程中发现max正常，min值为-32768，查资料得知数据缺失时用这个值填充。

### 3.对数据进行替换

```python
# 选择数据变量
data = dataset['variable'].values
# 替换标记值
data[data == -32768] = np.nan
```

解决问题

## 1.5 提取垂直剖面

之前的都是按层进行提取，提取该层整个平面的数据，按时间维度整理为tensor

现在需要提取一个点的24层垂直剖面

### 1. 给zeta补深度

由于zeta是海高异常，所以没有深度，形状(lon,lat)，而s，t，u，v等变量是(lv,lon,lat)，所以要用0给它补深度

```python
# 创建新的 zeta 数据数组，并初始化为 0
new_zeta_data = np.zeros((24, lat_dim, lon_dim))
# 将原来的 zeta 数据放入新的数组中的第一层
new_zeta_data[0, :, :] = zeta.values
```

### 2. 提取点剖面

在经纬网中提取一个点的数据，注意，虽然经过1.3的对齐坐标，但是坐标名没变，uv还是lonu(lonv)，s，t，zeta是lon

```python
profiles[var_name]r = ds[var_name].sel(lonu=lon, latu=lat, method='nearest')  #用于uv提取
profiles[var_name] = ds[var_name].sel(lon=lon, lat=lat, method='nearest')     #用于stzeta提取
```

### 3. 拓展维度

由于网络的输入要求是五维张量(N, T, C, H, W)，提取完后只有高度H，没有W，所以给点补维度。这里已经经过了时间长度的提取，现在向量形状为（N, H）所以

需要把它变为(N, H, 1)这里的1就是宽度W

```python
expanded_dict[key] = value.reshape(value.shape[0], value.shape[1], 1)
```

## 1.6 纠正错误

### 1. 深度顺序问题

这个数据集，数据是倒着的，depth0是最深层，而depth23是表面

### 2. 输入数据问题

模型的输入input=(SST, SSS, SLA ,SSWU, SSWV),都是表面数据也就是depth23，之前在读的时候（假如在对温度进行depth_n预测），输入的是4个表面数据SSS, SLA ,SSWU, SSWV和一个depth_n的SST，而此时的label也是depth_n的SST，所以模型的效果非常好，RMSE和CORR都很高。直到9月1号开完组会后才发现问题。及时修改了代码，输入的是表面5个变量，label设置为depth_n的SST（SSS）

在读取每个输入变量时，

```python
def read_raw_data(var_type, depth, time_step, data_dict):
    # 提取层训练数据
    raw_data_1 = extract_layer_data(data_dict, var_type, 23)
    # 提取层标签数据
    raw_data_2 = extract_layer_data(data_dict, var_type, depth)
```

### 3. 尝试把图转为60*80

```python
def resize_nc_variables(nc_file, new_shape=(60, 80)):
```

成功了，但是问题依然没有解决

# 2.网络

## 2.1 数据输入

## 2.2 位置编码

选用**Convlstm**作为位置编码

```python
Input:输入介绍
        A tensor of size [B, T, C, H, W] or [T, B, C, H, W]# 需要是5维的
    Output:输出介绍
        返回的是两个列表：layer_output_list，last_state_list
        列表0：layer_output_list--单层列表，每个元素表示一层LSTM层的输出h状态,每个元素的size=[B,T,hidden_dim,H,W]
        列表1：last_state_list--双层列表，每个元素是一个二元列表[h,c],表示每一层的最后一个timestamp的输出状态[h,c],h.size=c.size = [B,hidden_dim,H,W]
```

在这个例子中，输入是一个包含时序信息的五维张量，表示 [B, T, C, H, W] 或 [T, B, C, H, W] 这样包含时间、批次、通道、高度和宽度的数据。通过 LSTM 处理后，输出会是两个列表，`layer_output_list` 和 `last_state_list`。这两个列表的输出可以用来进行位置编码，以下是如何利用这些结果进行位置编码的解释：

### 1. **`layer_output_list` 的作用**
`layer_output_list` 是一个单层列表，每个元素对应 LSTM 每一层的输出 `h` 状态，表示了每一层 LSTM 在处理整个时间序列后，时间维度的输出。每个元素的形状是 `[B, T, hidden_dim, H, W]`，即保留了批次大小、时间序列长度、隐藏维度、高度和宽度。

- **位置编码**：`layer_output_list` 可以被视为时序上每一时刻的隐藏状态表示，因此可以用作时序的位置信息。每个时间步 T 对应不同的 LSTM 输出，这意味着 LSTM 内部已经隐式地编码了位置信息。因此，这些输出可以直接用于位置编码，将时序信息传递给后续网络。
- **编码时序依赖**：在 Transformer 中，原始输入是平行处理的，缺少时序依赖，而 `layer_output_list` 包含了每个时间步的隐状态信息，可以用于补充 Transformer 的时序依赖。

### 2. **`last_state_list` 的作用**
`last_state_list` 是一个双层列表，其中每个元素是一个二元列表 `[h, c]`，表示每一层 LSTM 的最后一个时间步的隐状态 `h` 和细胞状态 `c`。`h` 和 `c` 的形状均为 `[B, hidden_dim, H, W]`。

- **使用最后一个时间步的状态**：`last_state_list` 中的 `h` 和 `c` 表示 LSTM 在序列末尾（即最后一个时间步）的状态。对于序列处理任务，通常最后一个时间步的状态可以代表整个序列的信息，特别是在捕捉时序依赖方面。你可以将这个最后的隐状态 `h` 作为整个序列的时序位置信息的一种总结。
- **压缩时序信息**：通过 `last_state_list`，你能够获得压缩的时序信息，而不需要关注整个序列的所有时间步。因此，这可以用于简化位置编码的过程，将最后一个状态作为输入序列的时序位置表示。

### 3.利用 `layer_output_list` 中的时序输出
- **直接编码**：将 `layer_output_list` 作为 Transformer 的输入或与输入嵌入相加。由于 LSTM 的输出 `h` 自然包含了位置信息，这些输出向量本身可以起到位置编码的作用。
- **融合策略**：你可以选择使用全部时间步的 LSTM 输出 `layer_output_list`，也可以将时间步的输出向量 `h` 与原始的 Transformer 输入进行融合（如通过相加或拼接）。

## 2.3 Patch Embeding

将 ConvLSTM 的输出处理为适合 Transformer 编码器输入的格式，具体过程如下：

### 1. **将输出划分为非重叠的空间 Patch**
   - 从 ConvLSTM 处理的输出中，我们首先将每个通道划分为 `N` 个不重叠的空间 patch（块），其中 `N = HW/P²`，即通道的高度 `H` 和宽度 `W` 分别除以 `P`，然后得到每个 patch 的大小为 `P × P`。
   - `N` 代表空间位置的数量，而 `T` 是时间步数。也就是说，我们在时序的每一个时间步 `t` 上，将每个空间通道划分为 `N` 个 patch。

### 2. **将 Patch 拉平成向量**
   - 每个 patch 被拉平成一个向量，表示为 `x(n,t)`，其中 `n` 表示空间位置，`t` 表示时间序列。每个向量 `x(n,t)` 的长度为 `5P²`，其中 `5` 代表特征维度（通道维度），`P²` 是每个 patch 的面积（即 `P × P` 大小的 patch 被拉平成一维向量）。

### 3. **线性映射到编码器输入格式**
   - 为了将这些 patch 向量转换为适合 Transformer 编码器的输入格式，我们使用一个可学习的线性嵌入层，将每个 patch 向量 `x(n,t)` 映射为一个新的嵌入向量 `z(n,t)`，其中 `z(n,t)` 的大小为 `D`（即 Transformer 编码器所需的输入维度）。
   - 这个映射过程通过一个可学习的权重矩阵 `WE` 完成，其中 `WE ∈ R(5P² × D)`。该矩阵将 `x(n,t)` 从长度为 `5P²` 的向量映射为长度为 `D` 的嵌入向量。

### 4. **公式解释**
   - 映射过程如公式所示：`z(n,t) = x(n,t) · WE`，即将每个 patch 向量 `x(n,t)` 通过线性变换乘以权重矩阵 `WE`，得到嵌入向量 `z(n,t)`。
   - 这些嵌入向量 `z(n,t)` 之后会作为输入传递给 Transformer 的编码器。

通过将 ConvLSTM 的输出划分为多个 patch，并将每个 patch 拉平成向量，再通过线性变换将其嵌入到 Transformer 所需的格式，这样可以将时空信息转化为适合 Transformer 编码器处理的输入数据形式。




![image-20240913202826433](https://raw.githubusercontent.com/Liuyh451/PicRep/img/img/image-20240913202826433.png)

## 2.4 covlstmformer

该类下x = fold_tensor(x, (28, 52), (5, 5))，要把28*52，尺寸算对

该网络 `covlstmformer` 的设计结合了卷积层（Conv）、编码器层（EncoderLayer）、特征图展开与折叠操作，主要用于处理时空序列数据。以下是这个网络结构的总结：

### 1. **网络结构概述**
   - **输入数据**：时空序列数据，形状为 `[B, T, C, H, W]`，其中 `B` 是批次大小，`T` 是时间步长，`C` 是通道数，`H` 和 `W` 分别表示高度和宽度。
   - **卷积层（Cov）**：网络使用了两个卷积层来提取特征。第一个卷积层 `cov1` 在最初处理输入数据，第二个卷积层 `cov2` 在第一次编码器层之后对数据进一步处理。
   - **编码器层（EncoderLayer）**：网络包含两个编码器层 `encode1` 和 `encode2`，分别在卷积层之后对展开的特征图进行编码。

### 2. **前向传播过程**
   - **Step 1: 初始卷积层处理**：
     - 输入形状为 `[16, 3, 5, 28, 52]` 的数据，经过 `cov1` 卷积层，输出的特征图被展开为大小为 `(5, 5)` 的小块，生成 `[16, 250, 3, 25]` 的张量。
   
   - **Step 2: 跳跃连接与第一个编码器**：
     - 使用 `encode1` 对卷积后的数据进行编码，并加上跳跃连接，保持形状为 `[16, 250, 3, 25]`。
     - 将特征图折叠回原来的空间维度 `[28, 52]`，即恢复为原始输入的大小。

   - **Step 3: 第二个卷积与编码器层处理**：
     - 经过第二个卷积层 `cov2` 处理，添加跳跃连接，再次展开特征图为 `(5, 5)` 的小块，并使用 `encode2` 进行编码。
     - 最后，将数据折叠回 `[28, 52]` 的空间维度。

   - **Step 4: 最后一层卷积**：
     - 经过 `cov_last` 卷积层处理，得到最终输出，输出的形状为 `[16, 1, 28, 52]`。

### 3. **总结**
   - 网络通过两个卷积层和两个编码器层来逐步提取时空特征。
   - 卷积操作提取局部特征，展开操作允许特征图在不同的空间块上进行编码。
   - 编码器层与跳跃连接用于捕捉更复杂的时空信息，保持特征的连贯性。
   - 最终通过 `cov_last` 层输出经过编码和卷积处理的特征图。

该网络的设计特别适合时空数据的处理，结合卷积和编码器层，能够有效捕捉局部和全局的时空特征。

### 2.4.2 残差连接

![image-20240818001637856](https://raw.githubusercontent.com/Liuyh451/PicRep/img/img/image-20240818001637856.png)

### 2.4.3 fold_tensor

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

# 4. 实机训练

## 4.1 未修改前

### 4.1.1 温度nRMSE

这里的图还是错的，未改输入变量之前的

![img](https://raw.githubusercontent.com/Liuyh451/PicRep/img/img/809539b4b4370a4f0f07328e6363fae3.png)

一月份数据的均值走势

![img](https://raw.githubusercontent.com/Liuyh451/PicRep/img/img/eb22b865c47f695f8a50213cbdc781f0.png)

训练过程中的loss和lr变化

![image-20240913212437060](https://raw.githubusercontent.com/Liuyh451/PicRep/img/img/image-20240913212437060.png)

## 4.2 修改后

### loss和lr变化过程

![image-20240910161738265](C:\Users\86156\AppData\Roaming\Typora\typora-user-images\image-20240910161738265.png)

## 4.3 问题

修改后模型收敛很快，而且通常在50个epoch就收敛，性能不再提升。RMSE正常，但是CORR却低

存在 **RMSE 低但相关系数（CORR）也低** 的情况。这种情况通常出现在以下情境中：

1. **模型偏差小但趋势捕捉不准**：模型的预测值与真实值非常接近，误差较小（RMSE 低），但它可能无法很好地捕捉到数据的整体变化趋势，导致相关系数低。换句话说，模型虽然对每个数据点的预测值误差较小，但对数据的波动和趋势没有较好的跟踪。

2. **数据范围较小**：如果数据的整体变化范围较小，RMSE 可能会很低，因为数值差异不大；但相关系数取决于模型对趋势的拟合程度，如果模型没有反映出趋势，相关性就会较低。

3. **过于平滑的模型**：如果模型过于平滑化预测，导致输出值与真实值之间的偏差较小，RMSE 会变低，但这种平滑化可能无法反映数据的真实波动，导致相关系数下降。

因此，虽然 RMSE 和相关系数通常是一起衡量模型好坏的重要指标，但它们并不是完全线性相关的，低 RMSE 不一定意味着高相关系数。

还是CORR问题，甚至有的为负，怀疑是有bug

## 5.实验结果

看起来数据是正常的，温度有个突变层

![img](https://raw.githubusercontent.com/Liuyh451/PicRep/img/img/01f2dfbc4c5ab1879aa301d49f28b157.png)
