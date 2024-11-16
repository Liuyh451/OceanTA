# 1.数据集
## 1.1 数据范围

![image.png](https://picbed-1313037164.cos.ap-nanjing.myqcloud.com/202411141404607.png)

##  1.2 数据组成
### 1.2.1 使用的参数

| 变量名  | 维度        | 类型      | 填充值 (_FillValue) | 标准名 (standard_name) | 长名 (long_name)                                                  | 单位 (units) | 有效范围 (valid_range) |
| ---- | --------- | ------- | ---------------- | ------------------- | --------------------------------------------------------------- | ---------- | ------------------ |
| tm02 | ('time',) | float32 | 9999.0           | 平均波浪周期              | Mean wave period estimated from 0th and 2nd moment of spectrum  | s          | [0., 30.]          |
| mdir | ('time',) | float32 | 9999.0           | 平均波向                | Mean wave direction (from) calculated from directional spectrum | degree     | [0., 360.]         |
| Hm0  | ('time',) | float32 | 9999.0           | 有效波高                | Significant wave height estimate from spectrum                  | m          | [0., 25.]          |

## 1.2.2 浮标数据
使用的是五个地点的浮标数据，来源于[Bruk av observasjonsdata i SVV-E39-prosjektet](https://thredds.met.no/thredds/catalog/obs/buoy-svv-e39/catalog.html)
主要用于：浮标的观测数据被作为上下文（context）输入到条件变分自编码器（CVAE）中。这些数据为模型提供了关于当前时刻和过去一段时间内波浪场的基本信息
### 1.2.3 SWAN数据
来自于[SWAN250/Sula](https://thredds.met.no/thredds/catalog/e39_models/SWAN250/Sula/catalog.html)250m网格，时间分辨率为1小时，范围latitude 62.0◦ to 62.6◦ and longitude 5.3◦ to 6.8◦
主要用于：在条件变分自编码器（CVAE）中，历史的浮标观测数据被用作 **上下文输入**，而模型的目标是基于这些观测数据生成**未来时刻的波浪场**。波浪场数据作为目标输出或“标签”，用于训练模型，帮助模型学习如何从浮标观测数据中推断出未来的波浪场
## 1.3 数据处理
### 1.3.1 采样
时间分辨率为10min，每个数据集都是1个月的数据进行封装的；
空间分辨率：纬度和经度之间的间隔分别为 `9.54e-06` 和 `0.0001144`；
一共有网格点：Total number of grids: 19927296 (Latitude: 4464, Longitude: 4464)
>~~目前考虑对时空分辨率进行降低，降至每个网格大小为$250*250$~~

存在问题：这里的4464刚好就是每隔10分钟采样一次，31天的次数，同时波高等数据是1维数据，4464个。这里的经纬度分辨率很可能不是$4464*4464$，而是这个经纬度下共采样了4464次，而浮标在移动，所以有4464组经纬度
### 1.3.2 数据清洗
#### 1.去除异常值
浮标C和F有段时间是没有数据的，要用Nan进行填充