# 1.数据集
## 1.1 数据范围
浮标的数据范围如下：实际上浮标测量的只是一个点的数据，形成区域的原因可能是被水冲走了

![image.png](https://picbed-1313037164.cos.ap-nanjing.myqcloud.com/202411141404607.png)

之前一直在纠结的$128*128$的区域在哪，也被攻破，这幅图的经纬度范围给出了答案，也就是lon_min, lon_max = 5.6, 6.2  lat_min, lat_max = 62.2, 62.5，经过这个裁剪的结果是$134*124$，所以略微调整了经纬度范围即可找到目标区域62.200°N-62.486°N，5.603°E-6.219°E，$128\times128$
![image.png](https://picbed-1313037164.cos.ap-nanjing.myqcloud.com/202411192259867.png)

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
去除异常值：浮标C和F有段时间是没有数据的，要用Nan进行填充。由于观测点之间不同月份观测的时间可能不一样，所以选择将不足4464的直接用nan补为4464
![image.png](https://picbed-1313037164.cos.ap-nanjing.myqcloud.com/202411182330198.png)
### 1.3.3 重编码平均波方向

   - 由于 0° 和 360° 是相同的方向，所以在处理波浪数据时，我们不直接使用角度值。为了避免这个问题，我们将 `dirm` 编码为两个值：`cos(dirm)` 和 `sin(dirm)`。
   - 通过这种方式，`dirm` 就变成了一个二维的表示，表示为 `(cos(dirm), sin(dirm))`，这避免了直接使用角度时的问题。
   - 通过这种编码，我们可以轻松地将 `(cos(dirm), sin(dirm))` 转换回原始的 `dirm` 值，方法是通过反正切函数 `atan2`。
## 1.3.4 归一化
本来考虑的是在处理每个月的数据时进行归一化，但是考虑到后面可能要对数据进行反归一化，而又不确定每个月的数据长度，无法根据保存下来的最大最小值进行反归一化。因此考虑先将数据读取，然后对数据进行重编码；之后将数据全部拼接起来。最后对全部数据进行归一化，并把最大最小值保存下来

 ~~1.4 滑动窗口~~
~~使用了大小为3，步长为4的窗口进行数据集构建，下面的数据为拼接后（把5个站点的数据按时间维度）的数据~~
~~原始数据形状: torch.Size$([5, 113615, 3])$~~
~~滑动窗口后的数据形状: (5, 113613, 3, 3)~~
## 1.4 对齐数据
使用nan填充每个时间点的数据，使得每个浮标观测数据和后发数据在时间维度上是一致的。而且海浪场t时刻的数据是用浮标观测数据t，t-10，t-20分钟的数据作为上下文背景的。所以第一个数据没有上下文，考虑舍弃或者让nan作为上下文。其他的点都是用浮标观测数据的后30分钟数据作为上下文背景的。
# 2.网络结构
## 2.1 上下文编码器
原文中说的是将（3,4）其中3为特征数，4为时间步，中的特征拆为3个，按4个时间步（1，4）分别和id向量嵌入拼接起来，生成的是$1*20$的向量，分别输入到3个GRU中，其中第一个GRU的结果和生成的向量一起输入到第2个GRU中，同理第2个GRU的结果和生成的向量一起输入到第3个GRU中，要满足GRU是共享GRU，并且解码器可以满足任意个数浮标输入。
实际上，这个可以用矩阵进行并行运算：
1. 数据准备：输入数据格式调整为(num_buoys, time_steps, obs_dim)，也就是（浮标的数量，时间步，特征）
(num_buoys, time_steps, obs_dim + id_emb_dim)。拼成这个
```python
class ContextualEncoder(nn.Module):  
    def __init__(self, obs_dim=3, time_steps=4, id_emb_dim=16, gru_hidden_size=512):  
        super(ContextualEncoder, self).__init__()  
        # 浮标ID的嵌入层  
        self.embedding = nn.Embedding(num_embeddings=100, embedding_dim=id_emb_dim)  
        # GRU层（共享的）  
        self.gru = nn.GRU(input_size=obs_dim + id_emb_dim, hidden_size=gru_hidden_size, batch_first=True)  
        # 最大池化层  
        self.max_pool = nn.AdaptiveMaxPool1d(1)  
  
    def forward(self, buoy_obs, buoy_ids):  
        """  
        输入：  
        - buoy_obs: (num_buoys, time_steps, obs_dim), 每个浮标的观测数据  
        - buoy_ids: (num_buoys,), 每个浮标的ID  
        输出：  
        - context_vector: (gru_hidden_size,), 上下文向量  
        """        # 1. 获取浮标ID的嵌入  
        buoy_id_emb = self.embedding(buoy_ids)  # (num_buoys, id_emb_dim)  
        buoy_id_emb = buoy_id_emb.unsqueeze(1).repeat(1, buoy_obs.size(1), 1)  # (num_buoys, time_steps, id_emb_dim)  
  
        # 2. 拼接浮标观测数据和嵌入  
        combined_input = torch.cat([buoy_obs, buoy_id_emb], dim=-1)  # (num_buoys, time_steps, obs_dim + id_emb_dim)  
  
        # 3. 通过共享GRU进行编码  
        _, h_n = self.gru(combined_input)  # h_n: (1, num_buoys, gru_hidden_size)  
        buoy_embeddings = h_n.squeeze(0)  # (num_buoys, gru_hidden_size)  
  
        # 4. 最大池化聚合  
        buoy_embeddings = buoy_embeddings.unsqueeze(0).permute(0, 2, 1)  # (1, gru_hidden_size, num_buoys)  
        context_vector = self.max_pool(buoy_embeddings)  # (gru_hidden_size,)  
        # 5. 调整形状为 (gru_hidden_size, 1, 1)        context_vector = context_vector.permute(1, 0, 2)  # (gru_hidden_size, 1, 1)  
        return context_vector
```
### 2.1.2 GRU
**GRU有两个输入**，分别是：
1. **当前数据特征（X）**：这是模型在当前时间步接收到的输入数据。
2. **上一单元输出状态（旧H）**：这是模型在上一个时间步的输出状态，也被用作当前时间步的输入之一，以捕捉序列数据中的时间依赖性。
当前数据特征也就是观测值和位置编码拼接后的数据，上一个单元的状态就是512那个向量，是GRU自动向下传递的，不需要手动输入。
目前需要解决的是时间步
GRU中的512向量是

# 3.损失

这段文字详细介绍了**条件变分自编码器（Conditional Variational Autoencoder，CVAE）**与**对抗学习（Adversarial Learning，AL）**的结合（即CVAE-AL）。下面是对每个部分的解释：

1. **CVAE模型**：
   - **CVAE** 是在传统的变分自编码器（VAE）基础上的扩展，增加了条件变量 `c`，它使得模型在数据生成过程中能够控制特定的条件（如特定的波浪参数）。
   - CVAE由**编码器** `qφ(z|X, c)` 和**解码器** `pθ(X|z, c)` 组成：
     - **编码器**：将输入数据（X）和条件（c）转换为潜在变量（z）。
     - **解码器**：根据潜在变量（z）和条件（c）重建原始数据（X）。
   - 目标是最大化变分下界（ELBO），即：
    $$\log p_{\theta}(X | c) \geq \mathbb{E}[\log p_{\theta}(X | z, c)] - D_{KL}(q_{\phi}(z | X, c) || p_{\theta}(z | c))$$
     - 其中 `D_KL` 是Kullback-Leibler散度。

2. **重建损失（Reconstruction Loss）**：
   - 使用均方误差（MSE）作为重建损失，公式为：
   $$l_{rec} = \| X - \mathbb{E}[p_{\theta}(X | q_{\phi}(z | X, c), c)] \|^2$$
   - MSE会有正则化过度的风险，但期望较大的样本多样性，因此使用一个超参数来平衡重建损失与KL损失。

3. **KL损失（KL Loss）**：
   - 假设潜在变量 `z` 服从标准正态分布 `p_{\theta}(z | c) ∼ N(0, I)`。
   - 编码器 `q_{\phi}(z | X, c)` 输出潜在变量 `z` 的均值和方差，分别为 `μθ(X, c)` 和 `σ²θ(X, c)`。
   - 使用重参数化技巧从 `z ∼ q_{\phi}(z | X, c)` 进行采样，KL损失公式为：
    $$l_{kl} = \frac{1}{d} \sum \left( 2 \mu^2_{\theta}(X, c) + \sigma^2_{\theta}(X, c) - 1 - \log \sigma^2_{\theta}(X, c) \right)$$
     - 其中 `d` 是潜在向量 `z` 的维度。

4. **对抗学习（Adversarial Learning）**：
   - 在CVAE框架中，增加了对抗性训练。使用一个**鉴别器**来区分由解码器生成的输出和真实的波浪场。
   - 对抗损失的训练目标分别为：
     - **生成器损失**：$$l_{adv-G} = \text{lbce}(D_{\phi}(\hat{X}), 1)$$
     - **鉴别器损失**：$$l_{adv-D} = \frac{1}{2} \text{lbce}(D_{\phi}(\hat{X}), 0) + \frac{1}{2} \text{lbce}(D_{\phi}(X), 1)$$
   - 其中 `lbce` 是二元交叉熵损失函数，`Dφ` 是鉴别器。

5. **训练和推理**：
   - 训练过程中，生成器（包括上下文编码器、编码器、解码器）和鉴别器交替进行优化。
   - 训练损失公式为：
     $$l_G = l_{rec} + \lambda_{kl} l_{kl} + \lambda_{adv} l_{adv-G}$$
     $$l_D = l_{adv-D}$$
     - 其中 `λkl` 和 `λadv` 是平衡KL损失和对抗损失的超参数。
   - 在推理阶段，只有上下文编码器和解码器用于生成波浪场。

---
### 关键点：
- **CVAE**部分包括上下文编码器、编码器和解码器。训练时，生成器（包括上下文编码器、编码器和解码器）和鉴别器交替更新。
- **对抗损失**：通过鉴别器来优化生成的波浪场是否与真实波浪场相似。
- **KL损失**：对潜在空间的分布进行正则化，确保潜在变量遵循标准正态分布。

通过这种方法，CVAE-AL可以有效地结合生成模型和对抗训练，生成更逼真的波浪场预测。