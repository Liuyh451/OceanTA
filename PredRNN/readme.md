**复现代码来自https://github.com/thuml/predrnn-pytorch**

# 1.数据处理
## 1.1 对帧(Frame)展平
在对 **Moving MNIST** 进行预测时，使用 RNN 变体（例如 ConvLSTM、PredRNN、PredRNN++、MIM 等）来建模时序依赖关系，`self.frame_channel` 计算方式如下：

```python
self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
```

这是为了 **将输入帧（图像）拆分成多个小块（patch），并将它们视作时间步中的特征输入**，从而适应 RNN 变体的计算模式。

---

### **为什么对帧计算通道？**

在 RNN 变体（如 ConvLSTM）中，输入的 **每个时间步** 都需要一个张量，而 Moving MNIST 的原始输入是一组 2D 灰度图像（形状通常是 `(batch_size, time_steps, height, width, channels)`，其中 `channels=1` ）。如果直接输入 2D 图像，网络可能无法很好地学习局部运动模式，因此我们常使用 **patch-based** 方式对图像进行预处理，使模型能够更好地捕捉局部运动特征。

假设：

- Moving MNIST 每帧大小为 **64×64**
- `patch_size = 4`
- `img_channel = 1`（灰度图）

则：

```python
self.frame_channel = 4 * 4 * 1 = 16
```

这意味着，每个 4×4 的 patch 会被展平成 16 维的向量，乘在一起得到 **每个 patch 被展平后的特征通道数**，即每个 patch 的信息量，然后输入到 RNN 进行时序建模。

如果我们使用 `patchify` 技术将 **64×64** 的帧划分成 **16×16 个 patch（每个4×4）**，那么每帧的输入形状就会变为：

```python
(batch_size, time_steps, num_patches, frame_channel)
= (batch_size, time_steps, 16×16, 16)
= (batch_size, time_steps, 256, 16)
```

这种做法可以让 RNN 处理局部运动模式，而不是全局特征，从而提高预测效果。

---
>`patchify` 是一种 **将图像或序列划分为多个小块（patches）** 的方法，常用于计算机视觉和时序建模任务，例如 **ViT（Vision Transformer）** 或 **时空预测任务（如 Moving MNIST 预测）**。
>在 **RNN 变体（如 ConvLSTM、PredRNN、MIM）** 中，使用 `patchify` 主要是为了：
>- 让模型更容易学习局部运动模式，而不仅仅是全局信息。
>- 降低计算复杂度，提高训练效率。
>- 使输入数据适配 RNN 变体的输入格式（通常是**固定长度的向量**）。
## 1.2 数据载入
### 1.方式一 按patch方式
原作者是ims shape = (8, 20, 16, 16, 16)，这里形状为（batch，T，W，H，C）这里的C一般默认是1（灰色图），但是这里用了一个patch的方式进行了划分，见1.1部分的内容

### 2.方式二 默认通道为1

## 1.3 数据对比

---

Moving MNIST 数据集是视频预测任务（如 PredRNN）的经典基准数据集，其数据形状遵循时空序列的典型结构。以下是详细解析：

---

### **1. 单个样本的数据形状**
每个样本（即一个视频片段）的形状为：  
**`(num_frames, channels, height, width)`**  
具体数值为：  
**`(20, 1, 64, 64)`**  

| 维度         | 说明                                                   |
| ------------ | ------------------------------------------------------ |
| `num_frames` | 视频总帧数，通常为 20 帧（前 10 帧输入，后 10 帧预测） |
| `channels`   | 图像通道数，MNIST 为灰度图，通道数为 1                 |
| `height`     | 图像高度，标准为 64 像素                               |
| `width`      | 图像宽度，标准为 64 像素                               |

---

### **2. 数据集整体结构**
- **训练集**：包含 `N` 个样本，形状为 `(N, 20, 1, 64, 64)`  
- **测试集**：包含 `M` 个样本，形状为 `(M, 20, 1, 64, 64)`  

---

### **3. 与 Swan 海浪数据集的对比**
| **特性**       | **Moving MNIST**     | **Swan 海浪数据集**                           |
| -------------- | -------------------- | --------------------------------------------- |
| **数据内容**   | 运动数字（动态纹理） | 海浪参数（物理场演化）                        |
| **数据形状**   | `(20, 1, 64, 64)`    | 可能为 `(T, C, H, W)`，如 `(20, 3, 128, 128)` |
| **通道意义**   | 单通道（灰度强度）   | 多通道（如波高、流速分量）                    |
| **时间依赖性** | 简单线性运动 + 碰撞  | 复杂非线性流体动力学                          |

## **1.4 PredRNN的预测机制**
### **(1) 输入输出结构**
- **训练阶段**：  
  模型通过滑动窗口学习历史序列到未来序列的映射。  
  例如：输入前 `input_length=10` 帧，预测未来 `pred_length=10` 帧，总序列长度 `total_length=20`。
  
  ```
  Input : [t1, t2, ..., t10]  
  Output: [t11, t12, ..., t20]  # 一次性预测整个未来序列
  ```
  
- **测试阶段**：  
  使用相同逻辑，输入历史帧，模型直接输出完整的未来预测序列（而非逐帧迭代预测）。  
  例如：输入测试集的前10帧，直接得到未来10帧的预测结果。

### **(2) 预测模式对比**
| **预测模式** | **描述**                                               | **特点**                         |
| ------------ | ------------------------------------------------------ | -------------------------------- |
| **开环预测** | 输入完整历史帧，直接输出所有未来预测帧（一步预测）     | 无误差累积，依赖完整历史信息     |
| **闭环预测** | 逐步预测，将前一预测帧作为下一时间步的输入（递归预测） | 可能累积误差，更接近实际预测场景 |

**PredRNN默认采用开环预测**，即一次性生成全部预测帧。

### (3) 样本构造

1. **输入构造**：  
   从测试集中抽取连续的时间窗口，每个样本包含 `input_length` 帧作为输入。
   ```
   样本1: [t1, t2, ..., t10] → 预测 [t11, t12, ..., t20]  
   样本2: [t11, t12, ..., t20] → 预测 [t21, t22, ..., t30]  
   ```
2. **预测输出**：  
   模型对每个输入样本一次性输出对应的 `pred_length` 帧（如10帧）。

3. **评估方式**：  
   将预测的 `pred_length` 帧与真实未来帧逐帧对比，计算每个未来时间步（如t11到t20）的指标（RMSE、R²等）。

---

### **(4) swan测试场景示例**
#### **a. 数据说明**
- 测试集包含长度为100的海浪序列：`[t1, t2, ..., t100]`。
- 模型配置：`input_length=10`, `total_length=20` → `pred_length=10`。

#### **b. 预测过程**
1. **样本划分**：  
   将测试集划分为多个输入-输出对，每对包含连续的20帧（前10输入，后10真实值）：
   
   ```
   样本1: 输入 [t1-t10], 真实输出 [t11-t20]  
   样本2: 输入 [t11-t20], 真实输出 [t21-t30]  
   ...  
   样本9: 输入 [t91-t100], 真实输出 []（若总长度不足则舍弃）
   ```
   
2. **模型预测**：  
   对每个样本输入前10帧，模型输出未来10帧的预测结果。

PredRNN中的“预测序列为10”表示模型单次推理能生成未来10帧的完整预测。测试时，输入历史10帧，直接输出未来10帧，而非逐时间步迭代预测。评估需按预测步对齐所有样本的结果，统计每个未来位置（如第1步、第2步...第10步）的平均性能。

## 1.5 swan数据样本构造

由于原代码中是根据一个20个帧也就是一个样本有20个时间步进行划分的，用前10个时间步去预测后10个时间步，而swan数据每个样本只有一个时间步（一个样本其实就是一个时间步）。总结1.3和1.4的内容可以发现，我们需要对swan数据进行滑动窗口构造让每一个时间步变成20个时间步

# 2.训练

## 2.1 RSS过程
这一段代码用于 **遍历时间步，逐步生成预测帧**，并根据是否使用 **反向调度采样（Reverse Scheduled Sampling, RSS）** 选择当前时间步的输入帧。

---

###  (1) **代码解析**

```python
# 遍历时间步，逐步生成预测帧
for t in range(self.configs.total_length - 1):
    # 根据配置选择是否使用反向调度采样
    if self.configs.reverse_scheduled_sampling == 1:
        if t == 0:
            net = frames[:, t]  # 第一帧直接使用真实帧
        else:
            net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
    else:
        if t < self.configs.input_length:
            net = frames[:, t]  # 在输入长度内使用真实帧
        else:
            net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                  (1 - mask_true[:, t - self.configs.input_length]) * x_gen
```

---

### (2) RSS，反向调度采样

RSS 是 **用于提升模型泛化能力的一种数据采样策略**，可以 **控制预测时使用真实帧（ground truth）还是预测帧（x_gen）** 作为输入：

- **普通采样（非 RSS）**：训练时，前 `input_length` 帧使用真实数据，后面逐步过渡到预测数据。
- **RSS 采样**：在训练后期，让模型逐渐减少对真实帧的依赖，更早地开始使用预测数据作为输入，使其适应真实推理场景。

---

### (3) 代码逻辑

#### **a. 反向调度采样 **

```python
if self.configs.reverse_scheduled_sampling == 1:
    if t == 0:
        net = frames[:, t]  # 第一帧直接使用真实帧
    else:
        net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
```

- **第一帧 (`t == 0`) 直接使用真实帧** 作为输入，因为第一帧没有历史信息可参考。
- **后续帧 (`t > 0`) 采用**: $$\text{net} = \text{mask\_true} \times \text{真实帧} + (1 - \text{mask\_true}) \times \text{预测帧}$$
    - `mask_true` 是一个 **动态调整的掩码**，决定当前时间步是否使用真实帧（ground truth）。
    - **`mask_true = 1`** 时，使用真实帧 `frames[:, t]`。
    - **`mask_true = 0`** 时，使用预测帧 $x_{gen}$（即前一时间步模型的预测输出）。
    - **训练初期 `mask_true` 可能更接近 1（依赖真实数据），后期逐渐减少**。

---

#### **b. 非反向调度采样**

```python
if t < self.configs.input_length:
    net = frames[:, t]  # 在输入长度内使用真实帧
else:
    net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
          (1 - mask_true[:, t - self.configs.input_length]) * x_gen
```

- **前 `input_length` 帧直接使用真实数据**（训练时提供的真实帧）。
- **后面的帧开始用 `mask_true` 进行采样**：
    - `mask_true[:, t - self.configs.input_length]` **决定是否使用真实帧**。
    - **与 RSS 采样不同的是，这里的 `mask_true` 主要用于过渡，而 RSS 则用于更早地让模型适应预测任务**。

---

## **2.2 示例**

假设：

- `input_length = 5`
- `total_length = 10`
- `reverse_scheduled_sampling = 1`
- `mask_true` 变化如下：
  
    ```
    [1, 1, 0.8, 0.5, 0.2, 0, 0, 0, 0]
    ```
    

那么：

| 时间步 t | `frames[:, t]`（真实帧） | `x_gen`（预测帧） | `mask_true` | `net`                              |
| -------- | ------------------------ | ----------------- | ----------- | ---------------------------------- |
| 0        | ✅ 使用                   | ❌                 | 1           | `frames[:, 0]`                     |
| 1        | ✅ 使用                   | ❌                 | 1           | `frames[:, 1]`                     |
| 2        | ✅ 80%                    | ✅ 20%             | 0.8         | `0.8 * frames[:, 2] + 0.2 * x_gen` |
| 3        | ✅ 50%                    | ✅ 50%             | 0.5         | `0.5 * frames[:, 3] + 0.5 * x_gen` |
| 4        | ✅ 20%                    | ✅ 80%             | 0.2         | `0.2 * frames[:, 4] + 0.8 * x_gen` |
| 5        | ❌                        | ✅                 | 0           | `x_gen`                            |
| 6        | ❌                        | ✅                 | 0           | `x_gen`                            |
| 7        | ❌                        | ✅                 | 0           | `x_gen`                            |

可以看出：

- **前 `input_length`（5 帧）优先使用真实数据**。
- **训练中后期，模型更依赖自身预测的帧**，提升泛化能力。

---

## **2.3 总结**

1. **前 `input_length` 帧使用真实数据**，确保模型在训练初期获得足够的监督信号。
2. **`mask_true` 逐步减少真实数据的使用，过渡到预测数据**，提升模型的自回归能力。
3. **RSS 采样（`reverse_scheduled_sampling`）策略** 使得 **训练后期更早使用预测数据**，增强模型适应性。
4. **避免 "曝光偏差"（Exposure Bias）**，即训练时总是使用真实数据，导致测试时模型无法适应自身预测的误差。
# 3.模型
## 3.1 PredRNN

#### **1. 代码结构**

- **模型类 `RNN`**
    - 负责构建 **时空LSTM层**，处理输入帧并生成预测帧。
    - 计算损失，包括 **均方误差 (MSE Loss)** 和 **解耦损失**。

---

#### **2. 主要流程**

1. **初始化模型**
   
    - 解析 `configs` 配置，确定 **图像尺寸、通道数、LSTM层数、隐藏单元数量** 等。
    - 构建多个 **SpatioTemporalLSTMCell** 组成的 LSTM 层（使用 `ModuleList`）。
    - 定义 **1×1卷积层 (`conv_last`)**，用于生成最终的预测帧。
    - 定义 **适配器卷积 (`adapter`)**，用于特征归一化。
2. **前向传播 (`forward`)**
   
    1. **数据预处理**
       
        - 交换 `frames_tensor` 维度，使其适应网络输入格式 `[batch, length, height, width, channel] → [batch, length, channel, height, width]`。
        - 生成 **mask_true**，用于 **真实帧和预测帧的混合**，控制训练中使用多少真实数据。
    2. **初始化隐藏状态**
       
        - `h_t, c_t, memory` 初始化为 0（批量大小、隐藏单元、图像宽高）。
    3. **时间步循环（逐帧预测）**
       
        - **第一帧**：直接使用真实帧作为输入。
        - **后续帧**
            - 若使用 `reverse_scheduled_sampling`，根据 `mask_true` 混合真实帧与预测帧。
            - **层间传播**
                - 第一层 LSTM 直接处理输入数据，更新 `h_t[0], c_t[0], memory`。
                - 后续层 **接收前一层的隐藏状态作为输入**，逐层更新 `h_t[i], c_t[i]`。
            - 计算 **解耦损失**（衡量 `delta_c` 和 `delta_m` 的相似性）。
        - 最终 **通过 `conv_last` 生成预测帧 `x_gen`**，存入 `next_frames`。
    4. **计算损失**
       
        - **MSE Loss**：衡量预测帧和真实帧的误差。
        - **解耦损失**：防止信息冗余，提高预测的独立性。
        - 最终损失 `loss = MSE + β × 解耦损失`。
3. **可视化（可选）**
   
    - 通过 `tsne.visualization()` 生成 `delta_c` 和 `delta_m` 的可视化图像。

## 3.2 SpatioTemporalLSTMCell

#### **1. 目标**

该单元是 **时空LSTM（Spatio-Temporal LSTM, ST-LSTM）** 的核心计算单元，专用于 **视频帧预测、时空建模等任务**。  
它 **扩展** 了标准 LSTM，在 **细胞状态** (Cell State, $c_t$) 之外 **引入记忆单元** (Memory Cell, $m_t$)，以更好地处理时空信息。

---

#### **2. 计算流程**

假设输入 $x_t$ 形状为$(B, C, H, W)$，其中：

- $B$ 是 **批量大小**，
- $C$ 是 **通道数**，
- $H,W$是 **特征图的高和宽**。

模型执行如下计算：

1. **输入变换**
   
    - 计算 输入 $x_t$、隐藏状态 $h_t$、记忆状态 $m_t$的特征表示：
    $$X=convx(xt)(7N) \quad H = \text{conv}_h(h_t) \quad (4N)M = \text{conv}_m(m_t) \quad (3N)$$
    其中 `N` 是隐藏单元的通道数。
    
2. **门控机制**
   
    - ST-LSTM **定义了两个输入门、两个遗忘门、两个更新门**：
    - **细胞状态 $c_t$的更新**
     $$\begin{array}{l}
        i_t = \sigma(X_i + H_i) \quad \text{(输入门)} \\
        f_t = \sigma(X_f + H_f + b) \quad \text{(遗忘门)} \\
        g_t = \tanh(X_g + H_g) \quad \text{(候选状态)} \\
        \Delta c = i_t \odot g_t \\
        c_{t+1} = f_t \odot c_t + \Delta c
        \end{array}$$
      
   - **记忆状态 $m_t$的更新**
$$
\begin{array}{l}
i_t' = \sigma(X_i' + M_i) \\
f_t' = \sigma(X_f' + M_f + b) \\
g_t' = \tanh(X_g' + M_g) \\
\Delta m = i_t' \odot g_t' \\
m_{t+1} = f_t' \odot m_t + \Delta m
\end{array}
$$

3. **输出门和隐藏状态更新**
   - 计算 $c_t+1$ 和 $m_t+1$之后，拼接成 `mem`： $mem = \text{Concat}(c_{t+1}, m_{t+1})$
    - 计算输出门： $o_t = \sigma(X_o + H_o + \text{conv}_o(mem))$
    - 最终计算新的隐藏状态： $h_{t+1} = o_t \odot \tanh(\text{conv}_\text{last}(mem))$



## 3.3 PredRNN_v2

这段代码是 **PredRNN** 训练过程中用于 **scheduled sampling（调度采样）** 的关键部分，它决定了模型在训练时是使用真实数据还是自己的预测结果来生成下一帧。

## **📌 代码作用**

```python
# 遍历时间步，逐步生成预测帧，total_length: 总时间步数（输入帧 + 预测帧）
for t in range(self.configs.total_length - 1):

    # 反向调度采样
    if self.configs.reverse_scheduled_sampling == 1:
        if t == 0:
            net = frames[:, t]  # 第一帧直接使用真实帧
        else:
            net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
    else:
        # input_length: 仅输入的真实帧数量（训练时可见的帧）。
        if t < self.configs.input_length:
            net = frames[:, t]  # 在输入长度内使用真实帧
        else:
            print("mask_true shape:", mask_true.shape)
            print("frames shape:", frames.shape)
            print("x_gen shape:", x_gen.shape)
            net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                  (1 - mask_true[:, t - self.configs.input_length]) * x_gen
```

**这部分代码实现了：**

1. **对于前 `input_length` 帧（真实数据输入部分）**，始终使用真实数据 `frames[:, t]` 作为输入。
2. **对于 `input_length` 之后的帧（需要预测的部分）**：
   - 以 `mask_true` 为权重，在 **真实帧 `frames[:, t]`** 和 **预测帧 `x_gen`** 之间插值：
     - `mask_true` 近似 1 时，更多使用真实帧 `frames[:, t]`
     - `mask_true` 近似 0 时，更多使用模型自己预测的 `x_gen`
   - 这样可以 **逐步过渡到只使用模型预测结果**，避免训练时过度依赖真实数据，提升泛化能力。

------

## **📌 你的任务是否需要这个步骤？**

**你的任务**：对 **海浪的三个参数**（比如 Hm0、Tm02、SST）进行预测，不是视频预测，但本质上仍然是一个 **时序预测任务**。

### **情况 1️⃣：你的任务是序列到序列预测**

如果你的模型需要：

- **输入一段历史数据（input_length 帧）**
- **然后预测未来的波浪参数（total_length - input_length 帧）**

那 **这部分代码是需要的**，因为：

- `input_length` 之前，你希望模型用真实数据作为输入
- `input_length` 之后，模型只能依赖自己的预测，逐步生成未来的波浪数据
- 通过 `mask_true`，模型可以**从使用真实数据过渡到使用自己的预测**，提高稳定性

------

### **情况 2️⃣：你的任务是端到端直接预测未来帧**

如果你：

- 只输入一组波浪数据
- 直接输出整个未来时间段的预测结果（没有逐步生成）
- 你的模型不是 RNN 结构，而是 Transformer、CNN 之类的

那么**不需要这个步骤**，可以直接输入 `frames`，让模型输出完整预测值，而不用 `mask_true` 进行采样。

------

### **📌 总结**

✅ **如果你的模型是 PredRNN 结构，并且需要逐步预测波浪未来的变化趋势（比如 Hm0、Tm02、SST 的时序预测），那这部分代码是必要的。**
 ❌ **如果你的任务是一次性预测多个未来时间步，不是逐步生成，那可以去掉 `mask_true`，直接让模型学习 `input -> output` 的映射。**

如果你不确定，可以试着打印 `mask_true` 看看它的值，以及去掉这部分代码后模型是否还能稳定训练。

# 4.评估

## 4.1 RMSE

遇到一个bug，两个向量true和pred(160, N)在计算rmse = np.sqrt(mean_squared_error(true_flat, pred_flat))计算出来的rmse是一个标量

这是因为 `mean_squared_error` 函数默认会对整个输入数组计算一个单一的均方误差值，然后使用 `np.sqrt` 对这个单一的均方误差值求平方根，所以最终得到的 `RMSE` 也只有一个值。 

### 原因分析

 `mean_squared_error` 函数在计算时，会把输入的两个数组 `true_flat` 和 `pred_flat` 视为整体，计算它们对应元素之间的平方误差，然后求平均值。在你的代码里，`true_flat` 和 `pred_flat` 的形状是 `(160, 16384)`，`mean_squared_error` 会计算这两个数组所有元素的均方误差，得到一个单一的数值，再经过 `np.sqrt` 运算后，结果仍然是一个单一的数值。

 ### 解决方案 

如果希望得到每个样本（共 160 个样本）的 `RMSE`，可以对每个样本单独计算均方误差，再求平方根。以下是修改后的代码示例：

```python
import numpy as np 
from sklearn.metrics 
import mean_squared_error 
# 假设 true_flat 和 pred_flat 已经定义，形状为 (160, 16384) 
num_samples = true_flat.shape[0] 
rmse_values = [] 
for i in range(num_samples):    
    sample_true = true_flat[i]    
    sample_pred = pred_flat[i]    
    mse = mean_squared_error(sample_true, sample_pred)    
    rmse = np.sqrt(mse)    
    rmse_values.append(rmse) 
rmse_values = np.array(rmse_values) 
print("每个样本的 RMSE 形状:", rmse_values.shape)
print("每个样本的 RMSE:", rmse_values) 
```

### 代码解释 

1. **遍历每个样本**：通过 `for` 循环遍历 `true_flat` 和 `pred_flat` 中的每个样本。 
1.  **计算每个样本的均方误差**：在每次循环中，取出当前样本的真实值和预测值，使用 `mean_squared_error` 函数计算均方误差。 
1. **计算每个样本的 RMSE**：对每个样本的均方误差求平方根，得到该样本的 `RMSE`。 
1. **存储结果**：将每个样本的 `RMSE` 存储在 `rmse_values` 列表中。 
1. **转换为 `NumPy` 数组**：最后将 `rmse_values` 列表转换为 `NumPy` 数组，方便后续处理和分析。 通过这种方式，你就可以得到每个样本的 `RMSE`，结果的形状应该是 `(160,)`。 

# 5.流程图

![image-20250328001948568](https://picbed-1313037164.cos.ap-nanjing.myqcloud.com/image-20250328001948568.png)