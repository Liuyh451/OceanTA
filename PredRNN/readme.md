**复现代码来自https://github.com/thuml/predrnn-pytorch**

# 1.数据处理问题
## 1.1 对帧(Frame)展平
在对 **Moving MNIST** 进行预测时，使用 RNN 变体（例如 ConvLSTM、PredRNN、PredRNN++、MIM 等）来建模时序依赖关系，`self.frame_channel` 计算方式如下：

```python
self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
```

这是为了 **将输入帧（图像）拆分成多个小块（patch），并将它们视作时间步中的特征输入**，从而适应 RNN 变体的计算模式。

---

### **为什么对帧计算通道？**

在 RNN 变体（如 ConvLSTM）中，输入的 **每个时间步** 都需要一个张量，而 Moving MNIST 的原始输入是一组 2D 灰度图像（形状通常是 `(batch_size, time_steps, height, width, channels)`，其中 `channels=1` ）。如果直接输入 2D 图像，网络可能无法很好地学习局部运动模式，因此我们常使用 **patch-based** 方式对图像进行预处理，使模型能够更好地捕捉局部运动特征。
### **举例说明**

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
# 2.训练
## 2.1 RSS过程
这一段代码用于 **遍历时间步，逐步生成预测帧**，并根据是否使用 **反向调度采样（Reverse Scheduled Sampling, RSS）** 选择当前时间步的输入帧。

---

###  2.1.1 **代码解析**

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

### 2.1.2 **核心概念**

#### 1. total_length 和 input_length

- **`total_length`**: 总时间步数（输入帧 + 预测帧），即整个序列的长度。
- **`input_length`**: 仅输入的真实帧数量（训练时可见的帧）。

对于 **预测任务**（如视频帧预测），通常会：

- **前 `input_length` 帧使用真实数据（ground truth）** 作为输入。
- **后面的时间步需要模型自行预测**，并使用前一时间步的预测帧作为输入（自回归）。

---

#### 2. reverse_scheduled_sampling（RSS，反向调度采样）

RSS 是 **用于提升模型泛化能力的一种数据采样策略**，可以 **控制预测时使用真实帧（ground truth）还是预测帧（x_gen）** 作为输入：

- **普通采样（非 RSS）**：训练时，前 `input_length` 帧使用真实数据，后面逐步过渡到预测数据。
- **RSS 采样**：在训练后期，让模型逐渐减少对真实帧的依赖，更早地开始使用预测数据作为输入，使其适应真实推理场景。

---

### 2.1.3 代码逻辑

### **1. 反向调度采样 (`reverse_scheduled_sampling == 1`)**

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

### **2. 非反向调度采样 (`reverse_scheduled_sampling == 0`)**

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

## **示例**

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

## **总结**

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