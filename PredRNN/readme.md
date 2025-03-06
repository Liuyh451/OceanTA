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