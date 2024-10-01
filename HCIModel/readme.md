# 1.数据
## 1.1 数据来源
## 1.2 数据处理
### 1.滑动窗口
滑动窗口大小，dataset1为17，dataset2为21

数据在滑动之前的形状为(3000*32),N=3000,C=32，经过滑动之后形状变为（3000\*17*\*32），17为时间步数也叫窗口大小。

1. - 滑动窗口是一种常用的技术，尤其在处理时间序列数据、信号处理和数据分析中。它的基本思想是在数据集中使用一个固定大小的窗口，通过这个窗口逐步移动，以提取特征或进行计算。

     以下是一个简单的滑动窗口示例：

     假设我们有一个时间序列数据：

     \[ \text{数据} = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] \]

     ### 滑动窗口参数
     - **窗口大小**: 3
     - **滑动步长**: 1

     ### 滑动窗口操作
     1. 第一个窗口：\[ [1, 2, 3] \]
     2. 移动窗口（步长为1）：
        - 第二个窗口：\[ [2, 3, 4] \]
        - 第三个窗口：\[ [3, 4, 5] \]


### 2.标准化

## 1.3 数据加载

通常需要将数据划分为X，y。X为用于训练的数据（也可能用于测试），y是X对应的标签。在这里X就是波的四个参数，y就是wh（波高），所以需要划分出6份数据。X，y分别的train，val，test。

### 1.DataLoader

在 `train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)` 中，`DataLoader` 的作用是将你的训练数据（`self.train_data`）按批次加载，并支持随机打乱（如果 `shuffle=True`）。

如果你的 `self.train_data` 是由 `x_train`（输入特征）和 `y_train`（标签）组成的，通常有以下几种方式来构建 `self.train_data`，从而确保 `DataLoader` 能正确区分输入和标签：

1. **使用 `TensorDataset`**
   如果你的 `x_train` 和 `y_train` 都是 `torch.Tensor`，可以将它们组合成一个 `TensorDataset`，这样 `DataLoader` 可以轻松区分输入和标签。

   ```python
   from torch.utils.data import TensorDataset, DataLoader
   
   # 假设 x_train 和 y_train 是张量
   self.train_data = TensorDataset(x_train, y_train)
   train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
   ```

   在这种情况下，每次从 `train_loader` 中获取的数据是一个元组 `(batch_x, batch_y)`，分别表示输入数据和标签。

2. **自定义数据集类**
   如果你有更复杂的数据结构，可以通过继承 `Dataset` 类，创建自定义数据集，并在其中定义如何返回输入和标签。

   ```python
   from torch.utils.data import Dataset, DataLoader
   
   class CustomDataset(Dataset):
       def __init__(self, x_data, y_data):
           self.x_data = x_data
           self.y_data = y_data
   
       def __len__(self):
           return len(self.x_data)
   
       def __getitem__(self, idx):
           return self.x_data[idx], self.y_data[idx]
   
   # 初始化自定义数据集
   self.train_data = CustomDataset(x_train, y_train)
   train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
   ```

在上述两种情况下，`DataLoader` 都能够正确区分输入和标签，并在每个批次中返回对应的 `(input, label)` 对。

因此，如果你确保 `self.train_data` 是通过 `TensorDataset` 或自定义 `Dataset` 结构组合的，那么 `DataLoader` 会自动区分输入特征（`x_train`）和标签（`y_train`）。

3.在训练时使用

```python
train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
for batch_idx, (inputs, targets) in enumerate(train_loader):
    # 每10个批次打印一次
    if (batch_idx + 1) % 10 == 0:
        print(f"第 {batch_idx + 1} 批次已训练完成")
        inputs = inputs.to(self.device).float()  # 获取输入数据并转移到 GPU
        targets = targets.to(self.device).float()  # 获取目标数据并转移到 GPU
```

这里它会自动按批次大小加载数据，且加载的数据为元组(inputs, targets)，target就是标签

### 2.问题

dataloader默认是从0索引进行加载数据的，最大索引为dataloader的长度。如果你的张量并不是从0开始的，就会出问题。

```python
X_train = X_data[:train_size].reset_index(drop=True)
X_val = X_data[train_size:train_size + val_size].reset_index(drop=True)
X_test = X_data[train_size + val_size:train_size + val_size + test_size].reset_index(drop=True)
```

通过这种方式划分的数据（X_train = X_data[:train_size]），在转为numpy时索引是继承的

> NumPy 数组没有像 Pandas DataFrame 那样的索引概念，因此不需要重置索引。NumPy 数组的切片操作会返回一个新的数组，这个新数组会自动继承原始数组的结构和索引。

所以加.reset_index(drop=True)可以让索引重置，前提是X_data要是datafram格式

# 2.模型

## 2.1 感知模块
使用MVMD技术分解
## 2.2 分析模块

PyTorch 的卷积操作要求输入张量的形状为 `(N, C, L)`，其中 `N` 是批量大小，`C` 是通道数，`L` 是序列长度。如果你的输入形状是 `(N, L, input_channels)`，你需要对其进行转置，以满足卷积层的输入要求。

**时间步长的处理**：由于使用的是一维卷积（`Conv1d`），所以卷积会在时间步长维度（长度 `17`）上进行。这是处理时间序列数据的标准方法。

**输出形状**：卷积操作的输出形状为 `(N, output_channels, L)`，在这里 `L` 仍然是 `17`，确保你可以在后续的模型中正确地使用这些特征图

## 2.3 预测模块

在上个模块得到的结果为（40，8，17），但是预测模块结果要的是（40，）。有两个问题

### 2.3.1 通道维度

公式写的是8个通道相加，没说相加后求平均

### 2.3.2 时间维度

暂时不知道怎么处理，目前是简单相加