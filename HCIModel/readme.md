# 1.数据

## 1.1 文件结构

├── .gitignore
├── .idea
│   ├── .gitignore
│   ├── HCIModel.iml
│   ├── inspectionProfiles
│   │   └── profiles_settings.xml
│   ├── misc.xml
│   ├── modules.xml
│   ├── vcs.xml
│   └── workspace.xml
├── Ablation
│   ├── __pycache__
│   │   └── model.cpython-38.pyc
│   ├── checkpoint.chk
│   ├── data
│   │   ├── Multi
│   │   │   ├── predictions.npy
│   │   │   ├── predictions_inverse.npy
│   │   │   ├── predictions_inverse_final.npy
│   │   │   └── targets.npy
│   │   ├── loss
│   │   │   ├── train_loss_list.npy
│   │   │   ├── train_loss_list_10.npy
│   │   │   ├── train_loss_list_11.npy
│   │   │   ├── train_loss_list_2.npy
│   │   │   ├── train_loss_list_3.npy
│   │   │   ├── train_loss_list_4.npy
│   │   │   ├── train_loss_list_5.npy
│   │   │   ├── train_loss_list_6.npy
│   │   │   ├── train_loss_list_7.npy
│   │   │   ├── train_loss_list_8.npy
│   │   │   ├── train_loss_list_9.npy
│   │   │   ├── val_loss_list.npy
│   │   │   ├── val_loss_list_10.npy
│   │   │   ├── val_loss_list_11.npy
│   │   │   └── val_loss_list_9.npy
│   │   └── single
│   │       ├── predictions.npy
│   │       ├── predictions_inverse.npy
│   │       ├── predictions_inverse_final.npy
│   │       └── targets.npy
│   ├── eval_single.ipynb
│   ├── evaluate.ipynb
│   ├── main.py
│   ├── main_single.py
│   ├── model.py
│   └── net
│       ├── checkpoint1.chk
│       ├── checkpoint2.chk
│       ├── checkpoint3.chk
│       ├── checkpoint4.chk
│       ├── checkpoint5.chk
│       ├── checkpoint6.chk
│       ├── checkpoint7.chk
│       └── checkpoint8.chk
├── Baseline
│   ├── Baseline.py
│   └── main.py
├── Compare
│   ├── __pycache__
│   │   └── model_ori.cpython-38.pyc
│   ├── checkpoint_proposed.chk
│   ├── data
│   │   ├── MultiOutput
│   │   │   ├── predictions.npy
│   │   │   ├── predictions_inverse.npy
│   │   │   ├── predictions_inverse_final.npy
│   │   │   └── targets.npy
│   │   └── loss
│   │       └── proposed
│   │           ├── train_loss_list.npy
│   │           ├── train_loss_list_2.npy
│   │           ├── train_loss_list_3.npy
│   │           ├── val_loss_list.npy
│   │           ├── val_loss_list_2.npy
│   │           └── val_loss_list_3.npy
│   ├── eval_m.ipynb
│   ├── main_multi.py
│   └── model_ori.py
├── MVMD.ipynb
├── Residual
│   ├── Util.py
│   ├── __pycache__
│   │   └── Util.cpython-38.pyc
│   ├── checkpoint.chk
│   ├── data
│   │   ├── combined_predictions.npy
│   │   ├── loss
│   │   │   ├── train_loss_list.npy
│   │   │   └── val_loss_list.npy
│   │   ├── pre_original_data.npy
│   │   ├── predictions.npy
│   │   ├── predictions_res_inverse.npy
│   │   └── targets.npy
│   └── main_res.py
├── Util.py
├── __pycache__
│   └── Util.cpython-38.pyc
├── data
│   ├── dt1
│   │   ├── loss
│   │   │   ├── train_loss_list.npy
│   │   │   ├── train_loss_list_2.npy
│   │   │   ├── train_loss_list_3.npy
│   │   │   ├── train_loss_list_4.npy
│   │   │   ├── train_loss_list_5.npy
│   │   │   ├── train_loss_list_6.npy
│   │   │   ├── train_loss_list_7.npy
│   │   │   ├── val_loss_list.npy
│   │   │   ├── val_loss_list_2.npy
│   │   │   ├── val_loss_list_3.npy
│   │   │   ├── val_loss_list_4.npy
│   │   │   ├── val_loss_list_5.npy
│   │   │   ├── val_loss_list_6.npy
│   │   │   └── val_loss_list_7.npy
│   │   ├── mean.npy
│   │   ├── mode1
│   │   │   ├── predictions.npy
│   │   │   └── targets.npy
│   │   ├── mode2
│   │   │   ├── predictions.npy
│   │   │   └── targets.npy
│   │   ├── mode3
│   │   │   ├── predictions.npy
│   │   │   └── targets.npy
│   │   ├── mode4
│   │   │   ├── predictions.npy
│   │   │   └── targets.npy
│   │   ├── mode5
│   │   │   ├── predictions.npy
│   │   │   └── targets.npy
│   │   ├── mode6
│   │   │   ├── predictions.npy
│   │   │   └── targets.npy
│   │   ├── mode7
│   │   │   ├── predictions.npy
│   │   │   └── targets.npy
│   │   ├── mode8
│   │   │   ├── predictions.npy
│   │   │   └── targets.npy
│   │   ├── pre_original_data.npy
│   │   ├── predictions_inverse.npy
│   │   └── std.npy
│   └── dt2
│       ├── mode1
│       ├── mode2
│       ├── mode3
│       ├── mode4
│       ├── mode5
│       └── mode6
├── dataProcess.ipynb
├── eval.ipynb
├── main.ipynb
├── main.py
├── model.ipynb
├── net
│   ├── dt1
│   │   ├── checkpoint1.chk
│   │   ├── checkpoint2.chk
│   │   ├── checkpoint3.chk
│   │   ├── checkpoint4.chk
│   │   ├── checkpoint5.chk
│   │   ├── checkpoint6.chk
│   │   ├── checkpoint7.chk
│   │   └── checkpoint8.chk
│   └── dt2
├── readme.md
└── tools.py

Ablation是消融实验，为的是比较两种网络（relu）；Compare比较的是提出的网络3.1和3.2中提到的两种方法；

位于主目录中的是所提出的网络的整体结构，用的是3.1方法。每个文件夹下的data用于存loss值和预测值，部分文件夹下存有训练好的模型net。

## 1.2 数据处理

### 1.滑动窗口
[[T技术#4.3 确定滞后值Lag|确定步长]]
滑动窗口大小，dataset1为17，dataset2为21

数据在滑动之前的形状为(3000，32),N=3000,C=32，经过滑动之后形状变为（3000，17，32），17为时间步数也叫窗口大小。

---

**滑动窗口技术**

滑动窗口是一种常用的技术，尤其在处理时间序列数据、信号处理和数据分析中。它的基本思想是在数据集中使用一个固定大小的窗口，通过这个窗口逐步移动，以提取特征或进行计算。
假设我们有一个时间序列数据：$\text{数据} = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]$
**滑动窗口参数**：
- **窗口大小**: 3
- **滑动步长**: 1

**滑动窗口操作**：
1. 第一个窗口：$[1, 2, 3]$
2. 移动窗口（步长为1）：
   - 第二个窗口：$[2, 3, 4]$
   - 第三个窗口：$[3, 4, 5]$
   （注意：此示例仅展示了前三个窗口，实际操作中窗口会继续以步长1向右移动，直到覆盖整个数据集。）

---

![image.png](https://picbed-1313037164.cos.ap-nanjing.myqcloud.com/20241027004040.png)

### 2.标准化
用标准化函数`standardize`进行标准化，同时记录下每个模态的mean和std，保存为npy（600，8），方便后面进行反标准化`inverse_standardize`
注意：对各模态`prediction`相加后的结果用`dataset`的mean和std进行反标准化是不对的，这样的误差很大。
正确的应该是对各模态`prediction`的各自对应的mean和std进行反标准化，再相加后和`dataset`进行对比

## 1.3 数据加载

通常需要将数据划分为X，y。X为用于训练的数据（也可能用于测试），y是X对应的标签。在这里X就是波的四个参数，y就是wh（波高），所以需要划分出6份数据。X，y分别的train，val，test。

### 1.DataLoader

在 `train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)` 中，`DataLoader` 的作用是将你的训练数据（`self.train_data`）按批次加载，并支持随机打乱（如果 `shuffle=True`）。

如果你的 `self.train_data` 是由 `x_train`（输入特征）和 `y_train`（标签）组成的，通常有以下几种方式来构建 `self.train_data`，从而确保 `DataLoader` 能正确区分输入和标签：

1. **使用 `TensorDataset`** 如果你的 `x_train` 和 `y_train` 都是 `torch.Tensor`，可以将它们组合成一个 `TensorDataset`，这样 `DataLoader` 可以轻松区分输入和标签。

```python
from torch.utils.data import TensorDataset, DataLoader  
    ​  
    # 假设 x_train 和 y_train 是张量  
    self.train_data = TensorDataset(x_train, y_train)  
    train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
```

在这种情况下，每次从 `train_loader` 中获取的数据是一个元组 `(batch_x, batch_y)`，分别表示输入数据和标签。
    
2. **自定义数据集类** 如果你有更复杂的数据结构，可以通过继承 `Dataset` 类，创建自定义数据集，并在其中定义如何返回输入和标签。

```python
from torch.utils.data import Dataset, DataLoader  
    ​  
    class CustomDataset(Dataset):  
        def __init__(self, x_data, y_data):  
            self.x_data = x_data  
            self.y_data = y_data  
    ​  
        def __len__(self):  
            return len(self.x_data)  
    ​  
        def __getitem__(self, idx):  
            return self.x_data[idx], self.y_data[idx]  
    ​  
    # 初始化自定义数据集  
    self.train_data = CustomDataset(x_train, y_train)  
    train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
```

在上述两种情况下，`DataLoader` 都能够正确区分输入和标签，并在每个批次中返回对应的 `(input, label)` 对。

因此，如果你确保 `self.train_data` 是通过 `TensorDataset` 或自定义 `Dataset` 结构组合的，那么 `DataLoader` 会自动区分输入特征（`x_train`）和标签（`y_train`）。

3.在训练时使用

train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)  
for batch_idx, (inputs, targets) in enumerate(train_loader):  
    # 每10个批次打印一次  
    if (batch_idx + 1) % 10 == 0:  
        print(f"第 {batch_idx + 1} 批次已训练完成")  
        inputs = inputs.to(self.device).float()  # 获取输入数据并转移到 GPU  
        targets = targets.to(self.device).float()  # 获取目标数据并转移到 GPU

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
![image.png](https://picbed-1313037164.cos.ap-nanjing.myqcloud.com/20241012233718.png)
![image.png](https://picbed-1313037164.cos.ap-nanjing.myqcloud.com/20241012233727.png)


## 2.2 分析模块
$$\begin{aligned}
& A = \tanh(x_i \ast w_1 + b_1), \\
& B = \sigma(x_i \ast w_2 + b_2), \\
& C = \tanh(x_i \ast w_3 + b_3), \\
& D = \sigma(x_i \ast w_4 + b_4), \\
& E = A \otimes B + C \otimes D, \\
& \hat{y}_n \sim (E),
\end{aligned}$$

PyTorch 的卷积操作要求输入张量的形状为 `(N, C, L)`，其中 `N` 是批量大小，`C` 是通道数，`L` 是序列长度。如果你的输入形状是 `(N, L, input_channels)`，你需要对其进行转置，以满足卷积层的输入要求。

**时间步长的处理**：由于使用的是一维卷积（`Conv1d`），所以卷积会在时间步长维度（长度 `17`）上进行。这是处理时间序列数据的标准方法。![2daefe6492f957a5c7821b4d25f829a4_720.png](https://picbed-1313037164.cos.ap-nanjing.myqcloud.com/2daefe6492f957a5c7821b4d25f829a4_720.png)
在这个例子中，卷积层的每个卷积核会在 **时间步长维度**（即 `5` 这个维度）上滑动，将三步长的局部信息聚合成一个特征输出。通过以下计算，卷积核对每个时间步的邻域信息进行加权和，输出一个新的特征值。

- 当卷积核滑动到时间步 `1, 2, 3` 时，输出一个值
- 当卷积核滑动到时间步 `2, 3, 4` 时，输出一个值
- 当卷积核滑动到时间步 `3, 4, 5` 时，输出一个值

经过卷积操作后，输入张量的时间维度会保持不变（因为 `padding=1`），但每个时间步的数据被卷积聚合成了新的特征。因此，如果我们执行 `output = conv(x)`，得到的输出张量的形状为 `(1, 2, 5)`，保持时间步长 `5`，但数据被卷积操作更新了。

通过这种方式，**时间步的卷积**能够提取相邻时间步的局部特征，同时保持整个时间序列的步长不变。

**输出形状**：卷积操作的输出形状为 `(N, output_channels, L)`，在这里 `L` 仍然是 `17`，确保可以在后续的模型中正确地使用这些特征图
![image.png](https://picbed-1313037164.cos.ap-nanjing.myqcloud.com/20241027103615.png)
输入数据的形状为 `(40, 32, 17)`，其中：
- `40` 是批次大小
- `32` 是通道数
- `17` 是时间步长

以下是每个步骤中 `tensor` 的形状变化：

1. **`BasicUnit`中的卷积操作**：
    - 输入形状：`(40, 32, 17)`
    - 在 `BasicUnit` 类中，定义了四个卷积层 (`conv1`, `conv2`, `conv3`, `conv4`)，每个卷积层的卷积核大小为 `3`，并使用 `padding=1` 来保持时间步长的一致性。
    - 卷积操作不会改变通道数（仍然为 `32`）和时间步长（仍然为 `17`），因此经过每个卷积后的 `A`、`B`、`C` 和 `D` 的形状均为 `(40, 32, 17)`。
    - 在 `forward` 中计算 `A * B + C * D`，得到 `E`，其形状为 `(40, 32, 17)`。

2. **`BrainAnalysisModule` 中的第一个 `BasicUnit`**：
    - `output1 = self.basic_unit1(x)` 生成的 `output1` 形状为 `(40, 32, 17)`，即经过第一个基本单元后的输出。
  
3. **`BrainAnalysisModule` 中的第二个 `BasicUnit`**：
    - `output2 = self.basic_unit2(output1)`，`output1` 作为第二个基本单元的输入。
    - 第二个基本单元处理后的 `output2` 形状仍然为 `(40, 32, 17)`。

4. **消除时间步长的操作**：
    - 使用 `view` 函数将 `output2` 重新调整为二维形状 `(40, 32 * 17)`，即 `(批次大小, 通道数 * 时间步长)`。
    - `output2_flat = output2.view(output2.size(0), -1)`，此时 `output2_flat` 的形状为 `(40, 544)`。
    - 通过展平操作，将每个样本的时间步长信息编码为一个长向量，适配全连接层的输入要求。

5. **全连接层**：
    - `final_output = self.fc(output2_flat)`，通过全连接层将维度从 `(40, 544)` 压缩到 `(40, 1)`。
    - 输出的形状为 `(40, 1)`，适合作为一个标量输出。
## 2.3 预测模块

在上个模块得到的结果为（40，8，17），但是预测模块结果要的是（40，）。有两个问题

### 2.3.1 通道维度

公式写的是8个通道相加，没说相加后求平均

### 2.3.2 时间维度

通过全连接层将时间维度消去

# 3.实现
## 3.1 Plan A 单一输出
2024.10.3
用8个模态的数据(8，N，4)的数据作为训练集，（8，N，1）的数据作为标签，分别训练8种模态的8个模型，然后再在这个预测模块相加。目前来说是这样的
计划给dataset跑一遍
计划使用动态学习率
计划使用原来的数据作为label，目前使用的是分解后的数据作为的label

| Dataset  | Method              | RMSE   | MAE    | SSE     | MAPE(%) | TIC    |
| -------- | ------------------- | ------ | ------ | ------- | ------- | ------ |
| Dataset1 | Relu based method   | 0.1498 | 0.1156 | 13.4710 | 4.8301  | 0.0562 |
|          | The proposed method | 0.194  | 0.131  | 22.717  | 5.026   | 0.076  |

## 3.2 Plan B 多输出
2024.10.27
对每个模态的特征独立预测，将标签设置为 `(batch_size, 8)`，其中每个值对应一个模态的整体特征
**准备数据**：将8个模态的4列数据合并成一个32维的向量，作为每个样本的输入特征
**设置标签**：对于每个输入样本，标签可以是一个包含8个目标值的向量，代表每个模态对应的真实值
训练集 X 形状: (1906, 32, 17)，训练集 y 形状: torch.Size($[1906, 8]$)
验证集 X 形状: (477, 32, 17)，验证集 y 形状: torch.Size($[477, 8]$)
测试集 X 形状: (600, 32, 17)，测试集 y 形状: torch.Size($[600, 8]$)

| Dataset  | Method              | RMSE   | MAE    | SSE     | MAPE(%) | TIC    |
| -------- | ------------------- | ------ | ------ | ------- | ------- | ------ |
| Dataset1 | Relu based method   | 0.1546 | 0.1167 | 14.3537 | 4.7710  | 0.0581 |
|          | The proposed method | 0.2909 | 0.1910 | 50.7998 | 7.2714  | 0.1169 |
对比可知多输出的效果不如单输出的。目前的问题是为什么提出的方法不如Relu based method？

下面的实验结果均是单输出的proposed method

# 4.实验结果

## 4.1 Dataset 1
### 4.1.1 预测值和真实值对比

![image.png](https://picbed-1313037164.cos.ap-nanjing.myqcloud.com/20241027134036.png)

## 4.2 MVMD分解效果
### 4.2.1 各模态分解结果
![image.png](https://picbed-1313037164.cos.ap-nanjing.myqcloud.com/20241026112757.png)
### 4.2.2 各模态皮尔森系数热力图
![image.png](https://picbed-1313037164.cos.ap-nanjing.myqcloud.com/20241026114835.png)
### 4.2.3 各模态的具体预测
![预测](https://picbed-1313037164.cos.ap-nanjing.myqcloud.com/20241012233429.png)

### 4.2.4 分解模态的贡献
![image.png](https://picbed-1313037164.cos.ap-nanjing.myqcloud.com/20241026162802.png)
# 5. 讨论
## 5.1. 分解残差的影响
### 5.1.1 对残差的预测
![image.png](https://picbed-1313037164.cos.ap-nanjing.myqcloud.com/20241026232630.png)
### 5.1.2 模态加残差对模型的影响

![image.png](https://picbed-1313037164.cos.ap-nanjing.myqcloud.com/20241027002943.png)

| Dataset  | Method                           | RMSE   | MAE    | SSE     | MAPE(%) | TIC    |
| -------- | -------------------------------- | ------ | ------ | ------- | ------- | ------ |
| Dataset1 | The proposed method              | 0.1945 | 0.1314 | 22.7172 | 5.0262  | 0.0764 |
|          | The proposed method plus residue | 0.1785 | 0.1219 | 19.1229 | 4.6506  | 0.0699 |
| Dataset2 |                                  |        |        |         |         |        |
### 5.1.3 门控单元的影响

采用了PlanB
![image.png](https://picbed-1313037164.cos.ap-nanjing.myqcloud.com/20241027135309.png)

| Dataset  | Method              | RMSE   | MAE    | SSE     | MAPE(%) | TIC    |
| -------- | ------------------- | ------ | ------ | ------- | ------- | ------ |
| Dataset1 | Relu based method   | 0.1546 | 0.1167 | 14.3537 | 4.7710  | 0.0581 |
|          | The proposed method | 0.3127 | 0.2081 | 58.7042 | 8.0200  | 0.1268 |
