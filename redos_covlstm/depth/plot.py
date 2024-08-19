import matplotlib.pyplot as plt
import numpy as np

file1_path = './data_4080/test_true.npy'
file2_path = './data_4080/test_pred.npy'
data1 = np.load(file1_path)
data2 = np.load(file2_path)
class Plot:
    def __init__(self,data1,data2):
        self.data1 = data1
        self.data2 = data2
    def plot_comparison(self):
        """
        读取两个 .npy 文件的数据，对第二维度的数据进行均值计算，
        然后将计算后的均值折线图绘制在同一张图上进行对比。

        参数:
        file1_path (str): 第一个 .npy 文件的路径。
        file2_path (str): 第二个 .npy 文件的路径。
        """
        # 加载 .npy 文件
        data1,data2 = self.mean_over_time()

        # 压缩数据到 (N, M) 形状
        data1 = data1.squeeze()
        data2 = data2.squeeze()

        # 计算第二维度的均值
        mean1 = np.mean(data1, axis=1)
        mean2 = np.mean(data2, axis=1)

        # 确保数据具有相同的横轴维度
        if mean1.shape[0] != mean2.shape[0]:
            raise ValueError("两个数据的第一维度长度不同，无法进行对比")

        x_axis = np.arange(mean1.shape[0])

        plt.figure(figsize=(12, 6))

        # 绘制第一个数据集的均值折线图
        plt.plot(x_axis, mean1, label='true', color='b')

        # 绘制第二个数据集的均值折线图
        plt.plot(x_axis, mean2, label='pred', color='r', linestyle='--')

        plt.legend()
        plt.xlabel('date')
        plt.ylabel('Mean Value')
        plt.title('Comparison of Mean Values')
        plt.grid(True)
        plt.show()

    def plot_nrmse_vs_depth(self,depth, nrmse):
        """
        绘制深度（Depth）与nRMSE之间关系的折线图

        参数:
            depth (list): 深度列表，长度为24。
            nrmse (list): 对应的nRMSE值列表，长度为24。
        """
        plt.figure(figsize=(10, 6))
        plt.plot(depth, nrmse, marker='o', linestyle='-', color='orange')
        plt.title('nRMSE vs Depth')
        plt.xlabel('Depth')
        plt.ylabel('nRMSE')
        # 设置纵轴范围
        plt.ylim(0.0005, 0.0045)
        plt.grid(True)
        plt.show()

    import numpy as np

    def mean_over_time(self):

        """
        对每个时间步在 (28, 52) 的维度上计算均值，得到一个二维数组 (28, 52)。

        参数:
        data (np.ndarray): 输入数据，形状为 (500, 28, 52)。

        返回:
        np.ndarray: 压缩后的二维数组，形状为 (28, 52)。
        """
        data1=np.mean(self.data1, axis=0)
        data2=np.mean(self.data2, axis=0)
        data1 = np.squeeze(data1)  # 去除单个维度，确保形状为 (28, 52)
        data2 = np.squeeze(data2)  # 去除单个维度，确保形状为 (28, 52)
        return data1, data2

    def plot_heatmap_data(self,data1,data2):
        """
        绘制地理空间数据的热图。data:(lon,lat)
        """
        plt.imshow(data1, cmap='viridis', aspect='auto')
        plt.colorbar(label='Value')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('test_true')
        plt.show()
        plt.imshow(data2, cmap='viridis', aspect='auto')
        plt.colorbar(label='Value')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('test_pred')
        plt.show()
plot=Plot(data1,data2)
data1,data2=plot.mean_over_time()

# 示例调用
depth = list(range(1, 25))
nrmse = [0.002023, 0.002823, 0.003603, 0.003094, 0.002901, 0.002428, 0.002329, 0.002305, 0.002601, 0.002781, 0.002910,
             0.003067, 0.002822, 0.002604, 0.002252, 0.002057, 0.001860, 0.001657, 0.001448, 0.001764, 0.001785,
             0.001446,
             0.001717,
             0.002045]
# polt.plot_nrmse_vs_depth(depth, nrmse)
