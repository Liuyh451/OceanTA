import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
file1_path = './data_4080/test_true.npy'
file2_path = './data_4080/test_pred.npy'
def plot_mean_comparison(file1_path, file2_path):
    """
    读取两个 .npy 文件的数据，对第二维度的数据进行均值计算，
    然后将计算后的均值折线图绘制在同一张图上进行对比。

    参数:
    file1_path (str): 第一个 .npy 文件的路径。
    file2_path (str): 第二个 .npy 文件的路径。
    """
    # 加载 .npy 文件
    data1 = np.load(file1_path)
    data2 = np.load(file2_path)

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

# 示例用法
import numpy as np
import matplotlib.pyplot as plt

def plot_annual_mean_comparison(file1_path, file2_path, num_days_per_year):
    """
    读取两个 .npy 文件的数据，按年份提取1月份的数据，每年的数据按指定的天数进行分组，
    计算每年数据的均值，然后在同一张图上进行对比绘制折线图。

    参数:
    file1_path (str): 第一个 .npy 文件的路径。
    file2_path (str): 第二个 .npy 文件的路径。
    num_days_per_year (int): 每年1月份数据的天数。
    """
    # 加载 .npy 文件
    data1 = np.load(file1_path)
    data2 = np.load(file2_path)

    # 压缩数据到 (N, D) 形状
    data1 = data1.squeeze()
    data2 = data2.squeeze()
    data1 = np.pad(data1, (0, 2), mode='constant', constant_values=0)
    data2 = np.pad(data2, (0, 2), mode='constant', constant_values=0)
    num_data_points1 = data1.shape[0]
    num_data_points2 = data2.shape[0]
    print(num_data_points1, num_data_points2)

    if num_data_points1 % num_days_per_year != 0 or num_data_points2 % num_days_per_year != 0:
        raise ValueError("数据点总数不能被每年的天数整除，请检查数据。")

    num_years1 = num_data_points1 // num_days_per_year
    num_years2 = num_data_points2 // num_days_per_year

    if num_years1 != num_years2:
        raise ValueError("两个数据的年份数量不同，请检查数据。")

    # 创建年份数组
    years = np.arange(1992, 1992 + num_years1)

    # 计算每年数据的均值
    mean_values1 = np.zeros(num_years1)
    mean_values2 = np.zeros(num_years2)

    for i in range(num_years1):
        start_index = i * num_days_per_year
        end_index = (i + 1) * num_days_per_year
        mean_values1[i] = np.mean(data1[start_index:end_index])
        mean_values2[i] = np.mean(data2[start_index:end_index])

    # 绘制折线图
    plt.figure(figsize=(12, 6))
    plt.plot(years, mean_values1, marker='o', linestyle='-', color='b', label='File 1 January Mean')
    plt.plot(years, mean_values2, marker='x', linestyle='--', color='r', label='File 2 January Mean')

    plt.xlabel('Year')
    plt.ylabel('Mean Value')
    plt.title('Comparison of January Mean Values from Two Files')
    plt.grid(True)
    plt.legend()
    plt.show()

# 示例用法
num_days_per_year = 31  # 每年1月份的天数，根据实际情况调整
plot_annual_mean_comparison(file1_path, file2_path, num_days_per_year)
plot_mean_comparison(file1_path, file2_path)


