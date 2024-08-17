import matplotlib.pyplot as plt


def plot_nrmse_vs_depth(depth, nrmse):
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


# 示例调用
depth = list(range(1, 25))
nrmse = [0.002023, 0.002823, 0.003603, 0.003094, 0.002901, 0.002428, 0.002329, 0.002305, 0.002601, 0.002781, 0.002910,
         0.003067, 0.002822, 0.002604, 0.002252, 0.002057, 0.001860, 0.001657, 0.001448, 0.001764, 0.001785,
         0.001446,
         0.001717,
         0.002045]
plot_nrmse_vs_depth(depth, nrmse)
