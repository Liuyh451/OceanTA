import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn.functional as F
import numpy as np
import torch
import os
import shutil


def scatter(x, colors, file_name, class_num):
    """绘制带类别颜色的二维散点图并保存为图片
    
    Args:
        x: 二维坐标数据，形状为(N, 2)的numpy数组
        colors: 每个数据点对应的类别标签数组
        file_name: 输出图片的文件名前缀(无需扩展名)
        class_num: 要显示的类别总数
    """
    f = plt.figure(figsize=(226/15, 212/15))
    ax = plt.subplot(aspect='equal')
    color_pen = ['black', 'r']  # 类别颜色映射
    my_legend = ['Delta_C', 'Delta_M']  # 图例标签

    # 生成去重的类别标签集合
    label_set = []
    label_set.append(colors[0])
    for i in range(1, len(colors)):
        if colors[i] not in label_set:
            label_set.append(colors[i])

    # 按类别绘制散点
    for i in range(class_num):
        ax.scatter(x[colors == label_set[i], 0], x[colors == label_set[i], 1], 
                   lw=0, s=70, c=color_pen[i], label=str(my_legend[i]))
    
    # 设置坐标轴和图例
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('tight')
    ax.legend(loc='upper right')
    
    # 保存可视化结果
    f.savefig(file_name + ".png", bbox_inches='tight')
    print(file_name + ' save finished')


def plot_TSNE(data, label, path, title, class_num):
    """使用t-SNE进行降维可视化
    
    Args:
        data: 高维特征数据，形状为(N, D)的numpy数组
        label: 对应数据的类别标签数组 
        path: 图片保存路径
        title: 可视化标题(用作文件名前缀)
        class_num: 要显示的类别数量
    """
    colors = label
    # t-SNE降维处理
    tsne_features = TSNE(random_state=20190129).fit_transform(data)
    # 调用散点图绘制函数
    scatter(tsne_features, colors, os.path.join(path, title), class_num)


def visualization(length, layers, c, m, path, elements=10):
    """记忆细胞解耦可视化(生成t-SNE图)
    
    通过分析c(上下文记忆)和m(长期记忆)的top元素关系，可视化记忆组件的解耦情况
    
    Args:
        length: 序列总长度
        layers: 堆叠的预测层数
        c: 上下文记忆张量，形状应为(L*T, B, H, D)
        m: 长期记忆张量，形状应与c保持一致
        path: 结果保存路径
        elements: 选择用于分析的前k个重要元素，默认为10
    
    Note:
        会先清空并重新创建保存路径目录
        为每个时间步和层生成多个tsne可视化图片
    """
    # 初始化输出目录
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    
    # 遍历每个时间步和层
    for t in range(length - 1):
        for i in range(layers):
            data = []
            label = []
            # 处理每个batch和hidden单元
            for j in range(c[layers * t + i].shape[0]):
                for k in range(c[layers * t + i].shape[1]):
                    # 提取c和m的前k个重要元素
                    value1, index1 = torch.topk(c[layers * t + i, j, k], elements)
                    value2, index2 = torch.topk(m[layers * t + i, j, k], elements)
                    
                    # 构造c的关键特征组合: [c_topk + m_topk对应的c元素]
                    c_key = F.normalize(torch.cat([value1, c[layers * t + i, j, k, index2]], dim=0), 
                                      dim=0).detach().cpu().numpy().tolist()
                    data.append(c_key)
                    label.append(0)  # 0表示c类型
                    
                    # 构造m的关键特征组合: [c_topk对应的m元素 + m_topk]
                    m_key = F.normalize(torch.cat([m[layers * t + i, j, k, index1], value2], dim=0),
                                      dim=0).detach().cpu().numpy().tolist()
                    data.append(m_key)
                    label.append(1)  # 1表示m类型
                
                # 为每个case生成tsne图
                plot_TSNE(np.array(data), np.array(label), path, 
                         'case_' + str(j) + '_tsne_' + str(i) + '_' + str(t), 2)
