def unfold_StackOverChannel(img, kernel_size):
    """
    divide the original image to patches, then stack the grids in each patch along the channels
    Args:
        img (N, *, C, H, W): the last two dimensions must be the spatial dimension
        N:批量大小 (Batch Size)
        *：这是一个通配符，表示可能存在的任意数量的额外维度。通常在图像数据中，这些额外的维度可能代表时间步、序列长度或其他需要的维度
        C：通道数 (Channels)，表示每个样本的特征图的数量。
        H:高度 (Height)，表示每个特征图的高度。
        W:宽度 (Width)，表示每个特征图的宽度。
        kernel_size: tuple of length 2
    Returns:
        output (N, *, C*H_k*N_k, H_output, W_output)
    """
    T = img.size(1)
    n_dim = len(img.size())#ndim=5
    assert n_dim == 4 or n_dim == 5
    #对高度（-2 维度）进行 unfold 操作，使用 kernel_size[0] = 3 和 step=3，例如img 形状为 (N, 8, 22, 1)，
    #结果形状为 (N, 8, 8, 1, 3), 其中 8 是高度维度被划分的数量，3 是每个 patch 的高度
    pt = img.unfold(-2, size=kernel_size[0], step=kernel_size[0])
    #对宽度（-2 维度）进行 unfold 操作，使用 kernel_size[1] = 1 和 step=1
    #结果形状为 (N, 8, 8, 1, 3)，由于 kernel_size[1] 为 1，宽度不变，结果的 8 维度表示所有可能的宽度位置
    pt = pt.unfold(-2, size=kernel_size[1], step=kernel_size[1]).flatten(-2)  # (N, *, C, n0, n1, k0*k1)
    #然后将这两个 unfold 操作的结果展平得到形状 1.(16,3,1,8,1,3)
    #2.
    if n_dim == 4:  # (N, C, H, W)
        pt = pt.permute(0, 1, 4, 2, 3).flatten(1, 2)
        #4维图像，所以最终 reshape 和 permute 操作: (N, 1, 3, 22)
    elif n_dim == 5:  # (N, T, C, H, W)
        pt = pt.permute(0, 1, 2, 5, 3, 4).flatten(2, 3)
    assert pt.size(-3) == img.size(-3) * kernel_size[0] * kernel_size[1]
    #(16,3,3,8,1)
    # pt = pt.reshape(pt.size(0), T, 25, -1).permute(0, 3, 1, 2)
    #todo 测试
    pt = pt.reshape(pt.size(0), T, kernel_size[0] * kernel_size[1], -1).permute(0, 3, 1, 2)
    return pt
import torch

# 随机生成一个形状为 [16, 3, 5, 24, 1] 的张量
resdual1 = torch.randn(16, 3, 5, 24, 1)

# 输出张量的形状

resdual1 = unfold_StackOverChannel(resdual1, (3, 1))
print(resdual1.shape)