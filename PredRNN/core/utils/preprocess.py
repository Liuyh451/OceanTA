import numpy as np

def reshape_patch(img_tensor, patch_size):
    """
    将图像张量重塑为补丁张量。

    参数:
    img_tensor - 输入的图像张量，维度为(batch_size, seq_length, img_height, img_width, num_channels)。
    patch_size - 补丁的大小，即每个补丁的边长。

    返回:
    patch_tensor - 重塑后的补丁张量。
    """
    # 确保输入张量的维度正确
    assert 5 == img_tensor.ndim
    
    # 获取输入张量的各个维度大小
    batch_size = np.shape(img_tensor)[0]
    seq_length = np.shape(img_tensor)[1]
    img_height = np.shape(img_tensor)[2]
    img_width = np.shape(img_tensor)[3]
    num_channels = np.shape(img_tensor)[4]
    
    # 重塑张量，将其分成补丁
    a = np.reshape(img_tensor, [batch_size, seq_length,
                                img_height//patch_size, patch_size,
                                img_width//patch_size, patch_size,
                                num_channels])
    
    # 调整维度，以便补丁可以被连续地处理
    b = np.transpose(a, [0,1,2,4,3,5,6])
    
    # 重新塑造张量，将补丁视为单独的通道
    patch_tensor = np.reshape(b, [batch_size, seq_length,
                                  img_height//patch_size,
                                  img_width//patch_size,
                                  patch_size*patch_size*num_channels])
    
    # 返回重塑后的补丁张量
    return patch_tensor

def reshape_patch_back(patch_tensor, patch_size):
    """
    将补丁张量重塑回原始图像张量。

    参数:
    patch_tensor - 输入的补丁张量，维度为(batch_size, seq_length, patch_height, patch_width, channels)。
    patch_size - 补丁的大小，即每个补丁的边长。

    返回:
    img_tensor - 重塑后的图像张量。
    """
    # 确保输入补丁张量的维度正确
    assert 5 == patch_tensor.ndim
    
    # 获取补丁张量的各个维度大小
    batch_size = np.shape(patch_tensor)[0]
    seq_length = np.shape(patch_tensor)[1]
    patch_height = np.shape(patch_tensor)[2]
    patch_width = np.shape(patch_tensor)[3]
    channels = np.shape(patch_tensor)[4]
    
    # 计算原始图像的通道数
    img_channels = channels // (patch_size*patch_size)
    
    # 重塑补丁张量，将其分成单独的补丁
    a = np.reshape(patch_tensor, [batch_size, seq_length,
                                  patch_height, patch_width,
                                  patch_size, patch_size,
                                  img_channels])
    
    # 调整维度，以便补丁可以被连续地重组回图像
    b = np.transpose(a, [0,1,2,4,3,5,6])
    
    # 重新塑造张量，将补丁合并回原始图像尺寸
    img_tensor = np.reshape(b, [batch_size, seq_length,
                                patch_height * patch_size,
                                patch_width * patch_size,
                                img_channels])
    
    # 返回重塑后的图像张量
    return img_tensor
