import torch
import torch.nn as nn
from torch.nn.functional import one_hot
import torch.nn.functional as F
###loss function

smooth = 1.    #to avoid zero division

def dice_coef_anti(y_true, y_pred):
    y_true_anti = y_true[:,1,:]
    y_pred_anti = y_pred[:,1,:]
    intersection_anti = torch.sum(y_true_anti * y_pred_anti)
    return (2 * intersection_anti + smooth) / (torch.sum(y_true_anti) + torch.sum(y_pred_anti) + smooth)

def dice_coef_cyc(y_true, y_pred):
    y_true_cyc = y_true[:,2,:]
    y_pred_cyc = y_pred[:,2,:]
    intersection_cyc = torch.sum(y_true_cyc * y_pred_cyc)
    return (2 * intersection_cyc + smooth) / (torch.sum(y_true_cyc) + torch.sum(y_pred_cyc) + smooth)

def dice_coef_nn(y_true, y_pred):
    y_true_nn = y_true[:,0,:]
    y_pred_nn = y_pred[:,0,:]
    intersection_nn = torch.sum(y_true_nn * y_pred_nn)
    return (2 * intersection_nn + smooth) / (torch.sum(y_true_nn) + torch.sum(y_pred_nn) + smooth)

def mean_dice_coef(y_true, y_pred):
    return (dice_coef_anti(y_true, y_pred) + dice_coef_cyc(y_true, y_pred) + dice_coef_nn(y_true, y_pred))/3

def weighted_mean_dice_coef(y_true, y_pred):
    return (0.35 * dice_coef_anti(y_true, y_pred) + 0.62 * dice_coef_cyc(y_true,y_pred) + 0.03 * dice_coef_nn(y_true, y_pred))

def dice_coef_loss(y_true, y_pred):
    # 先将 target 进行 one-hot 处理，转换为 (N, C, H, W)
    # y_pred = y_pred.squeeze(1)

    # 对预测数据进行softmax处理
    y_pred_softmax = F.softmax(y_pred, dim=1)
    # y_pred = F.softmax(y_pred, dim=1)
    #二维变成一位数据
    num = y_pred.size(0)
    channel = y_pred.size(1)
    assert y_pred.size() == y_true.size(), "the size of predict and target must be equal."

    pre = y_pred.view(num,channel, -1)
    pre_softmax = y_pred_softmax.view(num,channel, -1)
    tar = y_true.view(num,channel, -1)
    intersection_anti = torch.sum(tar * pre)
    # loss_1结果不收敛 验证集出现nan
    N_dice_eff  = (2 * intersection_anti + smooth) / (torch.sum(tar) + torch.sum(pre) + smooth)
    loss_1 =  1 - N_dice_eff.sum() / num
    wei_mean_dice_coef = weighted_mean_dice_coef(tar, pre)
    # 验证集在0.4左右，无法提升
    loss2 =  1 - weighted_mean_dice_coef(tar, pre)
    # 使用softmax函数激活后，能够有效的提升训练准确率
    loss2_softmax =  1 - weighted_mean_dice_coef(tar, pre_softmax)

    return loss2_softmax
