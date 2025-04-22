import argparse
from functools import partial
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
# from torchvision import datasets
from torch.utils.data import DataLoader
from datasets.cityscapes import CityScapes
import numpy
import torch.nn.functional as F
from loss import JointEdgeSegLoss
import cv2
import torchvision
from torchvision.transforms import transforms
from torchvision.utils import save_image
from config import cfg, assert_and_infer_cfg
import logging
import math
import os
import sys
from my_functionals.Dloss import dice_coef_loss
import torch
import numpy as np
from utils.misc import AverageMeter, prep_experiment, evaluate_eval, fast_hist
from utils.f_boundary import eval_mask_boundary
import datasets
import loss
import network
import optimizer
from torch.utils.data import DataLoader
import torchvision.transforms as standard_transforms
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# 记录的内容保存在“./log”里面，log里面会生成events.out.tfevents的文件
writer = SummaryWriter('./log')
# Argument Parser
"""
通用配置:
--lr(float): 初始学习率，默认1e - 3
--arch(str): 模型架构导入路径，默认使用GSCNN网络
--trunk(str): 主干网络模型，可选resnet101 / resnet50，默认resnet101
--eval_thresholds(str): 边界评估阈值列表，默认'0.0005,0.001875,0.00375,0.005'
--rescale(float): 学习率缩放因子，默认1.0
--repoly(float): 多项式缩放因子，默认1.5
"""
parser = argparse.ArgumentParser(description='MFNET')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--arch', type=str, default='network.gscnn.GSCNN')
parser.add_argument('--trunk', type=str, default='resnet101',
                    help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('--eval_thresholds', type=str, default='0.0005,0.001875,0.00375,0.005',
                    help='Thresholds for boundary evaluation')
parser.add_argument('--rescale', type=float, default=1.0,
                    help='Rescaled LR Rate')
parser.add_argument('--repoly', type=float, default=1.5,
                    help='Rescaled Poly')
"""
损失函数权重配置:
--edge_weight(float): 边缘损失权重，默认1.0
--seg_weight(float): 分割损失权重，默认1.0
--att_weight(float): 注意力损失权重，默认1.0
--dual_weight(float): 双重损失权重，默认1.0
"""
parser.add_argument('--edge_weight', type=float, default=1.0,
                    help='Edge loss weight for joint loss')
parser.add_argument('--seg_weight', type=float, default=1.0,
                    help='Segmentation loss weight for joint loss')
parser.add_argument('--att_weight', type=float, default=1.0,
                    help='Attention loss weight for joint loss')
parser.add_argument('--dual_weight', type=float, default=1.0,
                    help='Dual loss weight for joint loss')
"""
训练过程控制:
     --evaluate (flag): 是否进入评估模式，默认False
     --max_epoch (int): 最大训练轮数，默认100
     --train_batch_size (int): 训练批次大小，默认64
     --val_batch_size (int): 验证批次大小，默认16
     --start_epoch (int): 起始训练轮数，默认0
"""
parser.add_argument('--evaluate', action='store_true', default=False)
parser.add_argument('--max_epoch', type=int, default=100)
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=16)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument("--local_rank", default=0, type=int)
"""
优化器配置:
--sgd(flag): 是否使用SGD优化器，默认False
--sgd_finetuned(flag): 是否使用微调后的SGD，默认False
--adam(flag): 是否使用Adam优化器，默认True
--amsgrad(flag): Adam是否使用amsgrad变体，默认False
--weight_decay(float): 权重衰减系数，默认1e - 4
--momentum(float): 动量参数，默认0.9
"""
parser.add_argument('--sgd', action='store_true', default=False)
parser.add_argument('--sgd_finetuned',action='store_true',default=False)
parser.add_argument('--adam', action='store_true', default=True)
parser.add_argument('--amsgrad', action='store_true', default=False)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
"""
学习率策略:
--lr_schedule(str): 学习率调整策略，默认'poly'
--poly_exp(float): 多项式学习率指数，默认1.0
"""
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=1.0,
                    help='polynomial LR exponent')
"""
数据预处理:
--crop_size(tuple): 训练图像裁剪尺寸，默认(101, 120)
--scale_min(float): 图像动态缩放下限，默认0.5
--scale_max(float): 图像动态缩放上限，默认2.0
"""
parser.add_argument('--bs_mult', type=int, default=16)
parser.add_argument('--bs_mult_val', type=int, default=4)
parser.add_argument('--crop_size', type=tuple, default=(101,120),
                    help='training crop size')
parser.add_argument('--scale_min', type=float, default=0.5,
                    help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=2.0,
                    help='dynamically scale training images up to this size')
"""
实验管理:
--exp(str): 实验目录名称，默认'default'
--tb_tag(str): TensorBoard日志标签
--ckpt(str): 模型检查点保存路径，默认'logs/ckpt'
--tb_path(str): TensorBoard日志路径，默认'logs/tb'
--syncbn(flag): 是否启用同步批量归一化，默认True
--test_mode(flag): 快速测试模式，默认False
"""
parser.add_argument('--snapshot', type=str, default=None)
parser.add_argument('--restore_optimizer', action='store_true', default=False)
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')
parser.add_argument('--tb_tag', type=str, default='',
                    help='add tag to tb dir')
parser.add_argument('--ckpt', type=str, default='logs/ckpt')
parser.add_argument('--tb_path', type=str, default='logs/tb')
parser.add_argument('--syncbn', action='store_true', default=True,
                    help='Synchronized BN')
parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                    help='Synchronized BN')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help='minimum testing (1 epoch run ) to verify nothing failed')
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0)
parser.add_argument('--maxSkip', type=int, default=0)
args = parser.parse_args()
args.best_record = {'epoch': -1, 'iter': 0, 'val_loss': 1e10, 'acc': 0,
                        'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}

#Enable CUDNN Benchmarking optimization
torch.backends.cudnn.benchmark = True
args.world_size = 1
# from keras.utils import np_utils
'''
Main Function
'''
def main():

    #Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)

    # train_loader1, val_loader2, train_obj = datasets.setup_loaders(args)
    net = network.get_eddynet(args)
    optim, scheduler = optimizer.get_optimizer(args, net)
    train_sampler = None
    val_sampler = None

    from datasets.dataset_npy import MyDataSet, valDataset

    val_data_path = "args.val_data_path"
    data_path = "args.data_path"

    transforms_ = [
        # transforms.Resize((img_height, img_width), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
    transforms_ = standard_transforms.Compose(transforms_)

    train_dataset = MyDataSet(
        data=data_path,
        # label=data_path,
        transform=transforms_
        )
    #
    val_dataset = valDataset(
        data=val_data_path,
        # label=val_data_path,
        transform=transforms_
        )
    #todo 原本这个num_workers=4，第二个是4//2
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                              num_workers=0, shuffle=(train_sampler is None), drop_last=True, sampler = train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size,
                            num_workers=0 , shuffle=False, drop_last=False, sampler = val_sampler)

    # 可视化数据
    edge_train_data = np.expand_dims(np.load('./datasets/edge_train_data.npy'), 3)[:, :, :, 0]
    SSH_train = np.expand_dims(np.load('./datasets/train_img_data.npy'), 3)[:, :, :, 0]

    Seg_train = np.expand_dims(np.load('./datasets/seg_train_data.npy'), 3)[:, :, :, 0]
    randindex = np.random.randint(0, len(SSH_train))
    import matplotlib.pyplot as plt

    torch.cuda.empty_cache()
    """
        dataloader
    """

    ## Data pipeline
    train_losses = []
    edge_losses = []
    acc = []
    good_score = 0
    #Main Loop
    for epoch in range(args.start_epoch, args.max_epoch):
    # Update EPOCH CTR
    #     cfg.immutable(False)
        cfg.EPOCH  = epoch
        # cfg.immutable(True)
        print(cfg.EPOCH)
        train_loss = 0
        acc_epoch = 0
        edge_loss = 0
        scheduler.step()
        acc_epoch = validate(val_loader, net, optim, epoch,acc_epoch)
        result_score = acc_epoch / len(val_loader)
        acc.append(acc_epoch/len(val_loader))
        train_loss,edge_loss = train(train_loader, net, optim, epoch,train_loss,edge_loss)
        train_losses.append(train_loss / len(train_loader))
        # print("-----------------------------------val_loader_length is "+str(len(val_loader)))
        edge_losses.append(edge_loss / len(train_loader))
        if result_score > good_score:
            torch.save(net.state_dict(), "/checkpoints/sample/model_%d.pth" % (epoch))
            good_score=result_score
            print("best F1 :"+str(good_score)+" best epoch:"+str(epoch))

    plt.plot(np.arange(len(train_losses)), train_losses, label="train loss")
    plt.plot(np.arange(len(edge_losses)), edge_losses, label="edge loss")
    plt.plot(np.arange(len(acc)), acc, label="ACC")
    plt.legend()  # 显示图例
    plt.xlabel('epoches')
    # plt.ylabel("epoch")
    plt.title('Model loss')
    plt.savefig("tensor_loss_filename.png")
    writer.close()
    # plt.show()
def contour_loss(inputs, targets):
    """
    Inputs:
        inputs: predicted tensor of shape (N, C, H, W)
        targets: target tensor of shape (N, H, W), values in [0, num_classes-1]
        num_classes: number of classes including background
        alpha: weight for class loss
        beta: weight for contour loss
        gamma: weight for L1 regularization
    """
    import torch
    import torch.nn as nn

    # 自定义每个类别的权重，其中第一类为背景，权重为0
    class_weight = torch.FloatTensor([0, 1, 1]).cuda()

    # 定义损失函数
    criterion = nn.CrossEntropyLoss(weight=class_weight)


    loss = criterion(inputs, targets)  # 计算损失函数

    return loss
# 中尺度涡旋轮廓损失函数
def bce_l1_loss(input, target):
    # 计算二分类交叉熵损失
    bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='mean')
    # 计算 L1 损失
    l1_loss = F.l1_loss(torch.sigmoid(input), target, reduction='mean')
    # 综合两个损失，可调整两个损失的权重
    loss = bce_loss + l1_loss
    return loss

def bce2d_loss(input, target):
    n, c, h, w = input.size()
    input = F.log_softmax(input, dim=1)
    log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)  # n,h,w,c
    target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
    target_trans = target_t.clone()
    pos_index = (target_t == 1)
    neg_index = (target_t == 0)
    fqx_index = (target_t == 2)
    ignore_index = (target_t > 2)

    target_trans[pos_index] = 1
    target_trans[neg_index] = 0
    target_trans[fqx_index] = 2

    pos_index = pos_index.data.cpu().numpy().astype(bool)
    neg_index = neg_index.data.cpu().numpy().astype(bool)
    fqx_index = fqx_index.data.cpu().numpy().astype(bool)
    ignore_index = ignore_index.data.cpu().numpy().astype(bool)

    weight = torch.Tensor(log_p.size()).fill_(0)
    weight = weight.numpy()
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    fanqixuan_num = fqx_index.sum()
    sum_num = pos_num + neg_num+fanqixuan_num
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num
    weight[fqx_index] = fanqixuan_num * 1.0 / sum_num

    weight[ignore_index] = 0

    weight = torch.from_numpy(weight)
    weight = weight.cuda()
    loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, size_average=True)
    return loss

def train(train_loader, net, optimizer, curr_epoch,train_loss,edge_loss):
    '''
    Runs the training loop per epoch
    train_loader: Data loader for train
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch 
    writer: tensorboard writer
    return: val_avg for step function if required
    '''
    net.train()

    curr_iter = curr_epoch * len(train_loader)

    for i, data in enumerate(train_loader):
        if i==0:
            print('running....')

        inputs, input2,mask, edge = data

        # testimg = cv2.imread(path_img, 0)
        mask = torch.transpose(torch.transpose(mask, 1, 3), 2, 3)
        edge = torch.transpose(torch.transpose(edge, 1, 3), 2, 3)
        # edge2 = torch.transpose(torch.transpose(edge2, 1, 3), 2, 3)
        # mask[(mask != 255) & (mask != 0)] = 1  # 注：eddy填充的公式  1-125 -qixuan
        # mask[(mask != 1) & (mask != 0)] = 2  # 注：eddy填充的公式    255-2--beijing
        if torch.sum(torch.isnan(inputs)) > 0:
            import pdb; pdb.set_trace()

        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)

        inputs, input2,mask, edge = inputs.cuda(), input2.cuda(), mask.cuda(), edge.cuda()
        inputs = inputs.type(torch.FloatTensor).cuda()
        input2 = input2.type(torch.FloatTensor).cuda()
        edge = edge.type(torch.FloatTensor).cuda()

        optimizer.zero_grad()

        main_loss = None

        loss2 = None

        L1_G = torch.nn.L1Loss()

        if torch.cuda.is_available():
            L1_G = L1_G.cuda()
        if args.max_epoch !=0:

            seg_out, edge_out = net(inputs,input2,edge)
            #原始交叉熵损失函数，只能识别一个
            loss_edge =bce2d_loss(edge_out,edge)
            # 改进的交叉熵损失函数
            # loss_edg_ceL =contour_loss(edge_out,edge)
            #L1损失函数
            loss_l1 = L1_G(edge_out,edge)
            # 中尺涡语义分割损失函数
            loss2  = dice_coef_loss(mask,seg_out)
            # 该损失函数能够有效得到分类信息
            loss_edge_class  = bce_l1_loss(edge,edge_out)
            # loss_edge_class  = bce_l1_loss(edge,edge_out)
            train_loss += loss2.item()
            edge_loss +=loss_edge.item()*3
            # main_loss=loss2
            # 添加权重系数，使网络关注与轮廓细节生成能力
            # main_loss=loss2+loss_edg_ceL
            main_loss=loss2+loss_edge_class*50
            main_loss_1 = loss2 + loss_edge_class * (curr_epoch - i)
            # main_loss=loss2
            # main_loss=loss2+loss_edge*20+loss_l1
            # main_loss=loss2+loss_attentions+loss_edge*20

        main_loss.backward()

        optimizer.step()

        curr_iter += 1

        if args.local_rank == 0:
            sys.stdout.write(
                "\r[Epoch %d/%d: batch %d/%d] [main_loss: %.3f, L1_loss: %.3f, dice_coef_loss: %.3f, edge_loss: %.3f ]"
                % (
                    args.max_epoch, curr_epoch, i, len(train_loader),
                    main_loss.item(), loss_l1.item(),loss2.item(),loss_edge_class.item()
                )
                )
            # Image Dumps

        seg_predictions = seg_out.data.max(1)[1].cpu()
        # if curr_epoch > 15:

        # if not i % 5:

    writer.add_scalar('loss_main_1', main_loss_1.item(), curr_epoch)
    writer.add_scalar('dice_coef_loss', loss2.item(), curr_epoch)
    writer.add_scalar('loss_edge', loss_edge_class.item(), curr_epoch)



    return train_loss,edge_loss



def precision(y_true, y_pred):
    true_positive = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    predicted_positive = torch.sum(torch.round(torch.clip(y_pred, 0, 1)))
    precision = true_positive / (predicted_positive+0.00001)
    return precision
def recall(y_true, y_pred):
    true_positive = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    possible_positive = torch.sum(torch.round(torch.clip(y_true, 0, 1)))
    recall = true_positive / (possible_positive +0.00001)
    return recall
def fbeta_score(y_true, y_pred, beta = 1):
    y_true_temp = y_true
    y_pred_temp = y_pred
    y_true = y_true.type(torch.FloatTensor)
    y_pred = y_pred.type(torch.FloatTensor)
    if beta < 0:
        raise ValueError('the lowest choosable beta is zero')
    temp = torch.clip(y_true, 0, 1)
    temp = torch.round(temp)
    if torch.sum(torch.round(torch.clip(y_true, 0, 2))) == 0:
    # if torch.sum(torch.round(torch.clip(y_true, 0, 1))) == 0:
        return 0
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    b = beta ** 2
    fbeta_score = (1 + b) * (p * r) / (b * p + r)
    return fbeta_score,r,p
def validate(val_loader, net, optimizer, curr_epoch,acc_epoch):
    '''
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch 
    write r: tensorboard writer
    return: 
    '''
    net.eval()

    for vi, data in enumerate(val_loader):
        input, input2,mask, edge = data
        assert len(input.size()) == 4 and len(mask.size()) == 4
        mask = torch.transpose(torch.transpose(mask, 1, 3), 2, 3)
        # assert input.size()[2:] == mask.size()[1:]
        # h, w = mask.size()[1:]

        # batch_pixel_size = input.size(0) * input.size(2) * input.size(3)
        input, input2,mask_cuda, edge_cuda = input.cuda(),input2.cuda(), mask.cuda(), edge.cuda()
        edge_cuda = torch.unsqueeze(edge_cuda, 1)
        edge_cuda = edge_cuda.type(torch.FloatTensor).cuda()
        input = input.type(torch.FloatTensor).cuda()
        input2 = input2.type(torch.FloatTensor).cuda()

        with torch.no_grad():
            seg_out, edge_out = net(input,input2,edge_cuda)    # output = (1, 19, 713, 713)
        seg_predictions = seg_out.argmax(1)
        # seg_predictions = seg_out.data.max(1)[1]
        fbeta_score1,recall,precision  = fbeta_score(mask_cuda,seg_out)
        local_rank = 0





        sys.stdout.write(
            "\r[Epoch %d/%d: batch %d/%d] [fbeta_score1: %.3f, recall: %.3f, precision: %.3f ]"
            % (args.max_epoch, curr_epoch, vi, len(val_loader),
                fbeta_score1.item(),recall.item(),precision.item() )
            )
        mask = mask.type(torch.FloatTensor).cuda()
        if curr_epoch==15:
            # img_sample_pred_real = torch.cat((seg_out2.data, mask.data), -2)
            mask = mask.type(torch.FloatTensor).cuda()
            edge=edge.type(torch.FloatTensor).cuda()


        if numpy.isnan(fbeta_score1):
            local_rank = 1
            fbeta_score1 = 0.4
        acc_epoch += fbeta_score1
        writer.add_scalar('accuracy', fbeta_score1, curr_epoch)
        writer.add_scalar('Recall', recall.item(), curr_epoch)
        writer.add_scalar('Precision', precision.item(), curr_epoch)
    return acc_epoch


def evaluate(val_loader, net):
    '''
    Runs the evaluation loop and prints F score
    val_loader: Data loader for validation
    net: thet network
    return: 
    '''
    net.eval()
    for thresh in args.eval_thresholds.split(','):
        mf_score1 = AverageMeter()
        mf_pc_score1 = AverageMeter()
        ap_score1 = AverageMeter()
        ap_pc_score1 = AverageMeter()
        Fpc = np.zeros((args.dataset_cls.num_classes))
        Fc = np.zeros((args.dataset_cls.num_classes))
        for vi, data in enumerate(val_loader):
            input, mask, edge = data
            assert len(input.size()) == 4 and len(mask.size()) == 4
            assert input.size()[2:] == mask.size()[1:]
            h, w = mask.size()[1:]

            batch_pixel_size = input.size(0) * input.size(2) * input.size(3)
            input, mask_cuda, edge_cuda = input.cuda(), mask.cuda(), edge.cuda()

            with torch.no_grad():
                seg_out, edge_out = net(input)

            seg_predictions = seg_out.data.max(1)[1].cpu()
            edge_predictions = edge_out.max(1)[0].cpu()

            logging.info('evaluating: %d / %d' % (vi + 1, len(val_loader)))
            _Fpc, _Fc = eval_mask_boundary(seg_predictions.numpy(), mask.numpy(), args.dataset_cls.num_classes, bound_th=float(thresh))
            Fc += _Fc
            Fpc += _Fpc

            del seg_out, edge_out, vi, data

        logging.info('Threshold: ' + thresh)
        logging.info('F_Score: ' + str(np.sum(Fpc/Fc)/args.dataset_cls.num_classes))
        logging.info('F_Score (Classwise): ' + str(Fpc/Fc))

if __name__ == '__main__':
    main()




