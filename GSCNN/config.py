from easydict import EasyDict as edict
import torch.nn as nn
import torch
cfg = edict()

# ---- DATASET 配置 ----
cfg.DATASET = edict()
cfg.MODEL= edict()
cfg.DATASET.NAME = 'cityscapes'
cfg.DATASET.CITYSCAPES_DIR = ''
#归一化层
cfg.MODEL.BNFUNC = nn.BatchNorm2d

def assert_and_infer_cfg(args):
    global cfg

    # 用 args 的值更新 cfg
    for key in vars(args):
        val = getattr(args, key)
        setattr(cfg, key, val)
        print(f"已设置参数 {key}={val}")

    # 多卡世界大小更新（如果用分布式）
    if hasattr(args, 'world_size'):
        cfg.world_size = args.world_size

    # crop_size 校验与推导（可选）
    if not isinstance(cfg.crop_size, tuple):
        try:
            cfg.crop_size = eval(cfg.crop_size)
        except:
            raise ValueError(f"⚠️ crop_size 格式错误: {cfg.crop_size}，应为 tuple 格式")

    # 设置 benchmark
    torch.backends.cudnn.benchmark = True

    return cfg

