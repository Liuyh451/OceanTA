import os
import torch
from torch.optim import Adam
from core.models import predrnn, predrnn_v2, action_cond_predrnn, action_cond_predrnn_v2

class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        networks_map = {
            'predrnn': predrnn.RNN,
            'predrnn_v2': predrnn_v2.RNN,
            'action_cond_predrnn': action_cond_predrnn.RNN,
            'action_cond_predrnn_v2': action_cond_predrnn_v2.RNN,
        }

        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)

        self.optimizer = Adam(self.network.parameters(), lr=configs.lr)

    def save(self, itr):
        stats = {}
        stats['net_param'] = self.network.state_dict()
        checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt'+'-'+str(itr))
        torch.save(stats, checkpoint_path)
        print("save model to %s" % checkpoint_path)

    def load(self, checkpoint_path):
        print('load model:', checkpoint_path)
        stats = torch.load(checkpoint_path)
        self.network.load_state_dict(stats['net_param'])

    def train(self, frames, mask):
        """
        训练模型的一个批次数据。

        参数:
        frames: 输入的帧数据，通常是一个序列，用于训练模型。
        mask: 掩码数据，指示哪些部分需要预测。

        返回值:
        当前批次的损失值（loss），以 numpy 数组形式返回，用于评估模型性能。
        """
        # 将输入帧数据转换为张量并移动到指定设备（如 GPU）
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        # 将掩码数据转换为张量并移动到指定设备
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        # 清空优化器中的梯度信息，避免梯度累积
        self.optimizer.zero_grad()
        # 将数据输入网络，得到预测帧和损失值
        next_frames, loss = self.network(frames_tensor, mask_tensor)
        # 反向传播计算梯度
        loss.backward()
        # 更新模型参数
        self.optimizer.step()
        # 返回损失值的数值形式（从 GPU 转移到 CPU 并转换为 numpy 格式）
        return loss.detach().cpu().numpy()


    def test(self, frames, mask):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        next_frames, _ = self.network(frames_tensor, mask_tensor)
        return next_frames.detach().cpu().numpy()