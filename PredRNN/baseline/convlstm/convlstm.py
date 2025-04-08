import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os

# ===================== 参数集中管理 =====================
params = {
    "train": '/root/autodl-fs/train.npy',  # 主数据文件
    "val": '/root/autodl-fs/val.npy',
    "test": '/root/autodl-fs/test.npy',
    "epochs": 400,
    "batch_size": 32,
    "history": 3,
    "input_channels": 3,
    "hidden_dim": 128,
    "kernel_size": 3,
    "lr": 1e-4,
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_model_path": "conv_lstm.pth",
    'save_preds_path': '/result/predictions.npy',
    'save_targets_path': '/result/targets.npy'
}


# ===================== 数据集 =====================
class WaveDataset(Dataset):
    def __init__(self, data, history=3):
        self.data = data
        self.history = history
        self.samples = self._create_samples()

    def _create_samples(self):
        samples = []
        total_frames = self.data.shape[0]
        for t in range(self.history, total_frames):
            x = self.data[t - self.history:t]
            y = self.data[t]
            samples.append((x, y))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        x = torch.tensor(x, dtype=torch.float).permute(3, 0, 1, 2)  # [C, T, H, W]
        y = torch.tensor(y, dtype=torch.float).permute(2, 0, 1)  # [C, H, W]
        return x, y

    # ===================== ConvLSTMCell =====================


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, padding, stride=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim,
                              kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x, states):
        h_cur, c_cur = states
        combined = torch.cat([x, h_cur], dim=1)
        conv_out = self.conv(combined)
        i, f, o, g = torch.chunk(conv_out, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, shape, device):
        return (torch.zeros(batch_size, self.hidden_dim, *shape, device=device),
                torch.zeros(batch_size, self.hidden_dim, *shape, device=device))


# ===================== ConvLSTM 多层模块 =====================
class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, padding, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            ConvLSTMCell(input_dim if i == 0 else hidden_dim,
                         hidden_dim, kernel_size, padding)
            for i in range(num_layers)
        ])

    def forward(self, x):  # x: [B, T, C, H, W]
        b, t, c, h, w = x.size()
        h_t, c_t = [], []
        for layer in self.layers:
            h, c = layer.init_hidden(b, (h, w), x.device)
            h_t.append(h)
            c_t.append(c)

        outputs = []
        for time_step in range(t):
            input_ = x[:, time_step]
            for i, layer in enumerate(self.layers):
                h_t[i], c_t[i] = layer(input_, (h_t[i], c_t[i]))
                input_ = h_t[i]
            outputs.append(h_t[-1].unsqueeze(1))
        return torch.cat(outputs, dim=1)  # [B, T, hidden, H, W]


# ===================== WavePredictor 主模型 =====================
class WavePredictor(nn.Module):
    def __init__(self, input_channels, hidden_dim, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.convlstm = ConvLSTM(input_dim=input_channels,
                                 hidden_dim=hidden_dim,
                                 kernel_size=kernel_size,
                                 padding=padding,
                                 num_layers=1)
        self.decoder = nn.Conv2d(hidden_dim, input_channels, kernel_size=1)

    def forward(self, x):  # [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)  # -> [B, T, C, H, W]
        out_seq = self.convlstm(x)  # -> [B, T, hidden, H, W]
        out = self.decoder(out_seq[:, -1])  # 取最后一帧作为预测
        return out


def data_load(path):
    # 加载 .npy 文件并提取字典
    data_dict = np.load(path, allow_pickle=True).item()  # 使用 .item() 或 [()]
    hs = data_dict['hs']
    tm02 = data_dict['tm02']
    theta0 = data_dict['theta0']  # (T, W, H)

    # 合并成 3 通道数据 (T, W, H, C)
    data = np.stack([hs, tm02, theta0], axis=-1).astype('float32')
    # 替换 NaN 为 0
    data = np.nan_to_num(data, nan=0.0)
    # 方式二：归一化（Min-Max Scaling）
    mins = np.min(data, axis=(0, 1, 2), keepdims=True)  # 各通道最小值
    maxs = np.max(data, axis=(0, 1, 2), keepdims=True)  # 各通道最大值
    data = (data - mins) / (maxs - mins + 1e-8)  # 缩放到 [0,1]
    print(f"Data shape: {data.shape}")
    return data


# ===================== 训练 + 验证 + 测试 =====================
def train_val(params):
    train_data = data_load(params['train'])  # [T, 128, 128, 3]
    val_data = data_load(params['val'])

    train_data = WaveDataset(train_data, history=params['history'])
    val_data = WaveDataset(val_data, history=params['history'])

    train_loader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=True,
                              num_workers=params['num_workers'])
    val_loader = DataLoader(val_data, batch_size=params['batch_size'], shuffle=False)

    model = WavePredictor(params['input_channels'], params['hidden_dim'], params['kernel_size']).to(params['device'])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])

    best_val_loss = float('inf')

    for epoch in range(params['epochs']):
        print(f"Epoch {epoch + 1}/{params['epochs']}")
        model.train()
        total_train_loss = 0
        for x, y in train_loader:
            x, y = x.to(params['device']), y.to(params['device'])
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # 验证
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(params['device']), y.to(params['device'])
                pred = model(x)
                loss = criterion(pred, y)
                total_val_loss += loss.item()

        print(f"[Epoch {epoch + 1}] Train Loss: {total_train_loss / len(train_loader):.4f} | "
              f"Val Loss: {total_val_loss / len(val_loader):.4f}")

        # 保存最优模型
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            torch.save(model.state_dict(), params['save_model_path'])
        # 每隔100轮保存一次模型
        if (epoch + 1) % 100 == 0:
            # 构建新的保存路径，包含epoch信息
            epoch_save_path = params['save_model_path'].replace('.pth', f'_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), epoch_save_path)

def test(params):
    test_data=data_load(params['val'])

    test_data = WaveDataset(test_data, history=params['history'])

    test_loader = DataLoader(test_data, batch_size=params['batch_size'], shuffle=False)

    model = WavePredictor(params['input_channels'], params['hidden_dim'], params['kernel_size']).to(params['device'])

    # ========== 测试 ==========
    model.load_state_dict(torch.load(params['save_model_path']))
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(params['device'])
            pred = model(x).cpu().numpy()
            preds.append(pred)
            targets.append(y.numpy())
    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    print(f"Test MSE: {np.mean((preds - targets)**2):.4f}")
    # 保存预测结果和真实值
    np.save(params.get('save_preds_path', 'predictions.npy'), preds)
    np.save(params.get('save_targets_path', 'targets.npy'), targets)
# train_val(params)
test(params)