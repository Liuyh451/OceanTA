
import torch
import torch.nn as nn

class ContextualEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, gru_hidden_size):
        super(ContextualEncoder, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.gru = nn.GRU(input_dim + embedding_dim, gru_hidden_size, batch_first=True)
        self.max_pool = nn.MaxPool1d(num_buoys)

    def forward(self, x, buoy_ids):
        batch_size, num_buoys, window_size, _ = x.size()
        # Encode buoy ids into positional embeddings
        buoy_id_embeddings = self.get_buoy_id_embeddings(buoy_ids, num_buoys, self.embedding_dim)
        # Concatenate positional embeddings with observations at each time step
        x_with_id = torch.cat([x, buoy_id_embeddings.unsqueeze(2).expand(-1, -1, window_size, -1)], dim=-1)
        print(x_with_id.shape)
        _, gru_hidden_states = self.gru(x_with_id)
        # Extract the last hidden state for each buoy
        buoy_embeddings = gru_hidden_states[:, -1, :]
        # Reshape for max pooling
        pooled_embeddings = buoy_embeddings.permute(0, 2, 1)
        # Apply max pooling across buoys
        context_vector = self.max_pool(pooled_embeddings).squeeze(-1)
        return context_vector

    def get_buoy_id_embeddings(self, buoy_ids, num_buoys, embedding_dim):
        embeddings = nn.Embedding(num_buoys, embedding_dim)
        return embeddings(buoy_ids)
# 假设一些示例输入
batch_size = 32
num_buoys = 5
window_size = 3
input_dim = 4
embedding_dim = 16
gru_hidden_size = 512

# 创建随机输入数据和 buoy ids
input_data = torch.randn(batch_size, num_buoys, window_size, input_dim)
buoy_ids = torch.randint(0, num_buoys, (batch_size, num_buoys))

encoder = ContextualEncoder(input_dim, embedding_dim, gru_hidden_size)
context_vector = encoder(input_data, buoy_ids)
print(context_vector.shape)

class ContextualEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, gru_hidden_size):
        """
        简化的上下文编码器。

        Args:
            input_dim: 每个时间步的输入特征维度（例如 4）。
            embedding_dim: 浮标 ID 的嵌入维度。
            gru_hidden_size: GRU 隐藏层大小。
        """
        super(ContextualEncoder, self).__init__()

        # 浮标 ID 嵌入层
        self.embedding = nn.Embedding(num_embeddings=100, embedding_dim=embedding_dim)

        # GRU 编码器（共享的）
        self.gru = nn.GRU(input_dim + embedding_dim, gru_hidden_size, batch_first=True)

        # 最大池化层（用于聚合浮标的上下文向量）
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, obs_data, buoy_ids):
        """
        前向传播。

        Args:
            obs_data: 滑动窗口后的观测数据，形状为 (batch_size * num_windows, window_size, input_dim)。
            buoy_ids: 浮标 ID，形状为 (batch_size,)。

        Returns:
            c: 全局上下文向量，形状为 (batch_size, gru_hidden_size)。
        """
        batch_size = buoy_ids.shape[0]
        num_windows = obs_data.size(0) // batch_size  # 每个浮标的窗口数
        window_size = obs_data.size(1)

        # Step 1: 浮标 ID 嵌入
        buoy_embeddings = self.embedding(buoy_ids)  # (batch_size, embedding_dim)
        buoy_embeddings = buoy_embeddings.unsqueeze(1).expand(-1, num_windows,
                                                              -1)  # (batch_size, num_windows, embedding_dim)
        buoy_embeddings = buoy_embeddings.reshape(-1,
                                                  buoy_embeddings.size(-1))  # (batch_size * num_windows, embedding_dim)

        # Step 2: 将浮标嵌入与窗口数据拼接
        concatenated_data = torch.cat([obs_data, buoy_embeddings.unsqueeze(1).expand(-1, window_size, -1)], dim=-1)

        # Step 3: 通过 GRU 编码每个浮标的窗口片段
        _, gru_hidden_states = self.gru(concatenated_data)  # (1, batch_size * num_windows, gru_hidden_size)
        buoy_embeddings = gru_hidden_states.squeeze(0)  # (batch_size * num_windows, gru_hidden_size)

        # Step 4: 恢复浮标维度，并进行最大池化
        buoy_embeddings = buoy_embeddings.reshape(batch_size, num_windows,
                                                  -1)  # (batch_size, num_windows, gru_hidden_size)
        max_pooled = self.max_pool(buoy_embeddings.transpose(1, 2)).squeeze(-1)  # (batch_size, gru_hidden_size)

        return max_pooled



"""
这是第3版的代码，对比第一版的代码更改的地方在于：
1.feature_dim特征数为4，而不是time_step=4
2.拼接的过程中提取3个时间步进行一次拼接，和id拼接后的向量为20
"""
class ContextualEncoder3(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_buoys, time_steps):
        """
        初始化 Contextual Encoder
        :param input_dim: 每个时间步的特征维度 (F)
        :param embedding_dim: 浮标 ID 嵌入向量的维度
        :param hidden_dim: GRU 隐藏层维度 (H)
        :param num_buoys: 最大浮标数量
        :param time_steps: 时间步数 (T)
        """
        super(ContextualEncoder3, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=num_buoys, embedding_dim=embedding_dim)
        self.gru = nn.GRU(input_size=input_dim + embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.max_pool = nn.AdaptiveMaxPool1d(1)  # 自适应池化到 (B, H, 1)

    def forward(self, buoy_obs, buoy_ids):
        """
        前向传播
        :param buoy_obs: 浮标观测数据，形状为 (B, T, F)
        :param buoy_ids: 浮标 ID，形状为 (B, T)
        :return: 上下文向量 c，形状为 (B, H)
        """
        batch_size, time_steps, _ = buoy_obs.shape

        # 1. 将浮标 ID 转换为嵌入向量
        buoy_emb = self.embedding(buoy_ids)  # (B, T, embedding_dim)

        # 2. 将浮标观测数据与嵌入向量拼接
        combined_input = torch.cat([buoy_obs, buoy_emb], dim=-1)  # (B, T, F + embedding_dim)
        print(combined_input.shape)
        # 3. 通过共享 GRU 编码
        gru_output, _ = self.gru(combined_input)  # gru_output: (B, T, H)

        # 4. MaxPooling 聚合浮标的时间步信息
        pooled_output = self.max_pool(gru_output.transpose(1, 2))  # (B, H, T) -> (B, H, 1)

        # 5. 压缩为最终的上下文向量
        context_vector = pooled_output.squeeze(-1)  # (B, H)

        return context_vector


"""
这是第2（也与是2）版的代码，对比第一版的代码更改的地方在于：
1.feature_dim特征数为4，而不是time_step=4
2.拼接的过程中提取3个时间步进行一次拼接，和id拼接后的向量为20
"""
class ContextualEncoder2(nn.Module):
    def __init__(self, feature_dim=4, id_emb_dim=16, gru_hidden_size=512):
        super(ContextualEncoder2, self).__init__()
        self.feature_dim = feature_dim
        self.id_emb_dim = id_emb_dim
        self.gru_hidden_size = gru_hidden_size

        # 浮标ID嵌入层
        self.embedding = nn.Embedding(num_embeddings=100, embedding_dim=id_emb_dim)

        # 共享GRU
        self.gru = nn.GRU(input_size=feature_dim + id_emb_dim, hidden_size=gru_hidden_size, batch_first=True)

        # 最大池化层
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, buoy_obs, buoy_ids):
        """
        输入：
        - buoy_obs: (num_buoys, time_steps, feature_dim), 每个浮标的观测数据
        - buoy_ids: (num_buoys,), 每个浮标的ID
        输出：
        - context_vector: (gru_hidden_size, 1, 1), 上下文向量
        """
        num_buoys, time_steps, feature_dim = buoy_obs.size()
        assert time_steps >= 3, "时间步必须至少为3。"

        # 获取浮标ID的嵌入
        buoy_id_emb = self.embedding(buoy_ids)  # (num_buoys, id_emb_dim)

        # 初始化 GRU 隐藏状态
        hidden_state = torch.zeros(1, num_buoys, self.gru_hidden_size).to(buoy_obs.device)

        # 用于存储每个时间步的 GRU 结果
        gru_outputs = []

        # 遍历时间步并提取特征向量，拼接位置嵌入，输入到 GRU
        for t in range(3):  # 每次提取3个时间步
            # 提取特征向量 (num_buoys, feature_dim)
            feature_vector = buoy_obs[:, t, :]  # (num_buoys, feature_dim)

            # 拼接位置嵌入
            combined_input = torch.cat([feature_vector, buoy_id_emb], dim=-1).unsqueeze(1)  # (num_buoys, 1, feature_dim + id_emb_dim)

            # 共享 GRU
            gru_out, hidden_state = self.gru(combined_input, hidden_state)  # gru_out: (num_buoys, 1, gru_hidden_size)

            # 将 GRU 的输出保存
            gru_outputs.append(gru_out.squeeze(1))  # (num_buoys, gru_hidden_size)

        # 第一个 GRU 的输出和第2个向量作为输入到第二个 GRU
        second_gru_input = torch.cat([gru_outputs[0], buoy_obs[:, 1, :]], dim=-1).unsqueeze(1)
        second_gru_out, hidden_state = self.gru(second_gru_input, hidden_state)

        # 第2个 GRU 的结果和第3个向量作为输入到第三个 GRU
        third_gru_input = torch.cat([second_gru_out.squeeze(1), buoy_obs[:, 2, :]], dim=-1).unsqueeze(1)
        third_gru_out, hidden_state = self.gru(third_gru_input, hidden_state)

        # 经过最大池化生成上下文向量
        third_gru_out = third_gru_out.permute(0, 2, 1)  # (num_buoys, gru_hidden_size, 1)
        context_vector = self.max_pool(third_gru_out).squeeze(-1).unsqueeze(0)  # (1, gru_hidden_size, 1)

        return context_vector




