import numpy as np
import torch
import Utils
import wave_filed_data_prepare

# 实例化模型
context_encoder = Utils.ContextualEncoder()
decoder = Utils.WaveFieldDecoder()

# 加载保存的参数
context_encoder.load_state_dict(torch.load("net/context_encoder_params.pth"))
decoder.load_state_dict(torch.load("net/decoder_params.pth"))
# 设置为评估模式
context_encoder.eval()
decoder.eval()
all_generated_wave_fields = []
# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("loading swan data and buoy data........")
buoy_data = np.load('data/buoy_data_test.npy')
buoy_data = torch.tensor(buoy_data)
swan_data = wave_filed_data_prepare.combine_monthly_data("/home/hy4080/met_waves/Swan_cropped/swanSula", 2019,
                                                         2021)
swan_data = torch.tensor(swan_data)
print("swan data and buoy data shape", buoy_data.shape, swan_data.shape)
# 初始化推理模块
inference = Utils.Inference(context_encoder, decoder, device)
# 创建数据集和数据加载器
dataset = Utils.TimeSeriesDataset(buoy_data, swan_data, batch_size=128)
dataloader = Utils.DataLoader(dataset, batch_size=None, shuffle=False)
for batch_idx, (buoy_data, _) in enumerate(dataloader):
    print(f"Processing batch {batch_idx + 1}/{len(dataloader)}...")
    generated_wave_fields = inference.generate_wave_field(buoy_data)
    all_generated_wave_fields.append(generated_wave_fields)

# 将所有生成的波场拼接为一个张量
all_generated_wave_fields = torch.cat(all_generated_wave_fields, dim=0)
print("所有生成波场的形状:", all_generated_wave_fields.shape)
generated_wave_fields = all_generated_wave_fields.cpu().numpy()
np.save('/home/hy4080/met_waves/data/generated_wave_fields.npy', generated_wave_fields)
print("数据已成功保存为.npy文件")
