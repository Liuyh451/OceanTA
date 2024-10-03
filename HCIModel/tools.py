import os

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# data文件夹路径
data_dir = os.path.join(current_dir, 'data')

# dt1和da2文件夹路径
dt1_dir = os.path.join(data_dir, 'dt1')
da2_dir = os.path.join(data_dir, 'da2')

# 在dt1中创建mode1到mode8文件夹
for i in range(1, 9):
    mode_dir = os.path.join(dt1_dir, f'mode{i}')
    os.makedirs(mode_dir, exist_ok=True)  # 如果文件夹不存在则创建

# 在da2中创建mode1到mode6文件夹
for i in range(1, 7):
    mode_dir = os.path.join(da2_dir, f'mode{i}')
    os.makedirs(mode_dir, exist_ok=True)  # 如果文件夹不存在则创建

print("文件夹创建完毕！")
