import os

# 定义 depthm 目录的路径
base_dir = '/path/to/depthm'  # 请替换为实际的路径

# 遍历 depth0-depth23 文件夹
for i in range(24):
    folder_name = f"depth{i}"
    file_path = os.path.join(base_dir, folder_name, 'main_4080.py')

    # 检查文件是否存在
    if os.path.exists(file_path):
        # 读取文件内容
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # 修改 dynamic_depth 和 full_path 的值
        with open(file_path, 'w') as file:
            for line in lines:
                if 'dynamic_depth' in line:
                    # 修改 dynamic_depth 为当前文件夹的值
                    line = f"dynamic_depth = {i}\n"
                elif 'full_path =' in line:
                    # 修改 full_path 中的 depth 值
                    line = f'full_path = "/home/hy4080/wplyh/depthm/depth{i}/temp"\n'
                file.write(line)

        print(f"已修改 {file_path} 中的 dynamic_depth 和 full_path 值为 {i}")
    else:
        print(f"{file_path} 不存在")

