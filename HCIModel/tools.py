# import os
#
# # 获取当前脚本所在目录
# current_dir = os.path.dirname(os.path.abspath(__file__))
#
# # data文件夹路径
# data_dir = os.path.join(current_dir, 'data')
#
# # dt1和da2文件夹路径
# dt1_dir = os.path.join(data_dir, 'dt1')
# da2_dir = os.path.join(data_dir, 'da2')
#
# # 在dt1中创建mode1到mode8文件夹
# for i in range(1, 9):
#     mode_dir = os.path.join(dt1_dir, f'mode{i}')
#     os.makedirs(mode_dir, exist_ok=True)  # 如果文件夹不存在则创建
#
# # 在da2中创建mode1到mode6文件夹
# for i in range(1, 7):
#     mode_dir = os.path.join(da2_dir, f'mode{i}')
#     os.makedirs(mode_dir, exist_ok=True)  # 如果文件夹不存在则创建
#
# print("文件夹创建完毕！")
import os


def print_directory_tree(root_path, indent=""):
    # 遍历指定目录的内容
    items = os.listdir(root_path)
    items.sort()  # 可选，按名称排序输出

    for idx, item in enumerate(items):
        item_path = os.path.join(root_path, item)
        is_last = (idx == len(items) - 1)

        # 根据当前级别输出文件夹或文件的前缀
        prefix = "└── " if is_last else "├── "

        # 打印文件夹或文件
        print(indent + prefix + item)

        # 如果是目录，递归调用
        if os.path.isdir(item_path):
            sub_indent = "    " if is_last else "│   "
            print_directory_tree(item_path, indent + sub_indent)


# 调用函数，指定根目录
root_directory = "./"  # 将此路径更改为你的根目录
print_directory_tree(root_directory)
