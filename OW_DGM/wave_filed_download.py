import requests
import os
from bs4 import BeautifulSoup
import re
import sys


def print_progress_bar(iteration, total, length=50):
    """
    手动实现一个进度条
    :param iteration: 当前进度
    :param total: 总进度
    :param length: 进度条的长度
    """
    percent = (iteration / total) * 100
    bar_length = int(length * iteration // total)
    bar = '=' * bar_length + '-' * (length - bar_length)

    # 打印进度条
    sys.stdout.write(f'\r[{bar}] {percent:.2f}%')
    sys.stdout.flush()


def download_file(url, save_path):
    """从 URL 下载文件并保存到指定路径"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))  # 获取文件总大小
    with open(save_path, 'wb') as f:
        for data in response.iter_content(chunk_size=1024):  # 分块下载
            f.write(data)
            print_progress_bar(f.tell(), total_size)  # 更新进度条
    print()  # 换行


def get_nc_files(base_url, start_year, end_year):
    """获取页面中的 .nc 文件链接"""
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # 正则表达式，用于提取文件名中的年份
    year_pattern = re.compile(r'swanSula(\d{6})\.nc')

    nc_files = []
    for link in soup.find_all('a', href=True):
        file_name = link['href']
        match = year_pattern.search(file_name)  # 搜索文件名中的年份
        if match:
            file_year = match.group(1)[:4]  # 提取年份部分
            if start_year <= file_year <= end_year:  # 判断是否在指定的年份范围内
                nc_files.append(base_url + file_name)
    return nc_files


def main():
    base_url = "https://thredds.met.no/thredds/catalog/e39_models/SWAN250/Sula/catalog.html"
    save_dir = "E:/Dataset/met_waves/swan"  # 文件保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 获取需要下载的.nc文件
    nc_files = get_nc_files(base_url, '201701', '201906')

    # 下载每个文件并显示进度
    total_files = len(nc_files)
    for i, file_url in enumerate(nc_files):
        file_name = file_url.split("/")[-1]  # 获取文件名
        save_path = os.path.join(save_dir, file_name)

        print(f"Downloading {file_name} ({i + 1}/{total_files})")
        download_file(file_url, save_path)
        print(f"Finished downloading {file_name}")


if __name__ == "__main__":
    main()
