import time
import requests
import os
from bs4 import BeautifulSoup
import re
import sys

# 初始页面 URL 和下载目录
base_url = "https://thredds.met.no/thredds/catalog/obs/buoy-svv-e39/catalog.html"
download_dir = "E:/Dataset/met_waves"


# 下载文件函数
def download_file(url, save_path):
    response = requests.get(url, stream=True)
    with open(save_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Downloaded: {save_path}")


# 获取年份目录的页面
def get_year_urls():
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, "html.parser")
    year_urls = []

    # 查找页面中包含年份的目录链接
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and "/catalog.html" in href:
            year_str = href.split("/")[0]
            if year_str.isdigit() and 2017 <= int(year_str) <= 2020:
                year_urls.append(f"https://thredds.met.no/thredds/catalog/obs/buoy-svv-e39/{year_str}/catalog.html")

    return year_urls


# 获取每月数据页面的 URL 列表
def get_month_urls(year_url):
    response = requests.get(year_url)
    soup = BeautifulSoup(response.text, "html.parser")
    month_urls = []

    # 查找页面中的月份目录链接
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and "/catalog.html" in href:
            month_urls.append(year_url.replace("catalog.html", href))

    return month_urls


# 获取Sulafjorden_wave.nc文件的下载链接
def get_nc_file_urls(month_url):
    response = requests.get(month_url)
    soup = BeautifulSoup(response.text, "html.parser")
    file_urls = []

    # 查找所有Sulafjorden_wave.nc文件
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and "Sulafjorden_wave.nc" in href:
            dataset_url = month_url.replace("catalog.html", href)
            dataset_page = requests.get(dataset_url)
            dataset_soup = BeautifulSoup(dataset_page.text, "html.parser")

            # 找到实际的下载链接
            for a in dataset_soup.find_all("a"):
                file_href = a.get("href")
                if file_href and "fileServer" in file_href:
                    file_urls.append(f"https://thredds.met.no{file_href}")

    return file_urls


def get_nc_file_urls_1(month_url):
    """
    Args:
        month_url: 获取Breisundet_wave.nc文件链接

    Returns:
        file_urls: 文件链接
    """
    response = requests.get(month_url)
    soup = BeautifulSoup(response.text, "html.parser")
    file_urls = []

    # 查找所有Sulafjorden_wave.nc文件
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and "Breisundet_wave.nc" in href:
            dataset_url = month_url.replace("catalog.html", href)
            dataset_page = requests.get(dataset_url)
            dataset_soup = BeautifulSoup(dataset_page.text, "html.parser")

            # 找到实际的下载链接
            for a in dataset_soup.find_all("a"):
                file_href = a.get("href")
                if file_href and "fileServer" in file_href:
                    file_urls.append(f"https://thredds.met.no{file_href}")

    return file_urls


def get_nc_file_urls_2(month_url):
    """
    Args:
        month_url: 获取Vartdalsfjorden_wave.nc文件链接

    Returns:
        file_urls: 文件链接
    """
    response = requests.get(month_url)
    soup = BeautifulSoup(response.text, "html.parser")
    file_urls = []

    # 查找所有Sulafjorden_wave.nc文件
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and "Vartdalsfjorden_wave.nc" in href:
            dataset_url = month_url.replace("catalog.html", href)
            dataset_page = requests.get(dataset_url)
            dataset_soup = BeautifulSoup(dataset_page.text, "html.parser")

            # 找到实际的下载链接
            for a in dataset_soup.find_all("a"):
                file_href = a.get("href")
                if file_href and "fileServer" in file_href:
                    file_urls.append(f"https://thredds.met.no{file_href}")

    return file_urls


# 主下载逻辑
def main_buoy():
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    # 获取年份页面列表
    year_urls = get_year_urls()
    for year_url in year_urls:
        # 获取每月页面列表
        month_urls = get_month_urls(year_url)
        for month_url in month_urls:
            # 获取.nc文件的实际下载链接
            file_urls = get_nc_file_urls_2(month_url)
            for file_url in file_urls:
                file_name = os.path.join(download_dir, os.path.basename(file_url))

                # 检查文件是否已下载
                if not os.path.exists(file_name):
                    print(file_url)
                    download_file(file_url, file_name)
                    time.sleep(5)  # 等待2秒再下载下一个文件，避免过多请求
                else:
                    print(f"File already exists: {file_name}")


# ---------------------------------------------------------------------------------
# 下面是下载swan数据的函数

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


def download_file_swan(url, save_path):
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
