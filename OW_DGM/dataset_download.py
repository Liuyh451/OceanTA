import os
import sys

import requests
from bs4 import BeautifulSoup
import time

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
def main():
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




# if __name__ == "__main__":
#     main()
