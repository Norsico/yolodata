import os
import requests
import zipfile
import shutil
from tqdm import tqdm

# === 1. 设置下载链接 ===
# 根据你提供的地址，推测有两个分卷文件 (.001 和 .002)
DATASET_URLS = [
    "https://github.com/Norsico/yolodata/releases/download/0.1.0/SODA10M_82_10k.zip.001",
    "https://github.com/Norsico/yolodata/releases/download/0.1.0/SODA10M_82_10k.zip.002"
]

# === 2. 路径配置 ===
# 合并后的完整压缩包存放路径
MERGED_ZIP_FILE = "/workspace/SODA10M_82_10k.zip"
# 解压后的数据集根目录
DATASET_DIR = "/workspace/datasets/SODA10M_82_10k" 

def download_file(url, filename):
    """使用 requests + tqdm 实现带进度条的下载"""
    print(f"⬇️ 正在下载分卷: {os.path.basename(filename)}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024 # 1 MB
    
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(filename))
    
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    
    if total_size != 0 and progress_bar.n != total_size:
        print("⚠️ 警告：下载可能不完整")
    else:
        print("✅ 分卷下载完成")

def merge_files(part_files, output_file):
    """将多个分片文件二进制合并为一个大文件"""
    print(f"🔗 正在合并 {len(part_files)} 个分片到 {output_file} ...")
    
    with open(output_file, 'wb') as outfile:
        for part in part_files:
            print(f"   + 读取分片并写入: {part}")
            with open(part, 'rb') as infile:
                shutil.copyfileobj(infile, outfile)
    
    print("✅ 合并完成！")

def fix_nested_dir(target_dir):
    """
    检查是否存在双层嵌套 (例如 target_dir/SODA10M_82_10k/images)，
    如果存在，将内部文件移动到 target_dir 并删除多余层级。
    """
    folder_name = os.path.basename(target_dir) # SODA10M_82_10k
    # 猜测解压后可能多了一层同名文件夹
    nested_path = os.path.join(target_dir, folder_name) 
    
    if os.path.exists(nested_path) and os.path.isdir(nested_path):
        print(f"⚠️ 检测到多层嵌套: {nested_path}")
        print("🔧 正在修正目录结构...")
        
        # 移动文件
        for item in os.listdir(nested_path):
            src = os.path.join(nested_path, item)
            dst = os.path.join(target_dir, item)
            shutil.move(src, dst)
            
        # 删除空文件夹
        os.rmdir(nested_path)
        print("✅ 目录结构修正完成！")
    else:
        pass

# ================= 主流程 =================

if not os.path.exists(DATASET_DIR):
    # --- A. 下载与合并逻辑 ---
    part_files = []
    
    # 如果还没有合并好的大包，就开始下载分卷
    if not os.path.exists(MERGED_ZIP_FILE):
        print("🚀 开始处理数据集下载任务...")
        
        # 1. 下载每个分卷
        for index, url in enumerate(DATASET_URLS):
            # 临时文件名，例如 /workspace/SODA10M_82_10k.zip.001
            part_name = f"/workspace/temp_SODA_part_{index+1:03d}" 
            part_files.append(part_name)
            
            if not os.path.exists(part_name):
                download_file(url, part_name)
            else:
                print(f"✅ 分卷 {os.path.basename(part_name)} 已存在，跳过下载")
        
        # 2. 合并分卷
        merge_files(part_files, MERGED_ZIP_FILE)
        
        # 3. 删除临时分卷释放空间
        print("🗑️ 删除临时分卷文件...")
        for part in part_files:
            if os.path.exists(part):
                os.remove(part)
    else:
        print(f"✅ 完整压缩包 {MERGED_ZIP_FILE} 已存在，跳过下载和合并")

    # --- B. 解压逻辑 ---
    print(f"📦 正在解压数据集到 {DATASET_DIR} ...")
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    try:
        with zipfile.ZipFile(MERGED_ZIP_FILE, 'r') as zip_ref:
            # 这里的解压可能会比较慢，3GB建议耐心等待
            for member in tqdm(zip_ref.infolist(), desc="正在解压"):
                zip_ref.extract(member, DATASET_DIR)
        print(f"✅ 解压完成")
        
        # --- C. 目录修正与清理 ---
        fix_nested_dir(DATASET_DIR)

        print(f"🗑️ 正在删除合并后的压缩包以释放空间: {MERGED_ZIP_FILE}")
        os.remove(MERGED_ZIP_FILE)
        print("✅ 空间已清理，数据集准备就绪！")

    except zipfile.BadZipFile:
        print("❌ 错误：压缩包损坏！可能是下载不完整或合并顺序错误。")
        # 如果出错，尝试删除坏包，方便重试
        if os.path.exists(MERGED_ZIP_FILE):
            os.remove(MERGED_ZIP_FILE)

else:
    print(f"✅ 数据集目录 {DATASET_DIR} 已存在，无需操作")
    # 再次检查目录结构，确保万无一失
    fix_nested_dir(DATASET_DIR)