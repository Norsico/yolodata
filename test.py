import os
import zipfile
import math
from tqdm import tqdm

# === 配置 ===
SOURCE_DIR = r"G:\ultralytics\datasets\SODA10M_82_10k"  # 你的数据集路径
OUTPUT_NAME = "SODA10M_82_10k.zip"                      # 临时大文件名
CHUNK_SIZE = 1500 * 1024 * 1024                         # 切片大小：1500MB (<2GB)

def zip_folder(source_dir, output_filename):
    """将文件夹压缩为标准zip文件"""
    print(f"📦 正在压缩文件夹: {source_dir}...")
    parent_folder = os.path.dirname(source_dir)
    contents = os.walk(source_dir)
    
    # 计算文件总数用于进度条
    total_files = sum([len(files) for r, d, files in os.walk(source_dir)])
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        with tqdm(total=total_files, desc="压缩进度", unit="file") as pbar:
            for root, dirs, files in contents:
                for file in files:
                    abs_path = os.path.join(root, file)
                    # 保持相对路径，不包含盘符等
                    rel_path = os.path.relpath(abs_path, os.path.dirname(source_dir))
                    zipf.write(abs_path, rel_path)
                    pbar.update(1)
    print("✅ 压缩完成！")

def split_file(file_path, chunk_size):
    """将大文件二进制切割为多个小文件"""
    print(f"✂️ 正在切割文件: {file_path}...")
    file_size = os.path.getsize(file_path)
    chunks = math.ceil(file_size / chunk_size)
    
    with open(file_path, 'rb') as f:
        for i in range(chunks):
            chunk_name = f"{file_path}.{i+1:03d}"  # 例如 .zip.001
            print(f"   -> 生成分片: {chunk_name}")
            
            with open(chunk_name, 'wb') as chunk_f:
                # 读取并写入数据，防止内存溢出，分块读取
                bytes_written = 0
                while bytes_written < chunk_size:
                    # 每次读 64MB
                    read_size = min(64 * 1024 * 1024, chunk_size - bytes_written)
                    data = f.read(read_size)
                    if not data:
                        break
                    chunk_f.write(data)
                    bytes_written += len(data)
    
    print(f"✅ 切割完成！共生成 {chunks} 个分片。")
    print("🚀 请将生成的 .zip.001, .zip.002 等文件上传到 GitHub Release。")

if __name__ == "__main__":
    # 1. 先压缩
    zip_folder(SOURCE_DIR, OUTPUT_NAME)
    
    # 2. 再切割
    split_file(OUTPUT_NAME, CHUNK_SIZE)
    
    # 3. 删除原始的大zip文件（可选，保留分片即可）
    # os.remove(OUTPUT_NAME) 
    print("🎉 本地处理结束。")