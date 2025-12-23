import os
import zipfile
from tqdm import tqdm

# === 1. 设置解压文件路径 ===
ZIP_FILE = '/root/yolodata/SODA10M_82_6k.zip'  # ZIP 文件路径
EXTRACT_DIR = '/root/yolodata/datasets'  # 解压到目标目录

# === 2. 解压文件 ===
def extract_zip(zip_file, extract_dir):
    print(f"📦 正在解压 {zip_file} 到 {extract_dir}...")
    
    # 确保目标解压目录存在
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
    
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # 获取文件列表
            file_list = zip_ref.namelist()
            # 使用 tqdm 显示进度条
            for file in tqdm(file_list, desc="解压中", unit="file"):
                zip_ref.extract(file, extract_dir)
        print("✅ 解压成功！")
    except Exception as e:
        print(f"❌ 解压失败: {e}")

# ================= 执行解压 =================
if __name__ == "__main__":
    # 检查文件是否存在
    if os.path.exists(ZIP_FILE):
        extract_zip(ZIP_FILE, EXTRACT_DIR)
    else:
        print(f"❌ 文件 {ZIP_FILE} 不存在，请检查路径是否正确。")
