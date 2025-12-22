import os
import shutil
import zipfile
import subprocess
import time

# === 1. 设置下载链接 ===
ORIGINAL_URLS = [
    'https://github.com/Norsico/yolodata/releases/download/0.2.0/SODA10M_82_6k.zip'
]

# === 2. 策略调整：优先用原始链接（求稳），其次才是镜像 ===
MIRRORS = [
    "",                           # <--- 空字符串代表使用原始 GitHub 链接 (最稳)
    "https://mirror.ghproxy.com/",# <--- 备用镜像1
    "https://ghproxy.net/"        # <--- 备用镜像2
]

# === 3. 路径配置 ===
MERGED_ZIP_FILE = "/workspace/SODA10M_82_10k.zip"
DATASET_DIR = "/workspace/datasets/SODA10M_82_10k" 

# === 4. 是否启用分卷合并 ===
USE_SPLIT_FILES = False  # 设置为 False 表示不进行分卷合并

def install_aria2():
    if shutil.which("aria2c") is None:
        try:
            subprocess.run(["apt-get", "update"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["apt-get", "install", "-y", "aria2"], check=True)
        except:
            pass

def download_with_aria2(url, filename, proxy_prefix):
    final_url = proxy_prefix + url
    
    # 打印当前尝试的策略
    source_name = "GitHub原源" if proxy_prefix == "" else proxy_prefix
    print(f"   [Aria2] 正在尝试: {source_name}")
    print(f"           (地址: {final_url})")
    
    cmd = [
        "aria2c", 
        "-c",                       # 断点续传
        "-x", "4",                  # <--- 降到 4 线程，防止被封 IP
        "-s", "4", 
        "-k", "1M", 
        "--max-tries=0",            # 无限重试
        "--retry-wait=2",           # 重试等待
        "--lowest-speed-limit=1K",  # 只要有速度就不杀
        "--connect-timeout=10",     # 连接超时
        "--check-certificate=false",
        "--console-log-level=warn", # 依然会显示警告，别怕
        "--summary-interval=5",     # 5秒刷新一次进度
        "--dir", os.path.dirname(filename), 
        "-o", os.path.basename(filename),
        final_url
    ]
    subprocess.run(cmd, check=True)

def smart_download(url, filename):
    print(f"⬇️ 检查/下载文件: {os.path.basename(filename)}")
    
    # 死循环模式：只要没下完，就一直换源重试
    attempt = 0
    while True:
        for proxy in MIRRORS:
            try:
                download_with_aria2(url, filename, proxy)
                print("✅ 成功完成！")
                return
            except subprocess.CalledProcessError:
                attempt += 1
                print(f"⚠️ 当前线路不稳定，自动切换... (第 {attempt} 次重试)")
                time.sleep(2)
            except KeyboardInterrupt:
                print("\n🛑 用户手动停止")
                exit()

def merge_files(part_files, output_file):
    print(f"🔗 正在合并 {len(part_files)} 个分片...")
    with open(output_file, 'wb') as outfile:
        for part in part_files:
            print(f"   + 合并: {os.path.basename(part)}")
            with open(part, 'rb') as infile:
                shutil.copyfileobj(infile, outfile)
    print("✅ 合并完成")

def fix_nested_dir(target_dir):
    folder_name = os.path.basename(target_dir)
    nested_path = os.path.join(target_dir, folder_name)
    if os.path.exists(nested_path) and os.path.isdir(nested_path):
        for item in os.listdir(nested_path):
            shutil.move(os.path.join(nested_path, item), os.path.join(target_dir, item))
        os.rmdir(nested_path)

# ================= 主流程 =================

if __name__ == "__main__":
    install_aria2()

    if not os.path.exists(DATASET_DIR):
        part_files = []
        
        if not os.path.exists(MERGED_ZIP_FILE):
            for index, url in enumerate(ORIGINAL_URLS):
                part_name = f"/workspace/temp_part_{index+1:03d}"
                part_files.append(part_name)
                # 这一步会卡住直到下载完成
                smart_download(url, part_name)
            
            if USE_SPLIT_FILES and len(ORIGINAL_URLS) > 1:
                # 合并分卷文件
                merge_files(part_files, MERGED_ZIP_FILE)
                
                # 清理临时分片
                for part in part_files:
                    if os.path.exists(part): os.remove(part)
                    if os.path.exists(part+".aria2"): os.remove(part+".aria2")
            
            # 如果只有一个文件，不需要合并分卷
            elif len(ORIGINAL_URLS) == 1:
                print(f"⚠️ 只有一个文件，跳过合并分卷。")
                shutil.move(part_files[0], MERGED_ZIP_FILE)
            
        # 解压
        print(f"📦 正在解压...") 
        os.makedirs(DATASET_DIR, exist_ok=True)
        try:
            with zipfile.ZipFile(MERGED_ZIP_FILE, 'r') as z:
                z.extractall(DATASET_DIR)
            fix_nested_dir(DATASET_DIR)
            os.remove(MERGED_ZIP_FILE)
            print("🎉🎉🎉 恭喜！终于搞定了！")
        except Exception as e:
            print(f"❌ 解压出错: {e}")
    else:
        print(f"✅ 目录已存在: {DATASET_DIR}")
        fix_nested_dir(DATASET_DIR)
