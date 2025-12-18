import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm  # 如果没有安装tqdm，可以注释掉相关行，改用普通print

# ================= 配置区域 =================
# 源数据集根目录
SRC_ROOT = Path(r"G:\ultralytics\datasets\SODA10M_82_10k")

# 目标数据集根目录
DST_ROOT = Path(r"G:\ultralytics\datasets\SODA10M_28_10k")

# 目标总数量
TARGET_TOTAL = 10000

# 训练集占比
#  (0.8 即 8:2)
TRAIN_RATIO = 0.2

# 支持的图片格式
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
# ===========================================

def main():
    print(f"--- 开始处理数据集 ---")
    print(f"源路径: {SRC_ROOT}")
    print(f"目标路径: {DST_ROOT}")

    # 1. 收集源数据集中所有的图片和标签对 (混合原train和val以保证随机性)
    all_pairs = []
    
    # 遍历源数据集的 train 和 val 文件夹
    for split in ['train', 'val']:
        img_dir = SRC_ROOT / 'images' / split
        lbl_dir = SRC_ROOT / 'labels' / split
        
        if not img_dir.exists() or not lbl_dir.exists():
            print(f"警告: 目录不存在 {img_dir} 或 {lbl_dir}，跳过。")
            continue

        print(f"正在扫描: {img_dir} ...")
        
        # 获取该目录下的文件
        files = os.listdir(img_dir)
        for f in files:
            file_path = Path(f)
            if file_path.suffix.lower() in IMG_EXTENSIONS:
                src_img_path = img_dir / f
                # 寻找对应的标签文件 (假设文件名相同，后缀为.txt)
                src_lbl_path = lbl_dir / f"{file_path.stem}.txt"
                
                if src_lbl_path.exists():
                    all_pairs.append((src_img_path, src_lbl_path))
    
    total_found = len(all_pairs)
    print(f"共找到有效数据对: {total_found} 组")

    if total_found < TARGET_TOTAL:
        print(f"错误: 源数据数量 ({total_found}) 小于目标数量 ({TARGET_TOTAL})，无法抽取。")
        return

    # 2. 随机打乱并抽取指定数量
    print(f"正在随机抽取 {TARGET_TOTAL} 组数据...")
    random.seed(42)  # 设置随机种子，保证每次运行结果一致（可选）
    random.shuffle(all_pairs)
    selected_pairs = all_pairs[:TARGET_TOTAL]

    # 3. 计算划分数量
    num_train = int(TARGET_TOTAL * TRAIN_RATIO)
    num_val = TARGET_TOTAL - num_train
    
    train_set = selected_pairs[:num_train]
    val_set = selected_pairs[num_train:]
    
    print(f"划分结果 -> Train: {len(train_set)}, Val: {len(val_set)}")

    # 4. 执行复制操作
    def copy_files(data_pairs, split_name):
        # 创建目标目录
        target_img_dir = DST_ROOT / 'images' / split_name
        target_lbl_dir = DST_ROOT / 'labels' / split_name
        
        target_img_dir.mkdir(parents=True, exist_ok=True)
        target_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"正在复制数据到 {split_name} 集...")
        
        # 使用 tqdm 显示进度条 (如果没有安装 tqdm，请将下行替换为普通 for 循环)
        # for img_src, lbl_src in data_pairs: 
        for img_src, lbl_src in tqdm(data_pairs, desc=f"Copying {split_name}"):
            shutil.copy2(img_src, target_img_dir / img_src.name)
            shutil.copy2(lbl_src, target_lbl_dir / lbl_src.name)

    copy_files(train_set, 'train')
    copy_files(val_set, 'val')

    # 5. 生成 yaml 配置文件建议
    print("\n--- 处理完成 ---")
    print(f"数据已保存至: {DST_ROOT}")
    print("\n[建议] 请检查你的 dataset.yaml 文件，将路径更新为：")
    print(f"path: {DST_ROOT}")
    print("train: images/train")
    print("val: images/val")

if __name__ == "__main__":
    main()