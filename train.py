import os
from ultralytics import YOLO

# ================= 配置区域 =================
# 1. 模型设置
# 权重微调: 'yolo11n.pt'
# 从头训练: 写 'ultralytics/cfg/models/11/yolo11-custom.yaml'
MODEL_CFG = r'G:\ultralytics\ultralytics\cfg\models\11\yolo11n-4head.yaml' 

# 2. 数据集设置
DATA_PATH = r'G:\ultralytics\datasets\SODA10M_82_1k\soda10m.yaml'

# 3. 训练参数
EPOCHS = 100            # 训练总轮数
BATCH_SIZE = 8         # 批次大小
IMG_SIZE = 640          # 图片大小
DEVICE = '0'            # 显卡编号
WORKERS = 6             # 数据加载线程数

# 4. ⭐ 保存路径设置
PROJECT_DIR = 'runs/train'  # 项目总目录
EXP_NAME = 'scratch_yolo11n_soda10m_1k_4head'  # 实验名称
# =======================================================

def main():
    # 构造上次训练的断点路径
    last_ckpt_path = os.path.join(PROJECT_DIR, EXP_NAME, 'weights', 'last.pt')
    
    # === 自动断点续训逻辑 ===
    if os.path.exists(last_ckpt_path):
        print(f"✅ 检测到上次未完成的训练，正在从断点恢复: {last_ckpt_path}")
        # 加载断点模型
        model = YOLO(last_ckpt_path)
        resume_training = True
    else:
        print(f"🆕 未检测到断点，开始新的训练: {MODEL_CFG}")
        # 加载新模型 (可以是 .pt 也可以是 .yaml)
        model = YOLO(MODEL_CFG)
        resume_training = False

    # === 开始训练 ===
    # 注意：如果是 resume=True，很多参数会直接沿用上次的设置，这里再次指定是为了保险
    model.train(
        data=DATA_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        project=PROJECT_DIR,
        name=EXP_NAME,
        resume=resume_training, # 关键参数：是否续训
        exist_ok=True,          # 允许覆盖同名文件夹(配合resume使用)
        cache=True              # 缓存图片到内存，加速训练
    )
    
    print(f"\n🎉 训练结束！最佳模型保存在: {os.path.join(PROJECT_DIR, EXP_NAME, 'weights', 'best.pt')}")

if __name__ == '__main__':
    main()
