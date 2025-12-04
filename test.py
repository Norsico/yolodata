import torch
import sys
from ultralytics import checks, YOLO

print("="*30)
print("1. 正在运行 Ultralytics 官方环境检查...")
# 这行代码会自动打印 OS, Python, PyTorch, CUDA 等信息
# 如果你是源码安装，它会显示 Git 的相关信息
checks()

print("\n" + "="*30)
print("2. 深度检查 PyTorch GPU 状态...")
if torch.cuda.is_available():
    print(f"✅ CUDA is available! (GPU可用)")
    print(f"   GPU Count: {torch.cuda.device_count()}")
    print(f"   Current Device: {torch.cuda.current_device()}")
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print(f"❌ CUDA is NOT available. (只能用CPU)")

print("\n" + "="*30)
print("3. 测试模型加载 (验证源码修改是否生效)...")
try:
    # 加载一个 nano 模型测试一下
    # 注意：如果你修改了 yolo11n.yaml 结构，这里应该加载 yaml 而不是 pt
    model = YOLO('yolo11n.pt') 
    print(f"✅ 模型加载成功！")
    
    # 打印一下模型会自动分配到哪个设备
    print(f"   模型默认分配设备: {model.device}")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")

print("="*30)
