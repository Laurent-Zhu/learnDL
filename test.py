import torch
import os

print("系统环境变量:")
print("CUDA_VISIBLE_DEVICES =", os.environ.get('CUDA_VISIBLE_DEVICES'))
print("CUDA_HOME =", os.environ.get('CUDA_HOME'))

print("\nPyTorch 信息:")
print("PyTorch 版本:", torch.__version__)
print("CUDA 是否可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("当前设备:", torch.cuda.current_device())
    print("设备名称:", torch.cuda.get_device_name(0))